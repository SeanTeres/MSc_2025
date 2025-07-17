import sys
import os
import gc
from typing import List
# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

from datasets.hdf_dataset import HDF5Dataset
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datasets.dataloader import get_dataloaders
import torchxrayvision as xrv
import torch
from train_utils import classes, helpers
import torch.nn.functional as F
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score
import seaborn as sns
from sklearn.calibration import calibration_curve
import io
import torchvision.transforms as transforms
import os
from tsne import visualize_tsne
import math
import random
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses, miners

import torch
from torch import device, nn, Tensor
import torch.nn.functional as F
from clf_manager import BinaryClassifier, MulticlassClassifier, XRVBasedClassifier
# Add after the existing imports at the top
import cl_metrics


def prepare_ilo_data(ilo_dataset, device):
    """Prepare ILO dataset for efficient access"""
    ilo_images = []
    ilo_labels = []

    for idx in range(len(ilo_dataset)):
        image, label = ilo_dataset[idx]
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).to(device)
        label_tensor = torch.tensor(label, dtype=torch.long).to(device)
        ilo_images.append(image_tensor)
        ilo_labels.append(label_tensor)

    return torch.cat(ilo_images, dim=0), torch.stack(ilo_labels)

class QuadrupletMarginLoss(nn.modules.loss._Loss):
    __constants__ = ['margin1', 'margin2', 'p', 'eps', 'swap', 'reduction']
    margin1: float
    margin2: float
    p: float
    eps: float
    swap: bool

    def __init__(self,
                 margin1: float = 1.0,
                 margin2: float = 1.0,
                 p: float = 2.,
                 eps: float = 1e-6,
                 swap: bool = False,
                 type: str = 'DoubleTriplet',  # 'DoubleTriplet' or 'Original'
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 mask_ilo_tb: bool = False):
        super().__init__(size_average, reduce, reduction)
        self.margin1 = margin1
        self.margin2 = margin2
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction
        assert type in ['DoubleTriplet', 'Original', 'TieredNegativeRanking', 'RelativeNegativeRanking']
        self.type = type
        self.mask_ilo_tb = mask_ilo_tb

    def forward(self, anchor: Tensor, positive: Tensor, negative1: Tensor, negative2: Tensor, 
                is_ilo_anchor: Tensor = None) -> Tensor:
        
        d_ap = torch.norm(anchor - positive, p=self.p, dim=1)
        d_an1 = torch.norm(anchor - negative1, p=self.p, dim=1)
        d_an2 = torch.norm(anchor - negative2, p=self.p, dim=1)
        d_n1n2 = torch.norm(negative1 - negative2, p=self.p, dim=1)
        
        # First term: standard triplet
        loss1 = F.triplet_margin_loss(anchor, positive, negative1,
                                      margin=self.margin1,
                                      p=self.p,
                                      eps=self.eps,
                                      swap=self.swap,
                                      reduction='none')

        if self.type == 'DoubleTriplet':
            # Second term: second negative pushed from anchor
            loss2 = F.triplet_margin_loss(anchor, positive, negative2,
                                          margin=self.margin2,
                                          p=self.p,
                                          eps=self.eps,
                                          swap=self.swap,
                                          reduction='none')
            
            # Apply masking if is_ilo_anchor provided
            if is_ilo_anchor is not None and self.mask_ilo_tb:
                # Zero out loss2 for ILO anchors (memory efficient - no additional tensors created)
                mask = (~is_ilo_anchor).float()
                loss2 = loss2 * mask


        elif self.type == 'Original':
            # Second term: negative2 pushed from negative1
            # Use D(N1, N2) instead of D(A, N2)
            d_ap = torch.norm(anchor - positive, p=self.p, dim=1)
            d_n1n2 = torch.norm(negative1 - negative2, p=self.p, dim=1)
            loss2 = torch.clamp(d_ap - d_n1n2 + self.margin2, min=0.0)

        elif self.type == 'TieredNegativeRanking':
        # First term: standard triplet loss (anchor-positive vs anchor-negative1)
            loss1 = torch.clamp(d_ap - d_an1 + self.margin1, min=0.0)
            
            # Second term: negative2 vs negative1 from anchor perspective
            loss2 = torch.clamp(d_an2 - d_an1 + self.margin2, min=0.0)
        
            
        elif self.type == 'RelativeNegativeRanking':
        # First term: standard triplet loss (anchor-positive vs anchor-negative1)
            loss1 = torch.clamp(d_ap - d_an2 + self.margin1, min=0.0)
            
            # Second term: negative2 vs negative1 from anchor perspective
            loss2 = torch.clamp(d_an2 - d_an1 + self.margin2, min=0.0)
      

        # Combine terms and apply reduction
        loss = loss1 + loss2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def build_label_to_indices_map(dataset):
    """
    Create a dictionary mapping each label to all indices with that label in the dataset.
    
    Args:
        dataset: PyTorch dataset with labels accessible via dataset[idx][1]
        
    Returns:
        Dictionary mapping label values to lists of indices
    """
    label_to_indices = {}
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
       
        
    return label_to_indices


def calculate_margin_violations(anchor_embedding, positive_embedding, negative_embedding, margin):
    """
    Calculate if a triplet violates the margin constraint.
    
    A triplet (A,P,N) violates the margin if:
    d(A,P) > d(A,N) - margin
    
    Args:
        anchor_embedding: Embedding of the anchor sample
        positive_embedding: Embedding of the positive sample
        negative_embedding: Embedding of the negative sample
        margin: The margin value to enforce
        
    Returns:
        is_violated: Boolean indicating if the margin is violated
        violation_amount: How much the constraint is violated by (if positive)
    """
    # Calculate distances
    dist_ap = F.pairwise_distance(anchor_embedding, positive_embedding)
    dist_an = F.pairwise_distance(anchor_embedding, negative_embedding)
    
    # Check violation
    violation_amount = dist_ap - (dist_an - margin)
    is_violated = violation_amount > 0
    
    return is_violated.item(), violation_amount.item()

def get_scheduled_p_ilo(current_epoch, initial_p_ilo_anchor, n_epochs, scheduling_fraction):
    """
    Calculate the scheduled probability of using ILO anchors.
    Starts at initial_p_ilo_anchor and decays to p_ilo_final over the first 
    scheduling_fraction portion of training.
    """
    # Calculate what fraction of the scheduling period we've completed
    schedule_end_epoch = n_epochs * scheduling_fraction
    schedule_point = min(1.0, current_epoch / schedule_end_epoch)
    sin_factor = math.sin(schedule_point * math.pi/2)

    # Calculate new probability
    new_p_ilo_anchor = initial_p_ilo_anchor - (initial_p_ilo_anchor - wandb.config.p_ilo_final) * sin_factor
    return new_p_ilo_anchor

def get_sin_scheduled_margin(current_epoch, margin_scheduling, initial_margin, final_margin, n_epochs, scheduling_fraction):
    if not margin_scheduling:
        return initial_margin
    
    schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
    sin_factor = math.sin(schedule_point * math.pi/2)
    
    if initial_margin > final_margin:
        current_margin = initial_margin - (initial_margin - final_margin) * sin_factor
    else:
        current_margin = initial_margin + (final_margin - initial_margin) * sin_factor
        
    return current_margin
    

def train_model_quadruplet(
    model,
    train_loader,
    val_loader,
    triplet_loss_fn,
    quadruplet_loss_fn,
    encoder_optimizer,
    device,
    n_epochs,
    experiment_name,
    ilo_dataset,
    mbod_merged_loader,
    clf_loss_fn=None,
    classifier_optimizer=None,
    use_classification=None,
    checkpoint_dir="checkpoints",
    tsne_interval=50,
    log_to_wandb=True,
    mining_strat="BSHN-v2",
    margin_scheduling=False,
    initial_margin=0.8,
    final_margin=0.2,
    scheduling_fraction=0.8,
    p_ilo_anchor=0.0,
    lambda_clf=0.25,
    active_classifier="multiclass_profusion"
):
    """
    Trains a model using quadruplet loss for multi-label tasks like 
    simultaneous pneumoconiosis and TB classification.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        triplet_loss_fn: Standard triplet loss function for comparison
        quadruplet_loss_fn: Quadruplet loss function
        optimizer: Optimizer for model parameter updates
        device: Device to run training on (cuda/cpu)
        n_epochs: Number of training epochs
        experiment_name: Name for saving checkpoints
        checkpoint_dir: Directory to save checkpoints
        tsne_interval: How often to run t-SNE visualization
        log_to_wandb: Whether to log metrics to wandb
        mining_strat: Mining strategy for finding negatives
        margin_scheduling: Enable/disable margin scheduling
        initial_margin: Initial margin value
        final_margin: Final margin value
        scheduling_fraction: Fraction of training to complete schedule
        
    Returns:
        dict: Dictionary containing trained model, best model state dict,
              training history and best validation metrics
    """
    # Define the mapping for multiclass_stb
    multiclass_stb_mapping = {
        0: "Profusion 0, No TB",
        1: "Profusion 1, No TB",
        2: "Profusion 2, No TB",
        3: "Profusion 3, No TB",
        4: "Profusion 0, With TB",
        5: "Profusion 1, With TB",
        6: "Profusion 2, With TB",
        7: "Profusion 3, With TB",
    }

    ilo_images, ilo_labels = prepare_ilo_data(ilo_dataset, device)  

    # Validate classification setup
    if use_classification:
        if clf_loss_fn is None:
            raise ValueError("clf_loss_fn must be provided when use_classification=True")
        if classifier_optimizer is None:
            raise ValueError("classifier_optimizer must be provided when use_classification=True")
        if (not hasattr(model, 'mc_prof_clf')) and (not hasattr(model, 'bin_prof_clf')):
            raise ValueError("Model must have 'mc_prof_clf' or 'bin_prof_clf' attribute when use_classification=True and active_classifier='binary'")
        print("🎯 Classification training enabled")
    else:
        print("🔄 Pure contrastive learning mode (no classification)")

    # from torch.optim.lr_scheduler import ReduceLROnPlateau
    # clf_scheduler = ReduceLROnPlateau(classifier_optimizer, 'min', factor=0.2, patience=3, verbose=True) # TO DO: Implement this properly (if it works)
    # encoder_scheduler = ReduceLROnPlateau(encoder_optimizer, 'min', factor=0.2, patience=5, verbose=True)


    # Create checkpoint directory
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tracking metrics
    num_classes = 8  # For multiclass_stb
    best_val_map = 0.0
    best_val_prof_bin_specificity = 0.0


    best_model_state = None
    history = {
        'train_loss': [],
        'train_map': [],
        'val_loss': [],
        'val_map': [],
        'train_class_map': [],
        'val_class_map': [],
        'train_prof_map': [],
        'val_prof_map': [],
        'train_quadruplet_loss': [],
        'val_quadruplet_loss': [],
        'train_specificity': [],  # Add these new keys
        'val_specificity': [],
        'train_class_specificity': [],
        'val_class_specificity': [],
        'train_embedding_ratio': [],
        'val_embedding_ratio': []
        
    }
    
    # Initialize class-specific metrics
    per_class_metrics = {class_id: {'train_ap': [], 'val_ap': []} for class_id in range(num_classes)}

    classifier_frozen = True
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("=" * 50)
        # if classifier_frozen and epoch >= 150:
        #     for param in model.mc_prof_clf.parameters():
        #         param.requires_grad = True
        #     classifier_frozen = False
        #     print("🔓 Unfroze classifier for training.")

        # # Freeze classifier for first half
        # if classifier_frozen:
        #     for param in model.mc_prof_clf.parameters():
        #         param.requires_grad = False

        # Apply margin scheduling if enabled
        if margin_scheduling and wandb.config.margin_schedule_scheme == "Epoch-Sine":
            current_margin = get_sin_scheduled_margin(epoch, margin_scheduling, initial_margin, final_margin, n_epochs, scheduling_fraction)
            triplet_loss_fn.margin = current_margin
            quadruplet_loss_fn.margin1 = current_margin
            quadruplet_loss_fn.margin2 = current_margin * wandb.config.beta_factor
            #print(f"Current margins: {current_margin:.4f}, {quadruplet_loss_fn.margin2:.4f}")

        if(wandb.config.p_ilo_scheduling):
            p_ilo_anchor = get_scheduled_p_ilo(epoch, wandb.config.p_ilo_anchor, n_epochs, wandb.config.scheduling_fraction)
        
        model.train()
        epoch_total_loss = 0.0
        epoch_quad_loss = 0.0
        epoch_clf_loss = 0.0

        epoch_batch_count = 0
        epoch_quadruplet_count = 0

        epoch_filtered_count = 0
        epoch_failed_formation_count = 0
        epoch_success_count = 0
        epoch_total_batches = 0

        all_embeddings = []
        all_labels = []

        train_prof_preds = []
        train_prof_labels = []
        
        # Training loop
        for batch_idx, sample in enumerate(train_loader):
                
            batch_filtered_count = 0
            epoch_total_batches += 1

            cpu_labels = sample[1].cpu().numpy()
            # if not can_form_quadruplets(cpu_labels):
            #     batch_filtered_count += 1
            #     epoch_filtered_count += 1

            #     if (batch_idx + 1):
            #         wandb.log({
            #             "batch_filtered_count": batch_filtered_count,
            #             "batch_idx": batch_idx,
            #         })

            #     print(f"Skipping batch {batch_idx + 1} due to insufficient labels for quadruplet formation.")
            #     print(f"Batch labels distribution: {np.bincount(cpu_labels, minlength=8)}")
            #     continue

      
            # Zero gradients for this batch
            encoder_optimizer.zero_grad()
            if classifier_optimizer is not None:
                classifier_optimizer.zero_grad()
            
            imgs = sample[0].to(device)
            labels = sample[1].to(device)
            
            feats = model.features(imgs)
            embeddings = F.normalize(feats, p=2, dim=1)
            
            # Track embeddings and labels for mAP calculation
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())

            current_batch_labels = labels.cpu().numpy()
            
            # Collect quadruplet components
            anchors = []
            positives = []
            negatives = []
            negatives_2 = []
            batch_quadruplet_count = 0
            batch_prof_0_anchors = 0
            batch_prof_pos_anchors = 0
            batch_prof_prioritized_negative_1 = 0
            n_ilo_anchors = 0

            is_ilo_flags = []
            batch_ap_failures = 0
            batch_shn_failures = 0
            batch_n2_failures = 0

            # For each sample in the batch, build a quadruplet
            for i, positive_label in enumerate(current_batch_labels):
                positive_embedding = embeddings[i].unsqueeze(0)

                if np.random.rand() < p_ilo_anchor:                        
                    # Find ILO anchors with the same label as the positive sample
                    ilo_indices = torch.where(ilo_labels == (positive_label % 4))[0]
                    if len(ilo_indices) > 0:
                        # Randomly select an ILO anchor
                        ilo_idx = np.random.choice(ilo_indices.cpu().numpy())
                        anchor_embedding = ilo_images[ilo_idx].unsqueeze(0)  # shape [1, C]
                        anchor_embedding = model.features(anchor_embedding)  # Get features of the anchor
                        anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                        anchor_label = ilo_labels[ilo_idx].item()  # Get the label of the anchor

                        n_ilo_anchors += 1
                        current_is_ilo = True

                        if(anchor_label % 4 > 0):
                            batch_prof_0_anchors += 0
                            batch_prof_pos_anchors += 1
                        else:
                            batch_prof_0_anchors += 1
                            batch_prof_pos_anchors += 0
                else:
                    current_is_ilo = False
                    # Find anchor with same label
                    batch_matching_indices = [j for j in range(len(current_batch_labels)) 
                                            if current_batch_labels[j] == positive_label and j != i]
                    
                    if batch_matching_indices:
                        # Found matching label in batch
                        batch_anchor_idx = np.random.choice(batch_matching_indices)
                        anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                        anchor_label = current_batch_labels[batch_anchor_idx]

                    else:
                        batch_ap_failures += 1
                        continue
                    
                
                # Find first negative using chosen mining strategy
                if mining_strat == "BSHN-v2":
                    # Look for negatives with different profusion score
                    prof_pos_score = positive_label % 4
                    pos_tb_status = 1 if positive_label >= 4 else 0

                    if(wandb.config.n1_selection == "Profusion-based"): # In line with our original approach. Strongly enforce profusion separation.
                        negative_indices = [j for j, label in enumerate(current_batch_labels) 
                                        if label != positive_label and (label % 4) != prof_pos_score]
                        
                    elif wandb.config.n1_selection == "MSTB-based":  # In line with the original quadruplet loss paper. They do not consider co-occurence of labels.
                        negative_indices = [j for j, label in enumerate(current_batch_labels) 
                                        if label != positive_label]
                        
                    

                if negative_indices:
                    negative_embeddings = embeddings[negative_indices]
                    anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)
                    
                    # Compute distances
                    dists = F.pairwise_distance(anchor_repeated, negative_embeddings)
                    positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
                    
                    # Find semi-hard negatives
                    semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + triplet_loss_fn.margin))
                    semi_hard_dists = dists[semi_hard_mask]
                    
                    if semi_hard_dists.numel() > 0:
                        # Use semi-hard negative
                        hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                        semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                        selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()


                        # if wandb.config.prioritize_prof_n1:
                        #     # Prioritize BSHN negatives with highest difference in profusion
                        #     semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                        #     neg_labels = [current_batch_labels[negative_indices[idx.item()]] for idx in semi_hard_indices]
                        #     
                        #     # Calculate profusion difference between each negative and anchor
                        #     anchor_prof = anchor_label % 4
                        #     prof_differences = np.array([abs((label % 4) - anchor_prof) for label in neg_labels])
                        #     
                        #     # Find the negative with highest profusion difference among semi-hard negatives
                        #     if len(prof_differences) > 0:
                        #         max_diff_idx = np.argmax(prof_differences)
                        #         selected_neg_idx = semi_hard_indices[max_diff_idx].item()
                        #     else:
                        #         # Fallback to regular semi-hard selection when no profusion differences
                        #         hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                        #         selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                        #         batch_prof_prioritized_negative_1 += 1

                        negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                        negative_label = current_batch_labels[negative_indices[selected_neg_idx]]

                        if wandb.config.n2_selection == "Proposed-v1":
                            # Find second negative with same profusion but different TB status
                            negative_indices_2 = [j for j, label in enumerate(current_batch_labels) 
                                            if (label % 4) == (negative_label % 4) and 
                                                (int(label >= 4) != int(negative_label >= 4))]
                            
                        elif wandb.config.n2_selection == "Original Paper":
                            # Find second negative with different label to Positive and Negative1
                            negative_indices_2 = [j for j, label in enumerate(current_batch_labels)
                                                if label != positive_label and label != negative_label]

                            
                        if negative_indices_2:
                            neg_2_idx = np.random.choice(negative_indices_2)
                            negative_embedding_2 = embeddings[neg_2_idx].unsqueeze(0)
                            negative_label_2 = current_batch_labels[neg_2_idx]
                            
                            # Valid quadruplet found!
                            anchors.append(anchor_embedding)
                            positives.append(positive_embedding)
                            negatives.append(negative_embedding)
                            negatives_2.append(negative_embedding_2)
                            is_ilo_flags.append(current_is_ilo)
                            batch_quadruplet_count += 1


                            
                            # Log first quadruplet for debugging
                            if batch_idx == 0 and len(anchors) == 1:
                                print(f"Quadruplet example:")
                                print(f"A: {anchor_label}, P: {positive_label}, N1: {negative_label}, N2: {negative_label_2}")
                                print(f"A-Prof: {anchor_label % 4}, P-Prof: {positive_label % 4}, N1-Prof: {negative_label % 4}, N2-Prof: {negative_label_2 % 4}")
                                print(f"A-TB: {anchor_label >= 4}, P-TB: {positive_label >= 4}, N1-TB: {negative_label >= 4}, N2-TB: {negative_label_2 >= 4}")
                        
                        else:
                            batch_n2_failures += 1
                            batch_quadruplet_count += 0
                        
                    else:
                        batch_shn_failures += 1
                        continue
            
            triplet_loss = torch.tensor(0.0, device=device)
            # Process collected quadruplets
            if batch_quadruplet_count > 0:
                
                epoch_success_count += 1
                # Combine all quadruplet components
                batch_anchors = torch.cat(anchors, dim=0)
                batch_positives = torch.cat(positives, dim=0)
                batch_negatives = torch.cat(negatives, dim=0)
                batch_negatives_2 = torch.cat(negatives_2, dim=0)

                is_ilo_anchor = torch.tensor(is_ilo_flags, dtype=torch.bool, device=device)

                if(wandb.config.mask_ilo_tb):
                    # Calculate batch loss using all quadruplets
                    quad_loss = quadruplet_loss_fn(batch_anchors, batch_positives, batch_negatives, batch_negatives_2, is_ilo_anchor=is_ilo_anchor)
                else:
                    # Calculate batch loss using all quadruplets
                    quad_loss = quadruplet_loss_fn(batch_anchors, batch_positives, batch_negatives, batch_negatives_2)
                    
                # For comparison/monitoring, calculate standard triplet loss
                triplet_loss = triplet_loss_fn(batch_anchors, batch_positives, batch_negatives)

            else:
                quad_loss = torch.tensor(0.0, device=device, requires_grad=True)
                epoch_failed_formation_count += 1
                print(f"No valid quadruplets found in batch {batch_idx + 1}.")
                print(f"REASONS:\n A-P failures: {batch_ap_failures} \n SHN failures: {batch_shn_failures} \n N2 failures: {batch_n2_failures}")



            clf_results = compute_classification_loss(model, embeddings, labels, clf_loss_fn, active_classifier=active_classifier)
            prof_labels = clf_results['prof_labels']
            prof_preds = clf_results['predictions']
            clf_loss = clf_results['loss']

            train_prof_labels.append(prof_labels.cpu())
            train_prof_preds.append(prof_preds.cpu())

            total_loss = quad_loss + lambda_clf * clf_loss

            # Use quadruplet loss for optimization
            total_loss.backward()

            encoder_optimizer.step()
            if classifier_optimizer is not None:
                classifier_optimizer.step()

            # Track metrics
            epoch_total_loss += total_loss.item()
            epoch_quad_loss += quad_loss.item()
            epoch_clf_loss += lambda_clf * clf_loss.item()
            epoch_batch_count += 1
            epoch_quadruplet_count += batch_quadruplet_count


            if log_to_wandb:
                wandb.log({
                    "batch_idx": batch_idx + 1,
                    "batch_quadruplet_loss": quad_loss.item(),
                    "batch_triplet_loss": triplet_loss.item(),
                    "batch_quadruplets": batch_quadruplet_count,
                    "batch": batch_idx + 1,
                    "batch_ilo_anchors": n_ilo_anchors,
                    "batch_clf_loss": lambda_clf * clf_loss.item(),
                    "batch_total_loss": total_loss.item(),
                    "batch_ap_failures": batch_ap_failures,
                    "batch_shn_failures": batch_shn_failures,
                    "batch_n2_failures": batch_n2_failures,
                })

        
        # End of epoch - calculate training metrics
        if epoch_batch_count > 0:
            train_loss = epoch_total_loss / epoch_batch_count
            train_quad_loss = epoch_quad_loss / epoch_batch_count
            train_clf_loss = epoch_clf_loss / epoch_batch_count

            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}")
            print(f"Total quadruplets formed: {epoch_quadruplet_count}")
            
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)


            # Calculate mAP for full labels and profusion-only
            train_map, train_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
            prof_all_labels = all_labels % 4
            train_prof_map, train_prof_class_map = helpers.compute_map_per_class(all_embeddings, prof_all_labels)
            
            # Ensure we track all classes
            train_class_map_full = {class_id: train_class_map.get(class_id, 0.0) for class_id in range(num_classes)}
            
            print(f"Train mAP: {train_map:.4f}")
            # print("- Per-Class Train mAP:")
            for class_id in range(num_classes):
                ap = train_class_map_full.get(class_id, 0.0)
                class_name = multiclass_stb_mapping.get(class_id, f"Class {class_id}")
                # print(f"  {class_name}: mAP = {ap:.4f}")
                
                # Store per-class metrics
                per_class_metrics[class_id]['train_ap'].append(ap)

            
            train_specificity, train_class_specificity = helpers.compute_specificity_at_sensitivity(
                all_embeddings, all_labels, sensitivity_target=0.90)
            

            all_prof_clf_labels = torch.cat(train_prof_labels, dim=0)
            all_prof_clf_preds = torch.cat(train_prof_preds, dim=0)

            train_misclass_confidences = cl_metrics.get_misclassification_confidences(all_prof_clf_preds, all_prof_clf_labels, adjacent_only=False)
            train_adj_misclass_confidences = cl_metrics.get_misclassification_confidences(all_prof_clf_preds, all_prof_clf_labels, adjacent_only=True)
            

            train_misclass_confidences_per_class = cl_metrics.get_misclassification_confidences_per_class(
                all_prof_clf_preds, all_prof_clf_labels, adjacent_only=False
            )
            train_adj_misclass_confidences_per_class = cl_metrics.get_misclassification_confidences_per_class(
                all_prof_clf_preds, all_prof_clf_labels, adjacent_only=True
            )

            for cls, confs in train_misclass_confidences_per_class.items():
                wandb.log({f"train_misclass_conf_mean_class_{cls}": np.mean(confs) if len(confs) > 0 else 0.0, "epoch": epoch + 1})

            for cls, confs in train_adj_misclass_confidences_per_class.items():
                wandb.log({f"train_adj_misclass_conf_mean_class_{cls}": np.mean(confs) if len(confs) > 0 else 0.0, "epoch": epoch + 1})




            if active_classifier == "multiclass_profusion":  # TO DO: Check if works
                # Change the task type
                task = "multiclass"
            elif active_classifier == "binary_profusion":
                # Change the task type
                task = "binary"

            train_prof_metrics = cl_metrics.calculate_classification_metrics(
                all_prof_clf_preds, all_prof_clf_labels, task_type=task
            )

            train_prof_binary_metrics = cl_metrics.calculate_classification_metrics(
                all_prof_clf_preds, all_prof_clf_labels, task_type=task, binary_target_class="profusion_present"
            )

            train_embedding_metrics = cl_metrics.calculate_embedding_alignment_metrics(all_embeddings, all_labels)



            # Update history
            history['train_embedding_ratio'].append(train_embedding_metrics['embedding_ratio'])
            history['train_specificity'].append(train_specificity)
            history['train_class_specificity'].append(train_class_specificity)
            history['train_loss'].append(train_loss)
            history['train_map'].append(train_map)
            history['train_class_map'].append(train_class_map_full)
            history['train_prof_map'].append(train_prof_map)
        else:
            print(f"Epoch {epoch + 1}/{n_epochs}: No valid quadruplets formed")
            train_prof_metrics = {}

        prof_labels = all_labels % 4
        with torch.no_grad():
            cm = get_confusion_matrix(predictions=all_prof_clf_preds,
                                    labels=all_prof_clf_labels)

            # prof_fig = create_conf_mat_plot(cm, clf_name="multiclass_profusion", epoch=epoch+1, set_name="train", log_to_wandb=True)

            if all_prof_clf_preds.dim() > 1:
                # Multiclass case
                prof_labels_binary = (all_prof_clf_labels > 0).long()
                prof_preds_binary = (torch.argmax(all_prof_clf_preds, dim=1) > 0).long()
            else:
                # Binary case
                prof_labels_binary = (all_prof_clf_labels > 0).long()
                prof_preds_binary = (torch.sigmoid(all_prof_clf_preds) > 0.5).long()

            cm_binary = get_confusion_matrix(prof_preds_binary, prof_labels_binary)

            train_combined_fig = create_combined_conf_mat_plot(             # Just for TP,FN, FP, TN logging  ---> TO DO: Fix this
                multiclass_cm=cm, 
                binary_cm=cm_binary, 
                set_name="train", 
                epoch=epoch+1, 
                log_to_wandb=False
            )
        
        # Run t-SNE visualization at regular intervals
        if (epoch + 1) % tsne_interval == 0:
            visualize_tsne(model, device, ilo_dataset, train_loader, 
                          trained=True, log_to_wandb=log_to_wandb, 
                          n_epochs=epoch+1, set_name="training", entire_dataset=False)
            visualize_tsne(model, device, ilo_dataset, val_loader, 
                          trained=True, log_to_wandb=log_to_wandb,
                          n_epochs=epoch+1, set_name="validation", entire_dataset=False)
            # train_binary_conf_mat_fig = create_conf_mat_plot(cm_binary, clf_name="binary_profusion", set_name="train", epoch=epoch+1, log_to_wandb=True)
            
            # Use the new combined function
            train_combined_fig = create_combined_conf_mat_plot(
                multiclass_cm=cm, 
                binary_cm=cm_binary, 
                set_name="train", 
                epoch=epoch+1, 
                log_to_wandb=True
            )
        # Validation loop
        print("\nVALIDATION\n")

        wandb.log({
        "encoder_lr": encoder_optimizer.param_groups[0]['lr'],
        "classifier_lr": classifier_optimizer.param_groups[0]['lr']
        })


        val_metrics = validate_quadruplet(
            model=model,
            val_loader=val_loader,
            device=device,
            triplet_loss_fn=triplet_loss_fn,
            quadruplet_loss_fn=quadruplet_loss_fn,
            num_classes=num_classes,
            multiclass_stb_mapping=multiclass_stb_mapping,
            mining_strat=mining_strat,
            clf_loss_fn=clf_loss_fn,
        )

        val_loss = val_metrics['val_quadruplet_loss']
        val_map = val_metrics['val_map']
        val_class_map = val_metrics['val_class_map']
        val_prof_map = val_metrics['val_prof_map']
        val_embeddings = val_metrics['val_embeddings']
        val_labels = val_metrics['val_labels']
        val_prof_metrics = val_metrics['val_prof_metrics']
        val_prof_binary_metrics = val_metrics['val_prof_binary_metrics']
        val_embedding_ratio = val_metrics['val_embedding_ratio']
        val_inter_class_distance = val_metrics['val_inter_class_distance']
        val_intra_class_distance = val_metrics['val_intra_class_distance']
        val_silhouette_score = val_metrics['val_silhouette_score']

        # Use validation predictions that were already calculated
        val_prof_preds = val_metrics['val_prof_preds']
        val_prof_labels = val_metrics['val_prof_labels']

        val_adj_misclass_confidences = cl_metrics.get_misclassification_confidences(val_prof_preds, val_prof_labels, adjacent_only=True)
        val_misclass_confidences = cl_metrics.get_misclassification_confidences(val_prof_preds, val_prof_labels, adjacent_only=False)

        val_misclass_confidences_per_class = cl_metrics.get_misclassification_confidences_per_class(
            val_prof_preds, val_prof_labels, adjacent_only=False
        )
        val_adj_misclass_confidences_per_class = cl_metrics.get_misclassification_confidences_per_class(
            val_prof_preds, val_prof_labels, adjacent_only=True
        )

        for cls, confs in val_misclass_confidences_per_class.items():
            wandb.log({f"val_misclass_conf_mean_class_{cls}": np.mean(confs) if len(confs) > 0 else 0.0, "epoch": epoch + 1})

        for cls, confs in val_adj_misclass_confidences_per_class.items():
            wandb.log({f"val_adj_misclass_conf_mean_class_{cls}": np.mean(confs) if len(confs) > 0 else 0.0, "epoch": epoch + 1})

        # Create confusion matrix from accumulated predictions
        cm = get_confusion_matrix(val_prof_preds, val_prof_labels)

        if all_prof_clf_preds.dim() > 1:
            # Multiclass case
            prof_labels_binary = (val_prof_labels > 0).long()
            prof_preds_binary = (torch.argmax(val_prof_preds, dim=1) > 0).long()
        else:
            # Binary case
            prof_labels_binary = (val_prof_labels > 0).long()
            prof_preds_binary = (torch.sigmoid(val_prof_preds) > 0.5).long()
            
        # Use consistent binary conversion
        cm_binary = get_confusion_matrix(prof_preds_binary, prof_labels_binary)
        
        val_combined_fig = create_combined_conf_mat_plot(
            multiclass_cm=cm, 
            binary_cm=cm_binary, 
            set_name="validation", 
            epoch=epoch+1, 
            log_to_wandb=False
        )

        # clf_scheduler.step(val_metrics['val_clf_loss']) # TO DO: Implement this properly (if it works)
        # encoder_scheduler.step(val_loss)


        # Save best model based on validation mAP
        if epoch == 0 or (val_prof_binary_metrics.get('specificity', 0.0) > best_val_prof_bin_specificity and epoch > 50):
            best_val_prof_bin_specificity = val_prof_binary_metrics.get('specificity', 0.0)
            print(f"Saving best model with validation profusion binary specificity: {best_val_prof_bin_specificity:.4f}")
            best_model_state = model.state_dict().copy()
            
            # Create checkpoint dictionary with both optimizers
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            }
            
            # Add classifier optimizer state if it exists
            if classifier_optimizer is not None:
                checkpoint['classifier_optimizer_state_dict'] = classifier_optimizer.state_dict()
            
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"best_bin_spec_model.pth"))

            best_spec_fig = create_combined_conf_mat_plot(
                multiclass_cm=cm, 
                binary_cm=cm_binary, 
                set_name="best val binary specificity", 
                epoch=epoch+1, 
                log_to_wandb=True
            )



        # Generate confusion matrices at visualization intervals
        if (epoch+1) % tsne_interval == 0:
            # Use the accumulated predictions from validation
            with torch.no_grad():

                #val_conf_mat_fig = create_conf_mat_plot(cm, clf_name="multiclass_profusion", set_name="validation", epoch=epoch+1, log_to_wandb=True)
                #val_binary_conf_mat_fig = create_conf_mat_plot(cm_binary, clf_name="binary_profusion", set_name="validation", epoch=epoch+1, log_to_wandb=True)

                # Use the new combined function
                val_combined_fig = create_combined_conf_mat_plot(
                    multiclass_cm=cm, 
                    binary_cm=cm_binary, 
                    set_name="validation", 
                    epoch=epoch+1, 
                    log_to_wandb=True
                )

        # Store validation metrics for all classes
        for class_id in range(num_classes):
            per_class_metrics[class_id]['val_ap'].append(val_class_map.get(class_id, 0.0))


        # Update history
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map)
        history['val_class_map'].append(val_class_map)
        history['val_prof_map'].append(val_metrics['val_prof_map'])
        history['val_quadruplet_loss'].append(val_loss)
        history['val_specificity'].append(val_metrics['val_specificity'])
        history['val_class_specificity'].append(val_metrics['val_class_specificity'])

        # Log metrics to wandb
        if log_to_wandb:
            wandb_log_dict = {
                "epoch": epoch + 1,
                "epoch_filtered_count": epoch_filtered_count,
                "epoch_failed_formation_count": epoch_failed_formation_count,
                "epoch_success_count": epoch_success_count,
                "epoch_total_batches": epoch_total_batches,
                "train_loss": train_loss if epoch_batch_count > 0 else 0.0,
                "train_map": train_map if epoch_batch_count > 0 else 0.0,
                "train_quadruplet_loss": train_quad_loss if epoch_batch_count > 0 else 0.0,
                "train_clf_loss": train_clf_loss if epoch_batch_count > 0 else 0.0,
                "val_clf_loss": val_metrics['val_clf_loss'] * lambda_clf,
                "val_loss": val_loss,
                "val_map": val_map,
                "val_prof_map": val_prof_map,
                "train_prof_map": train_prof_map if epoch_batch_count > 0 else 0.0,
                "current_margin1": quadruplet_loss_fn.margin1,
                "current_margin2": quadruplet_loss_fn.margin2,
                "current_p_ilo_anchor": p_ilo_anchor if wandb.config.p_ilo_scheduling else wandb.config.p_ilo_anchor,
                "train_quadruplets": epoch_quadruplet_count,
                "train_spec@sens": train_specificity if epoch_batch_count > 0 else 0.0,
                "val_spec@sens": val_metrics['val_specificity'],
                'val_embedding_ratio': val_embedding_ratio,
                'val_intra_class_distance': val_intra_class_distance,
                'val_inter_class_distance': val_inter_class_distance,
                'val_silhouette_score': val_silhouette_score,
                'val_misclass_conf_mean': np.mean(val_misclass_confidences) if len(val_misclass_confidences) > 0 else 0.0,
                'val_adj_misclass_conf_mean': np.mean(val_adj_misclass_confidences) if len(val_adj_misclass_confidences) > 0 else 0.0,
                'train_misclass_conf_mean': np.mean(train_misclass_confidences) if len(train_misclass_confidences) > 0 else 0.0,
                'train_adj_misclass_conf_mean': np.mean(train_adj_misclass_confidences) if len(train_adj_misclass_confidences) > 0 else 0.0,
            }
            
            # Add training profusion classification metrics
            if train_prof_metrics:
                wandb_log_dict.update({
                    "train_prof_accuracy": train_prof_metrics.get('accuracy', 0.0),
                    "train_prof_f1": train_prof_metrics.get('f1', 0.0),
                    "train_prof_precision": train_prof_metrics.get('precision', 0.0),
                    "train_prof_recall": train_prof_metrics.get('recall', 0.0),
                    "train_prof_auc": train_prof_metrics.get('auc', 0.0),
                    "train_prof_kappa": train_prof_metrics.get('kappa', 0.0),
                    "train_prof_specificity": train_prof_metrics.get('specificity', 0.0),
                    "train_prof_bin_accuracy": train_prof_binary_metrics.get('accuracy', 0.0),
                    "train_prof_bin_f1": train_prof_binary_metrics.get('f1', 0.0),
                    "train_prof_bin_specificity": train_prof_binary_metrics.get('specificity', 0.0),
                    "train_prof_bin_auc": train_prof_binary_metrics.get('auc', 0.0),
                    "train_prof_bin_spec_at_sens": train_prof_binary_metrics.get('spec_at_sens', 0.0),
                    "train_prof_bin_kappa": train_prof_binary_metrics.get('kappa', 0.0),
                    "train_prof_bin_precision": train_prof_binary_metrics.get('precision', 0.0),
                    "train_prof_bin_recall": train_prof_binary_metrics.get('recall', 0.0),
                    "train_intra_class_distance": train_embedding_metrics.get('intra_class_distance', 0.0),
                    "train_inter_class_distance": train_embedding_metrics.get('inter_class_distance', 0.0),
                    "train_embedding_ratio": train_embedding_metrics.get('embedding_ratio', 0.0),
                    "train_silhouette_score": train_embedding_metrics.get('silhouette_score', 0.0),
                    "train_davies_bouldin": train_embedding_metrics.get('davies_bouldin', 0.0),                                       
                })

            # Add validation profusion classification metrics
            wandb_log_dict.update({
                "val_prof_accuracy": val_prof_metrics.get('accuracy', 0.0),
                "val_prof_f1": val_prof_metrics.get('f1', 0.0),
                "val_prof_precision": val_prof_metrics.get('precision', 0.0),
                "val_prof_recall": val_prof_metrics.get('recall', 0.0),
                "val_prof_auc": val_prof_metrics.get('auc', 0.0),
                "val_prof_kappa": val_prof_metrics.get('kappa', 0.0),
                "val_prof_specificity": val_prof_metrics.get('specificity', 0.0),
                "val_prof_bin_accuracy": val_prof_binary_metrics.get('accuracy', 0.0),
                "val_prof_bin_f1": val_prof_binary_metrics.get('f1', 0.0),
                "val_prof_bin_specificity": val_prof_binary_metrics.get('specificity', 0.0),
                "val_prof_bin_auc": val_prof_binary_metrics.get('auc', 0.0),
                "val_prof_bin_spec_at_sens": val_prof_binary_metrics.get('spec_at_sens', 0.0),
                "val_prof_bin_kappa": val_prof_binary_metrics.get('kappa', 0.0),
                "val_prof_bin_precision": val_prof_binary_metrics.get('precision', 0.0),
                "val_prof_bin_recall": val_prof_binary_metrics.get('recall', 0.0),
            })
            
            # Add per-class metrics
            for class_id in range(num_classes):
                train_ap = train_class_map_full.get(class_id, 0.0) if epoch_batch_count > 0 else 0.0
                val_ap = val_class_map.get(class_id, 0.0)
                train_sens = train_class_specificity.get(class_id, 0.0) if epoch_batch_count > 0 else 0.0
                val_sens = val_metrics['val_class_specificity'].get(class_id, 0.0)
                
                wandb_log_dict[f"train_class_{class_id}_map"] = train_ap
                wandb_log_dict[f"val_class_{class_id}_map"] = val_ap
                wandb_log_dict[f"train_class_{class_id}_specificity"] = train_sens
                wandb_log_dict[f"val_class_{class_id}_specificity"] = val_sens
            
            wandb.log(wandb_log_dict)

        
        # Save best model based on validation mAP
        if epoch == 0 or (val_map > best_val_map and epoch > 50):
            best_val_map = val_map
            print(f"Saving best model with validation mAP: {best_val_map:.4f}")
            best_model_state = model.state_dict().copy()
            
            # Create checkpoint dictionary with both optimizers
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            }
            
            # Add classifier optimizer state if it exists
            if classifier_optimizer is not None:
                checkpoint['classifier_optimizer_state_dict'] = classifier_optimizer.state_dict()
            
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"best_model.pth"))

            visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, True, True, n_epochs=epoch+1, set_name="best val mAP", entire_dataset=True)

        if (torch.cuda.memory_allocated()/1e9) > 3:
            torch.cuda.empty_cache()
            gc.collect()
            
        
        # Save latest model
        final_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        }

        # Add classifier optimizer state if it exists
        if classifier_optimizer is not None:
            final_checkpoint['classifier_optimizer_state_dict'] = classifier_optimizer.state_dict()

        torch.save(final_checkpoint, os.path.join(checkpoint_dir, f"final_model.pth"))

    visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, True, True, n_epochs=n_epochs+1, set_name="final", entire_dataset=True)

    
    # Return training results
    return {
        'model': model,
        'best_model_state': best_model_state,
        'history': history,
        'best_val_map': best_val_map,
        'per_class_metrics': per_class_metrics
    }


def validate_quadruplet(model, val_loader, device, triplet_loss_fn, quadruplet_loss_fn, num_classes=8, multiclass_stb_mapping=None, mining_strat="BSHN-v2", clf_loss_fn=None, active_classifier="multiclass_profusion"):
    """
    Validation loop for quadruplet loss model - FIXED for experimental integrity
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation set
        device: Device to run validation on
        triplet_loss_fn: Standard triplet loss function
        quadruplet_loss_fn: Quadruplet loss function
        num_classes: Number of classes (default: 8)
        multiclass_stb_mapping: Dictionary mapping class IDs to names
        mining_strat: Mining strategy for triplet formation
    """
    if multiclass_stb_mapping is None:
        multiclass_stb_mapping = {i: f"Class {i}" for i in range(num_classes)}
    
    model.eval()
    running_quad_loss = 0.0
    running_triplet_loss = 0.0
    running_clf_loss = 0.0
    all_embeddings = []
    all_labels = []
    all_prof_preds = []
    all_prof_labels = []
    val_quadruplet_count = 0
    batch_with_quadruplets = 0
    total_batches = 0
    failed_anchor_batches = 0
    
    # 🚨 REMOVED DATA LEAKAGE: No longer using dataset-wide label mapping
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            total_batches += 1
            
            # Get validation batch
            imgs = sample[0].to(device)
            labels = sample[1].to(device)
            
            # Extract features and normalize embeddings
            features = model.features(imgs)
            embeddings = F.normalize(features, p=2, dim=1)
            
            # Store for mAP calculation - ALWAYS store, regardless of quadruplet formation
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

            val_clf_results = compute_classification_loss(model, embeddings, labels, clf_loss_fn, active_classifier=active_classifier)
            
            # 🎯 ALWAYS compute classification metrics (like training)
            prof_labels = val_clf_results['prof_labels']
            prof_preds = val_clf_results['predictions']
            clf_loss = val_clf_results['loss']

            all_prof_labels.append(prof_labels.detach().cpu())
            all_prof_preds.append(prof_preds.detach().cpu())
            running_clf_loss += clf_loss.item()
            
            # Get labels from batch
            current_batch_labels = labels.cpu().numpy()
            
            # Collect quadruplet components
            anchors = []
            positives = []
            negatives = []
            negatives_2 = []
            batch_quadruplet_count = 0
            batch_failed_anchors = 0
            
            # Form quadruplets ONLY within current batch - NO dataset access
            for i, positive_label in enumerate(current_batch_labels):
                # Positive embedding from current batch
                positive_embedding = embeddings[i].unsqueeze(0)
                
                # 🔧 FIXED: Only look for anchors within current batch
                batch_matching_indices = [j for j in range(len(current_batch_labels)) 
                                        if current_batch_labels[j] == positive_label and j != i]
                
                if batch_matching_indices:
                    # Found anchor in current batch
                    batch_anchor_idx = np.random.choice(batch_matching_indices)
                    anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                    anchor_label = current_batch_labels[batch_anchor_idx]
                else:
                    # 🚨 CRITICAL FIX: Skip instead of using validation dataset
                    batch_failed_anchors += 1
                    continue
                
                # Find first negative using the mining strategy
                if mining_strat == "BSHN-v2":
                    prof_pos_score = positive_label % 4
                    
                    negative_indices = [j for j, label in enumerate(current_batch_labels) 
                                      if label != positive_label and (label % 4) != prof_pos_score]
                
                if negative_indices:
                    negative_embeddings = embeddings[negative_indices]
                    anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)
                    
                    # Compute distances
                    dists = F.pairwise_distance(anchor_repeated, negative_embeddings)
                    positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
                    
                    # Find semi-hard negatives
                    semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + triplet_loss_fn.margin))
                    semi_hard_dists = dists[semi_hard_mask]
                    
                    if semi_hard_dists.numel() > 0:
                        # Use semi-hard negative
                        hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                        semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                        selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                        
                        negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                        negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                        
                        # Find second negative with same profusion but different TB status
                        negative_indices_2 = [j for j, label in enumerate(current_batch_labels) 
                                          if (label % 4) == (negative_label % 4) and 
                                             (int(label >= 4) != int(negative_label >= 4))]
                        
                        if negative_indices_2:
                            neg_2_idx = np.random.choice(negative_indices_2)
                            negative_embedding_2 = embeddings[neg_2_idx].unsqueeze(0)
                            
                            # Add to collections
                            anchors.append(anchor_embedding)
                            positives.append(positive_embedding)
                            negatives.append(negative_embedding)
                            negatives_2.append(negative_embedding_2)
                            batch_quadruplet_count += 1
            
            if batch_failed_anchors > 0:
                failed_anchor_batches += 1
            
            # Process collected quadruplets (similar to training)
            if batch_quadruplet_count > 0:
                # Combine all quadruplet components
                batch_anchors = torch.cat(anchors, dim=0)
                batch_positives = torch.cat(positives, dim=0)
                batch_negatives = torch.cat(negatives, dim=0)
                batch_negatives_2 = torch.cat(negatives_2, dim=0)
                
                # Calculate losses
                quad_loss = quadruplet_loss_fn(batch_anchors, batch_positives, batch_negatives, batch_negatives_2)
                triplet_loss = triplet_loss_fn(batch_anchors, batch_positives, batch_negatives)
                
                running_quad_loss += quad_loss.item()
                running_triplet_loss += triplet_loss.item()
                batch_with_quadruplets += 1
                val_quadruplet_count += batch_quadruplet_count
            # Note: No else clause needed - we always process classification
    
    # 🎯 ALWAYS calculate validation metrics (even if no quadruplets formed)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_prof_preds = torch.cat(all_prof_preds, dim=0)
    all_prof_labels = torch.cat(all_prof_labels, dim=0)


    # Calculate mAP for full labels
    val_map, val_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
    
    # Calculate mAP for profusion scores only
    prof_all_labels = all_labels % 4
    val_prof_map, val_prof_class_map = helpers.compute_map_per_class(all_embeddings, prof_all_labels)
    
    # Calculate average validation losses
    avg_quad_loss = running_quad_loss / max(1, batch_with_quadruplets)
    avg_triplet_loss = running_triplet_loss / max(1, batch_with_quadruplets)
    avg_clf_loss = running_clf_loss / max(1, total_batches)  # Always computed
    
    if wandb.config.active_classifier == "multiclass_profusion":  # TO DO: Check if works
        # Change the task type
        task = "multiclass"
    elif wandb.config.active_classifier == "binary_profusion":
        # Change the task type
        task = "binary"

    # Calculate classification metrics
    val_prof_metrics = cl_metrics.calculate_classification_metrics(
        all_prof_preds, all_prof_labels, task_type=task
    )
    
    # Calculate binary metrics (Prof 0 vs Prof 1-3) - CONSISTENT approach
    val_prof_binary_metrics = cl_metrics.calculate_classification_metrics(
        all_prof_preds, all_prof_labels,  # ✅ Use accumulated data from all batches
        task_type=task, 
        binary_target_class="profusion_present"  # ✅ Same binary problem as training
    )

    val_embedding_metrics = cl_metrics.calculate_embedding_alignment_metrics(all_embeddings, all_labels)  

    print(f"\nValidation Summary:")
    print(f"- Total batches processed: {total_batches}")
    print(f"- Batches with failed anchor formation: {failed_anchor_batches}")
    print(f"- Total quadruplets formed: {val_quadruplet_count}")
    print(f"- Avg Quadruplet Loss: {avg_quad_loss:.4f}")
    print(f"- Avg Classification Loss: {avg_clf_loss:.4f}")
    print(f"- Validation mAP: {val_map:.4f}")
    print(f"- Validation Profusion mAP: {val_prof_map:.4f}")
    print(f"- Validation Prof Accuracy: {val_prof_metrics.get('accuracy', 0.0):.4f}")
    
    # Create full map including any missing classes
    val_class_map_full = {class_id: val_class_map.get(class_id, 0.0) for class_id in range(num_classes)}
    
    # Calculate sensitivity at specificity
    val_specificity, val_class_specificity = helpers.compute_specificity_at_sensitivity(
        all_embeddings, all_labels, sensitivity_target=0.90)

    print(f"- Validation Specificity@0.90Sensitivity: {val_specificity:.4f}")
    
    # Log validation metrics to wandb
    wandb.log({
        "val_quadruplet_loss": avg_quad_loss,
        "val_triplet_loss": avg_triplet_loss,
        "val_quadruplets": val_quadruplet_count,
        "val_failed_anchor_batches": failed_anchor_batches,
        "val_total_batches": total_batches,
        "val_clf_loss": avg_clf_loss
    })
    
    # Return comprehensive metrics
    return {
        'val_quadruplet_loss': avg_quad_loss,
        'val_triplet_loss': avg_triplet_loss,
        'val_clf_loss': avg_clf_loss,
        'val_map': val_map,
        'val_class_map': val_class_map_full,
        'val_prof_map': val_prof_map,
        'val_prof_class_map': val_prof_class_map,
        'val_embeddings': all_embeddings,
        'val_labels': all_labels,
        'val_prof_preds': all_prof_preds,
        'val_prof_labels': all_prof_labels,
        'val_prof_metrics': val_prof_metrics,
        'val_prof_binary_metrics': val_prof_binary_metrics,
        'val_specificity': val_specificity,
        'val_class_specificity': val_class_specificity,
        'val_quadruplet_count': val_quadruplet_count,
        'val_failed_anchor_batches': failed_anchor_batches,
        'val_total_batches': total_batches,
        "val_embedding_ratio": val_embedding_metrics['embedding_ratio'],
        "val_intra_class_distance": val_embedding_metrics['intra_class_distance'],
        "val_inter_class_distance": val_embedding_metrics['inter_class_distance'],
        "val_silhouette_score": val_embedding_metrics['silhouette_score']
                }


def test_quadruplet_model(
    model, 
    test_loader, 
    device, 
    triplet_loss_fn, 
    quadruplet_loss_fn, 
    ilo_dataset,
    experiment_name,
    num_classes=8, 
    multiclass_stb_mapping=None, 
    mining_strat="BSHN-v2",
    log_to_wandb=True,
    lambda_clf=0.25,
    clf_loss_fn=None,
    active_classifier="multiclass_profusion",
    n_epochs=1000,
):
    """
    Comprehensive test function that mirrors validation metrics and provides
    detailed analysis with t-SNE visualization and confusion matrices.
    
    Args:
        model: The trained model to test
        test_loader: DataLoader for test set
        device: Device to run testing on
        triplet_loss_fn: Standard triplet loss function
        quadruplet_loss_fn: Quadruplet loss function
        ilo_dataset: ILO dataset for t-SNE visualization
        experiment_name: Name for logging purposes
        num_classes: Number of classes (default: 8)
        multiclass_stb_mapping: Dictionary mapping class IDs to names
        mining_strat: Mining strategy for triplet formation
        log_to_wandb: Whether to log metrics to wandb
        lambda_clf: Weight for classification loss
        
    Returns:
        dict: Comprehensive test metrics matching validation output
    """
    if multiclass_stb_mapping is None:
        multiclass_stb_mapping = {
            0: "Profusion 0, No TB",
            1: "Profusion 1, No TB", 
            2: "Profusion 2, No TB",
            3: "Profusion 3, No TB",
            4: "Profusion 0, With TB",
            5: "Profusion 1, With TB",
            6: "Profusion 2, With TB", 
            7: "Profusion 3, With TB",
        }
    
    print("=" * 60)
    print(f"TESTING MODEL: {experiment_name}")
    print("=" * 60)
    
    model.eval()
    running_quad_loss = 0.0
    running_triplet_loss = 0.0
    running_clf_loss = 0.0
    all_embeddings = []
    all_labels = []
    all_prof_preds = []
    all_prof_labels = []
    test_quadruplet_count = 0
    batch_with_quadruplets = 0
    total_batches = 0
    failed_anchor_batches = 0
    
    # Detailed failure tracking
    total_ap_failures = 0
    total_shn_failures = 0
    total_n2_failures = 0
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            total_batches += 1
            
            # Get test batch
            imgs = sample[0].to(device)
            labels = sample[1].long().to(device)
            
            # Extract features and normalize embeddings
            features = model.features(imgs)
            embeddings = F.normalize(features, p=2, dim=1)
            
            # Store for mAP calculation - ALWAYS store
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
            # ALWAYS compute classification metrics (matching validation)
            # Compute classification loss
            test_clf_results = compute_classification_loss(model, embeddings, labels, clf_loss_fn, active_classifier=active_classifier)
            prof_labels = test_clf_results['prof_labels']
            prof_preds = test_clf_results['predictions']
            clf_loss = test_clf_results['loss']

            all_prof_labels.append(prof_labels.cpu())
            all_prof_preds.append(prof_preds.cpu())
            running_clf_loss += clf_loss.item()
            
            # Get labels from batch
            current_batch_labels = labels.cpu().numpy()
            
            # Collect quadruplet components (same logic as validation)
            anchors = []
            positives = []
            negatives = []
            negatives_2 = []
            batch_quadruplet_count = 0
            batch_failed_anchors = 0
            batch_ap_failures = 0
            batch_shn_failures = 0
            batch_n2_failures = 0
            
            # Form quadruplets ONLY within current batch - NO dataset access
            for i, positive_label in enumerate(current_batch_labels):
                positive_embedding = embeddings[i].unsqueeze(0)
                
                # Find anchors within current batch only
                batch_matching_indices = [j for j in range(len(current_batch_labels)) 
                                        if current_batch_labels[j] == positive_label and j != i]
                
                if batch_matching_indices:
                    batch_anchor_idx = np.random.choice(batch_matching_indices)
                    anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                    anchor_label = current_batch_labels[batch_anchor_idx]
                else:
                    batch_failed_anchors += 1
                    batch_ap_failures += 1
                    continue
                
                # Find first negative using mining strategy
                if mining_strat == "BSHN-v2":
                    prof_pos_score = positive_label % 4
                    negative_indices = [j for j, label in enumerate(current_batch_labels) 
                                      if label != positive_label and (label % 4) != prof_pos_score]
                
                if negative_indices:
                    negative_embeddings = embeddings[negative_indices]
                    anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)
                    
                    # Compute distances
                    dists = F.pairwise_distance(anchor_repeated, negative_embeddings)
                    positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
                    
                    # Find semi-hard negatives
                    semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + triplet_loss_fn.margin))
                    semi_hard_dists = dists[semi_hard_mask]
                    
                    if semi_hard_dists.numel() > 0:
                        # Use semi-hard negative
                        hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                        semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                        selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                        
                        negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                        negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                        
                        # Find second negative with same profusion but different TB status
                        negative_indices_2 = [j for j, label in enumerate(current_batch_labels) 
                                          if (label % 4) == (negative_label % 4) and 
                                             (int(label >= 4) != int(negative_label >= 4))]
                        
                        if negative_indices_2:
                            neg_2_idx = np.random.choice(negative_indices_2)
                            negative_embedding_2 = embeddings[neg_2_idx].unsqueeze(0)
                            
                            # Add to collections
                            anchors.append(anchor_embedding)
                            positives.append(positive_embedding)
                            negatives.append(negative_embedding)
                            negatives_2.append(negative_embedding_2)
                            batch_quadruplet_count += 1
                        else:
                            batch_n2_failures += 1
                    else:
                        batch_shn_failures += 1
                else:
                    batch_shn_failures += 1
            
            # Track failures
            total_ap_failures += batch_ap_failures
            total_shn_failures += batch_shn_failures  
            total_n2_failures += batch_n2_failures
            
            if batch_failed_anchors > 0:
                failed_anchor_batches += 1
            
            # Process collected quadruplets
            if batch_quadruplet_count > 0:
                # Combine all quadruplet components
                batch_anchors = torch.cat(anchors, dim=0)
                batch_positives = torch.cat(positives, dim=0)
                batch_negatives = torch.cat(negatives, dim=0)
                batch_negatives_2 = torch.cat(negatives_2, dim=0)
                
                # Calculate losses
                quad_loss = quadruplet_loss_fn(batch_anchors, batch_positives, batch_negatives, batch_negatives_2)
                triplet_loss = triplet_loss_fn(batch_anchors, batch_positives, batch_negatives)
                
                running_quad_loss += quad_loss.item()
                running_triplet_loss += triplet_loss.item()
                batch_with_quadruplets += 1
                test_quadruplet_count += batch_quadruplet_count
    
    # Calculate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_prof_preds = torch.cat(all_prof_preds, dim=0)
    all_prof_labels = torch.cat(all_prof_labels, dim=0)
    
    # Calculate mAP for full labels
    test_map, test_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
    
    # Calculate mAP for profusion scores only
    prof_all_labels = all_labels % 4
    test_prof_map, test_prof_class_map = helpers.compute_map_per_class(all_embeddings, prof_all_labels)
    
    # Calculate average test losses
    avg_quad_loss = running_quad_loss / max(1, batch_with_quadruplets)
    avg_triplet_loss = running_triplet_loss / max(1, batch_with_quadruplets)
    avg_clf_loss = running_clf_loss / max(1, total_batches)

    if active_classifier == "multiclass_profusion":  # TO DO: Check if works
        # Change the task type
        task = "multiclass"
    elif active_classifier == "binary_profusion":
        # Change the task type
        task = "binary"
    
    # Calculate classification metrics
    test_prof_metrics = cl_metrics.calculate_classification_metrics(
        all_prof_preds, all_prof_labels, task_type=task
    )
    
    # Calculate binary metrics (Prof 0 vs Prof 1-3) - Same as validation
    test_prof_binary_metrics = cl_metrics.calculate_classification_metrics(
        all_prof_preds, all_prof_labels,
        task_type=task, 
        binary_target_class="profusion_present"  # ✅ Same binary problem as validation
    )
    
    # Create full map including any missing classes
    test_class_map_full = {class_id: test_class_map.get(class_id, 0.0) for class_id in range(num_classes)}
    
    # Calculate specificity at sensitivity
    test_specificity, test_class_specificity = helpers.compute_specificity_at_sensitivity(
        all_embeddings, all_labels, sensitivity_target=0.90)
    
    # Print comprehensive test summary
    print(f"\n{'='*60}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total batches processed: {total_batches}")
    print(f"Batches with failed anchor formation: {failed_anchor_batches}")
    print(f"Total quadruplets formed: {test_quadruplet_count}")
    print(f"Avg Quadruplet Loss: {avg_quad_loss:.4f}")
    print(f"Avg Triplet Loss: {avg_triplet_loss:.4f}")
    print(f"Avg Classification Loss: {avg_clf_loss:.4f}")
    print(f"Test mAP: {test_map:.4f}")
    print(f"Test Profusion mAP: {test_prof_map:.4f}")
    print(f"Test Specificity@0.90Sensitivity: {test_specificity:.4f}")
    print(f"\nProfusion Classification Metrics:")
    print(f"- Accuracy: {test_prof_metrics.get('accuracy', 0.0):.4f}")
    print(f"- F1-Score: {test_prof_metrics.get('f1', 0.0):.4f}")
    print(f"- AUC: {test_prof_metrics.get('auc', 0.0):.4f}")
    print(f"- Kappa: {test_prof_metrics.get('kappa', 0.0):.4f}")
    print(f"\nBinary Profusion Metrics (Prof 0 vs Prof 1-3):")
    print(f"- Binary Accuracy: {test_prof_binary_metrics.get('accuracy', 0.0):.4f}")
    print(f"- Binary F1: {test_prof_binary_metrics.get('f1', 0.0):.4f}")
    print(f"- Binary AUC: {test_prof_binary_metrics.get('auc', 0.0):.4f}")
    print(f"- Binary Specificity: {test_prof_binary_metrics.get('specificity', 0.0):.4f}")
    
    print(f"\nPer-Class Test mAP:")
    for class_id in range(num_classes):
        ap = test_class_map_full.get(class_id, 0.0)
        class_name = multiclass_stb_mapping.get(class_id, f"Class {class_id}")
        print(f"  {class_name}: mAP = {ap:.4f}")
    
    print(f"\nFailure Analysis:")
    print(f"- Anchor-Positive failures: {total_ap_failures}")
    print(f"- Semi-hard negative failures: {total_shn_failures}")
    print(f"- Second negative failures: {total_n2_failures}")
    
    # Generate t-SNE visualization
    if log_to_wandb:
        print(f"\nGenerating t-SNE visualization...")
        visualize_tsne(model, device, ilo_dataset, test_loader, 
                      trained=True, log_to_wandb=True, 
                      n_epochs="TEST", set_name="test", entire_dataset=True)
    
    # Generate confusion matrices
    print(f"Generating confusion matrices...")
    
    # Multiclass profusion confusion matrix
    # And in test_quadruplet_model function:
    cm_multiclass = get_confusion_matrix(all_prof_preds, all_prof_labels)
    if all_prof_preds.dim() > 1:
        # Multiclass case
        prof_labels_binary = (all_prof_labels > 0).long()
        prof_preds_binary = (torch.argmax(all_prof_preds, dim=1) > 0).long()
    else:
        # Binary case
        prof_labels_binary = (all_prof_labels > 0).long()
        prof_preds_binary = (torch.sigmoid(all_prof_preds) > 0.5).long()
    cm_binary = get_confusion_matrix(prof_preds_binary, prof_labels_binary)

    # Use the new combined function
    test_combined_fig = create_combined_conf_mat_plot(
        multiclass_cm=cm_multiclass,
        binary_cm=cm_binary,
        set_name="test",
        epoch=None,
        log_to_wandb=log_to_wandb,
        log_tp_tn_fp_fn=False
    )


    
    # Log comprehensive metrics to wandb
    if log_to_wandb:
        wandb_test_metrics = {
            # Core test metrics
            "test_quadruplet_loss": avg_quad_loss,
            "test_triplet_loss": avg_triplet_loss,
            "test_clf_loss": avg_clf_loss * lambda_clf,
            "test_map": test_map,
            "test_prof_map": test_prof_map,
            "test_specificity": test_specificity,
            "test_quadruplets": test_quadruplet_count,
            "test_failed_anchor_batches": failed_anchor_batches,
            "test_total_batches": total_batches,
            
            # Profusion classification metrics
            "test_prof_accuracy": test_prof_metrics.get('accuracy', 0.0),
            "test_prof_f1": test_prof_metrics.get('f1', 0.0),
            "test_prof_precision": test_prof_metrics.get('precision', 0.0),
            "test_prof_recall": test_prof_metrics.get('recall', 0.0),
            "test_prof_auc": test_prof_metrics.get('auc', 0.0),
            "test_prof_kappa": test_prof_metrics.get('kappa', 0.0),
            "test_prof_specificity": test_prof_metrics.get('specificity', 0.0),
            
            # Binary profusion metrics
            "test_prof_bin_accuracy": test_prof_binary_metrics.get('accuracy', 0.0),
            "test_prof_bin_f1": test_prof_binary_metrics.get('f1', 0.0),
            "test_prof_bin_specificity": test_prof_binary_metrics.get('specificity', 0.0),
            "test_prof_bin_auc": test_prof_binary_metrics.get('auc', 0.0),
            "test_prof_bin_spec_at_sens": test_prof_binary_metrics.get('spec_at_sens', 0.0),
            "test_prof_bin_kappa": test_prof_binary_metrics.get('kappa', 0.0),
            "test_prof_bin_precision": test_prof_binary_metrics.get('precision', 0.0),
            "test_prof_bin_recall": test_prof_binary_metrics.get('recall', 0.0),
            
            # Failure analysis
            "test_ap_failures": total_ap_failures,
            "test_shn_failures": total_shn_failures,
            "test_n2_failures": total_n2_failures,
        }
        
        # Add per-class metrics
        for class_id in range(num_classes):
            test_ap = test_class_map_full.get(class_id, 0.0)
            test_sens = test_class_specificity.get(class_id, 0.0)
            
            wandb_test_metrics[f"test_class_{class_id}_map"] = test_ap
            wandb_test_metrics[f"test_class_{class_id}_specificity"] = test_sens
        
        wandb.log(wandb_test_metrics)
        print(f"✅ Test metrics logged to wandb")
    
    # Return comprehensive test results (matching validation structure)
    return {
        'test_quadruplet_loss': avg_quad_loss,
        'test_triplet_loss': avg_triplet_loss,
        'test_clf_loss': avg_clf_loss,
        'test_map': test_map,
        'test_class_map': test_class_map_full,
        'test_prof_map': test_prof_map,
        'test_prof_class_map': test_prof_class_map,
        'test_embeddings': all_embeddings,
        'test_labels': all_labels,
        'test_prof_preds': all_prof_preds,
        'test_prof_labels': all_prof_labels,
        'test_prof_metrics': test_prof_metrics,
        'test_prof_binary_metrics': test_prof_binary_metrics,
        'test_specificity': test_specificity,
        'test_class_specificity': test_class_specificity,
        'test_quadruplet_count': test_quadruplet_count,
        'test_failed_anchor_batches': failed_anchor_batches,
        'test_total_batches': total_batches,
        'test_failure_analysis': {
            'ap_failures': total_ap_failures,
            'shn_failures': total_shn_failures,
            'n2_failures': total_n2_failures
        }
    }


def compute_classification_loss(model, embeddings, labels, clf_loss_fn=None, active_classifier="multiclass_profusion"):
    """
    Compute classification loss and predictions for profusion classes.
    
    Args:
        model: The model containing the mc_prof_clf classifier
        embeddings: Normalized embeddings from the model
        labels: Original labels (with combined profusion and TB information)
        clf_loss_fn: Optional loss function (defaults to cross_entropy)
        
    Returns:
        dict: Dictionary containing:
            - loss: The computed classification loss
            - predictions: Model predictions (logits)
            - prof_labels: Profusion labels (labels % 4)
    """

    if active_classifier == "multiclass_profusion":
        # Extract profusion labels (0-3)
        prof_labels = labels % 4
        # Get predictions from profusion classifier
        prof_preds = model.mc_prof_clf(embeddings)

    elif active_classifier == "binary_profusion":
        # Extract binary profusion labels (0 or 1)
        prof_labels = (labels > 0).float().view(-1)
        # Get predictions from binary classifier
        prof_preds = model.mc_prof_clf(embeddings).view(-1)


    else:
        raise ValueError(f"Unsupported active classifier: {active_classifier}")
    
    # Compute loss using provided function or default to cross entropy
    if clf_loss_fn is None:
        raise ValueError("Classification loss function must be provided.")
    else:
        # print(f"Preds shape: {prof_preds.shape}\n Labels shape: {prof_labels.shape}")
        clf_loss = clf_loss_fn(prof_preds, prof_labels)
        
    return {
        'loss': clf_loss,
        'predictions': prof_preds,
        'prof_labels': prof_labels
    }


def form_strict_quadruplets(embeddings, current_batch_labels, ilo_images, ilo_labels, model, device, p_ilo_anchor, mining_strat, triplet_loss_fn):
    """Form quadruplets with strict requirements - no fallbacks"""
    anchors = []
    positives = []
    negatives = []
    negatives_2 = []
    is_ilo_flags = []
    
    # Failure counters for logging
    ap_failures = 0
    ilo_fallback_count = 0
    shn_failures = 0
    n2_failures = 0
    
    for i, positive_label in enumerate(current_batch_labels):
        positive_embedding = embeddings[i].unsqueeze(0)
        anchor_found = False
        
        # 1. Try ILO anchor first if selected
        if np.random.rand() < p_ilo_anchor:
            ilo_indices = torch.where(ilo_labels == (positive_label % 4))[0]
            if len(ilo_indices) > 0:
                ilo_idx = np.random.choice(ilo_indices.cpu().numpy())
                anchor_embedding = ilo_images[ilo_idx].unsqueeze(0)
                anchor_embedding = model.features(anchor_embedding)
                anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                anchor_label = ilo_labels[ilo_idx].item()
                current_is_ilo = True
                anchor_found = True
            else:
                # NO FALLBACK - if no ILO anchor available, skip this sample
                ap_failures += 1
                continue
        else:
            # 2. Try in-batch anchor
            batch_matching_indices = [j for j in range(len(current_batch_labels)) 
                                    if current_batch_labels[j] == positive_label and j != i]
            if batch_matching_indices:
                batch_anchor_idx = np.random.choice(batch_matching_indices)
                anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                anchor_label = current_batch_labels[batch_anchor_idx]
                current_is_ilo = False
                anchor_found = True
            else:
                # NO FALLBACK - if no in-batch anchor, skip
                ap_failures += 1
                continue
        
        if not anchor_found:
            continue
            
        # 3. Find first negative with strict criteria
        if mining_strat == "BSHN-v2":
            prof_pos_score = positive_label % 4
            negative_indices = [j for j, label in enumerate(current_batch_labels) 
                              if label != positive_label and (label % 4) != prof_pos_score]
        
        if not negative_indices:
            # NO FALLBACK - skip if no valid negatives
            shn_failures += 1
            continue
            
        # 4. Apply semi-hard mining
        negative_embeddings = embeddings[negative_indices]
        anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)
        dists = F.pairwise_distance(anchor_repeated, negative_embeddings)
        positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
        
        semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + triplet_loss_fn.margin))
        semi_hard_dists = dists[semi_hard_mask]
        
        if semi_hard_dists.numel() == 0:
            # NO FALLBACK - skip if no semi-hard negatives
            shn_failures += 1
            continue
            
        # Select semi-hard negative
        hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
        semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
        selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
        negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
        negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
        
        # 5. Find second negative with strict criteria
        negative_indices_2 = [j for j, label in enumerate(current_batch_labels) 
                           if (label % 4) == (negative_label % 4) and 
                              (int(label >= 4) != int(negative_label >= 4))]
        
        if not negative_indices_2:
            # NO FALLBACK - skip if can't find second negative
            n2_failures += 1
            continue
            
        # Successfully formed quadruplet
        neg_2_idx = np.random.choice(negative_indices_2)
        negative_embedding_2 = embeddings[neg_2_idx].unsqueeze(0)
        
        anchors.append(anchor_embedding)
        positives.append(positive_embedding)
        negatives.append(negative_embedding)
        negatives_2.append(negative_embedding_2)
        is_ilo_flags.append(current_is_ilo)
    
    return {
        'anchors': anchors,
        'positives': positives,
        'negatives': negatives,
        'negatives_2': negatives_2,
        'is_ilo_flags': is_ilo_flags,
        'failures': {
            'ap_failures': ap_failures,
            'shn_failures': shn_failures,
            'n2_failures': n2_failures
        }
    }


def can_form_quadruplets(batch_labels, verbose=False):
    """Check if batch can form valid quadruplets"""
    label_counts = np.bincount(batch_labels, minlength=8)
    
    # Check for anchor-positive pairs
    has_anchor_positive = False
    for label in range(8):
        if label_counts[label] >= 2:
            has_anchor_positive = True
            break
    
    if not has_anchor_positive:
        if verbose:
            print("No anchor-positive pairs found in batch.")
        return False
    
    # Check for different profusion scores for BSHN-v2
    profusion_counts = np.zeros(4)
    for i in range(4):
        profusion_counts[i] = label_counts[i] + label_counts[i+4]
    
    if sum(1 for count in profusion_counts if count > 0) < 2:
        if verbose:
            print("Not enough profusion scores found.")
        return False
    
    # Check for TB+/TB- pairs
    for prof in range(4):
        if label_counts[prof] > 0 and label_counts[prof+4] > 0:
            return True
    
    return False


def create_combined_conf_mat_plot(multiclass_cm, binary_cm, set_name, epoch, log_to_wandb=False, log_tp_tn_fp_fn=True):
    """
    Create a single figure with both binary and multiclass confusion matrices side by side
    
    Args:
        multiclass_cm: Confusion matrix for multiclass profusion
        binary_cm: Confusion matrix for binary profusion
        set_name: Name of the dataset (train, validation, test)
        epoch: Current epoch or identifier
        log_to_wandb: Whether to log the figure to wandb
        
    Returns:
        fig: The created matplotlib figure
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define class names for both types
    mc_class_names = ["Prof 0", "Prof 1", "Prof 2", "Prof 3"]
    binary_class_names = ["No Profusion", "Profusion Present"]
    
    # MULTICLASS CONFUSION MATRIX (LEFT)
    ax = axes[0]
    im = ax.imshow(multiclass_cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    # Set title
    ax.set_title(f'Multiclass Profusion Classification\n{set_name.title()} Set - Epoch {epoch}', fontsize=14, pad=20)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(mc_class_names)))
    ax.set_yticks(np.arange(len(mc_class_names)))
    ax.set_xticklabels(mc_class_names, rotation=45, ha='right')
    ax.set_yticklabels(mc_class_names)
    
    # Add text annotations
    thresh = multiclass_cm.max() / 2
    for i in range(multiclass_cm.shape[0]):
        for j in range(multiclass_cm.shape[1]):
            text_color = "white" if multiclass_cm[i, j] > thresh else "black"
            ax.text(j, i, format(multiclass_cm[i, j], 'd'),
                   ha="center", va="center", color=text_color, fontsize=12)
    
    # Set labels
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    # KAPPA CALCULATION
    n_classes = multiclass_cm.shape[0]
    n_samples = np.sum(multiclass_cm)

    row_sums = np.sum(multiclass_cm, axis=1)
    col_sums = np.sum(multiclass_cm, axis=0)
    pe = np.sum(row_sums * col_sums) / (n_samples * n_samples)
    po = np.trace(multiclass_cm) / n_samples

    multiclass_kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    
    # Calculate and add accuracy text
    multiclass_accuracy = np.trace(multiclass_cm) / np.sum(multiclass_cm) if np.sum(multiclass_cm) > 0 else 0.0
    ax.text(0.02, 0.98, f'Accuracy: {multiclass_accuracy*100:.3f}%\n Kappa: {multiclass_kappa:.3f}', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # BINARY CONFUSION MATRIX (RIGHT)
    ax = axes[1]
    im = ax.imshow(binary_cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    # Set title
    ax.set_title(f'Binary Profusion Classification\n{set_name.title()} Set - Epoch {epoch}', fontsize=14, pad=20)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(binary_class_names)))
    ax.set_yticks(np.arange(len(binary_class_names)))
    ax.set_xticklabels(binary_class_names, rotation=45, ha='right')
    ax.set_yticklabels(binary_class_names)
    
    # Add text annotations
    thresh = binary_cm.max() / 2
    for i in range(binary_cm.shape[0]):
        for j in range(binary_cm.shape[1]):
            text_color = "white" if binary_cm[i, j] > thresh else "black"
            ax.text(j, i, format(binary_cm[i, j], 'd'),
                   ha="center", va="center", color=text_color, fontsize=12)
    
    # Set labels
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Calculate and add detailed metrics for binary
    tn, fp, fn, tp = binary_cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    po = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / ((tp + tn + fp + fn) ** 2)
    binary_kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
    
    if log_tp_tn_fp_fn:
        wandb.log({
            f"{set_name}_TP": tp,
            f"{set_name}_TN": tn,
            f"{set_name}_FP": fp,
            f"{set_name}_FN": fn,
            "epoch": epoch + 1 if epoch is not None else 0,
        })

    metrics_text = f'Accuracy: {accuracy*100:.3f}%\nSensitivity: {sensitivity*100:.3f}%\nSpecificity: {specificity*100:.3f}%\nKappa: {binary_kappa:.3f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Log to wandb if requested
    if log_to_wandb:
        wandb.log({
            f"{set_name}_combined_confusion_matrix": wandb.Image(fig)
        })
    
    return fig


def get_confusion_matrix(predictions, labels):
    """Compute confusion matrix for given predictions and labels"""
    if predictions is None or labels is None:
        return None
    
    # Convert to numpy for sklearn compatibility
    if isinstance(predictions, torch.Tensor):
        if predictions.dim() > 1:
            # Multi-class predictions (logits)
            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
        else:
            # Binary predictions (probabilities)
            pred_classes = (predictions > 0.5).cpu().numpy().astype(int)
    else:
        pred_classes = predictions
    
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Compute confusion matrix
    cm = confusion_matrix(labels_np, pred_classes)
    
    return cm



def plot_confusion_matrices_combined(prof_preds, prof_labels, set_name, epoch, log_to_wandb=False):
    """
    Create and log combined confusion matrices for multiclass and binary classification
    
    Args:
        prof_preds: Predictions from the model
        prof_labels: True labels
        set_name: Name of the dataset split (e.g., "train", "validation", "test")
        epoch: Current epoch number
        log_to_wandb: Whether to log the figures to Weights & Biases
    
    Returns:
        Combined figure with both confusion matrices
    """
    # Calculate multiclass confusion matrix
    multiclass_cm = get_confusion_matrix(prof_labels, prof_preds)
    prof_labels_binary = (prof_labels > 0).long()  # Always use > 0
    prof_preds_binary = (torch.argmax(prof_preds, dim=1) > 0).long()  # Always use > 0
    
    binary_cm = get_confusion_matrix(prof_preds_binary, prof_labels_binary)

     # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define class names for both types
    mc_class_names = ["Prof 0", "Prof 1", "Prof 2", "Prof 3"]
    binary_class_names = ["No Profusion", "Profusion Present"]
    
    # MULTICLASS CONFUSION MATRIX (LEFT)
    ax = axes[0]
    im = ax.imshow(multiclass_cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    # Set title
    ax.set_title(f'Multiclass Profusion Classification\n{set_name.title()} Set - Epoch {epoch}', fontsize=14, pad=20)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(mc_class_names)))
    ax.set_yticks(np.arange(len(mc_class_names)))
    ax.set_xticklabels(mc_class_names, rotation=45, ha='right')
    ax.set_yticklabels(mc_class_names)
    
    # Add text annotations
    thresh = multiclass_cm.max() / 2
    for i in range(multiclass_cm.shape[0]):
        for j in range(multiclass_cm.shape[1]):
            text_color = "white" if multiclass_cm[i, j] > thresh else "black"
            ax.text(j, i, format(multiclass_cm[i, j], 'd'),
                   ha="center", va="center", color=text_color, fontsize=12)
    
    # Set labels
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Calculate and add accuracy text
    multiclass_accuracy = np.trace(multiclass_cm) / np.sum(multiclass_cm) if np.sum(multiclass_cm) > 0 else 0.0
    ax.text(0.02, 0.98, f'Accuracy: {multiclass_accuracy*100:.3f}%', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # BINARY CONFUSION MATRIX (RIGHT)
    ax = axes[1]
    im = ax.imshow(binary_cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    # Set title
    ax.set_title(f'Binary Profusion Classification\n{set_name.title()} Set - Epoch {epoch}', fontsize=14, pad=20)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(binary_class_names)))
    ax.set_yticks(np.arange(len(binary_class_names)))
    ax.set_xticklabels(binary_class_names, rotation=45, ha='right')
    ax.set_yticklabels(binary_class_names)
    
    # Add text annotations
    thresh = binary_cm.max() / 2
    for i in range(binary_cm.shape[0]):
        for j in range(binary_cm.shape[1]):
            text_color = "white" if binary_cm[i, j] > thresh else "black"
            ax.text(j, i, format(binary_cm[i, j], 'd'),
                   ha="center", va="center", color=text_color, fontsize=12)
    
    # Set labels
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Calculate and add detailed metrics for binary
    tn, fp, fn, tp = binary_cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics_text = f'Accuracy: {accuracy*100:.3f}%\nSensitivity: {sensitivity*100:.3f}%\nSpecificity: {specificity*100:.3f}%'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Log to wandb if requested
    if log_to_wandb:
        wandb.log({
            f"{set_name}_combined_confusion_matrix": wandb.Image(fig)
        })
    
    return fig



def create_conf_mat_plot(cm, clf_name, set_name, epoch, log_to_wandb=False):
        if clf_name == "binary_profusion":
            class_names = ["No Profusion", "Profusion Present"]
            title = "Binary Profusion Classification"
        elif clf_name == "binary_tb":
            class_names = ["No TB", "TB"]
            title = "Binary TB Classification"
        elif clf_name == "multiclass_profusion":
            class_names = ["Prof 0", "Prof 1", "Prof 2", "Prof 3"]
            title = "Multiclass Profusion Classification"
        elif clf_name == "multiclass_mstb":
            class_names = ["Prof 0, TB-", "Prof 1, TB-", "Prof 2, TB-", "Prof 3, TB-", "Prof 0, TB+", "Prof 1, TB+", "Prof 2, TB+", "Prof 3, TB+"]
            title = "Multiclass MSTB Classification"
        else:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
            title = f"{clf_name} Classification"


         # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20)
        
        # Set title
        ax.set_title(f'{title}\n{set_name.title()} Set - Epoch {epoch}', fontsize=14, pad=20)
        
        # Set tick labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color=text_color, fontsize=12)
        
        # Set labels
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # Calculate and add accuracy text
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            metrics_text = f'Accuracy: {accuracy*100:.3f}%\nSensitivity: {sensitivity*100:.3f}%\nSpecificity: {specificity*100:.3f}%'
        else:
            accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
            metrics_text = f'Accuracy: {accuracy*100:.3f}%'

        # Add metrics text box
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()

        if log_to_wandb:
            wandb.log({
                f"{set_name}_{clf_name}_confusion_matrix": wandb.Image(fig)
            })     
            
        return fig