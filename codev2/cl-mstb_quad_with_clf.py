import sys
import os
import gc
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
from tsne import visualize_tsne, MultiClassBaseClassifier
import math
import random
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses, miners


import torch
from torch import nn, Tensor
import torch.nn.functional as F

    
class BinaryClassifier(nn.Module):
    def __init__(self, in_features):
        super(BinaryClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, 1024)  # Input layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc1 = nn.Linear(1024, 512)  # First hidden layer
        self.dropout1 = nn.Dropout(0.3)  # Add dropout for regularization
        self.fc2 = nn.Linear(512, 256)   # Second hidden layer
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 1)     # Output layer (single node for binary classification)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)  # Apply dropout after the first layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # Raw logit output (no sigmoid here)
        return x

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
                 type: str = 'anchor-push',  # 'anchor-push' or 'structured'
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)
        self.margin1 = margin1
        self.margin2 = margin2
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction
        assert type in ['anchor-push', 'structured']
        self.type = type

    def forward(self, anchor: Tensor, positive: Tensor, negative1: Tensor, negative2: Tensor) -> Tensor:
        # First term: standard triplet
        loss1 = F.triplet_margin_loss(anchor, positive, negative1,
                                      margin=self.margin1,
                                      p=self.p,
                                      eps=self.eps,
                                      swap=self.swap,
                                      reduction='none')

        if self.type == 'anchor-push':
            # Second term: second negative pushed from anchor
            loss2 = F.triplet_margin_loss(anchor, positive, negative2,
                                          margin=self.margin2,
                                          p=self.p,
                                          eps=self.eps,
                                          swap=self.swap,
                                          reduction='none')
        elif self.type == 'structured':
            # Second term: negative2 pushed from negative1
            # Use D(N1, N2) instead of D(A, N2)
            d_ap = torch.norm(anchor - positive, p=self.p, dim=1)
            d_n1n2 = torch.norm(negative1 - negative2, p=self.p, dim=1)
            loss2 = torch.clamp(d_ap - d_n1n2 + self.margin2, min=0.0)

        # Combine terms and apply reduction
        loss = loss1 + loss2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def build_profusion_tb_map(dataset):
    """
    Create dictionaries mapping samples by their profusion score and TB status separately.
    
    Args:
        dataset: PyTorch dataset with multiclass_stb labels (0-7) accessible via dataset[idx][1]
        
    Returns:
        Dictionary with two sub-dictionaries:
        - 'profusion': Maps profusion scores (0-3) to lists of indices
        - 'tb_status': Maps TB status (0=negative, 1=positive) to lists of indices
        - 'combined': Maps (profusion, tb_status) tuples to lists of indices
    """
    profusion_map = {0: [], 1: [], 2: [], 3: []}
    tb_map = {0: [], 1: []}  # 0=TB negative, 1=TB positive
    combined_map = {}
    
    print("Building profusion and TB status maps...")
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        # For multiclass_stb: 0-3 are profusion with no TB, 4-7 are profusion with TB
        profusion_score = label % 4
        tb_status = 1 if label >= 4 else 0
        
        # Store in profusion map
        profusion_map[profusion_score].append(idx)
        
        # Store in TB status map
        tb_map[tb_status].append(idx)
        
        # Store in combined map
        combined_key = (profusion_score, tb_status)
        if combined_key not in combined_map:
            combined_map[combined_key] = []
        combined_map[combined_key].append(idx)
    
    # Print statistics
    # print("\nProfusion score distribution:")
    # for prof_score, indices in profusion_map.items():
    #     print(f"  Profusion {prof_score}: {len(indices)} samples")
    
    # print("\nTB status distribution:")
    # print(f"  TB Negative: {len(tb_map[0])} samples")
    # print(f"  TB Positive: {len(tb_map[1])} samples")
    
    # print("\nCombined (profusion, TB) distribution:")
    # for (prof_score, tb_status), indices in combined_map.items():
    #     tb_text = "TB+" if tb_status == 1 else "TB-"
    #     print(f"  Profusion {prof_score}, {tb_text}: {len(indices)} samples")
    
    return {
        'profusion': profusion_map,
        'tb_status': tb_map,
        'combined': combined_map
    }



def build_label_to_indices_map(dataset):
    """
    Create a dictionary mapping each label to all indices with that label in the dataset.
    
    Args:
        dataset: PyTorch dataset with labels accessible via dataset[idx][1]
        
    Returns:
        Dictionary mapping label values to lists of indices
    """
    label_to_indices = {}
    
    print("Building label-to-indices map...")
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    # Print statistics
    # print(f"Label distribution in dataset:")
    # for label, indices in label_to_indices.items():
    #     print(f"Label {label}: {len(indices)} samples")
        
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



def train_model_quadruplet_with_clf(
    model,
    train_loader,
    val_loader,
    triplet_loss_fn,
    quadruplet_loss_fn,
    optimizer,
    device,
    n_epochs,
    experiment_name,
    checkpoint_dir="checkpoints",
    tsne_interval=50,
    log_to_wandb=True,
    mining_strat="BSHN-v2",
    margin_scheduling=False,
    initial_margin=0.8,
    final_margin=0.2,
    scheduling_fraction=0.8,
    p_ilo_anchor=0.5
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

    # Add binary classifier to the model
    model.binary_classifier = BinaryClassifier(in_features=2048).to(device)
    
    # Update optimizer to include binary classifier parameters
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': model.binary_classifier.parameters()}
    ], lr=optimizer.param_groups[0]['lr'], weight_decay=optimizer.param_groups[0]['weight_decay'])

    # Create label-to-indices map for finding appropriate anchors
    labels_to_indices = build_label_to_indices_map(train_loader.dataset.dataset)
    prof_tb_labels = build_profusion_tb_map(train_loader.dataset.dataset)

    # Margin scheduling functions
    def get_sin_scheduled_margin(current_epoch):
        if not margin_scheduling:
            return triplet_loss_fn.margin
        
        schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
        sin_factor = math.sin(schedule_point * math.pi/2)
        
        if initial_margin > final_margin:
            current_margin = initial_margin - (initial_margin - final_margin) * sin_factor
        else:
            current_margin = initial_margin + (final_margin - initial_margin) * sin_factor
            
        return current_margin
    
    def get_scheduled_p_ilo(current_epoch, initial_p_ilo_anchor):
        """
        Calculate the scheduled probability of using ILO anchors.
        Starts at p_ilo_anchor and decays to 0.0 over the first half of training.
        """
        schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
        sin_factor = math.sin(schedule_point * math.pi/2)

        new_p_ilo_anchor = initial_p_ilo_anchor * (1 - sin_factor)
        return new_p_ilo_anchor

    # Create checkpoint directory
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tracking metrics
    num_classes = 8  # For multiclass_stb
    best_val_map = 0.0
    best_model_state = None
    history = {
        'train_loss': [],
        'train_binary_loss': [],
        'train_map': [],
        'val_loss': [],
        'val_map': [],
        'train_class_map': [],
        'val_class_map': [],
        'train_prof_map': [],
        'val_prof_map': [],
        'train_quadruplet_loss': [],
        'val_quadruplet_loss': [],
        'train_binary_acc': [],
        'val_binary_acc': [],
        'train_binary_auc': [],
        'val_binary_auc': [],
    }
    
    # Initialize class-specific metrics
    per_class_metrics = {class_id: {'train_ap': [], 'val_ap': []} for class_id in range(num_classes)}

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("=" * 50)

        # Apply margin scheduling if enabled
        if margin_scheduling:
            current_margin = get_sin_scheduled_margin(epoch)
            triplet_loss_fn.margin = current_margin
            quadruplet_loss_fn.margin1 = current_margin
            quadruplet_loss_fn.margin2 = current_margin * 0.25
            print(f"Current margins: {current_margin:.4f}, {quadruplet_loss_fn.margin2:.4f}")
        
        model.train()
        epoch_total_loss = 0.0
        epoch_binary_loss = 0.0
        epoch_binary_acc = 0.0
        epoch_binary_auc = 0.0
        epoch_batch_count = 0
        epoch_quadruplet_count = 0

        all_embeddings = []
        all_labels = []
        
        # Lists to accumulate binary classification metrics
        all_binary_preds = []
        all_binary_labels = []
        
        # Training loop
        for batch_idx, sample in enumerate(train_loader):
            # Zero gradients for this batch
            optimizer.zero_grad()
            
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
            n_ilo_anchors = 0
            
            # For each sample in the batch, build a quadruplet
            for i, positive_label in enumerate(current_batch_labels):
                positive_embedding = embeddings[i].unsqueeze(0)

                if(wandb.config.p_ilo_scheduling):
                    p_ilo_anchor = get_scheduled_p_ilo(epoch, p_ilo_anchor)
            

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
                else:
                
                    # Find anchor with same label
                    batch_matching_indices = [j for j in range(len(current_batch_labels)) 
                                            if current_batch_labels[j] == positive_label and j != i]
                    
                    if batch_matching_indices:
                        # Found matching label in batch
                        batch_anchor_idx = np.random.choice(batch_matching_indices)
                        anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                        anchor_label = current_batch_labels[batch_anchor_idx]
                    else:
                        # Try finding matching sample in dataset
                        matching_indices = labels_to_indices.get(positive_label.item(), [])
                        if i in matching_indices:
                            matching_indices.remove(i)
                        
                        if matching_indices:
                            chosen_index = np.random.choice(matching_indices)
                            anchor_img, anchor_label = train_loader.dataset.dataset[chosen_index]
                            anchor_img = anchor_img.unsqueeze(0).to(device)
                            anchor_embedding = model.features(anchor_img)
                            anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                        else:
                            # Skip if no matching anchor found
                            continue
                
                # Find first negative using chosen mining strategy
                if mining_strat == "BSHN-v2":
                    # Look for negatives with different profusion score
                    prof_pos_score = positive_label % 4
                    pos_tb_status = 1 if positive_label >= 4 else 0
                    
                    negative_indices = [j for j, label in enumerate(current_batch_labels) 
                                      if label != positive_label and (label % 4) != prof_pos_score]
                    
                    if not negative_indices:
                        # Fallback to any negative
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
                        
                        negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                        negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                        
                        # Find second negative with same profusion but different TB status
                        negative_indices_2 = [j for j, label in enumerate(current_batch_labels) 
                                           if (label % 4) == (negative_label % 4) and 
                                              (int(label >= 4) != int(negative_label >= 4))]
                        
                        if negative_indices_2:
                            neg_2_idx = np.random.choice(negative_indices_2)
                            negative_embedding_2 = embeddings[neg_2_idx].unsqueeze(0)
                            negative_label_2 = current_batch_labels[neg_2_idx]
                            
                            # Valid quadruplet found!
                            anchors.append(anchor_embedding)
                            positives.append(positive_embedding)
                            negatives.append(negative_embedding)
                            negatives_2.append(negative_embedding_2)
                            batch_quadruplet_count += 1
                            
                            # Log first quadruplet for debugging
                            if batch_idx == 0 and len(anchors) == 1:
                                print(f"Quadruplet example:")
                                print(f"A: {anchor_label}, P: {positive_label}, N1: {negative_label}, N2: {negative_label_2}")
                                print(f"A-Prof: {anchor_label % 4}, P-Prof: {positive_label % 4}, N1-Prof: {negative_label % 4}, N2-Prof: {negative_label_2 % 4}")
                                print(f"A-TB: {anchor_label >= 4}, P-TB: {positive_label >= 4}, N1-TB: {negative_label >= 4}, N2-TB: {negative_label_2 >= 4}")
            
            # Process collected quadruplets
            if batch_quadruplet_count > 0:
                # Combine all quadruplet components
                batch_anchors = torch.cat(anchors, dim=0)
                batch_positives = torch.cat(positives, dim=0)
                batch_negatives = torch.cat(negatives, dim=0)
                batch_negatives_2 = torch.cat(negatives_2, dim=0)
                
                # Calculate batch loss using all quadruplets
                quad_loss = quadruplet_loss_fn(batch_anchors, batch_positives, batch_negatives, batch_negatives_2)
                
                # For comparison/monitoring, calculate standard triplet loss
                triplet_loss = triplet_loss_fn(batch_anchors, batch_positives, batch_negatives)

                # Get binary labels (Profusion status: 1 if label % 4 > 0, else 0)
                binary_labels = torch.tensor([1 if (l % 4 > 0 ) else 0 for l in current_batch_labels], 
                                            device=device).float().unsqueeze(1)
                
                # Forward pass through binary classifier
                binary_logits = model.binary_classifier(feats)
                binary_preds = torch.sigmoid(binary_logits)
                
                # Calculate binary classification loss
                binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_labels)
                
                # Track binary predictions and labels for metrics
                all_binary_preds.append(binary_preds.detach().cpu())
                all_binary_labels.append(binary_labels.detach().cpu())
                
                # Combine losses with a weighting factor
                lambda_clf = 0.5  # Weight for classification loss
                total_loss = quad_loss + lambda_clf * binary_loss

                # Use total loss for optimization
                total_loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_total_loss += total_loss.item()
                epoch_binary_loss += binary_loss.item()
                epoch_batch_count += 1
                epoch_quadruplet_count += batch_quadruplet_count
                
                if log_to_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        "batch_quadruplet_loss": quad_loss.item(),
                        "batch_triplet_loss": triplet_loss.item(),
                        "batch_binary_loss": binary_loss.item(),
                        "batch_total_loss": total_loss.item(),
                        "batch_quadruplets": batch_quadruplet_count,
                        "batch": batch_idx + epoch * len(train_loader),
                        "batch_ilo_anchors": n_ilo_anchors,

                    })
            else:
                print(f"No valid quadruplets found in batch {batch_idx + 1}. Skipping.")
                print(f"Batch labels distribution: {np.bincount(current_batch_labels, minlength=8)}")
                torch.cuda.empty_cache()

        
        # End of epoch - calculate training metrics
        if epoch_batch_count > 0:
            train_loss = epoch_total_loss / epoch_batch_count
            train_binary_loss = epoch_binary_loss / epoch_batch_count
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Binary Loss: {train_binary_loss:.4f}")
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
            print("- Per-Class Train mAP:")
            for class_id in range(num_classes):
                ap = train_class_map_full.get(class_id, 0.0)
                class_name = multiclass_stb_mapping.get(class_id, f"Class {class_id}")
                print(f"  {class_name}: mAP = {ap:.4f}")
                
                # Store per-class metrics
                per_class_metrics[class_id]['train_ap'].append(ap)
            
            # Calculate binary classification metrics at end of epoch
            if len(all_binary_preds) > 0:
                all_binary_preds = torch.cat(all_binary_preds, dim=0)
                all_binary_labels = torch.cat(all_binary_labels, dim=0)
                
                # Calculate accuracy
                binary_pred_labels = (all_binary_preds >= 0.5).float()
                train_binary_acc = accuracy_score(all_binary_labels.numpy(), binary_pred_labels.numpy())
                
                # Calculate AUC
                try:
                    train_binary_auc = roc_auc_score(all_binary_labels.numpy(), all_binary_preds.numpy())
                except:
                    train_binary_auc = 0.0
                
                history['train_binary_acc'].append(train_binary_acc)
                history['train_binary_auc'].append(train_binary_auc)
                
                print(f"Binary Classification - Accuracy: {train_binary_acc:.4f}, AUC: {train_binary_auc:.4f}")
        
            # Update history
            history['train_loss'].append(train_loss)
            history['train_binary_loss'].append(train_binary_loss)
            
            history['train_map'].append(train_map)
            history['train_class_map'].append(train_class_map_full)
            history['train_prof_map'].append(train_prof_map)
        else:
            print(f"Epoch {epoch + 1}/{n_epochs}: No valid quadruplets formed")
        
        # Run t-SNE visualization at regular intervals
        if (epoch + 1) % tsne_interval == 0:
            visualize_tsne(model, device, ilo_dataset, train_loader, 
                          trained=True, log_to_wandb=log_to_wandb, 
                          n_epochs=epoch+1, set_name="training", entire_dataset=False)
            visualize_tsne(model, device, ilo_dataset, val_loader, 
                          trained=True, log_to_wandb=log_to_wandb,
                          n_epochs=epoch+1, set_name="validation", entire_dataset=False)
        
        # Validation loop
        print("\nVALIDATION\n")
        val_metrics = validate_quadruplet_with_clf(
            model=model,
            val_loader=val_loader,
            device=device,
            triplet_loss_fn=triplet_loss_fn,
            quadruplet_loss_fn=quadruplet_loss_fn,
            num_classes=num_classes,
            multiclass_stb_mapping=multiclass_stb_mapping,
            mining_strat=mining_strat
        )
        
        val_loss = val_metrics['val_quadruplet_loss']
        val_map = val_metrics['val_map']
        val_class_map = val_metrics['val_class_map']
        val_prof_map = val_metrics['val_prof_map']
        val_binary_acc = val_metrics.get('val_binary_acc', 0.0)
        val_binary_auc = val_metrics.get('val_binary_auc', 0.0)
        
        # Store validation metrics for all classes
        for class_id in range(num_classes):
            per_class_metrics[class_id]['val_ap'].append(val_class_map.get(class_id, 0.0))
        
        # Update history
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map)
        history['val_class_map'].append(val_class_map)
        history['val_prof_map'].append(val_prof_map)
        history['val_quadruplet_loss'].append(val_loss)
        history['val_binary_acc'].append(val_binary_acc)
        history['val_binary_auc'].append(val_binary_auc)
        
        # Log metrics to wandb
        if log_to_wandb:
            wandb_log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss if epoch_batch_count > 0 else 0.0,
                "train_map": train_map if epoch_batch_count > 0 else 0.0,
                "val_loss": val_loss,
                "val_map": val_map,
                "val_prof_map": val_prof_map,
                "train_prof_map": train_prof_map if epoch_batch_count > 0 else 0.0,
                "current_margin1": quadruplet_loss_fn.margin1,
                "current_margin2": quadruplet_loss_fn.margin2,
                "current_p_ilo_anchor": p_ilo_anchor if wandb.config.p_ilo_scheduling else wandb.config.p_ilo_anchor,
                "train_binary_acc": train_binary_acc if epoch_batch_count > 0 else 0.0,
                "train_binary_auc": train_binary_auc if epoch_batch_count > 0 else 0.0,
                "val_binary_acc": val_binary_acc,
                "val_binary_auc": val_binary_auc,
            }

            
            # Log per-class metrics
            for class_id in range(num_classes):
                train_ap = train_class_map_full.get(class_id, 0.0) if epoch_batch_count > 0 else 0.0
                val_ap = val_class_map.get(class_id, 0.0)
                
                wandb_log_dict[f"train_class_{class_id}_map"] = train_ap
                wandb_log_dict[f"val_class_{class_id}_map"] = val_ap
            
            wandb.log(wandb_log_dict)
        
        # Save best model based on validation mAP
        if epoch == 0 or (val_map > best_val_map and epoch > 50):
            best_val_map = val_map
            print(f"Saving best model with validation mAP: {best_val_map:.4f}")
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, f"best_model.pth"))


            visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, True, True, n_epochs=epoch+1, set_name="best val mAP", entire_dataset=True)
        
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        
        # Save latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, f"final_model.pth"))

    visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, True, True, n_epochs=n_epochs+1, set_name="final", entire_dataset=True)

    
    # Return training results
    return {
        'model': model,
        'best_model_state': best_model_state,
        'history': history,
        'best_val_map': best_val_map,
        'per_class_metrics': per_class_metrics
    }

def validate_quadruplet_with_clf(model, val_loader, device, triplet_loss_fn, quadruplet_loss_fn, num_classes=8, multiclass_stb_mapping=None, mining_strat="BSHN-v2"):
    """
    Validation loop for quadruplet loss model
    
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
    running_total_loss = 0.0
    running_triplet_loss = 0.0
    running_binary_loss = 0.0
    all_embeddings = []
    all_labels = []
    all_binary_preds = []
    all_binary_labels = []
    val_quadruplet_count = 0
    batch_with_quadruplets = 0
    
    # Build a map of validation labels to their indices
    labels_to_indices = build_label_to_indices_map(val_loader.dataset.dataset)
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            # Get validation batch
            imgs = sample[0].to(device)
            labels = sample[1].long().to(device)
            
            # Extract features and normalize embeddings
            features = model.features(imgs)
            embeddings = F.normalize(features, p=2, dim=1)
            
            # Store for mAP calculation
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
            # Get labels from batch
            current_batch_labels = labels.cpu().numpy()
            
            # Collect quadruplet components
            anchors = []
            positives = []
            negatives = []
            negatives_2 = []
            batch_quadruplet_count = 0
            
            # Form quadruplets using the same strategy as in training
            for i, positive_label in enumerate(current_batch_labels):
                # Positive embedding from current batch
                positive_embedding = embeddings[i].unsqueeze(0)
                
                # Find anchor with same label
                batch_matching_indices = [j for j in range(len(current_batch_labels)) 
                                        if current_batch_labels[j] == positive_label and j != i]
                
                if batch_matching_indices:
                    batch_anchor_idx = np.random.choice(batch_matching_indices)
                    anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                    anchor_label = current_batch_labels[batch_anchor_idx]
                else:
                    # Try finding matching sample in dataset
                    matching_indices = labels_to_indices.get(positive_label.item(), [])
                    if i in matching_indices:
                        matching_indices.remove(i)
                    
                    if matching_indices:
                        chosen_index = np.random.choice(matching_indices)
                        anchor_img, anchor_label = val_loader.dataset.dataset[chosen_index]
                        anchor_img = anchor_img.unsqueeze(0).to(device)
                        anchor_embedding = model.features(anchor_img)
                        anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                    else:
                        continue
                
                # Find first negative using the mining strategy
                if mining_strat == "BSHN-v2":
                    prof_pos_score = positive_label % 4
                    
                    negative_indices = [j for j, label in enumerate(current_batch_labels) 
                                      if label != positive_label and (label % 4) != prof_pos_score]
                    
                    if not negative_indices:
                        # Fallback to any negative
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
            
            # Process collected quadruplets
            if batch_quadruplet_count > 0:
                # Combine all quadruplet components
                batch_anchors = torch.cat(anchors, dim=0)
                batch_positives = torch.cat(positives, dim=0)
                batch_negatives = torch.cat(negatives, dim=0)
                batch_negatives_2 = torch.cat(negatives_2, dim=0)

                # Get binary labels (Profusion status: 1 if label % 4 > 0, else 0)
                binary_labels = torch.tensor([1 if (l % 4 > 0) else 0 for l in current_batch_labels], 
                                            device=device).float().unsqueeze(1)
                
                # Forward pass through binary classifier
                binary_logits = model.binary_classifier(features)
                binary_preds = torch.sigmoid(binary_logits)
                
                # Calculate binary classification loss
                binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_labels)
                
                # Store binary predictions and labels for metrics
                all_binary_preds.append(binary_preds.cpu())
                all_binary_labels.append(binary_labels.cpu())
                
                # Calculate losses
                quad_loss = quadruplet_loss_fn(batch_anchors, batch_positives, batch_negatives, batch_negatives_2)
                triplet_loss = triplet_loss_fn(batch_anchors, batch_positives, batch_negatives)

                total_loss = quad_loss + binary_loss

                running_quad_loss += quad_loss.item()
                running_total_loss += total_loss.item()
                running_triplet_loss += triplet_loss.item()
                running_binary_loss += binary_loss.item()
                batch_with_quadruplets += 1
                val_quadruplet_count += batch_quadruplet_count
    
    # Calculate validation metrics
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate mAP for full labels
        val_map, val_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
        
        # Calculate mAP for profusion scores only
        prof_all_labels = all_labels % 4
        val_prof_map, val_prof_class_map = helpers.compute_map_per_class(all_embeddings, prof_all_labels)
        
        # Calculate average validation losses
        avg_quad_loss = running_quad_loss / max(1, batch_with_quadruplets)
        avg_triplet_loss = running_triplet_loss / max(1, batch_with_quadruplets)
        avg_total_loss = running_total_loss / max(1, batch_with_quadruplets)
        
        print(f"\nValidation Summary:")
        print(f"- Total quadruplets formed: {val_quadruplet_count}")
        print(f"- Avg Quadruplet Loss: {avg_quad_loss:.4f}")
        print(f"- Avg Triplet Loss: {avg_triplet_loss:.4f}")
        print(f"- Validation mAP: {val_map:.4f}")
        print(f"- Validation Profusion mAP: {val_prof_map:.4f}")
        print("- Per-Class Validation mAP:")
        
        # Create full map including any missing classes
        val_class_map_full = {class_id: val_class_map.get(class_id, 0.0) for class_id in range(num_classes)}
        
        for class_id in range(num_classes):
            ap = val_class_map_full.get(class_id, 0.0)
            class_name = multiclass_stb_mapping.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: mAP = {ap:.4f}")
        
        # Calculate binary classification metrics
        val_binary_acc = 0.0
        val_binary_auc = 0.0
        val_binary_loss = running_binary_loss / max(1, len(val_loader))
        
        if all_binary_preds:
            all_binary_preds = torch.cat(all_binary_preds, dim=0)
            all_binary_labels = torch.cat(all_binary_labels, dim=0)
            
            # Calculate accuracy
            binary_pred_labels = (all_binary_preds >= 0.5).float()
            val_binary_acc = accuracy_score(all_binary_labels.numpy(), binary_pred_labels.numpy())
            
            # Calculate AUC
            try:
                val_binary_auc = roc_auc_score(all_binary_labels.numpy(), all_binary_preds.numpy())
            except Exception as e:
                print(f"Error calculating AUC: {e}")
                val_binary_auc = 0.0
                
            print(f"- Validation Binary Classification Metrics:")
            print(f"  Accuracy: {val_binary_acc:.4f}")
            print(f"  AUC: {val_binary_auc:.4f}")
            print(f"  Loss: {val_binary_loss:.4f}")
        
        # Log validation metrics
        wandb.log({
            "val_quadruplet_loss": avg_quad_loss,
            "val_triplet_loss": avg_triplet_loss,
            "val_quadruplets": val_quadruplet_count,
            "val_total_loss": avg_total_loss,
            "val_binary_loss": val_binary_loss,
            "val_binary_acc": val_binary_acc,
            "val_binary_auc": val_binary_auc,
        })
        
        return {
            'val_quadruplet_loss': avg_quad_loss,
            'val_triplet_loss': avg_triplet_loss,
            'val_map': val_map,
            'val_class_map': val_class_map_full,
            'val_prof_map': val_prof_map,
            'val_prof_class_map': val_prof_class_map,
            'val_embeddings': all_embeddings,
            'val_labels': all_labels,
            'val_binary_acc': val_binary_acc,
            'val_binary_auc': val_binary_auc,
            'val_binary_loss': val_binary_loss
        }
    
    # Return empty metrics if no quadruplets were formed
    return {
        'val_quadruplet_loss': 0.0,
        'val_triplet_loss': 0.0,
        'val_map': 0.0,
        'val_class_map': {class_id: 0.0 for class_id in range(num_classes)},
        'val_prof_map': 0.0,
        'val_prof_class_map': {class_id: 0.0 for class_id in range(4)},
        'val_embeddings': None,
        'val_labels': None,
        'val_binary_acc': 0.0,
        'val_binary_auc': 0.0,
        'val_binary_loss': 0.0
    }

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")

    preprocess = transforms.Compose([
    # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]
     

        # Create an HDF5SilicosisDataset instance
        mbod_dataset_merged = HDF5Dataset(
            hdf5_path=hdf5_file_path,
            labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )


        ilo_dataset = HDF5Dataset(
            hdf5_path=ilo_hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )

        wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
        wandb.init(project='MBOD-cl', name='mstb_quad-p_ilo_50-m_005_5',
            config={
                "experiment_type": "Quadruplet with Binary Profusion Classifier",
                "batch_size": 24,
                "n_epochs": 1500,
                "learning_rate": 1e-4,
                "oversample": True,
                "initial_margin": 0.05,      
                "final_margin": 0.5,        
                "margin_scheduling": True,   # Enable margin scheduling
                "scheduling_fraction": 0.75,  # Complete scheduling in first x% of training
                "mining": "BSHN-v2",
                "augmentations": True,
                "filtered_dataset": True,
                "loss_function": "Triplet",
                "p_ilo_anchor": 0.5,
                "p_ilo_final": 0.0,
                "num_classes": 8,  # Explicitly specify 8 classes
                "OS_factor": 0.65,  # Oversampling factor
                "p_ilo_scheduling": False,  # Enable p_ilo scheduling
            })

        experiment_name = wandb.run.name

        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=wandb.config.learning_rate,  # Try a smaller learning rate
            weight_decay=wandb.config.learning_rate  # Add L2 regularization
        )
        triplet_loss_fn = nn.TripletMarginLoss(margin=wandb.config.initial_margin, p=2)

        margin_1 = wandb.config.initial_margin
        margin_2 = wandb.config.initial_margin * 0.25
        quadruplet_loss_fn = QuadrupletMarginLoss(margin1=margin_1, margin2=margin_2, p=2, type='structured')

        n_epochs = wandb.config.n_epochs
        margin = wandb.config.initial_margin
        batch_size = wandb.config.batch_size
        
        preprocess = transforms.Compose([
           # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
           # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        mbod_merged_loader = torch.utils.data.DataLoader(mbod_dataset_merged, batch_size=wandb.config.batch_size, shuffle=True)

        print("+"*50)
        print(f"MBOD dataset size: {len(mbod_dataset_merged)}")
        print(f"ILO dataset size: {len(ilo_dataset)}")
        print("+"*50)
        

        if(wandb.config.augmentations):

            augmentations_list = transforms.Compose([
                transforms.RandomRotation(degrees=10, expand=False, fill=0),
                # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
            ])
            # Get the dataloaders
            train_loader, _, _ = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=augmentations_list,
                oversample=wandb.config.oversample,
                balanced_batches=True if wandb.config.loss_function == "PCCT" else False,
                scaling_factor = wandb.config.OS_factor 
            )

            _, val_loader, test_loader = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=None,
                oversample=None,
                balanced_batches=True if wandb.config.loss_function == "PCCT" else False,
                scaling_factor = wandb.config.OS_factor 
            )

        else:
            train_loader, _, _ = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=None,
                oversample=wandb.config.oversample,
                balanced_batches=True if wandb.config.loss_function == "PCCT" else False,
                scaling_factor = wandb.config.OS_factor
            )

            _, val_loader, test_loader = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=None,
                oversample=None,
                balanced_batches=True if wandb.config.loss_function == "PCCT" else False,
                scaling_factor = wandb.config.OS_factor
            )

        print("Preloading ILO images onto the GPU...")
        ilo_images = []
        ilo_labels = []

        for idx in range(len(ilo_dataset)):
            image, label = ilo_dataset[idx]
            
            # Convert image to a PyTorch tensor and move to GPU
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).to(device)
            label_tensor = torch.tensor(label, dtype=torch.long).to(device)

            ilo_images.append(image_tensor)
            ilo_labels.append(label_tensor)

        # Stack all tensors into a single tensor for efficient access
        ilo_images = torch.cat(ilo_images, dim=0)  # Shape: (N, 1, H, W)
        ilo_labels = torch.stack(ilo_labels)       # Shape: (N,)

        print(f"ILO images loaded onto GPU: {ilo_images.shape}")
        print(f"ILO labels loaded onto GPU: {ilo_labels.shape}")

        visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, trained=False, log_to_wandb=True, set_name="pre-training", entire_dataset=True, is_mstb=False)
        visualize_tsne(model, device, ilo_dataset, train_loader, trained=False, log_to_wandb=True, set_name="pre-training", entire_dataset=False, is_mstb=False)
        
        
        results = train_model_quadruplet_with_clf(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            triplet_loss_fn=triplet_loss_fn,
            quadruplet_loss_fn=quadruplet_loss_fn,
            optimizer=optimizer,
            device=device,
            n_epochs=n_epochs,
            experiment_name=experiment_name,
            margin_scheduling=wandb.config.margin_scheduling,
            initial_margin=wandb.config.initial_margin,
            final_margin=wandb.config.final_margin,
            scheduling_fraction=wandb.config.scheduling_fraction,
            mining_strat=wandb.config.mining,
            p_ilo_anchor=wandb.config.p_ilo_anchor
        )

       

    except KeyError as e:
        print(f"Missing configuration: {e}")


