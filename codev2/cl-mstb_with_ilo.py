import sys
import os

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
    print("\nProfusion score distribution:")
    for prof_score, indices in profusion_map.items():
        print(f"  Profusion {prof_score}: {len(indices)} samples")
    
    print("\nTB status distribution:")
    print(f"  TB Negative: {len(tb_map[0])} samples")
    print(f"  TB Positive: {len(tb_map[1])} samples")
    
    print("\nCombined (profusion, TB) distribution:")
    for (prof_score, tb_status), indices in combined_map.items():
        tb_text = "TB+" if tb_status == 1 else "TB-"
        print(f"  Profusion {prof_score}, {tb_text}: {len(indices)} samples")
    
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
    print(f"Label distribution in dataset:")
    for label, indices in label_to_indices.items():
        print(f"Label {label}: {len(indices)} samples")
        
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



def train_model_mstb(
    model,
    train_loader,
    val_loader,
    ilo_dataset,
    ilo_images,
    ilo_labels,
    triplet_loss_fn,
    optimizer,
    device,
    n_epochs,
    experiment_name,
    checkpoint_dir="checkpoints",
    tsne_interval=50,
    log_to_wandb=True,
    mining_strat="Random",  # Mining strategy for triplet loss
    margin_scheduling=False,  # Enable/disable margin scheduling
    initial_margin=0.8,       # Initial larger margin
    final_margin=0.2,         # Final smaller margin
    scheduling_fraction=0.8,   # Fraction of training to complete schedule
    p_ilo_anchor=0.5          # Probability of using ILO image as anchor (vs in-batch)

):
    """
    Trains a model using contrastive learning with triplet loss.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        ilo_dataset: Dataset containing ILO standard images
        ilo_images: Preloaded ILO images on device
        ilo_labels: Preloaded ILO labels on device
        triplet_loss_fn: Loss function for triplet loss
        optimizer: Optimizer for model parameter updates
        device: Device to run training on (cuda/cpu)
        n_epochs: Number of training epochs
        experiment_name: Name for saving checkpoints
        checkpoint_dir: Directory to save checkpoints
        tsne_interval: How often to run t-SNE visualization
        log_to_wandb: Whether to log metrics to wandb
        experiment_name: Name of the experiment for wandb logging
        p_ilo_anchor: Probability (0-1) of using ILO image as anchor vs in-batch sample
        
    Returns:
        dict: Dictionary containing trained model, best model state dict, 
              training history and best validation metrics
    """

    # Define the mapping for multiclass_stb (TB and Silicosis combined classification)
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

    labels_to_indices = build_label_to_indices_map(train_loader.dataset.dataset)
    prof_tb_labels = build_profusion_tb_map(train_loader.dataset.dataset)

    if(initial_margin > final_margin):  # Change condition to check initial > final

        def get_linear_scheduled_margin(current_epoch):
            """Calculate margin based on current training progress using a linear function"""
            if not margin_scheduling:
                return triplet_loss_fn.margin  # Return the original margin
            
            # Calculate how far we are through the scheduled part of training
            schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
            
            # Linearly interpolate between initial and final margin
            current_margin = initial_margin - (initial_margin - final_margin) * schedule_point
            return current_margin
        
        def get_sin_scheduled_margin(current_epoch):
            """Calculate margin based on current training progress using a sinusoidal function"""
            if not margin_scheduling:
                return triplet_loss_fn.margin  # Return the original margin
            
            # Calculate how far we are through the scheduled part of training
            schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
            
            # Use a sin function that starts at 0 and ends at π/2
            # sin(0) = 0 and sin(π/2) = 1
            sin_factor = math.sin(schedule_point * math.pi/2)
            
            # Interpolate between initial and final margin using the sin factor
            current_margin = initial_margin - (final_margin - initial_margin) * sin_factor
            return current_margin

    else:
        def get_sin_scheduled_margin(current_epoch):
            """Calculate margin based on current training progress using a sinusoidal function"""
            if not margin_scheduling:
                return triplet_loss_fn.margin  # Return the original margin
            
            # Calculate how far we are through the scheduled part of training
            schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
            
            # Use a sin function that starts at 0 and ends at π/2
            # sin(0) = 0 and sin(π/2) = 1
            sin_factor = math.sin(schedule_point * math.pi/2)
            
            # Interpolate between initial and final margin using the sin factor
            current_margin = initial_margin + (final_margin - initial_margin) * sin_factor
            return current_margin
        
        def get_linear_scheduled_margin(current_epoch):
            """Calculate margin based on current training progress using a linear function"""
            if not margin_scheduling:
                return triplet_loss_fn.margin  # Return the original margin
            
            # Calculate how far we are through the scheduled part of training
            schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
            
            # Linearly interpolate between initial and final margin
            current_margin = initial_margin + (initial_margin - final_margin) * schedule_point
            return current_margin
    
    # Ensure we're tracking metrics for all 8 classes
    num_classes = 8
    
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    
    model.train()
    
    # Tracking metrics
    best_val_map = 0.0
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
    }
    
    # Initialize class-specific metrics for all 8 classes
    per_class_metrics = {class_id: {'train_ap': [], 'val_ap': []} for class_id in range(num_classes)}

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("=" * 50)

         # Apply margin scheduling if enabled
        if margin_scheduling:
            current_margin = get_sin_scheduled_margin(epoch)
            triplet_loss_fn.margin = current_margin
            print(f"Current margin: {current_margin:.4f}")
        
        epoch_total_loss = 0.0
        epoch_batch_count = 0
        epoch_margin_violations = []

        all_embeddings = []
        all_labels = []
        
        # Training loop
        for batch_idx, sample in enumerate(train_loader):
            # Zero gradients once at the start of each batch
            optimizer.zero_grad()
            
            imgs = sample[0]
            labels = sample[1]
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            feats = model.features(imgs)
            
            embeddings = F.normalize(feats, p=2, dim=1)
            # Detach embeddings for tracking to avoid graph retention
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            current_batch_labels = labels.cpu().numpy()
            
            # Collect all triplets for this batch before computing loss
            anchors = []
            positives = []
            negatives = []
            batch_triplet_count = 0
            
            n_mbod_anchors = 0
            n_ilo_anchors = 0

            batch_matches = 0
            dataset_matches = 0

            n_bshn = 0
            n_relaxed_bshn = 0
            n_random = 0
            n_bhn = 0

            n_tb_anchors = 0
            n_prof_anchors = 0

            n_fallbacks = 0
            
            # For each sample in the batch, build a triplet
            for i, positive_label in enumerate(current_batch_labels):
                # print(f"Batch labels: {set(current_batch_labels)}")
                # Positive embedding (from the current batch)
                positive_embedding = embeddings[i].unsqueeze(0)  # shape [1, C]


                # Decide whether to use ILO anchor or in-batch anchor
                if np.random.rand() < p_ilo_anchor:
                    # Find ILO anchors with the same label as the positive sample
                    ilo_indices = torch.where(ilo_labels == (positive_label%4))[0]
                    if len(ilo_indices) > 0:
                        # Randomly select an ILO anchor
                        ilo_idx = np.random.choice(ilo_indices.cpu().numpy())
                        anchor_embedding = ilo_images[ilo_idx].unsqueeze(0)  # shape [1, C]
                        anchor_embedding = model.features(anchor_embedding)  # Get features of the anchor
                        anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                        anchor_label = ilo_labels[ilo_idx].item()  # Get the label of the anchor

                        n_ilo_anchors += 1
                    else:
                        print(f"No ILO anchor found for label {positive_label}. Skipping.")
                        continue

                else:
                    # First attempt: Search within the current batch for another sample with the same label
                    batch_matching_indices = [j for j in range(len(current_batch_labels)) 
                                            if current_batch_labels[j] == positive_label and j != i]
                    
                    if batch_matching_indices:
                        # Found a matching label in the batch
                        batch_anchor_idx = np.random.choice(batch_matching_indices)
                        anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                        anchor_label = current_batch_labels[batch_anchor_idx]
                        n_mbod_anchors += 1
                        batch_matches += 1
                    else:
                        # Fallback: Search the dataset for a sample with the same label
                        # print(f"No matching label found in batch for label {positive_label}. Searching dataset...")
                        matching_indices = labels_to_indices.get(positive_label.item(), [])
                        
                        if i in matching_indices:
                            matching_indices.remove(i)
                        
                        if matching_indices:
                            chosen_index = np.random.choice(matching_indices)
                            anchor_img, anchor_label = train_loader.dataset.dataset[chosen_index]
                            anchor_embedding = model.features(anchor_img.unsqueeze(0).to(device))
                            anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                            n_mbod_anchors += 1
                            dataset_matches += 1

                        else:
                            print('No matching indices found for label:', positive_label.item())
                            continue  # Skip this sample and move to next one
                
                # Mining strategies for finding negative samples
                negative_embedding = None
                negative_label = None

                # Find all negatives in the batch
                tb_pos_label = 1 if positive_label >= 4 else 0  # TB status (0=negative, 1=positive)


                if(mining_strat == "BSHN"):
                    n_diff_tb = 0
                    n_diff_mstb = 0
                    n_diff_both = 0
                    negative_indices = [j for j, label in enumerate(current_batch_labels) if label != positive_label]
                    n_diff_mstb = len(negative_indices)


                elif mining_strat == "BSHN-v2":
                    # Additional constraint: Profusion score must be different.
                    prof_pos_score = positive_label % 4
                    pos_tb_status = 1 if positive_label >= 4 else 0

                    bshn_indices = [j for j, label in enumerate(current_batch_labels) if label != positive_label]
                    n_diff_mstb = len(bshn_indices)

                    negative_indices = []
                    n_diff_tb = 0
                    n_diff_both = 0
                    n_diff_prof = 0

                    n_ilo_negs = 0

                    for j, label in enumerate(current_batch_labels):
                        if label != positive_label:
                            neg_prof_score = label % 4
                            neg_tb_status = 1 if label >= 4 else 0

                            if(neg_prof_score != prof_pos_score):
                                n_diff_prof += 1
                                negative_indices.append(j)

                
                else:
                    raise ValueError(f"Unknown mining strategy: {mining_strat}")


                if len(negative_indices) > 0:

                    wandb.log({
                        "batch_negatives": len(negative_indices),
                    })

                    ilo_neg_indices = [j for j, label in enumerate(ilo_labels) if label != positive_label]
                    n_ilo_negatives = len(ilo_neg_indices)

                    ilo_neg_imgs = ilo_images[ilo_neg_indices]
                    ilo_neg_embeddings = model.features(ilo_neg_imgs.to(device))
                    ilo_neg_embeddings = F.normalize(ilo_neg_embeddings, p=2, dim=1)
                    
                    negative_embeddings = embeddings[negative_indices]  # shape [N_neg, C]

                    # Repeat anchor to match shape [N_neg, C]
                    anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)

                    # Compute pairwise distances: anchor vs. each negative
                    dists = F.pairwise_distance(anchor_repeated, negative_embeddings)

                    ilo_dists = F.pairwise_distance(anchor_embedding.repeat(n_ilo_negatives, 1), ilo_neg_embeddings)
                    ilo_pos_dists = F.pairwise_distance(anchor_embedding, positive_embedding)

                    # Compute anchor-positive distance
                    positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
                    
                    # Try increasingly relaxed criteria:
                    # 1. First attempt: strict semi-hard negatives (original condition)
                    semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + triplet_loss_fn.margin))
                    semi_hard_dists = dists[semi_hard_mask]

                    ilo_semi_hard_mask = (ilo_dists > ilo_pos_dists) & (ilo_dists < (ilo_pos_dists + triplet_loss_fn.margin))
                    ilo_semi_hard_dists = ilo_dists[ilo_semi_hard_mask]

                    if semi_hard_dists.numel() > 0:
                        # Pick the hardest among semi-hard (closest to positive)
                        hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                        
                        # Map back to the original indices
                        semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                        selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                        
                        negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                        negative_label = current_batch_labels[negative_indices[selected_neg_idx]]

                        hard_ilo_idx_in_masked = torch.argmin(ilo_semi_hard_dists).item()

                        if (ilo_semi_hard_dists.numel() > 0 and
                            ilo_semi_hard_dists[hard_ilo_idx_in_masked] < semi_hard_dists[hard_idx_in_masked]
                        ):
                            # Use ILO negative if it's closer than the batch negative
                            negative_embedding = ilo_neg_embeddings[ilo_neg_indices[hard_ilo_idx_in_masked]].unsqueeze(0)
                            negative_label = ilo_labels[ilo_neg_indices[hard_ilo_idx_in_masked]]
                            n_ilo_negs += 1

                        n_bshn += 1
                    
                    # 2. Relaxed attempt: Increase margin
                    else:
                        larger_margin = triplet_loss_fn.margin * 1.5 
                        semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + larger_margin))
                        semi_hard_dists = dists[semi_hard_mask]
                        
                        if semi_hard_dists.numel() > 0:
                            hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                            semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                            selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                            
                            negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                            negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                            n_relaxed_bshn += 1
                        
                        # 3. Last resort: Just use the hardest negative that's farther than positive
                        else:
                            hard_negatives_mask = dists > positive_distance
                            hard_negative_dists = dists[hard_negatives_mask]
                            
                            if hard_negative_dists.numel() > 0:
                                hard_idx_in_masked = torch.argmin(hard_negative_dists).item()
                                hard_indices = torch.nonzero(hard_negatives_mask).squeeze(1)
                                selected_neg_idx = hard_indices[hard_idx_in_masked].item()
                                
                                negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                                negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                                n_bhn += 1
                            
                            # 4. Final fallback: Random negative
                            else:
                                neg_idx = np.random.choice(negative_indices)
                                negative_embedding = embeddings[neg_idx].unsqueeze(0)
                                negative_label = current_batch_labels[neg_idx]
                                n_random += 1                    

                else:
                    print(f"No negatives found for positive label {positive_label}!!!")
                    negative_embedding = None

                    wandb.log({
                        "batch_negatives": 0,
                    })

                # If we have valid triplet components, add them to the collection
                if anchor_embedding is not None and positive_embedding is not None and negative_embedding is not None:
                    anchors.append(anchor_embedding)
                    positives.append(positive_embedding)
                    negatives.append(negative_embedding)
                    batch_triplet_count += 1

                    n_tb_anchors += 1 if anchor_label >= 4 else 0
                    n_prof_anchors += 1 if (anchor_label % 4 > 0) else 0
                    
                    # Track margin violations for monitoring (not for gradient)
                    with torch.no_grad():
                        is_violated, violation_amount = calculate_margin_violations(
                            anchor_embedding, 
                            positive_embedding, 
                            negative_embedding, 
                            triplet_loss_fn.margin
                        )
                        epoch_margin_violations.append(is_violated)
                    
                    # Log examples only for the first few triplets
                    if batch_idx == 0 and (i == 0):
                        print(f"A: {anchor_label}, P: {positive_label}, N: {negative_label}")
                        # print(f"Margin Violated: {is_violated}, Violation Amount: {violation_amount:.4f}")
            
            # Only process if we found valid triplets
            if batch_triplet_count > 0:
                # Combine all triplet components
                batch_anchors = torch.cat(anchors, dim=0)
                batch_positives = torch.cat(positives, dim=0)
                batch_negatives = torch.cat(negatives, dim=0)
                
                # Calculate batch loss once using all triplets
                batch_loss = triplet_loss_fn(batch_anchors, batch_positives, batch_negatives)
                
                # Backward pass - only once per batch
                batch_loss.backward()
                optimizer.step()
                
                # Track loss value (detached to avoid retaining computation graph)
                avg_loss = batch_loss.item()
                epoch_total_loss += batch_loss.item()
                epoch_batch_count += 1
                
                # Log batch metrics
                if log_to_wandb:
                    wandb.log({
                        "loss": avg_loss,
                        "batch": batch_idx,
                        "batch_mbod_anchors": n_mbod_anchors,
                        "batch_ilo_anchors": n_ilo_anchors,
                        "pos_batch_matches": batch_matches,
                        "pos_dataset_matches": dataset_matches,
                        "bshn_matches": n_bshn,
                        "bhn_matches": n_bhn,
                        "relaxed_bshn_matches": n_relaxed_bshn,
                        "b_rand_matches": n_random,
                        "batch_triplets": batch_triplet_count,
                        "batch_tb_anchors": n_tb_anchors,
                        "batch_prof_anchors": n_prof_anchors,
                        "batch_fallbacks": n_fallbacks if (mining_strat != "BSHN") else 0,
                        "batch_mstb_diff": n_diff_mstb,
                        "batch_tb_diff": n_diff_tb,
                        "batch_both_diff": n_diff_both,
                        "batch_prof_diff": n_diff_prof,
                        "batch_ilo_negatives": n_ilo_negs,
                    })
            else:
                print(f"Batch {batch_idx + 1}: No valid triplets found. Labels: {set(current_batch_labels)}")

                wandb.log({
                    "batch_triplets": 0,
                })
            
            # Clean up to free memory
            del imgs, labels, feats, embeddings
            if 'batch_anchors' in locals():
                del batch_anchors, batch_positives, batch_negatives, batch_loss
            torch.cuda.empty_cache()
        
        # Calculate training metrics (only if we had any valid batches)
        if epoch_batch_count > 0:
            train_loss = epoch_total_loss / epoch_batch_count
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}")
        else:
            train_loss = float('nan')
            print(f"Epoch {epoch + 1}/{n_epochs}, No valid triplets found!")
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        train_map, train_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
        prof_all_labels = all_labels % 4
        train_prof_map, train_prof_class_map = helpers.compute_map_per_class(all_embeddings, prof_all_labels)
        
        # Make sure we log metrics for all classes even if they don't appear in the current epoch
        train_class_map_full = {class_id: train_class_map.get(class_id, 0.0) for class_id in range(num_classes)}
        
        print(f"Train mAP: {train_map:.4f}")
        print("- Per-Class Train mAP:")
        for class_id in range(num_classes):
            ap = train_class_map_full.get(class_id, 0.0)
            class_name = multiclass_stb_mapping.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: mAP = {ap:.4f}")
            
            # Store per-class metrics
            per_class_metrics[class_id]['train_ap'].append(ap)

        # Calculate margin violation rate for this epoch
        if epoch_margin_violations:
            violation_rate = sum(epoch_margin_violations) / len(epoch_margin_violations)
            print(f"Margin violation rate: {violation_rate:.4f} ({sum(epoch_margin_violations)}/{len(epoch_margin_violations)})")
        else:
            violation_rate = 0.0
            print("No triplets were formed to check for margin violations")

        # Reset violation tracking for next epoch
        epoch_margin_violations = []

        # Log to WandB
        if log_to_wandb:
            wandb.log({
                "margin_violation_rate": violation_rate,
                "epoch": epoch + 1
            })
        
        # Run t-SNE visualization at regular intervals
        if (epoch + 1) % tsne_interval == 0:
            visualize_tsne(model, device, ilo_dataset, train_loader, 
                          trained=True, log_to_wandb=log_to_wandb, n_epochs=epoch+1, set_name="training", entire_dataset=False, color_by_profusion=True)
            visualize_tsne(model, device, ilo_dataset, val_loader, 
                          trained=True, log_to_wandb=log_to_wandb, n_epochs=epoch+1, set_name="validation", entire_dataset=False, color_by_profusion=True)
        
        # Run validation
        print("\nVALIDATION\n")
        val_loss, val_map, val_class_map, val_embeddings, val_labels, val_violation, val_prof_map, _ = validate(
            model, val_loader, device, triplet_loss_fn, ilo_images, ilo_labels, 
            num_classes=num_classes, mining_strat=mining_strat
        )
        
        # Fill in missing classes in validation metrics
        val_class_map_full = {class_id: val_class_map.get(class_id, 0.0) for class_id in range(num_classes)}
        
        # Store validation metrics for all classes
        for class_id in range(num_classes):
            per_class_metrics[class_id]['val_ap'].append(val_class_map_full.get(class_id, 0.0))
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_map'].append(train_map)
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map)
        history['train_class_map'].append(train_class_map)
        history['val_class_map'].append(val_class_map)
        history['train_prof_map'].append(train_prof_map)
        history['val_prof_map'].append(val_prof_map)

        
        # Log metrics to wandb
        if log_to_wandb:
            wandb_log_dict = {
                "train_loss": train_loss,
                "train_map": train_map,
                "val_loss": val_loss,
                "val_map": val_map,
                "epoch": epoch + 1,
                "current_margin": current_margin if margin_scheduling else triplet_loss_fn.margin,\
                "train_prof_map": train_prof_map,
                "val_prof_map": val_prof_map
            }
            # Log per-class metrics for all 8 classes
            for class_id in range(num_classes):
                class_name = multiclass_stb_mapping.get(class_id, f"class_{class_id}")
                train_ap = train_class_map_full.get(class_id, 0.0)
                val_ap = val_class_map_full.get(class_id, 0.0)
                
                wandb_log_dict[f"train_class_{class_id}_map"] = train_ap
                wandb_log_dict[f"val_class_{class_id}_map"] = val_ap
                #wandb_log_dict[f"train_{class_name}_map"] = train_ap
                #wandb_log_dict[f"val_{class_name}_map"] = val_ap
            
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
        
        # Save latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, f"final_model.pth"))

    
    visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, True, True, n_epochs=n_epochs+1, set_name="final", entire_dataset=True)
    
    # Return training results with class-specific metrics
    return {
        'model': model,
        'best_model_state': best_model_state,
        'history': history,
        'best_val_map': best_val_map,
        'final_train_map': train_map,
        'final_val_map': val_map,
        'per_class_metrics': per_class_metrics
    }

def validate(model, val_loader, device, triplet_loss_fn, ilo_images, ilo_labels, num_classes=8, mining_strat="Random"):
    """
    Validation loop using the same triplet formation strategy as training.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation set
        device: Device to run validation on
        triplet_loss_fn: Triplet loss function
        ilo_images: Preloaded ILO images on GPU (kept for compatibility but not used)
        ilo_labels: Preloaded ILO labels on GPU (kept for compatibility but not used)
        num_classes: Number of classes (default: 8)
        mining_strat: Mining strategy for triplet loss (default: "Random")
        
    Returns:
        val_loss: Average validation loss
        val_map: Mean Average Precision on validation set
        val_class_map: Per-class Average Precision
    """

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
    
    model.eval()
    running_loss = 0.0
    all_embeddings = []
    all_labels = []
    val_margin_violations = []
    
    print("Running validation loop...")
    batch_with_triplets = 0  # Count of batches with valid triplets
    total_triplets = 0  # Total valid triplets formed
    
    # First, build a map of validation labels to their indices
    labels_to_indices = build_label_to_indices_map(val_loader.dataset.dataset)
    
    with torch.no_grad():  # No gradient tracking for validation
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
            
            # Accumulators for batch loss
            batch_triplet_loss = 0.0
            n_triplets = 0
            
            # Collect all triplets for this batch
            anchors = []
            positives = []
            negatives = []
            
            batch_matches = 0
            dataset_matches = 0
            
            # Form triplets using the same approach as in training
            for i, positive_label in enumerate(current_batch_labels):
                # Positive embedding (from the current batch)
                positive_embedding = embeddings[i].unsqueeze(0)  # shape [1, C]

                # First attempt: Search within the current batch for another sample with the same label
                batch_matching_indices = [j for j in range(len(current_batch_labels)) 
                                         if current_batch_labels[j] == positive_label and j != i]
                
                if batch_matching_indices:
                    # Found a matching label in the batch
                    batch_anchor_idx = np.random.choice(batch_matching_indices)
                    anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                    anchor_label = current_batch_labels[batch_anchor_idx]
                    batch_matches += 1
                else:
                    # Fallback: Search the dataset for a sample with the same label
                    matching_indices = labels_to_indices.get(positive_label.item(), [])
                    
                    if i in matching_indices:
                        matching_indices.remove(i)
                    
                    if matching_indices:
                        chosen_index = np.random.choice(matching_indices)
                        anchor_img, anchor_label = val_loader.dataset.dataset[chosen_index]
                        anchor_embedding = model.features(anchor_img.unsqueeze(0).to(device))
                        anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                        dataset_matches += 1
                    else:
                        # Skip if no matching samples found
                        continue
                
                # Mining strategies for finding negative samples
                negative_embedding = None
                negative_label = None
                

                    
                if(mining_strat == "BSHN"):
                    n_diff_tb = 0
                    n_diff_mstb = 0
                    n_diff_both = 0
                    negative_indices = [j for j, label in enumerate(current_batch_labels) if label != positive_label]
                    n_diff_mstb = len(negative_indices)


                
                elif mining_strat == "BSHN-v2":
                    prof_pos_score = positive_label % 4
                    pos_tb_status = 1 if positive_label >= 4 else 0

                    negative_indices = []
                    n_diff_tb = 0
                    n_diff_mstb = 0
                    n_diff_both = 0
                    n_diff_prof = 0

                    for j, label in enumerate(current_batch_labels):
                        if label != positive_label:
                            neg_prof_score = label % 4

                            if(neg_prof_score != prof_pos_score):
                                n_diff_prof += 1
                                negative_indices.append(j)
                    
                    if(len(negative_indices) == 0):
                        # If no negatives with different profusion and TB status, try different TB status
                        print(f"No validation negatives found using BSHN-v2. Using BSHN.")
                        negative_indices = [j for j, label in enumerate(current_batch_labels) if label != positive_label]
                        n_diff_mstb = len(negative_indices)

                else:
                    raise ValueError(f"Unknown validation mining strategy: {mining_strat}")
                
                if len(negative_indices) > 0:
                    negative_embeddings = embeddings[negative_indices]  # shape [N_neg, C]

                    # Repeat anchor to match shape [N_neg, C]
                    anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)

                    # Compute pairwise distances: anchor vs. each negative
                    dists = F.pairwise_distance(anchor_repeated, negative_embeddings)

                    # Compute anchor-positive distance
                    positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
                    
                    # Try increasingly relaxed criteria:
                    # 1. First attempt: strict semi-hard negatives
                    semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + triplet_loss_fn.margin))
                    semi_hard_dists = dists[semi_hard_mask]

                    if semi_hard_dists.numel() > 0:
                        # Pick the hardest among semi-hard (closest to positive)
                        hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                        
                        # Map back to the original indices
                        semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                        selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                        
                        negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                        negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                    
                    # 2. Relaxed attempt: Increase margin
                    else:
                        larger_margin = triplet_loss_fn.margin * 1.5 
                        semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + larger_margin))
                        semi_hard_dists = dists[semi_hard_mask]
                        
                        if semi_hard_dists.numel() > 0:
                            hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                            semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                            selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                            
                            negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                            negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                        
                        # 3. Last resort: Just use the hardest negative that's farther than positive
                        else:
                            hard_negatives_mask = dists > positive_distance
                            hard_negative_dists = dists[hard_negatives_mask]
                            
                            if hard_negative_dists.numel() > 0:
                                hard_idx_in_masked = torch.argmin(hard_negative_dists).item()
                                hard_indices = torch.nonzero(hard_negatives_mask).squeeze(1)
                                selected_neg_idx = hard_indices[hard_idx_in_masked].item()
                                
                                negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                                negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                            
                            # 4. Final fallback: Random negative
                            else:
                                neg_idx = np.random.choice(negative_indices)
                                negative_embedding = embeddings[neg_idx].unsqueeze(0)
                                negative_label = current_batch_labels[neg_idx]
                
                
                # If we have valid triplet components, add them to the collection
                if anchor_embedding is not None and positive_embedding is not None and negative_embedding is not None:
                    anchors.append(anchor_embedding)
                    positives.append(positive_embedding)
                    negatives.append(negative_embedding)
                    n_triplets += 1
                    
                    # Track margin violations for monitoring
                    is_violated, violation_amount = calculate_margin_violations(
                        anchor_embedding, 
                        positive_embedding, 
                        negative_embedding, 
                        triplet_loss_fn.margin
                    )
                    val_margin_violations.append(is_violated)
                    
                    # Log triplet details for the first batch
                    if batch_idx == 0 and i == 0:
                        print(f"A: {anchor_label}, P: {positive_label}, N: {negative_label}")
                        print(f"A (TB): {anchor_label >= 4}, P (TB): {positive_label >= 4}, N (TB): {negative_label >= 4}")
                        print(f"Loss: {violation_amount:.4f}")
            
            # Process collected triplets
            if n_triplets > 0:
                # Combine all triplet components
                batch_anchors = torch.cat(anchors, dim=0)
                batch_positives = torch.cat(positives, dim=0)
                batch_negatives = torch.cat(negatives, dim=0)
                
                # Calculate batch loss once using all triplets
                batch_loss = triplet_loss_fn(batch_anchors, batch_positives, batch_negatives)
                
                running_loss += batch_loss.item()
                batch_with_triplets += 1
                total_triplets += n_triplets
            else:
                print(f"Validation Batch {batch_idx + 1}: No valid triplets found. Labels: {set(current_batch_labels)}")
    
    # Calculate validation metrics
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        val_map, val_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)

        val_prof_labels = all_labels % 4
        val_prof_map, val_prof_class_map = helpers.compute_map_per_class(all_embeddings, val_prof_labels)
        
        # Fill in any missing classes
        val_class_map_full = {class_id: val_class_map.get(class_id, 0.0) for class_id in range(num_classes)}
        
        # Calculate average validation loss
        val_loss = running_loss / max(1, batch_with_triplets)
        
        print(f"\nValidation Summary:")
        print(f"- Total batches with valid triplets: {batch_with_triplets}/{batch_idx+1}")
        print(f"- Total valid triplets formed: {total_triplets}")
        print(f"- Validation Loss: {val_loss:.4f}")
        print(f"- Validation mAP: {val_map:.4f}")
        print("- Per-Class Validation mAP:")
        
        # Print metrics for all 8 classes
        for class_id in range(num_classes):
            ap = val_class_map_full.get(class_id, 0.0)
            class_name = multiclass_stb_mapping.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: mAP = {ap:.4f}")
        
        # Calculate validation margin violation rate
        if val_margin_violations:
            val_violation_rate = sum(val_margin_violations) / len(val_margin_violations)
            print(f"- Validation margin violation rate: {val_violation_rate:.4f} ({sum(val_margin_violations)}/{len(val_margin_violations)})")
        else:
            val_violation_rate = 0.0
            
        wandb.log({
            "val_margin_violation_rate": val_violation_rate,
            "val_batch_matches": batch_matches,
            "val_dataset_matches": dataset_matches
        })

        return val_loss, val_map, val_class_map_full, all_embeddings, all_labels, val_violation_rate, val_prof_map, val_prof_class_map
    
    return 0.0, 0.0, {class_id: 0.0 for class_id in range(num_classes)}, None, None, None



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
        

        train_loader, val_loader, test_loader = get_dataloaders(
        hdf5_path=hdf5_file_path,
        preprocess=preprocess,
        batch_size=16,
        labels_key="multiclass_stb",
        split_file="stratified_split_filt.json",
        augmentations=None,
        oversample=True,
        balanced_batches=False,
        scaling_factor = 0.5
        )

        wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
        wandb.init(project='MBOD-cl', name='mstb_v2-p_ilo_50-sin_m_005_05',
            config={
                "experiment_type": "CL-MSTB with ILO",
                "batch_size": 16,
                "n_epochs": 1000,
                "learning_rate": 1e-4,
                "oversample": True,
                "initial_margin": 0.05,      
                "final_margin": 0.5,        
                "margin_scheduling": True,   # Enable margin scheduling
                "scheduling_fraction": 0.85,  # Complete scheduling in first x% of training
                "mining": "BSHN-v2",
                "augmentations": True,
                "filtered_dataset": True,
                "loss_function": "Triplet",
                "p_ilo_anchor": 0.5,
                "num_classes": 8,  # Explicitly specify 8 classes
                "OS_factor": 0.65,  # Oversampling factor
                "ilo_scheduling": False,
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
                split_file="stratified_split_filt.json",
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
                split_file="stratified_split_filt.json",
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
                split_file="stratified_split_filt.json",
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
                split_file="stratified_split_filt.json",
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


        # results = train_triplet_model(
        #     model=model,
        #     device=device,
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     optimizer=optimizer,
        #     triplet_loss_fn=triplet_loss_fn,
        #     n_epochs=n_epochs,
        #     oversample=wandb.config.oversample,
        #     augmentations=wandb.config.augmentations,
        #     experiment_name=experiment_name,
        #     checkpoint_dir="/home/sean/MSc_2025/codev2/checkpoints",
        #     tsne_interval=10,  # Save t-SNE every 10 epochs
        #     log_to_wandb=True,
        #     margin_scheduling=wandb.config.margin_scheduling,
        #     initial_margin=wandb.config.initial_margin,
        #     final_margin=wandb.config.final_margin,
        #     scheduling_fraction=wandb.config.scheduling_fraction,
        #     p_ilo_anchor=wandb.config.p_ilo_anchor
        # )

        results = train_model_mstb(
            model,
            train_loader,
            val_loader,
            ilo_dataset,
            ilo_images,
            ilo_labels,
            triplet_loss_fn,
            optimizer,
            device,
            n_epochs,
            experiment_name,
            margin_scheduling=wandb.config.margin_scheduling,
            initial_margin=wandb.config.initial_margin,
            final_margin=wandb.config.final_margin,
            scheduling_fraction=wandb.config.scheduling_fraction,
            mining_strat=wandb.config.mining,
            p_ilo_anchor=wandb.config.p_ilo_anchor
        )


                


    except KeyError as e:
        print(f"Missing configuration: {e}")


