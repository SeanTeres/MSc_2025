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

def validate(model, val_loader, device, triplet_loss_fn, ilo_images, ilo_labels):
    """
    Validation loop using ILO anchors and in-batch negatives.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation set
        device: Device to run validation on
        triplet_loss_fn: Triplet loss function
        ilo_images: Preloaded ILO images on GPU
        ilo_labels: Preloaded ILO labels on GPU
        
    Returns:
        val_loss: Average validation loss
        val_map: Mean Average Precision on validation set
        val_class_map: Per-class Average Precision
    """
    model.eval()
    running_loss = 0.0
    all_embeddings = []
    all_labels = []
    val_margin_violations = []
    
    print("Running validation loop...")
    batch_with_triplets = 0  # Count of batches with valid triplets
    total_triplets = 0  # Total valid triplets formed
    
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
            
            # Form triplets using ILO anchors and in-batch negatives
            for i, positive_label in enumerate(current_batch_labels):
                # Positive embedding (from the current batch)
                positive_embedding = embeddings[i].unsqueeze(0)  # shape [1, C]

                # Find ILO anchors with the same label as the positive sample
                ilo_indices = torch.where(ilo_labels == positive_label)[0]
                if len(ilo_indices) > 0:
                    # Randomly select an ILO anchor
                    ilo_idx = np.random.choice(ilo_indices.cpu().numpy())
                    anchor_embedding = ilo_images[ilo_idx].unsqueeze(0)  # shape [1, C]
                    anchor_embedding = model.features(anchor_embedding)  # Get the features of the anchor
                    anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                else:
                    print(f"No ILO anchor found for label {positive_label}. Skipping.")
                    continue

                # Find negative samples in the batch with a different label
                negative_indices = [j for j, label in enumerate(current_batch_labels) if label != positive_label]
                if len(negative_indices) > 0:
                    neg_idx = np.random.choice(negative_indices)
                    negative_embedding = embeddings[neg_idx].unsqueeze(0)  # shape [1, C]
                else:
                    print(f"No negative sample found for label {positive_label}. Skipping.")
                    continue



                loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
                batch_triplet_loss += loss.item()
                n_triplets += 1

                # Track margin violations
                is_violated, _ = calculate_margin_violations(
                    anchor_embedding, 
                    positive_embedding, 
                    negative_embedding, 
                    triplet_loss_fn.margin
                )
                val_margin_violations.append(is_violated)

                # Log triplet details for the first batch
                if batch_idx == 0 and i == 0:
                    print(f"A: {positive_label}, P: {positive_label}, N: {current_batch_labels[neg_idx]}")
                    print(f"Loss: {loss.item():.4f}")
            
            # Update running loss
            if n_triplets > 0:
                batch_loss = batch_triplet_loss / n_triplets
                running_loss += batch_loss
                batch_with_triplets += 1
                total_triplets += n_triplets
            else:
                print(f"Validation Batch {batch_idx + 1}: No valid triplets found. Labels: {set(current_batch_labels)}")
    
    # Calculate validation metrics
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        val_map, val_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
        
        # Calculate average validation loss
        val_loss = running_loss / max(1, batch_with_triplets)
        
        print(f"\nValidation Summary:")
        print(f"- Total batches with valid triplets: {batch_with_triplets}/{batch_idx+1}")
        print(f"- Total valid triplets formed: {total_triplets}")
        print(f"- Validation Loss: {val_loss:.4f}")
        print(f"- Validation mAP: {val_map:.4f}")
        print("- Per-Class Validation mAP:")
        for class_id, ap in val_class_map.items():
            print(f"  Class {class_id}: mAP = {ap:.4f}")
        
        # Calculate validation margin violation rate
        if val_margin_violations:
            val_violation_rate = sum(val_margin_violations) / len(val_margin_violations)
            print(f"- Validation margin violation rate: {val_violation_rate:.4f} ({sum(val_margin_violations)}/{len(val_margin_violations)})")
        else:
            val_violation_rate = 0.0
            
        wandb.log({
            "val_margin_violation_rate": val_violation_rate
        })

        return val_loss, val_map, val_class_map, all_embeddings, all_labels, val_violation_rate
    
    return 0.0, 0.0, {}, None, None, None

def train_model_alt(
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
    mbod_merged_loader,
    checkpoint_dir="checkpoints",
    tsne_interval=30,
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
        'val_class_map': []
    }

    
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
            
            # For each sample in the batch, build a triplet
            for i, positive_label in enumerate(current_batch_labels):
                # Positive embedding (from the current batch)
                positive_embedding = embeddings[i].unsqueeze(0)  # shape [1, C]
                
                # Decide whether to use ILO anchor or in-batch anchor
                if np.random.rand() < p_ilo_anchor:
                    # Find ILO anchors with the same label as the positive sample
                    ilo_indices = torch.where(ilo_labels == positive_label)[0]
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
                    # Find other samples with the same label to use as anchor
                    pos_indices = np.where(current_batch_labels == positive_label)[0]
                    # Remove the current sample to avoid anchor=positive
                    pos_indices = pos_indices[pos_indices != i]
                    
                    if len(pos_indices) > 0:
                        # Randomly select a different positive sample as anchor
                        pos_idx = np.random.choice(pos_indices)
                        anchor_embedding = embeddings[pos_idx].unsqueeze(0)  # shape [1, C]
                        anchor_label = positive_label

                        n_mbod_anchors += 1
                    else:
                        # print(f"{i}/{len(current_batch_labels)}: No other MBOD samples with label {positive_label} found. Using ILO instead.")
                        # Find ILO anchors with the same label as the positive sample
                        ilo_indices = torch.where(ilo_labels == positive_label)[0]
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
                
                # Mining strategies for finding negative samples
                negative_embedding = None
                negative_label = None
                
                if mining_strat == "Random":
                    # Find negative samples in the batch with a different label
                    negative_indices = [j for j, label in enumerate(current_batch_labels) if label != positive_label]
                    if len(negative_indices) > 0:
                        neg_idx = np.random.choice(negative_indices)
                        negative_embedding = embeddings[neg_idx].unsqueeze(0)  # shape [1, C]
                        negative_label = current_batch_labels[neg_idx]
                    else:
                        print(f"No negative sample found for label {positive_label}. Skipping.")
                        continue
               
                elif mining_strat == "BHN":
                    # Find all negatives in the batch
                    negative_indices = [j for j, label in enumerate(current_batch_labels) if label != positive_label]
                    if len(negative_indices) > 0:
                        negative_embeddings = embeddings[negative_indices]  # shape [N_neg, C]

                        # Repeat anchor to match shape [N_neg, C]
                        anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)

                        # Compute pairwise distances
                        dists = F.pairwise_distance(anchor_repeated, negative_embeddings)

                        # Find hardest negative (smallest distance)
                        hard_idx = torch.argmin(dists).item()
                        negative_embedding = negative_embeddings[hard_idx].unsqueeze(0)
                        negative_label = current_batch_labels[negative_indices[hard_idx]]
                    else:
                        print(f"No hard negative found for label {positive_label}. Skipping.")
                        continue

                elif mining_strat == "BSHN":
                    # Find all negatives in the batch
                    negative_indices = [j for j, label in enumerate(current_batch_labels) if label != positive_label]
                    if len(negative_indices) > 0:
                        negative_embeddings = embeddings[negative_indices]  # shape [N_neg, C]

                        # Repeat anchor to match shape [N_neg, C]
                        anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)

                        # Compute pairwise distances: anchor vs. each negative
                        dists = F.pairwise_distance(anchor_repeated, negative_embeddings)

                        # Compute anchor-positive distance
                        positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
                        
                        # Try increasingly relaxed criteria:
                        # 1. First attempt: strict semi-hard negatives (original condition)
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
                    batch_triplet_count += 1
                    
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
                    if batch_idx == 0 and i == 0:
                        print(f"A: {anchor_label}, P: {positive_label}, N: {negative_label}")
                        print(f"Margin Violated: {is_violated}, Violation Amount: {violation_amount:.4f}")
            
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
                        "batch_ilo_anchors": n_ilo_anchors
                    })
            else:
                print(f"Batch {batch_idx + 1}: No valid triplets found. Labels: {set(current_batch_labels)}")
            
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
        
        print(f"Train mAP: {train_map:.4f}")
        print("- Per-Class Train mAP:")
        for class_id, ap in train_class_map.items():
            print(f"  Class {class_id}: mAP = {ap:.4f}")

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
                          trained=True, log_to_wandb=log_to_wandb, n_epochs=epoch+1, set_name="training", entire_dataset=False)
            visualize_tsne(model, device, ilo_dataset, val_loader, 
                          trained=True, log_to_wandb=log_to_wandb, n_epochs=epoch+1, set_name="validation", entire_dataset=False)
        
        # Run validation
        print("\nVALIDATION\n")
        val_loss, val_map, val_class_map, val_embeddings, val_labels, val_violation = validate(
            model, val_loader, device, triplet_loss_fn, ilo_images, ilo_labels
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_map'].append(train_map)
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map)
        history['train_class_map'].append(train_class_map)
        history['val_class_map'].append(val_class_map)
        
        # Log metrics to wandb
        if log_to_wandb:
            wandb_log_dict = {
                "train_loss": train_loss,
                "train_map": train_map,
                "val_loss": val_loss,
                "val_map": val_map,
                "epoch": epoch + 1,
                "current_margin": current_margin if margin_scheduling else triplet_loss_fn.margin
            }
            # Log per-class metrics
            for class_id, ap in train_class_map.items():
                wandb_log_dict[f"train_class_{class_id}_map"] = ap
            
            for class_id, ap in val_class_map.items():
                wandb_log_dict[f"val_class_{class_id}_map"] = ap
            
            wandb.log(wandb_log_dict)
        
        # Save best model based on validation mAP
        if epoch == 0 or val_map > best_val_map:
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
    
    # Return training results
    return {
        'model': model,
        'best_model_state': best_model_state,
        'history': history,
        'best_val_map': best_val_map,
        'final_train_map': train_map,
        'final_val_map': val_map
    }
def sweep_train():
    """
    Training function to be called by the sweep agent.
    This runs one training with parameters from the sweep.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run = wandb.init()
    
    # Load configuration
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")
    hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
    ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]
    
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Setup augmentations if enabled
    augmentations = None
    if wandb.config.augmentations:
        augmentations = transforms.Compose([
            transforms.RandomRotation(degrees=10, expand=False, fill=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
        ])
    
    # Create datasets
    mbod_dataset_merged = HDF5Dataset(
        hdf5_path=hdf5_file_path,
        labels_key="profusion_score",
        images_key="images",
        augmentations=None,
        preprocess=preprocess
    )
    
    ilo_dataset = HDF5Dataset(
        hdf5_path=ilo_hdf5_file_path,
        labels_key="profusion_score",
        images_key="images",
        augmentations=None,
        preprocess=preprocess
    )
    
    # Get dataloaders
    train_loader, _, _ = get_dataloaders(
        hdf5_path=hdf5_file_path,
        preprocess=preprocess,
        batch_size=wandb.config.batch_size,
        labels_key="profusion_score",
        split_file="stratified_split_filt.json",
        augmentations=augmentations,
        oversample=wandb.config.oversample,
        balanced_batches=False
    )

    _, val_loader, test_loader = get_dataloaders(
        hdf5_path=hdf5_file_path,
        preprocess=preprocess,
        batch_size=wandb.config.batch_size,
        labels_key="profusion_score",
        split_file="stratified_split_filt.json",
        augmentations=None,
        oversample=False,
        balanced_batches=False
    )
    
    mbod_merged_loader = torch.utils.data.DataLoader(
        mbod_dataset_merged,
        batch_size=wandb.config.batch_size,
        shuffle=False
    )
    
    # Create model
    model = xrv.models.ResNet(weights="resnet50-res512-all")
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.learning_rate  # L2 regularization
    )
    
    # Create loss function
    triplet_loss_fn = nn.TripletMarginLoss(margin=wandb.config.initial_margin, p=2)
    
    # Create unique experiment name for this sweep run
    experiment_name = f"sweep-{wandb.run.name}-m{wandb.config.initial_margin:.2f}-{wandb.config.final_margin:.2f}"
    
    # Preload ILO images
    print("Preloading ILO images onto the GPU...")
    ilo_images = []
    ilo_labels = []
    
    for idx in range(len(ilo_dataset)):
        image, label = ilo_dataset[idx]
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).to(device)
        label_tensor = torch.tensor(label, dtype=torch.long).to(device)
        ilo_images.append(image_tensor)
        ilo_labels.append(label_tensor)
    
    # Stack all tensors into a single tensor for efficient access
    ilo_images = torch.cat(ilo_images, dim=0)
    ilo_labels = torch.stack(ilo_labels)
    
    print(f"ILO images loaded onto GPU: {ilo_images.shape}")
    print(f"ILO labels loaded onto GPU: {ilo_labels.shape}")
    try:
    # Train the model
        results = train_model_alt(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            ilo_dataset=ilo_dataset,
            ilo_images=ilo_images,
            ilo_labels=ilo_labels,
            triplet_loss_fn=triplet_loss_fn,
            optimizer=optimizer,
            device=device,
            n_epochs=wandb.config.n_epochs,
            experiment_name=experiment_name,
            mbod_merged_loader=mbod_merged_loader,
            margin_scheduling=wandb.config.margin_scheduling,
            initial_margin=wandb.config.initial_margin,
            final_margin=wandb.config.final_margin,
            scheduling_fraction=wandb.config.scheduling_fraction,
            mining_strat=wandb.config.mining,
            p_ilo_anchor=wandb.config.p_ilo_anchor,
            tsne_interval=30  # Reduced frequency for sweep runs
        )
        
        # Log best results at the end of run
        wandb.log({
            "best_val_map": results['best_val_map'],
            "final_train_map": results['final_train_map'],
            "final_val_map": results['final_val_map']
        })
    finally:
        # Explicit cleanup after each run
        if 'model' in locals():
            del model
        if 'ilo_images' in locals():
            del ilo_images
        if 'ilo_labels' in locals():
            del ilo_labels
        if 'results' in locals():
            del results
        
        # Force CUDA to release all memory
        torch.cuda.empty_cache()
        
        # Optionally force garbage collection
        import gc
        gc.collect()



# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # Can use 'random', 'grid', or 'bayes'
    'metric': {
        'name': 'best_val_map',
        'goal': 'maximize'
    },
    'parameters': {
        'initial_margin': {
            'values': [0.01, 0.02, 0.05, 0.07, 0.1, 0.2]
        },
        'final_margin': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        },
        # Fixed parameters
        'batch_size': {'value': 16},
        'n_epochs': {'value': 250},  # Reduced for faster sweep
        'learning_rate': {'value': 1e-4},
        'oversample': {'value': True},
        'margin_scheduling': {'value': True},
        'scheduling_fraction': {'value': 0.85},
        'mining': {'value': 'BSHN'},
        'augmentations': {'value': True},
        'p_ilo_anchor': {'value': 0.5}
    }
}

if __name__ == "__main__":
    # Initialize wandb
    wandb.login(key='176da722bd80e35dbc4a8cea0567d495b7307688')
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="MBOD-cl-margin-sweep")
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_train, count=15)  