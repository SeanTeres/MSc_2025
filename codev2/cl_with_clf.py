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

class MultiClassBaseClassifier(nn.Module):
    def __init__(self, in_features, num_classes=4):
        """
        Multi-class classifier for pneumoconiosis profusion scoring.

        Args:
            in_features (int): Number of input features from the backbone model
            num_classes (int): Number of classes for classification (default: 4 for profusion levels 0-3)
        """
        super(MultiClassBaseClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)  # Output layer

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function for logits
        return x


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


def test_model_with_clf(
    model,
    test_loader,
    device,
    ilo_dataset,
    ilo_images,
    ilo_labels,
    triplet_loss_fn,
    multi_class_loss_fn,
    alpha,
    experiment_name,
):
    
    model.eval()
    running_triplet_loss = 0.0
    running_classifier_loss = 0.0
    running_total_loss = 0.0

    all_embeddings = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            imgs = sample[0].to(device)
            labels = sample[1].long().to(device)

            # Extract features and normalize embeddings
            features = model.features(imgs)
            embeddings = F.normalize(features, p=2, dim=1)
            

            # Classification loss (for all samples in batch)
            predictions = model.classifier(features)
            _, predicted_classes = torch.max(predictions, 1)
            classifier_loss = multi_class_loss_fn(predictions, labels)
            running_classifier_loss += classifier_loss.item()

            # Store for mAP calculation
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(predicted_classes.cpu())
        
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_preds = torch.cat(all_preds, dim=0)

            test_map, test_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
            wandb.log({
                "test_map": test_map,
                "test_class_map": test_class_map
            })

            # Calculate test accuracy
            all_labels_np = all_labels.cpu().numpy()  # Convert to numpy
            all_preds_np = all_preds.cpu().numpy()    # Convert to numpy
            test_accuracy = accuracy_score(all_labels_np, all_preds_np)
            test_recall = recall_score(all_labels_np, all_preds_np, average='macro')
            test_precision = precision_score(all_labels_np, all_preds_np, average='macro')
            test_f1 = f1_score(all_labels_np, all_preds_np, average='macro')
            test_kappa = cohen_kappa_score(all_labels_np, all_preds_np)

            wandb.log({
                "test_accuracy": test_accuracy,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "test_f1": test_f1,
                "test_kappa": test_kappa
            })

            # log confusion matrix
            cm = confusion_matrix(all_labels_np, all_preds_np)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            # Make sure the directory exists
            os.makedirs(os.path.join("checkpoints", experiment_name), exist_ok=True)

            # Save the confusion matrix with standard file operations
            cm_file_path = os.path.join("checkpoints", experiment_name, "confusion_matrix.png")
            plt.savefig(cm_file_path, bbox_inches='tight', dpi=300)
            plt.close()  # Close the figure to release memory

            # Log using explicit file path
            wandb.log({"test_confusion_matrix": wandb.Image(cm_file_path)})

        return test_map, test_class_map, test_accuracy, test_recall, test_precision, test_f1, test_kappa
            

def train_model_with_clf(
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
    alpha,
    checkpoint_dir="checkpoints",
    tsne_interval=100,
    log_to_wandb=True,
    mining_strat="Random",  # Mining strategy for triplet loss
    margin_scheduling=False,  # Enable/disable margin scheduling
    ilo_scheduling=True,  # Enable/disable ILO anchor scheduling
    initial_margin=0.8,       # Initial larger margin
    final_margin=0.2,         # Final smaller margin
    scheduling_fraction=0.8,   # Fraction of training to complete schedule
    p_ilo_anchor=0.5,          # Probability of using ILO image as anchor (vs in-batch)
    initial_p_ilo=1.0,  # Initial probability of using ILO anchor
    final_p_ilo=0.1,    # Final probability of using ILO anchor
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
        
    def get_linear_scheduled_p_ilo(current_epoch):
        """Calculate margin based on current training progress using a linear function"""
        if not ilo_scheduling:
            return p_ilo_anchor  # Return the original margin
        
        # Calculate how far we are through the scheduled part of training
        schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
        
        # Linearly interpolate between initial and final margin
        current_margin = initial_p_ilo - (initial_p_ilo - final_p_ilo) * schedule_point
        return current_margin
    
    def get_sin_scheduled_p_ilo(current_epoch):
        """Calculate margin based on current training progress using a sinusoidal function"""
        if not ilo_scheduling:
            return p_ilo_anchor  # Return the original margin
        
        # Calculate how far we are through the scheduled part of training
        schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
        
        # Use a sin function that starts at 0 and ends at π/2
        # sin(0) = 0 and sin(π/2) = 1
        sin_factor = math.sin(schedule_point * math.pi/2)
        
        # Interpolate between initial and final margin using the sin factor
        current_margin = initial_p_ilo + (final_p_ilo - initial_p_ilo) * sin_factor
        return current_margin
        
    
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    multi_class_loss_fn = nn.CrossEntropyLoss()
    
    
    model.train()
    
    # Tracking metrics
    best_val_map = 0.0
    best_val_f1 = 0.0
    best_val_kappa = 0.0
    best_model_state = None
    history = {
        'train_loss': [],
        'train_map': [],
        'val_loss': [],
        'val_map': [],
        'train_class_map': [],
        'val_class_map': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_clf_loss': [],
        'train_recall': [],       
        'train_precision': [],    
        'train_f1': [],           
        'train_kappa': [],        
        'val_recall': [],         
        'val_precision': [],      
        'val_f1': [],             
        'val_kappa': []           
    }

    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("=" * 50)

         # Apply margin scheduling if enabled
        if margin_scheduling:
            current_margin = get_sin_scheduled_margin(epoch)
            triplet_loss_fn.margin = current_margin
            print(f"Current margin: {current_margin:.4f}")

        if ilo_scheduling:
            current_p_ilo = get_sin_scheduled_p_ilo(epoch)
            print(f"Current p_ilo_anchor: {current_p_ilo:.4f}")
        else:
            current_p_ilo = p_ilo_anchor
            print(f"Using fixed p_ilo_anchor: {current_p_ilo:.4f}")
        
        epoch_total_loss = 0.0
        epoch_batch_count = 0
        epoch_margin_violations = []
        epoch_correct_preds = 0
        epoch_total_preds = 0

        all_embeddings = []
        all_labels = []
        all_preds = []
        
        # Training loop
        for batch_idx, sample in enumerate(train_loader):
            epoch_batch_count += 1
            # Zero gradients once at the start of each batch
            optimizer.zero_grad()
            
            imgs = sample[0]
            labels = sample[1]
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            feats = model.features(imgs)
            
            # Track classification accuracy for this batch
            predictions = model.classifier(feats)
            _, predicted_classes = torch.max(predictions, 1)
            batch_correct = (predicted_classes == labels).sum().item()
            epoch_correct_preds += batch_correct
            epoch_total_preds += labels.size(0)
            
            embeddings = F.normalize(feats, p=2, dim=1)
            # Detach embeddings for tracking to avoid graph retention
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_preds.append(predicted_classes.detach().cpu())
            
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
                if np.random.rand() < current_p_ilo:
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
                
                # Multi-class classification loss (optional)
                predictions = model.classifier(feats)
                multi_class_loss = multi_class_loss_fn(predictions, labels)               

                # Combine losses
                total_loss = batch_loss + alpha *  multi_class_loss
                total_loss.backward()
                optimizer.step()

                avg_loss = total_loss.item()
                epoch_total_loss += avg_loss

                # Log batch metrics
                if log_to_wandb and batch_idx % 10 == 0:  # Log every 10 batches to avoid cluttering
                    batch_accuracy = batch_correct / labels.size(0)
                    wandb.log({
                        "total_loss": total_loss.item(),
                        "loss": batch_loss.item(),
                        "clf_loss": multi_class_loss.item(),
                        "batch_accuracy": batch_accuracy,
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
        
    
        train_loss = epoch_total_loss / epoch_batch_count
        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}")

                
        all_embeddings = torch.cat(all_embeddings, dim=0)
        # all_labels = torch.cat(all_labels, dim=0)

        # Calculate training accuracy
        all_labels_np = torch.cat(all_labels, dim=0).cpu().numpy()  # Convert to numpy
        all_preds_np = torch.cat(all_preds).cpu().numpy()    # Convert to numpy


            
        # Calculate training accuracy
        train_accuracy = accuracy_score(all_labels_np, all_preds_np)
        train_recall = recall_score(all_labels_np, all_preds_np, average='macro')
        train_precision = precision_score(all_labels_np, all_preds_np, average='macro')
        train_f1 = f1_score(all_labels_np, all_preds_np, average='macro')
        train_kappa = cohen_kappa_score(all_labels_np, all_preds_np)



        print(f"Training Accuracy: {train_accuracy:.4f} ({epoch_correct_preds}/{epoch_total_preds})")

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

        
        # Run t-SNE visualization at regular intervals
        if (epoch + 1) % tsne_interval == 0:
            # Generate confusion matrix for validation set
            model.eval()
            
            # Initialize arrays to store validation predictions and labels
            val_preds = []
            val_labels_np = []
            
            # Collect predictions and labels from validation set
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    features = model.features(imgs)
                    predictions = model.classifier(features)
                    _, predicted = torch.max(predictions, 1)
                    val_preds.append(predicted.cpu().numpy())
                    val_labels_np.append(labels.cpu().numpy())
            
            # Convert lists to numpy arrays
            val_preds_np = np.concatenate(val_preds)
            val_labels_np = np.concatenate(val_labels_np)
            
            # Create binary labels
            val_binary_labels = (val_labels_np > 0).astype(int)
            val_binary_preds = (val_preds_np > 0).astype(int)
            
            # Get multi-class and binary confusion matrices
            val_cm = confusion_matrix(val_labels_np, val_preds_np)
            binary_val_cm = confusion_matrix(val_binary_labels, val_binary_preds)
            
            # Plot multi-class and binary confusion matrices side by side
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Multi-class confusion matrix
            class_labels = ["0", "1", "2", "3"]  # Or use a more appropriate list of class names

            sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            axes[0].set_title(f'Multi-class Confusion Matrix - Epoch {epoch+1}')
            
            # Binary confusion matrix (Class 0 vs Others)
            binary_labels = ["Class 0", "Other Classes"]
            sns.heatmap(binary_val_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=binary_labels, yticklabels=binary_labels, ax=axes[1])
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')
            axes[1].set_title(f'Binary Confusion Matrix (Class 0 vs Others) - Epoch {epoch+1}')
            
            plt.tight_layout()
            val_cm_path = os.path.join(checkpoint_dir, f"val_cm.png")
            plt.savefig(val_cm_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Check if file exists and has size
            if os.path.exists(val_cm_path) and os.path.getsize(val_cm_path) > 0:
                print(f"val CM saved successfully: {val_cm_path}")
                wandb.log({"val_confusion_matrix": wandb.Image(val_cm_path)})
            else:
                print(f"WARNING: val CM file is empty or missing: {val_cm_path}")
            
            # Similarly for training confusion matrices - use the already collected training data
            train_binary_labels = (all_labels_np > 0).astype(int)
            train_binary_preds = (all_preds_np > 0).astype(int)
            
            train_cm = confusion_matrix(all_labels_np, all_preds_np)
            binary_train_cm = confusion_matrix(train_binary_labels, train_binary_preds)
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Multi-class training confusion matrix
            class_labels = ["0", "1", "2", "3"]  # Or use a more appropriate list of class names

            sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            axes[0].set_title(f'Multi-class Training Confusion Matrix - Epoch {epoch+1}')
            
            # Binary training confusion matrix
            sns.heatmap(binary_train_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=binary_labels, yticklabels=binary_labels, ax=axes[1])
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('True')
            axes[1].set_title(f'Binary Training Confusion Matrix (Class 0 vs Others) - Epoch {epoch+1}')
            
            plt.tight_layout()
            train_cm_path = os.path.join(checkpoint_dir, f"train_cm.png")
            plt.savefig(train_cm_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Check if file exists and has size
            if os.path.exists(train_cm_path) and os.path.getsize(train_cm_path) > 0:
                print(f"Training CM saved successfully: {train_cm_path}")
                wandb.log({"train_confusion_matrix": wandb.Image(train_cm_path)})
            else:
                print(f"WARNING: Training CM file is empty or missing: {train_cm_path}")
            visualize_tsne(model, device, ilo_dataset, train_loader, 
                          trained=True, log_to_wandb=log_to_wandb, n_epochs=epoch+1, set_name="training", entire_dataset=False)
            visualize_tsne(model, device, ilo_dataset, val_loader, 
                          trained=True, log_to_wandb=log_to_wandb, n_epochs=epoch+1, set_name="validation", entire_dataset=False)
        
        # Run validation
        print("\nVALIDATION\n")
        val_loss, val_map, val_class_map, all_embeddings, all_labels, val_violation_rate, val_accuracy, val_recall, val_precision, val_f1, val_kappa, val_avg_clf_loss = validate_with_clf(
            model, val_loader, device, triplet_loss_fn, ilo_images, ilo_labels, multi_class_loss_fn
        )
        
       
        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['train_recall'].append(train_recall)
        history['train_precision'].append(train_precision)
        history['train_f1'].append(train_f1)
        history['train_kappa'].append(train_kappa)
        history['train_map'].append(train_map)
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map)
        history['train_class_map'].append(train_class_map)
        history['val_class_map'].append(val_class_map)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['val_clf_loss'].append(val_avg_clf_loss)
        history['val_recall'].append(val_recall)
        history['val_precision'].append(val_precision)
        history['val_f1'].append(val_f1)
        history['val_kappa'].append(val_kappa)
        
        # Log metrics to wandb
        if log_to_wandb:
            wandb_log_dict = {
                "margin_violation_rate": violation_rate,
                "train_loss": train_loss,
                "train_map": train_map,
                "train_accuracy": train_accuracy,
                "train_recall": train_recall,
                "train_precision": train_precision,
                "train_f1": train_f1,
                "train_kappa": train_kappa,
                "val_loss": val_loss,
                "val_map": val_map,
                "val_accuracy": val_accuracy,
                "val_clf_loss": val_avg_clf_loss,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "val_f1": val_f1,
                "val_kappa": val_kappa,
                "epoch": epoch + 1,
                "current_margin": current_margin if margin_scheduling else triplet_loss_fn.margin,
                "p_ilo_anchor": current_p_ilo
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
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"Saving best model with validation F1: {best_val_f1:.4f}")
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, f"best_model_f1.pth"))


        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            print(f"Saving best model with validation Kappa: {best_val_kappa:.4f}")
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, f"best_model_kappa.pth"))
        
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
        'final_val_map': val_map,
        'final_train_accuracy': train_accuracy,
        'final_val_accuracy': val_accuracy
    }

def validate_with_clf(model, val_loader, device, triplet_loss_fn, ilo_images, ilo_labels, multi_class_loss_fn):
    """
    Validation loop using ILO anchors and in-batch negatives.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation set
        device: Device to run validation on
        triplet_loss_fn: Triplet loss function
        ilo_images: Preloaded ILO images on GPU
        ilo_labels: Preloaded ILO labels on GPU
        multi_class_loss_fn: Loss function for classification
        
    Returns:
        val_loss: Average total validation loss
        val_map: Mean Average Precision on validation set
        val_class_map: Per-class Average Precision
        all_embeddings: Embeddings for validation set
        all_labels: Labels for validation set
        val_violation_rate: Rate of margin violations
        val_accuracy: Classification accuracy on validation set
    """
    model.eval()
    running_triplet_loss = 0.0
    running_classifier_loss = 0.0
    running_total_loss = 0.0
    all_embeddings = []
    val_all_labels = []
    val_all_preds = []
    val_margin_violations = []
    
    # Track classification accuracy
    total_correct = 0
    total_samples = 0
    
    print("Running validation loop...")
    batch_with_triplets = 0  # Count of batches with valid triplets
    total_triplets = 0  # Total valid triplets formed
    total_batches = 0  # Total batches processed
    
    with torch.no_grad():  # No gradient tracking for validation
        for batch_idx, sample in enumerate(val_loader):
            # Get validation batch
            imgs = sample[0].to(device)
            labels = sample[1].long().to(device)
            total_batches += 1
            total_samples += labels.size(0)
            
            # Extract features and normalize embeddings
            features = model.features(imgs)
            embeddings = F.normalize(features, p=2, dim=1)
            
            # Classification loss (for all samples in batch)
            predictions = model.classifier(features)
            classifier_loss = multi_class_loss_fn(predictions, labels)
            running_classifier_loss += classifier_loss.item()
            
            # Calculate accuracy
            _, predicted_classes = torch.max(predictions, 1)
            batch_correct = (predicted_classes == labels).sum().item()
            total_correct += batch_correct
            
            # Store for mAP calculation
            all_embeddings.append(embeddings.cpu())
            val_all_labels.append(labels.cpu())
            val_all_preds.append(predicted_classes.cpu())
            
            # Get labels from batch
            current_batch_labels = labels.cpu().numpy()
            
            # Accumulators for batch triplet loss
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

                # Calculate triplet loss
                triplet_loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
                batch_triplet_loss += triplet_loss.item()
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
                    print(f"Triplet Loss: {triplet_loss.item():.4f}")
            
            # Update running loss
            if n_triplets > 0:
                avg_batch_triplet_loss = batch_triplet_loss / n_triplets
                running_triplet_loss += avg_batch_triplet_loss
                
                # Calculate total loss (triplet + classification)
                total_batch_loss = avg_batch_triplet_loss + classifier_loss.item()
                running_total_loss += total_batch_loss
                
                batch_with_triplets += 1
                total_triplets += n_triplets
                
                if batch_idx == 0:
                    print(f"Classifier Loss: {classifier_loss.item():.4f}, Total Loss: {total_batch_loss:.4f}")
            else:
                # If no triplets formed, still count the classification loss
                running_total_loss += classifier_loss.item()
                print(f"Validation Batch {batch_idx + 1}: No valid triplets found. Only using classifier loss.")
    
    # Calculate validation metrics
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        val_all_labels = torch.cat(val_all_labels, dim=0)
        val_all_preds = torch.cat(val_all_preds, dim=0)  # Make sure to concatenate predictions too
        
        # Convert to numpy arrays before passing to sklearn metrics
        val_all_labels_np = val_all_labels.cpu().numpy()
        val_all_preds_np = val_all_preds.cpu().numpy()
        
        val_map, val_class_map = helpers.compute_map_per_class(all_embeddings, val_all_labels)
        
        # Calculate average validation losses
        avg_triplet_loss = running_triplet_loss / max(1, batch_with_triplets)
        avg_classifier_loss = running_classifier_loss / total_batches
        val_loss = running_total_loss / total_batches  # Total loss normalized by all batches
        
        # Use numpy arrays for metric calculations
        val_accuracy = accuracy_score(val_all_labels_np, val_all_preds_np)
        val_recall = recall_score(val_all_labels_np, val_all_preds_np, average='macro')
        val_precision = precision_score(val_all_labels_np, val_all_preds_np, average='macro')
        val_f1 = f1_score(val_all_labels_np, val_all_preds_np, average='macro')
        val_kappa = cohen_kappa_score(val_all_labels_np, val_all_preds_np)
        

        
        print(f"\nValidation Summary:")
        print(f"- Total batches: {total_batches}")
        print(f"- Batches with valid triplets: {batch_with_triplets}/{total_batches}")
        print(f"- Total valid triplets formed: {total_triplets}")
        print(f"- Avg Triplet Loss: {avg_triplet_loss:.4f}")
        print(f"- Avg Classifier Loss: {avg_classifier_loss:.4f}")
        print(f"- Validation Total Loss: {val_loss:.4f}")
        print(f"- Validation Accuracy: {val_accuracy:.4f} ({total_correct}/{total_samples})")
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
            
        # Log all metrics
        # wandb.log({
        #     "val_loss": avg_triplet_loss,
        #     "val_classifier_loss": avg_classifier_loss,
        #     "val_total_loss": val_loss,
        #     "val_accuracy": val_accuracy,
        #     "val_margin_violation_rate": val_violation_rate,
        # })

        return avg_triplet_loss, val_map, val_class_map, all_embeddings, val_all_labels, val_violation_rate, val_accuracy, val_recall, val_precision, val_f1, val_kappa, avg_classifier_loss
    
    return 0.0, 0.0, {}, None, None, 0.0, 0.0



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
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
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
        labels_key="profusion_score",
        split_file="stratified_split_filt.json",
        augmentations=None,
        oversample=True,
        balanced_batches=False
        )

        wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
        wandb.init(project='MBOD-cl', name='clf_025-p_ilo_sched-sin_m_01_05',
            config={
                "experiment_type": "end-end CL with ILO",
                "num_classes": 4,
                "batch_size": 24,
                "n_epochs": 1000,
                "learning_rate": 1e-4,
                "oversample": True,
                "initial_margin": 0.1,      
                "final_margin": 0.5,        
                "margin_scheduling": True,   # Enable margin scheduling
                "scheduling_fraction": 0.75,  # Complete scheduling in first x% of training
                "mining": "BSHN",
                "augmentations": False,
                "filtered_dataset": True,
                "loss_function": "Triplet",
                "p_ilo_anchor": 1.0,
                "clf_alpha": 0.25,
                "ilo_scheduling": True,
                "initial_ilo": 0.5,
                "final_ilo": 0.0,
                "OS_factor": 0.75
            })

        experiment_name = wandb.run.name

        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model.classifier = MultiClassBaseClassifier(in_features=2048, num_classes=4)
        model = model.to(device)



        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=wandb.config.learning_rate,  # Try a smaller learning rate
            weight_decay=wandb.config.learning_rate  # Add L2 regularization
        )
        triplet_loss_fn = nn.TripletMarginLoss(margin=wandb.config.initial_margin, p=2)
        multi_class_loss_fn = nn.CrossEntropyLoss()

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
                labels_key="profusion_score",
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
                labels_key="profusion_score",
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
                labels_key="profusion_score",
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
                labels_key="profusion_score",
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

        visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, trained=False, log_to_wandb=True, set_name="pre-training", entire_dataset=True)
        visualize_tsne(model, device, ilo_dataset, train_loader, trained=False, log_to_wandb=True, set_name="pre-training", entire_dataset=False)


        results = train_model_with_clf(
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
            alpha = wandb.config.clf_alpha,
            margin_scheduling=wandb.config.margin_scheduling,
            initial_margin=wandb.config.initial_margin,
            final_margin=wandb.config.final_margin,
            scheduling_fraction=wandb.config.scheduling_fraction,
            mining_strat=wandb.config.mining,
            p_ilo_anchor=wandb.config.p_ilo_anchor,
            ilo_scheduling=wandb.config.ilo_scheduling,
            initial_p_ilo=wandb.config.initial_ilo,
            final_p_ilo=wandb.config.final_ilo,
        )

        test_map, test_class_map, test_accuracy, test_recall, test_precision, test_f1, test_kappa = test_model_with_clf(
            model, test_loader, device, ilo_dataset, ilo_images, ilo_labels, triplet_loss_fn, multi_class_loss_fn, wandb.config.clf_alpha, wandb.run.name)
         


                


    except KeyError as e:
        print(f"Missing configuration: {e}")


