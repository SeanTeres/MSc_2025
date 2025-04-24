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


def train_model(
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
    tsne_interval=10,
    log_to_wandb=True,
    mining_strat="Random",  # Mining strategy for triplet loss
    margin_scheduling=False,  # Enable/disable margin scheduling
    initial_margin=0.8,       # Initial larger margin
    final_margin=0.2,         # Final smaller margin
    scheduling_fraction=0.8   # Fraction of training to complete schedule
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
        
    Returns:
        dict: Dictionary containing trained model, best model state dict, 
              training history and best validation metrics
    """
    def get_scheduled_margin(current_epoch):
        """Calculate margin based on current training progress"""
        if not margin_scheduling:
            return triplet_loss_fn.margin  # Return the original margin
        
        # Calculate how far we are through the scheduled part of training
        schedule_point = min(1.0, current_epoch / (n_epochs * scheduling_fraction))
        
        # Linearly interpolate between initial and final margin (increasing)
        current_margin = initial_margin + (final_margin - initial_margin) * schedule_point
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
            current_margin = get_scheduled_margin(epoch)
            triplet_loss_fn.margin = current_margin
            print(f"Current margin: {current_margin:.4f}")
            
            # Log margin to wandb

        
        running_loss = 0.0
        running_loss = 0.0
        
        all_embeddings = []
        all_labels = []
        
        # Training loop
        for batch_idx, sample in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = sample[0]
            labels = sample[1]
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            feats = model.features(imgs)
            
            embeddings = F.normalize(feats, p=2, dim=1)
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            current_batch_labels = labels.cpu().numpy()
            
            batch_triplet_loss = 0.0
            n_triplets = 0
            
            all_losses = []
            # For each sample in the batch, build a triplet
            for i, positive_label in enumerate(current_batch_labels):
                # Positive embedding (from the current batch)
                positive_embedding = embeddings[i].unsqueeze(0)  # shape [1, C]
                
                # Find ILO anchors with the same label as the positive sample
                ilo_indices = torch.where(ilo_labels == positive_label)[0]
                if len(ilo_indices) > 0:
                    # Randomly select an ILO anchor
                    ilo_idx = np.random.choice(ilo_indices.cpu().numpy())
                    anchor_embedding = ilo_images[ilo_idx].unsqueeze(0)  # shape [1, C]
                    anchor_embedding = model.features(anchor_embedding)  # Get features of the anchor
                    anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                    anchor_label = ilo_labels[ilo_idx].item()  # Get the label of the anchor
                else:
                    print(f"No ILO anchor found for label {positive_label}. Skipping.")
                    continue
                
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

                        # print(f"NUMBER OF NEGATIVES: {len(dists)}")

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
                        
                        # Print distance diagnostics for the first batch
                        if batch_idx == 0 and i < 3:
                            print(f"Sample {i}: A-P distance: {positive_distance.item():.4f}")
                            print(f"Sample {i}: A-N distances: min={dists.min().item():.4f}, max={dists.max().item():.4f}, mean={dists.mean().item():.4f}")
                        
                        
                        # Try increasingly relaxed criteria:
                        # 1. First attempt: strict semi-hard negatives (original condition)
                        semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + current_margin))
                        semi_hard_dists = dists[semi_hard_mask]

                        if semi_hard_dists.numel() > 0:
                            # Pick the hardest among semi-hard (closest to positive)
                            hard_idx_in_masked = torch.argmin(semi_hard_dists).item()  # CHANGE: Use argmin instead of argmax
                            
                            # Map back to the original indices
                            semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                            selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                            
                            negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                            negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                            
                            if batch_idx == 0 and i < 3:
                                print(f"Sample {i}: Found semi-hard negative with distance {semi_hard_dists[hard_idx_in_masked].item():.4f}")
                        
                        # 2. Relaxed attempt: Increase margin
                        else:
                            larger_margin = current_margin * 2.0  # Double the margin
                            semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + larger_margin))
                            semi_hard_dists = dists[semi_hard_mask]
                            
                            if semi_hard_dists.numel() > 0:
                                hard_idx_in_masked = torch.argmin(semi_hard_dists).item()  # CHANGE: Use argmin instead of argmax
                                semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                                selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()
                                
                                negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                                negative_label = current_batch_labels[negative_indices[selected_neg_idx]]
                                
                                if batch_idx == 0 and i < 3:
                                    print(f"Sample {i}: Found relaxed semi-hard negative with distance {semi_hard_dists[hard_idx_in_masked].item():.4f}")
                            
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
                                    
                                    if batch_idx == 0 and i < 3:
                                        print(f"Sample {i}: Using hardest negative with distance {hard_negative_dists[hard_idx_in_masked].item():.4f}")
                                
                                # 4. Final fallback: Random negative
                                else:
                                    if batch_idx == 0 and i < 3:
                                        print(f"Sample {i}: No suitable negatives found, using random. A-P: {positive_distance.item():.4f}, all negatives closer")
                                    
                                    neg_idx = np.random.choice(negative_indices)
                                    negative_embedding = embeddings[neg_idx].unsqueeze(0)
                                    negative_label = current_batch_labels[neg_idx]
                # Compute triplet loss
                loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
                all_losses.append(loss)
                batch_triplet_loss += loss.item()
                n_triplets += 1
                
                if batch_idx == 0 and i == 0:
                    print(f"A: {anchor_label}, P: {positive_label}, N: {negative_label}")
                    print(f"Loss: {loss.item():.4f}")
                    # print(f"Candidate negatives: {len(dists)}, mining: {wandb.config.mining}")
            
            if n_triplets > 0:
                # Average the losses and backprop
                total_loss = torch.stack(all_losses).mean()
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
            else:
                print(f"Batch {batch_idx + 1}: No valid triplets found. Labels: {set(current_batch_labels)}")
                total_loss = torch.tensor(0.0)  # Default value when no triplets found
            
            # Log batch metrics
            if log_to_wandb:
                wandb.log({
                    "loss": total_loss.item() if n_triplets > 0 else 0,
                    "batch": batch_idx
                })
            
            del imgs, labels
            torch.cuda.empty_cache()
        
        # Calculate training metrics
        train_loss = running_loss / (batch_idx + 1)
        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}")
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        train_map, train_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
        
        print(f"Train mAP: {train_map:.4f}")
        print("- Per-Class Train mAP:")
        for class_id, ap in train_class_map.items():
            print(f"  Class {class_id}: mAP = {ap:.4f}")
        
        # Run t-SNE visualization at regular intervals
        if (epoch + 1) % tsne_interval == 0:
            visualize_tsne(model, device, ilo_dataset, train_loader, 
                          trained=True, log_to_wandb=log_to_wandb, n_epochs=epoch+1, set_name="training", entire_dataset=False)
            visualize_tsne(model, device, ilo_dataset, val_loader, 
                          trained=True, log_to_wandb=log_to_wandb, n_epochs=epoch+1, set_name="validation", entire_dataset=False)
        
        # Run validation
        print("\nVALIDATION\n")
        val_loss, val_map, val_class_map, val_embeddings, val_labels = validate(
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

                # Compute triplet loss
                loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
                batch_triplet_loss += loss.item()
                n_triplets += 1

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
        
        return val_loss, val_map, val_class_map, all_embeddings, all_labels
    
    return 0.0, 0.0, {}, None, None

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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



        wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
        wandb.init(project='MBOD-cl', name='BSHN-OS-m_03_05',
            config={
                "batch_size": 16,
                "n_epochs": 250,
                "learning_rate": 1e-4,
                "oversample": True,
                "initial_margin": 0.3,      
                "final_margin": 0.5,        
                "margin_scheduling": True,   # Enable margin scheduling
                "scheduling_fraction": 0.9,  # Complete scheduling in first x% of training
                "mining": "BSHN",
                "augmentations": False
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

        

        if(wandb.config.augmentations):
            augmentations_list = transforms.Compose([
                # transforms.CenterCrop(np.round(224 * 0.9).astype(int)),  # Example crop
                transforms.RandomRotation(degrees=(-5, 5)),  
               # transforms.Lambda(lambda img: helpers.salt_and_pepper_noise_tensor(img, prob=0.02)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.025, 0.025))
            ])
            # Get the dataloaders
            train_loader, _, _ = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="profusion_score",
                split_file="stratified_split.json",
                augmentations=augmentations_list,
                oversample=wandb.config.oversample
            )

            _, val_loader, test_loader = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="profusion_score",
                split_file="stratified_split.json",
                augmentations=None,
                oversample=None
            )

        else:
            train_loader, _, _ = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="profusion_score",
                split_file="stratified_split.json",
                augmentations=None,
                oversample=wandb.config.oversample
            )

            _, val_loader, test_loader = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="profusion_score",
                split_file="stratified_split.json",
                augmentations=None,
                oversample=None
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

        


        model.train()

        results = train_model(
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
            mining_strat=wandb.config.mining
        )

        
                

    except KeyError as e:
        print(f"Missing configuration: {e}")