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
from tsne import visualize_tsne
import math
import random
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses, miners

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def print_dataloader_label_distribution(dataloader, label_mapping=None, wrapped=False):
    """
    Print the distribution of labels in a dataloader
    
    Args:
        dataloader: PyTorch dataloader
        label_mapping: Optional dict mapping label indices to human-readable names
    """
    label_counts = {}
    
    # Collect all labels
    if not wrapped:
        all_labels = []
        for _, labels in dataloader:
            if isinstance(labels, torch.Tensor):
                batch_labels = labels.cpu().numpy()
            else:
                batch_labels = np.array(labels)
                
            if len(batch_labels.shape) > 1:
                # For multi-label case, take argmax
                batch_labels = np.argmax(batch_labels, axis=1)
                
            all_labels.extend(batch_labels)
        
        # Count occurrences
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        
        # Print distribution
        total_samples = len(all_labels)
        print(f"Total samples: {total_samples}")
        print("Label distribution:")
        
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_samples) * 100
            if label_mapping and label in label_mapping:
                label_name = f"{label} ({label_mapping[label]})"
            else:
                label_name = str(label)
            
            print(f"  {label_name}: {count} samples ({percentage:.2f}%)")
    else:
        all_labels = []
        for _, labels in dataloader:
            if isinstance(labels, torch.Tensor):
                batch_labels = labels.cpu().numpy()
            else:
                batch_labels = np.array(labels)
                
            if len(batch_labels.shape) > 1:
                # For multi-label case, take argmax
                batch_labels = np.argmax(batch_labels, axis=1)
                
            all_labels.extend(batch_labels)
        
        # Count occurrences
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        
        # Print distribution
        total_samples = len(all_labels)
        print(f"Total samples: {total_samples}")
        print("Label distribution:")
        
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_samples) * 100
            if label_mapping and label in label_mapping:
                label_name = f"{label} ({label_mapping[label]})"
            else:
                label_name = str(label)
            
            print(f"  {label_name}: {count} samples ({percentage:.2f}%)")


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: Feature vectors of shape [batch_size, feature_dim]
            labels: Ground truth labels of shape [batch_size]
        Returns:
            A loss scalar
        """
        # Normalize feature vectors
        features = F.normalize(features, p=2, dim=1)
        
        # Device setup
        batch_size = features.shape[0]
        device = features.device
        
        # Create mask for matching and non-matching labels
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Create logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast cases (diagonal)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive samples
        # Mask out the "self" examples first
        mask = mask * logits_mask
        
        # Compute loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        
        # Return scalar loss
        return loss.mean()


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



def train_model_supcon(
    model,
    train_loader,
    val_loader,
    supcon_loss_fn,
    optimizer,
    device,
    n_epochs,
    experiment_name,
    checkpoint_dir="checkpoints",
    tsne_interval=50,
    log_to_wandb=True
):
    """
    Train a model using Supervised Contrastive Loss
    """
    # Create checkpoint directory
    checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tracking metrics
    num_classes = 8  # For multiclass_stb
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
        'train_sensitivity': [],
        'val_sensitivity': [],
        'train_class_sensitivity': [],
        'val_class_sensitivity': []
    }
    
    # Initialize class-specific metrics
    per_class_metrics = {class_id: {'train_ap': [], 'val_ap': []} for class_id in range(num_classes)}
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("=" * 50)
        
        model.train()
        epoch_total_loss = 0.0
        epoch_batch_count = 0
        
        all_embeddings = []
        all_labels = []
        
        # Training loop
        for batch_idx, sample in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()
            
            imgs = sample[0].to(device)
            labels = sample[1].to(device)
            
            # Forward pass through the model
            feats = model.features(imgs)
            
            # Calculate loss - no mining needed!
            loss = supcon_loss_fn(feats, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track embeddings and labels for evaluation
            embeddings = F.normalize(feats, p=2, dim=1)
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            # Update metrics
            epoch_total_loss += loss.item()
            epoch_batch_count += 1
            
            if log_to_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "batch_supcon_loss": loss.item(),
                    "batch": batch_idx + epoch * len(train_loader),
                })
        
        # End of epoch calculations
        train_loss = epoch_total_loss / epoch_batch_count
        
        # Calculate training metrics (mAP, etc)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
            
        # Calculate mAP for full labels and profusion-only
        train_map, train_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
        prof_all_labels = all_labels % 4
        train_prof_map, train_prof_class_map = helpers.compute_map_per_class(all_embeddings, prof_all_labels)
        
        # Calculate sensitivity
        train_sensitivity, train_class_sensitivity = helpers.compute_sensitivity_at_specificity(
            all_embeddings, all_labels, specificity_target=0.70)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_map'].append(train_map)
        history['train_class_map'].append(train_class_map)
        history['train_prof_map'].append(train_prof_map)
        history['train_sensitivity'].append(train_sensitivity)
        history['train_class_sensitivity'].append(train_class_sensitivity)
        
        # Run validation
        val_metrics = validate_supcon(
            model=model,
            val_loader=val_loader,
            device=device,
            supcon_loss_fn=supcon_loss_fn,
            num_classes=num_classes
        )
        
        # Update history with validation metrics
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_map'].append(val_metrics['val_map'])
        history['val_class_map'].append(val_metrics['val_class_map'])
        history['val_prof_map'].append(val_metrics['val_prof_map'])
        history['val_sensitivity'].append(val_metrics['val_sensitivity'])
        history['val_class_sensitivity'].append(val_metrics['val_class_sensitivity'])
        
        # Log metrics to wandb
        if log_to_wandb:
            wandb_log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_map": train_map,
                "val_loss": val_metrics['val_loss'],
                "val_map": val_metrics['val_map'],
                "val_prof_map": val_metrics['val_prof_map'],
                "train_prof_map": train_prof_map,
                "train_sens@spec": train_sensitivity,
                "val_sens@spec": val_metrics['val_sensitivity']
            }
            
            # Add per-class metrics
            for class_id in range(num_classes):
                wandb_log_dict[f"train_class_{class_id}_map"] = train_class_map.get(class_id, 0.0)
                wandb_log_dict[f"val_class_{class_id}_map"] = val_metrics['val_class_map'].get(class_id, 0.0)
                wandb_log_dict[f"train_class_{class_id}_sensitivity"] = train_class_sensitivity.get(class_id, 0.0)
                wandb_log_dict[f"val_class_{class_id}_sensitivity"] = val_metrics['val_class_sensitivity'].get(class_id, 0.0)
            
            wandb.log(wandb_log_dict)
        
        # Run t-SNE visualization at regular intervals
        if (epoch + 1) % tsne_interval == 0:
            visualize_tsne(model, device, ilo_dataset, train_loader, 
                          trained=True, log_to_wandb=log_to_wandb, 
                          n_epochs=epoch+1, set_name="training", entire_dataset=False)
            visualize_tsne(model, device, ilo_dataset, val_loader, 
                          trained=True, log_to_wandb=log_to_wandb,
                          n_epochs=epoch+1, set_name="validation", entire_dataset=False)
        
        # Save best model
        if epoch == 0 or (val_metrics['val_map'] > best_val_map and epoch > 50):
            best_val_map = val_metrics['val_map']
            print(f"Saving best model with validation mAP: {best_val_map:.4f}")
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(checkpoint_dir, f"best_model.pth"))
            
            visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, True, True, 
                          n_epochs=epoch+1, set_name="best val mAP", entire_dataset=True)
        
        # Memory management
        if (torch.cuda.memory_allocated()/1e9) > 3:
            torch.cuda.empty_cache()
            gc.collect()
            
        # Save latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, f"final_model.pth"))

    # Final TSNE visualization
    visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, True, True, 
                  n_epochs=n_epochs, set_name="final", entire_dataset=True)
    
    # Return training results
    return {
        'model': model,
        'best_model_state': best_model_state,
        'history': history,
        'best_val_map': best_val_map,
        'per_class_metrics': per_class_metrics
    }


def validate_supcon(model, val_loader, device, supcon_loss_fn, num_classes=8):
    """
    Validation function for Supervised Contrastive Loss
    """
    model.eval()
    running_loss = 0.0
    batch_count = 0
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            # Get validation batch
            imgs = sample[0].to(device)
            labels = sample[1].to(device)
            
            # Extract features
            features = model.features(imgs)
            
            # Calculate loss
            loss = supcon_loss_fn(features, labels)
            running_loss += loss.item()
            batch_count += 1
            
            # Store embeddings for metrics
            embeddings = F.normalize(features, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    # Calculate metrics
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate mAP for full labels
        val_map, val_class_map = helpers.compute_map_per_class(all_embeddings, all_labels)
        
        # Calculate mAP for profusion scores only
        prof_all_labels = all_labels % 4
        val_prof_map, val_prof_class_map = helpers.compute_map_per_class(all_embeddings, prof_all_labels)
        
        # Calculate sensitivity
        val_sensitivity, val_class_sensitivity = helpers.compute_sensitivity_at_specificity(
            all_embeddings, all_labels, specificity_target=0.70)
            
        # Return metrics
        return {
            'val_loss': running_loss / batch_count,
            'val_map': val_map,
            'val_class_map': {class_id: val_class_map.get(class_id, 0.0) for class_id in range(num_classes)},
            'val_prof_map': val_prof_map,
            'val_prof_class_map': val_prof_class_map,
            'val_sensitivity': val_sensitivity,
            'val_class_sensitivity': val_class_sensitivity
        }
    
    # Return empty metrics if no batches were processed
    return {
        'val_loss': 0.0,
        'val_map': 0.0,
        'val_class_map': {class_id: 0.0 for class_id in range(num_classes)},
        'val_prof_map': 0.0,
        'val_prof_class_map': {class_id: 0.0 for class_id in range(4)},
        'val_sensitivity': 0.0,
        'val_class_sensitivity': {class_id: 0.0 for class_id in range(num_classes)}
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

        
        multiclass_stb_mapping = {
            0: "Profusion 0, No TB",
            1: "Profusion 1, No TB",
            2: "Profusion 2, No TB",
            3: "Profusion 3, No TB",
            4: "Profusion 0, With TB",
            5: "Profusion 1, With TB",
            6: "Profusion 2, With TB",
            7: "Profusion 3, With TB"
        }

        wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
        wandb.init(project='MBOD-cl-2', name='mstb_supcon',
            config={
                "experiment_type": "SupCon (single view)",
                "beta_factor": 0.25,
                "dataset": "MBOD ONLY",
                "labeling_scheme": "MSTB",
                "batch_size": 24,
                "n_epochs": 1000,
                "learning_rate": 1e-4,
                "oversample": True,
                "initial_margin": 0.05,      
                "final_margin": 0.4,        
                "margin_scheduling": True,   # Enable margin scheduling
                "scheduling_fraction": 0.75,  # Complete scheduling in first x% of training
                "mining": "BSHN-v2",
                "augmentations": True,
                "filtered_dataset": True,
                "loss_function": "Triplet",
                "p_ilo_anchor": 0.5,
                "p_ilo_final": 0.1,
                "num_classes": 8,  # Explicitly specify 8 classes
                "OS_factor": 0.75,  # Oversampling factor
                "p_ilo_scheduling": True,  # Enable p_ilo scheduling
                "supcon_temp": 0.07,  # Temperature for SupCon loss
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
        margin_2 = wandb.config.initial_margin * wandb.config.beta_factor


        if "RNR" in wandb.config.experiment_type:
            loss_type = "RelativeNegativeRanking"
        elif "TNR" in wandb.config.experiment_type:
            loss_type = "TieredNegativeRanking"
        elif "DoubleTriplet" in wandb.config.experiment_type:
            loss_type = "DoubleTriplet"
        else:
            assert ValueError(f"Unknown experiment type: {wandb.config.experiment_type}")
            loss_type="NONE"

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
        
        supcon_loss_fn = SupConLoss(temperature=wandb.config.supcon_temp)

        # Train with SupCon loss
        results = train_model_supcon(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            supcon_loss_fn=supcon_loss_fn,
            optimizer=optimizer,
            device=device,
            n_epochs=wandb.config.n_epochs,
            experiment_name=experiment_name
        )

       

    except KeyError as e:
        print(f"Missing configuration: {e}")


