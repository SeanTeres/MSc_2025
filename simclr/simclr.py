
# TO DO: 
# - Layered learning rates
# - Stronger augmentations?
# - Gradient accumulation: SimCLR works better if you simulate large batches.
# - When accumulating, must calculate metrics on the effective batch, not per physical batch.

import random
import glob
import os
import sys
from datetime import datetime
import torch
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


from sklearn.manifold import TSNE
import yaml
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb

# Add paths - fix the correct path to MedVAE
sys.path.append(os.path.abspath("../mbod-data-processor"))
sys.path.append(os.path.abspath("../domain_adaptation"))
sys.path.append(os.path.abspath("../classification"))

from metrics import get_accuracy, get_specificity, get_sensitivity, get_f1_score, get_cm_for_class, cosine_alignment_loss, specificity_at_sensitivity, multiclass_specificity_at_sensitivity

import torch.utils
import torch.utils.data
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from utils import load_config

import torchxrayvision as xrv
import torch.nn as nn
import plotly.graph_objects as go  

from simclr_dataset import SimCLRDataset
from nt_xent_loss import nt_xent_loss

class SimCLR(nn.Module):
    # TO DO: Optional dropout, normalization, etc.
    def __init__(self, model, out_dim=128):
        super().__init__()
        self.model = model

        self.projector = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, out_dim)
        )


    def forward(self, x):
        h = self.model.features(x)
        z = self.projector(h)
        return h, z
    
def log_sample_images(aug1_batch, aug2_batch, epoch, batch_idx, num_samples=4):
    """Log sample images to wandb to verify data loading"""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(min(num_samples, aug1_batch.size(0))):
        # Convert from torch tensor to numpy and handle normalization for display
        img1 = aug1_batch[i].cpu().numpy()
        img2 = aug2_batch[i].cpu().numpy()
        
        # If single channel, squeeze the channel dimension
        if img1.shape[0] == 1:
            img1 = img1.squeeze(0)
            img2 = img2.squeeze(0)
        
        # Normalize for display (map to [0, 1] range)
        def normalize_for_display(img):
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return (img - img_min) / (img_max - img_min)
            return img
        
        img1_norm = normalize_for_display(img1)
        img2_norm = normalize_for_display(img2)
        
        # Plot augmented images
        axes[0, i].imshow(img1, cmap='gray')
        axes[0, i].set_title(f'Aug 1 - Sample {i+1}\n Unnormalized')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(img2, cmap='gray')
        axes[1, i].set_title(f'Aug 2 - Sample {i+1}\n Unnormalized')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Sample Images - Epoch {epoch+1}, Batch {batch_idx+1}')
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({
        f"sample_images/epoch_{epoch+1}": wandb.Image(plt),
        "sample_images/epoch": epoch + 1,
        "sample_images/batch": batch_idx + 1
    })
    
    plt.close(fig)

def compute_alignment_uniformity_v2(z1, z2, alpha=2, t=2):
    """
    Compute alignment and uniformity metrics matching
    'Understanding Contrastive Representation Learning' paper.
    """
    # Ensure float32
    z1, z2 = z1.float(), z2.float()

    # Normalize embeddings
    z1_norm = torch.nn.functional.normalize(z1, dim=1)
    z2_norm = torch.nn.functional.normalize(z2, dim=1)

    # Alignment: squared Euclidean distance between positive pairs
    alignment = ((z1_norm - z2_norm).norm(p=2, dim=1) ** alpha).mean()

    # Uniformity: how well embeddings are spread on hypersphere
    z_all = torch.cat([z1_norm, z2_norm], dim=0)
    pairwise_dists = torch.pdist(z_all, p=2).pow(2)
    uniformity = torch.log(torch.exp(-t * pairwise_dists).mean())

    return alignment.item(), uniformity.item()

def compute_alignment_uniformity(z1, z2, temperature=0.1):
    """
    Compute alignment and uniformity metrics for contrastive learning
    These are key metrics from "Understanding Contrastive Representation Learning"
    """
    # TO DO - For alignment, the original computes the squared Euclidean distance between normalized positive pairs, while your code computes the dot product (cosine similarity) instead.
    # Not entirely incorrect, but not consistent with the paper.
    # Does using the same temperature as the loss make sense? Should I be using a different scaling factor?

    batch_size = z1.shape[0]

    if z1.dtype == torch.float16:
        z1 = z1.float()
    if z2.dtype == torch.float16:
        z2 = z2.float()
    
    # Normalize embeddings
    z1_norm = torch.nn.functional.normalize(z1, dim=1)
    z2_norm = torch.nn.functional.normalize(z2, dim=1)
    
    # Alignment: how well positive pairs align
    alignment = (z1_norm * z2_norm).sum(dim=1).mean()
    
    # Uniformity: how uniformly distributed embeddings are on hypersphere
    z_all = torch.cat([z1_norm, z2_norm], dim=0)
    n = z_all.shape[0]
    dists = torch.pdist(z_all, p=2).pow(2)
    uniformity = dists.mul(-2).exp().mean().log()
    
    return alignment.item(), uniformity.item()

def train_one_epoch(model, dataloader, optimizer, device, epoch, scaler=None, log_to_wandb=True, log_batch_freq=50, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    total_alignment = 0.0
    total_uniformity = 0.0
    num_batches = 0
    accumulated_loss = 0.0

    # Add gradient scaling info
    if scaler is not None:
        print(f"Using mixed precision training with GradScaler")
    else:
        print("Using standard FP32 training")
    
    # Print gradient accumulation info
    effective_batch_size = dataloader.batch_size * accumulation_steps
    print(f"Physical batch size: {dataloader.batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Negatives per positive: {(effective_batch_size * 2) - 2}")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=130)
    
    for batch_idx, (aug1, aug2) in enumerate(pbar):
        aug1 = aug1.to(device, non_blocking=True)
        aug2 = aug2.to(device, non_blocking=True)

        if scaler is not None:
            # Mixed precision training with gradient accumulation
            with torch.amp.autocast(device_type='cuda'):
                _, aug1_z = model(aug1)
                _, aug2_z = model(aug2)
                loss = nt_xent_loss(aug1_z, aug2_z, temperature=TEMPERATURE)
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
        else:
            # Standard training with gradient accumulation
            _, aug1_z = model(aug1)
            _, aug2_z = model(aug2)
            loss = nt_xent_loss(aug1_z, aug2_z, temperature=TEMPERATURE)
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            loss.backward()
            accumulated_loss += loss.item()

        # Compute alignment and uniformity metrics (use original embeddings, not scaled loss)
        with torch.no_grad():
            alignment, uniformity = compute_alignment_uniformity(aug1_z, aug2_z, temperature=TEMPERATURE)
            total_alignment += alignment
            total_uniformity += uniformity

        # Update weights after accumulating gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            # Log the accumulated loss (scaled back up)
            total_loss += accumulated_loss * accumulation_steps
            num_batches += 1
            
            # Reset accumulated loss
            accumulated_loss = 0.0
        
        # Update progress bar with current metrics
        current_step = (batch_idx % accumulation_steps) + 1
        pbar.set_postfix({
            "Loss": f"{(loss.item() * accumulation_steps):.4f}",  # Show unscaled loss
            "Align": f"{alignment:.3f}",
            "Uniform": f"{uniformity:.3f}",
            "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
            "Step": f"{current_step}/{accumulation_steps}"  # Show accumulation progress
        })

        # Log batch-level metrics every N accumulation cycles
        if log_to_wandb and (batch_idx + 1) % (log_batch_freq * accumulation_steps) == 0:
            global_step = epoch * len(dataloader) + batch_idx
            
            wandb.log({
                "batch/loss": loss.item() * accumulation_steps,  # Unscaled loss
                "batch/alignment": alignment,
                "batch/uniformity": uniformity,
                "batch/learning_rate": optimizer.param_groups[0]['lr'],
                "batch/step": global_step,
                "batch/epoch": epoch + 1,
                "batch/effective_batch_size": effective_batch_size,
                "batch/accumulation_steps": accumulation_steps
            })

            if(epoch == 0):
                log_sample_images(aug1, aug2, epoch, batch_idx)

    # Handle any remaining gradients at end of epoch
    if (batch_idx + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss * accumulation_steps
        num_batches += 1

    # Log epoch-level metrics
    avg_loss = total_loss / max(num_batches, 1)
    avg_alignment = total_alignment / len(dataloader)
    avg_uniformity = total_uniformity / len(dataloader)
    
    if log_to_wandb:
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch/alignment": avg_alignment,
            "epoch/uniformity": avg_uniformity,
            "epoch/learning_rate": optimizer.param_groups[0]['lr'],
            "epoch/number": epoch + 1,
            "epoch/effective_batch_size": effective_batch_size,
            "epoch/accumulation_steps": accumulation_steps
        })

    return avg_loss


def save_checkpoint(model, optimizer, save_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

config = load_config("/home/sean/MSc_2025/simclr/config.yaml")

mbod_dataset_path = config["merged_silicosis_output"]["hdf5_file"]
rand_v3_dataset_path = config["rand_v3_output"]["hdf5_file"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def denormalize_xray(tensor):
    """Convert normalized tensor to torchxrayvision expected range [-1024, 1024]"""
    
    # Check current range
    current_min = tensor.min().item()
    current_max = tensor.max().item()
    
    # print(f"Input range: [{current_min:.3f}, {current_max:.3f}]")
    
    # If data is in [0, 1] range (common after ToTensor)
    if current_min >= 0 and current_max <= 1:
        # Map [0, 1] to [-1024, 1024]
        tensor = (tensor - 0.5) * 2048
    
    # If data is in [-1, 1] range
    elif current_min >= -1 and current_max <= 1:
        # Map [-1, 1] to [-1024, 1024]
        tensor = tensor * 1024
    
    # If data is in some other normalized range, map to [-1024, 1024]
    else:
        # Generic normalization to [-1024, 1024]
        tensor = (tensor - current_min) / (current_max - current_min)  # Map to [0, 1]
        tensor = (tensor - 0.5) * 2048  # Map to [-1024, 1024]
    
    #print(f"Output range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    return tensor

# Updated preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(denormalize_xray),
])
# Create datasets
mbod_merged = HDF5Dataset(
    hdf5_path=mbod_dataset_path,
    labels_key="silicosis",
    images_key="images",
    preprocess=preprocess
)


rand_v3 = HDF5Dataset(
    hdf5_path=rand_v3_dataset_path,
    labels_key="silicosis",
    images_key="images",
    preprocess=preprocess
)


# vae_model = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device)
model = xrv.models.ResNet(weights="resnet50-res512-all").to(device)

rand_v3_loader = torch.utils.data.DataLoader(
    rand_v3, batch_size=8, shuffle=False
)
mbod_loader = torch.utils.data.DataLoader(
    mbod_merged, batch_size=8, shuffle=False
)

simclr_transform = transforms.Compose([
    # Core augmentations (always apply these)
    transforms.RandomResizedCrop(512, scale=(0.3, 0.9)),  # Less aggressive cropping
    transforms.RandomHorizontalFlip(p=0.5),               # Keep this as-is
    
    # Optional augmentations (randomly applied)
    transforms.RandomApply([
        transforms.RandomRotation(15, fill=0)             # Only rotate 40% of time
    ], p=0.4),
        
    #     transforms.RandomApply([
    #         transforms.ColorJitter(brightness=0.01, contrast=0.01)  # Only jitter 50% of time
    #     ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))  # Only blur 30% of time
        ], p=0.3),
    
])
model = xrv.models.ResNet(weights="resnet50-res512-all").to(device)

simclr_model = SimCLR(model).to(device)


with open("simclr_cfg.yaml", "r") as f:
    exp_cfg = yaml.safe_load(f)

EXP_NAME = exp_cfg["EXPERIMENT"]["NAME"]
SAVE_DIR = exp_cfg["EXPERIMENT"]["SAVE_DIR"]
RANDOM_SEED = exp_cfg["EXPERIMENT"]["SEED"]

TRAIN_SET = exp_cfg["DATA"]["TRAIN_SET"]

EPOCHS = exp_cfg["TRAINING"]["EPOCHS"]
BATCH_SIZE = exp_cfg["TRAINING"]["BATCH_SIZE"]
LEARNING_RATE = exp_cfg["TRAINING"]["LEARNING_RATE"]
WEIGHT_DECAY = exp_cfg["TRAINING"]["WEIGHT_DECAY"]
TEMPERATURE = exp_cfg["TRAINING"]["TEMPERATURE"]
USE_MIXED_PRECISION = exp_cfg["TRAINING"]["USE_MIXED_PRECISION"]
GRADIENT_ACCUMULATION_STEPS = exp_cfg["TRAINING"].get("GRADIENT_ACCUMULATION_STEPS", 1)


WANDB_API_KEY = exp_cfg["WANDB"]["LOGIN"]
PROJECT_NAME = exp_cfg["WANDB"]["PROJECT_NAME"]

MODEL = exp_cfg["MODEL"]["SOURCE"]
WEIGHTS = exp_cfg["MODEL"]["WEIGHTS"]
PROJ_OUT_DIM = exp_cfg["MODEL"]["PROJ_OUT_DIM"]
FREEZE_ENC = exp_cfg["MODEL"]["FREEZE_ENCODER"]
PRETRAINED = exp_cfg["MODEL"]["PRETRAINED"]

if FREEZE_ENC:
    for param in simclr_model.model.parameters():
        param.requires_grad = False
    print("Encoder frozen. Training projector only.")
    
    # Only optimize projector parameters
    optimizer = torch.optim.Adam(
        simclr_model.projector.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
else:
    print("Fine-tuning all weights...")
    # Optimize all parameters
    optimizer = torch.optim.Adam(
        simclr_model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )

# Replace the current scaler=None with:
print(f"Mixed precision enabled: {USE_MIXED_PRECISION}")

if USE_MIXED_PRECISION:
    scaler = GradScaler()
    print("✅ GradScaler initialized for mixed precision training")
else:
    scaler = None
    print("⚠️ Using standard FP32 training")    

print(TRAIN_SET)

if(TRAIN_SET == "RAND_V3"):
    simclr_dataset = SimCLRDataset(rand_v3, transform=simclr_transform, supervised=False)
elif(TRAIN_SET == "MBOD"):
    simclr_dataset = SimCLRDataset(mbod_merged, transform=simclr_transform, supervised=False)
else:
    raise ValueError(f"Unsupported TRAIN_SET: {TRAIN_SET}")


loader = torch.utils.data.DataLoader(
    simclr_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)


wandb.login()
wandb.init(project=PROJECT_NAME, name=EXP_NAME, config=exp_cfg)

best_loss = float('inf')
os.makedirs(SAVE_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*60}")

    loss = train_one_epoch(
        simclr_model, loader, optimizer, device, epoch, 
        scaler=scaler, log_to_wandb=True, log_batch_freq=25,  # Reduced freq for accumulation
        accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    if loss < best_loss:
        best_loss = loss
        checkpoint_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_best_simclr_model.pth")
        save_checkpoint(simclr_model, optimizer, checkpoint_path)

    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_epoch_{epoch+1}.pth")
        save_checkpoint(simclr_model, optimizer, checkpoint_path)


final_checkpoint_path = os.path.join(SAVE_DIR, f"{EXP_NAME}_final_simclr_model.pth")
save_checkpoint(simclr_model, optimizer, final_checkpoint_path)

print(f"BEST LOSS: {best_loss}")

wandb.finish()
