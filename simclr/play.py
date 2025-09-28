
# TO DO: 
# - Layered learning rates
# - Mixed Precision
# - Stronger augmentations?

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
from metrics import get_accuracy, get_specificity, get_sensitivity, get_f1_score, get_cm_for_class, cosine_alignment_loss, specificity_at_sensitivity, multiclass_specificity_at_sensitivity, multilabel_specificity_at_sensitivity

import torch.utils
import torch.utils.data
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from utils import load_config

import torchxrayvision as xrv
import torch.nn as nn
import plotly.graph_objects as go  

from simclr_dataset import SimCLRDataset
from simclr.nt_xent_loss import nt_xent_loss

class SimCLR(nn.Module):
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
    

config = load_config("/home/sean/MSc/classification/config.yaml")

mbod_dataset_path = config["merged_silicosis_output"]["hdf5_file"]
rand_v1_dataset_path = config["rand_v1_output"]["hdf5_file"]
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

rand_v1 = HDF5Dataset(
    hdf5_path=rand_v1_dataset_path,
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

rand_v1_loader = torch.utils.data.DataLoader(
    rand_v1, batch_size=8, shuffle=False
)
rand_v3_loader = torch.utils.data.DataLoader(
    rand_v3, batch_size=8, shuffle=False
)
mbod_loader = torch.utils.data.DataLoader(
    mbod_merged, batch_size=8, shuffle=False
)

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),  # encourages focus on different regions
    transforms.RandomHorizontalFlip(),                     # safe and common for CXR
    transforms.RandomRotation(5),                          # small rotations to avoid misalignment
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # mild photometric changes
])

model = xrv.models.ResNet(weights="resnet50-res512-all").to(device)

simclr_model = SimCLR(model).to(device)



