from datetime import datetime
import glob
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchxrayvision as xrv
import wandb
import numpy as np
import yaml
import torch.nn.functional as F
import sklearn.metrics as skmetrics
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from dataloader import get_dataloaders

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from utils import load_config
# Add codev2 and DomainAdaptation to path
sys.path.append(os.path.abspath("../DomainAdaptation"))
from da_utils import reinitialize_weights, visualize_tsne_with_kaggle_tb, initialize_weights

sys.path.append(os.path.abspath("../classification"))
import metrics
from clf_manager import BinaryClassifier, MulticlassClassifier, XRVBasedClassifier
import scipy.stats as stats

def plot_weight_distributions(model, layer_name):
    # Find the layer by name
    layer = None
    for name, module in model.named_modules():
        if layer_name in name:
            layer = module
            break
    
    if layer is None or not hasattr(layer, 'weight'):
        print(f"Layer {layer_name} not found or has no weights")
        return
        
    weights = layer.weight.data.flatten().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=50, alpha=0.7)
    plt.title(f"Weight Distribution for {layer_name}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # For Kaiming init with ReLU, weights should follow N(0, sqrt(2/n_in))
    if isinstance(layer, nn.Conv2d):
        fan_in = layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
        std = (2 / fan_in) ** 0.5
        x = np.linspace(min(weights), max(weights), 1000)
        y = np.exp(-x**2/(2*std**2)) / (std * np.sqrt(2*np.pi)) * len(weights) * (max(weights)-min(weights))/50
        plt.plot(x, y, 'r-', label=f'Expected N(0, {std:.4f})')
        plt.legend()
    
    plt.show()
    plt.savefig(f"{layer_name}_weight_distribution.png")

model = xrv.models.ResNet(weights="resnet50-res512-all")

model.classifier = XRVBasedClassifier(input_dim=2048, num_classes=1, name="XRV-Base")

initialize_weights(model, init_backbone=True)

cfg = load_config("cl_config.yaml")

mbod_dataset = HDF5Dataset(
    hdf5_path = cfg["DATA_PATH_MBOD"],
    labels_key=cfg["LABELS_KEY_MBOD"],
    images_key="images",
    augmentations=None,
    preprocess=preprocess,
)