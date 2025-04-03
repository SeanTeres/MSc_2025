import wandb
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchxrayvision as xrv
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import utils.helpers as helpers
import utils.classes as classes
import utils.train_utils as train_utils


def log_augmented_images(dataset, num_images=5):
    images = []
    for i in range(num_images):
        img, label = dataset[i]
        images.append(wandb.Image(img.permute(1, 2, 0).numpy(), caption=f"Label: {label}"))
    wandb.log({"Augmented Images": images})

# Load the configuration file
with open('classification/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Define model-resolution pairs
model_configs = [
    {
        'model': 'densenet121-res224-all',
        'resolution': 224
    },
    {
        'model': 'resnet50-res512-all',
        'resolution': 512
    }
]

# Update sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'distribution': 'uniform',
            'min': 0.0009,
            'max': 0.002
        },
        'n_epochs': {
            'distribution': 'categorical',
            'values': [30]
        },
        'batch_size': {
            'distribution': 'categorical',
            'values': [16]
        },
        'train_dataset': {
            'distribution': 'categorical',
            'values': ['MBOD 1', 'MBOD 2']
        },
        'augmentation': {
            'distribution': 'categorical',
            'values': [False, True]
        },
        'model_resolution': {
            'distribution': 'categorical',
            'values': [512]
        },
        'oversampling': {
            'distribution': 'categorical',
            'values': [True]
        },
        'classifier': {
            'distribution': 'categorical',
            'values': ['Base512Classifier']
        },
        'loss_function': {
            'distribution': 'categorical',
            'values': ['CrossEntropyLoss']
        },
        'BCE_pos_weight': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }, 
        'target': {
            'distribution': 'categorical',
            'values': ['Profusion']
        }
    }
}