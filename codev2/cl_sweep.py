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
from tsne import visualize_tsne2, visualize_tsne

from cl import train_model, validate
from sweep_config import sweep_config



def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Get the path to the HDF5 files
    hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
    ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]

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

    # Set up data loaders with the current sweep parameters
    train_loader, val_loader, test_loader = get_dataloaders(
        hdf5_path=hdf5_file_path,
        preprocess=preprocess,
        batch_size=wandb.config.batch_size,
        labels_key="profusion_score",
        split_file="stratified_split.json",
        augmentations=None
    )

    # Create model
    model = xrv.models.ResNet(weights="resnet50-res512-all")
    model = model.to(device)

    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.learning_rate
    )
    
    # Create loss function with current margin value
    triplet_loss_fn = nn.TripletMarginLoss(margin=wandb.config.initial_margin, p=2)

    # Generate a unique experiment name including the margin
    experiment_name = f"sweep-{wandb.config.mining}-m{wandb.config.initial_margin}"

    # Preload ILO images
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
    ilo_images = torch.cat(ilo_images, dim=0)
    ilo_labels = torch.stack(ilo_labels)

    print(f"ILO images loaded onto GPU: {ilo_images.shape}")
    print(f"ILO labels loaded onto GPU: {ilo_labels.shape}")

    # Generate initial t-SNE visualization
    visualize_tsne(model, device, ilo_dataset, train_loader, trained=False, log_to_wandb=True, set_name="pre-training")

    # Train model
    results = train_model(
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
        margin_scheduling=wandb.config.margin_scheduling,
        initial_margin=wandb.config.initial_margin,
        final_margin=wandb.config.final_margin if hasattr(wandb.config, 'final_margin') else wandb.config.initial_margin,
        scheduling_fraction=wandb.config.scheduling_fraction if hasattr(wandb.config, 'scheduling_fraction') else 0.5,
        mining_strat=wandb.config.mining
    )

    # Log final metrics
    wandb.log({
        "best_val_map": results["best_val_map"],
        "final_train_map": results["final_train_map"],
        "final_val_map": results["final_val_map"]
    })

# Define the sweep agent function
def sweep_agent():
    wandb.init()
    main()

# Start the sweep
if __name__ == "__main__":
    wandb.login(key='176da722bd80e35dbc4a8cea0567d495b7307688')
    sweep_id = wandb.sweep(sweep_config, project='MBOD-cl')
    wandb.agent(sweep_id, function=sweep_agent)