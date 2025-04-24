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
import seaborn as sns
import io
import torchvision.transforms as transforms
import os
from sklearn.manifold import TSNE
import matplotlib.cm as cm

def visualize_tsne(model, device, ilo_dataset, mbod_loader, trained=False, log_to_wandb=False, n_epochs=0, set_name="Training", entire_dataset=False):
    """
    Generate t-SNE visualization for comparing embeddings from ILO and MBOD datasets
    
    Args:
        model: The model to extract features with
        device: Device to run computations on (cpu/cuda)
        ilo_dataset: Dataset of ILO images (same format as MBOD but no DataLoader)
        mbod_loader: DataLoader for MBOD dataset
        trained: Whether the model is trained or not (for filename)
        log_to_wandb: Whether to log results to W&B
        n_epochs: Number of epochs the model was trained (for title)
    """
    print("Starting t-SNE visualization generation...")
    model.eval()

    ilo_feats = []
    ilo_labels = []

    # Process ILO dataset
    if ilo_dataset is not None:
        print(f"\nProcessing {len(ilo_dataset)} ILO images...\n")
        for idx in range(len(ilo_dataset)):
            sample = ilo_dataset[idx]
            img = sample[0].unsqueeze(0).to(device)  # Add batch dimension
            label = sample[1]
            # print(f"label: {label}")

            if isinstance(label, (torch.Tensor, np.ndarray)):
                label = label.item() if hasattr(label, 'item') else float(label)

            with torch.no_grad():
                feats = model.features(img)
                feats = torch.flatten(feats, start_dim=1)

            ilo_feats.append(feats.cpu().numpy())
            ilo_labels.append(label)

        ilo_feats = np.concatenate(ilo_feats, axis=0)
        ilo_labels = np.array(ilo_labels)
        print(f"Processed {len(ilo_feats)} ILO images.")

    mbod_feats = []
    mbod_labels = []

    # Process MBOD DataLoader
    print("Processing MBOD batches...")
    for batch in mbod_loader:
        imgs = batch[0].to(device)  # Add channel dim if missing
        labels = batch[1]

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        with torch.no_grad():
            feats = model.features(imgs)
            feats = torch.flatten(feats, start_dim=1)

        mbod_feats.append(feats.cpu().numpy())
        mbod_labels.append(labels)

    mbod_feats = np.concatenate(mbod_feats, axis=0)
    mbod_labels = np.concatenate(mbod_labels, axis=0)
    print(f"Processed {len(mbod_feats)} MBOD images.")

    # Combine ILO and MBOD features and labels
    if ilo_dataset is not None:
        all_feats = np.concatenate([ilo_feats, mbod_feats], axis=0)
        all_labels = np.concatenate([ilo_labels, mbod_labels], axis=0)
    else:
        all_feats = mbod_feats
        all_labels = mbod_labels

    print("Fitting t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, verbose=1)
    all_feats_2d = tsne.fit_transform(all_feats)

    # Create directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Get unique class labels and set up a discrete colormap
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    
    print(f"Found {n_classes} unique classes: {unique_labels}")
    
    # Use tab10 for up to 10 classes, tab20 for up to 20
    cmap_name = 'tab10' if n_classes <= 10 else 'tab20'
    cmap = plt.cm.get_cmap(cmap_name, max(n_classes, 10))  # Ensure at least 10 colors
    
    # Create a mapping from class labels to colors
    label_to_color = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}
    
    # Separate ILO and MBOD features
    n_ilo = len(ilo_feats)
    ilo_coords = all_feats_2d[:n_ilo]
    mbod_coords = all_feats_2d[n_ilo:]
    ilo_labels_subset = all_labels[:n_ilo]
    mbod_labels_subset = all_labels[n_ilo:]
    
    # Create file names
    file_name_by_class = f"visualizations/tsne_by_class{'_trained' if trained else '_untrained'}.png"
    
    # Visualization by class and source
    plt.figure(figsize=(14, 10))
    
    # Plot each class with a different color, and ILO/MBOD with different markers
    for label in unique_labels:
        # ILO points for this class
        idx_ilo = np.where(ilo_labels_subset == label)[0]
        if len(idx_ilo) > 0:
            plt.scatter(
                ilo_coords[idx_ilo, 0], 
                ilo_coords[idx_ilo, 1],
                c=[label_to_color[label]], 
                marker='*', 
                s=120, 
                label=f'ILO - {int(label)}',
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )
        
        # MBOD points for this class
        idx_mbod = np.where(mbod_labels_subset == label)[0]
        if len(idx_mbod) > 0:
            plt.scatter(
                mbod_coords[idx_mbod, 0], 
                mbod_coords[idx_mbod, 1],
                c=[label_to_color[label]], 
                marker='o', 
                s=40, 
                label=f'MBOD - {int(label)}',
                alpha=0.6,
                edgecolors='white',
                linewidths=0.2
            )
    
    if entire_dataset:
        title_to_add = "Entire Dataset"
    else:
        title_to_add = ""

    plt.title(f"t-SNE Visualization {'(Trained Model)' if trained else '(Untrained Model)'}\n{title_to_add} - Epoch {n_epochs}", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create a custom legend with one entry per class and source combination
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(file_name_by_class, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE visualizations to {file_name_by_class}")
    
    # Log to wandb if requested
    if log_to_wandb:
        print("Logging t-SNE visualizations to wandb...")
        wandb.log({
            f"{set_name} tsne": wandb.Image(file_name_by_class)
        })
        print("Logged visualizations to wandb successfully")


def visualize_tsne2(model, device, ilo_dataset, mbod_loader, trained=False, log_to_wandb=False, n_epochs=0, set_name="pre-training"):
    """
    Generate t-SNE visualization for comparing embeddings from ILO and MBOD datasets
    
    Args:
        model: The model to extract features with
        device: Device to run computations on (cpu/cuda)
        ilo_dataset: Dataset of ILO images
        mbod_loader: DataLoader for MBOD dataset
        trained: Whether the model is trained or not (for filename)
        log_to_wandb: Whether to log results to W&B
        n_epochs: Number of epochs the model was trained (for title)
    """
    print("Starting t-SNE visualization generation...")
    model.eval()

    ilo_feats = []
    ilo_labels = []

    if ilo_dataset is not None:
        print(f"Processing {len(ilo_dataset)} ILO images...")
        for i in range(len(ilo_dataset)):
            img, label, _ = ilo_dataset[i]

            if isinstance(label, (torch.Tensor, np.ndarray)):
                label = label.item() if hasattr(label, 'item') else float(label)

            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                feats = model.features(img)
                feats = torch.flatten(feats, start_dim=1)

            ilo_feats.append(feats.cpu().numpy())
            ilo_labels.append(label)

        ilo_feats = np.concatenate(ilo_feats, axis=0)
        ilo_labels = np.array(ilo_labels)
        print(f"Processed {len(ilo_feats)} ILO images.")

    mbod_feats = []
    mbod_labels = []

    print("Processing MBOD batches...")
    for batch in mbod_loader:
        imgs = batch[0].unsqueeze(1).to(device)  # add channel dim if missing
        labels = batch[1]

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        with torch.no_grad():
            feats = model.features(imgs)
            feats = torch.flatten(feats, start_dim=1)

        mbod_feats.append(feats.cpu().numpy())
        mbod_labels.append(labels)

    mbod_feats = np.concatenate(mbod_feats, axis=0)
    mbod_labels = np.concatenate(mbod_labels, axis=0)
    print(f"Processed {len(mbod_feats)} MBOD images.")

    # Combine

    if ilo_dataset is not None:
        all_feats = np.concatenate([ilo_feats, mbod_feats], axis=0)
        all_labels = np.concatenate([ilo_labels, mbod_labels], axis=0)
    else:
        all_feats = mbod_feats
        all_labels = mbod_labels

    print("Fitting t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, verbose=1)
    all_feats_2d = tsne.fit_transform(all_feats)

    # Create directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Get unique class labels and set up a discrete colormap
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    
    print(f"Found {n_classes} unique classes: {unique_labels}")
    
    # Use tab10 for up to 10 classes, tab20 for up to 20
    cmap_name = 'tab10' if n_classes <= 10 else 'tab20'
    cmap = plt.cm.get_cmap(cmap_name, max(n_classes, 10))  # Ensure at least 10 colors
    
    # Create a mapping from class labels to colors
    label_to_color = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}
    
    # Separate ILO and MBOD features
    n_ilo = len(ilo_feats)
    ilo_coords = all_feats_2d[:n_ilo]
    mbod_coords = all_feats_2d[n_ilo:]
    ilo_labels_subset = all_labels[:n_ilo]
    mbod_labels_subset = all_labels[n_ilo:]
    
    # Create file names
    file_name_by_class = f"visualizations/tsne_by_class{'_trained' if trained else '_untrained'}.png"
    file_name_by_source = f"visualizations/tsne_by_source{'_trained' if trained else '_untrained'}.png"
    
    # FIRST PLOT: Visualization by class and source
    plt.figure(figsize=(14, 10))
    
    # Plot each class with a different color, and ILO/MBOD with different markers
    for i, label in enumerate(unique_labels):
        # ILO points for this class
        idx_ilo = np.where(ilo_labels_subset == label)[0]
        if len(idx_ilo) > 0:
            plt.scatter(
                ilo_coords[idx_ilo, 0], 
                ilo_coords[idx_ilo, 1],
                c=[label_to_color[label]], 
                marker='*', 
                s=100, 
                label=f'ILO - {int(label)}',
                alpha=0.8,
                edgecolors='white',
                linewidths=0.5
            )
        
        # MBOD points for this class
        idx_mbod = np.where(mbod_labels_subset == label)[0]
        if len(idx_mbod) > 0:
            plt.scatter(
                mbod_coords[idx_mbod, 0], 
                mbod_coords[idx_mbod, 1],
                c=[label_to_color[label]], 
                marker='o', 
                s=40, 
                label=f'MBOD - {int(label)}',
                alpha=0.6,
                edgecolors='white',
                linewidths=0.2
            )
    
    plt.title(f"t-SNE Visualization {'(Trained Model)' if trained else '(Untrained Model)'}\nEpoch {n_epochs}", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create a custom legend with one entry per class and source combination
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(file_name_by_class, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE visualizations to {file_name_by_class}")
    
    # Log to wandb if requested
    if log_to_wandb:
        print("Logging t-SNE visualizations to wandb...")
        wandb.log({
            f"{set_name} tsne": wandb.Image(file_name_by_class)
        })
        print("Logged visualizations to wandb successfully")

def check_empty_study_ids(hdf5_path):
    """
    Check how many samples have an empty study_id column
    
    Args:
        hdf5_path: Path to the HDF5 file
    """
    with h5py.File(hdf5_path, "r") as f:
        total_samples = f["study_id"].shape[0]
        empty_count = 0
        problematic_indices = []
        
        print(f"Checking {total_samples} study IDs for empty values...")
        
        for idx in range(total_samples):
            study_id = f["study_id"][idx]
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            
            # Check if study_id is empty or just whitespace
            if not study_id or study_id.strip() == '':
                empty_count += 1
                problematic_indices.append(idx)
                
        print(f"\nFound {empty_count} empty study IDs out of {total_samples} samples ({empty_count/total_samples:.2%})")
        
        if empty_count > 0 and empty_count <= 10:
            print("\nIndices with empty study IDs:")
            for idx in problematic_indices:
                print(f"  Sample {idx}")
        elif empty_count > 10:
            print("\nFirst 10 indices with empty study IDs:")
            for idx in problematic_indices[:10]:
                print(f"  Sample {idx}")
                
        # Also check for study IDs that don't have enough parts when split by '.'
        problem_format_count = 0
        for idx in range(total_samples):
            study_id = f["study_id"][idx]
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            
            parts = study_id.split('.')
            if len(parts) < 3:
                problem_format_count += 1
                
        print(f"\nFound {problem_format_count} study IDs without at least 3 parts when split by '.' ({problem_format_count/total_samples:.2%})")




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config()
    
    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        

        check_empty_study_ids(hdf5_file_path)


        # Create an HDF5SilicosisDataset instance
        mbod_dataset_merged = HDF5Dataset(
            hdf5_path=hdf5_file_path,
            labels_key="lab",  # Main pathology labels, 'lab' for all labels
            image_key="images",
            label_metadata=LABEL_SCHEMES,
            data_aug=None,
            transform=None
        )

        augmentations_list = [
            # transforms.CenterCrop(np.round(224 * 0.9).astype(int)),  # Example crop
            transforms.RandomRotation(degrees=(-5, 5)),  
            transforms.Lambda(lambda img: helpers.salt_and_pepper_noise_tensor(img, prob=0.02)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.025, 0.025))
        ]


        ilo_imgs = classes.ILOImagesDataset(
            "/home/sean/MSc_2025/data/ilo-radiographs-dicom",
            target_size=512,
            transform=None,
            filter_one_per_label=False,
            augment_label_0=False
        )

        # Get the dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            hdf5_file_path,
            batch_size=16,
            labels_key="profusion_score",
            image_key="images",
            oversample=False
        )

        # wandb.login()
        # wandb.init(project='MBOD-cl', name='img_test')
        
        # Load the saved model checkpoint
        checkpoint_path = "/home/sean/MSc_2025/mbod-data-processor/checkpoints/cl-final_model.pth"
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Initialize model architecture (same as used during training)
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = model.to(device)
        
        # Initialize optimizer (needed for loading state)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Load checkpoint with model state, optimizer state, and epoch
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
                
        raw_model = xrv.models.ResNet(weights="resnet50-res512-all")
        raw_model = raw_model.to(device)
        
        print(f"Successfully loaded model from epoch {epoch}")

        # wandb.init(project="tsne-visualization")
        #

        visualize_tsne(raw_model, device, ilo_imgs, train_loader, trained=False, log_to_wandb=False)
        visualize_tsne(model, device, ilo_imgs, train_loader, trained=True, log_to_wandb=False)

        

    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error loading model or generating visualizations: {e}")
        raise
    