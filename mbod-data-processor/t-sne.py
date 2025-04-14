from datasets.hdf_dataset import HDF5SilicosisDataset
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from dataloader import get_dataloaders, get_stratified_dataloaders
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
        mbod_dataset_merged = HDF5SilicosisDataset(
            hdf5_file_path=hdf5_file_path,
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
            "C:/Users/user-pc/Masters/MSc_2025/data/ilo-radiographs-dicom",
            target_size=512,
            transform=None,
            filter_one_per_label=False,
            augment_label_0=False
        )

        # Get the dataloaders
        train_loader, val_loader, test_loader = get_stratified_dataloaders(
            hdf5_file_path,
            batch_size=16,
            labels_key="profusion_score",
            image_key="images",
            oversample=False
        )

        

    except KeyError as e:
        print(f"Missing configuration: {e}")