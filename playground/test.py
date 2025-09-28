import sys
import os

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

import torch.utils
import torch.utils.data
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datasets.dataloader import get_dataloaders, get_dataloaders_with_files
import torchxrayvision as xrv
import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import io
import torchvision.transforms as transforms
import os
from sklearn.manifold import TSNE


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

# Extract "best" or "final" from checkpoint path
def extract_model_type(checkpoint_path):
    # Using string split approach
    filename = os.path.basename(checkpoint_path)  # Gets "best_model.pth" or "final_model.pth"
    model_type = filename.split('_')[0]  # Gets "best" or "final"
    return model_type

def list_h5_keys(file_path):
    """
    List all top-level keys (datasets) in an HDF5 file, with details about potential label fields
    
    Args:
        file_path: Path to the HDF5 file
    """
    with h5py.File(file_path, "r") as f:
        print(f"\nKeys in {os.path.basename(file_path)}:")
        for key in f.keys():
            # Get information about the dataset
            if isinstance(f[key], h5py.Dataset):
                shape = f[key].shape
                dtype = f[key].dtype
                
                # Identify potential label fields (1D arrays with limited unique values)
                if len(shape) == 1:
                    # For small datasets, examine all values
                    if shape[0] < 1000:
                        values = f[key][:]
                    # For large datasets, sample first 100 values
                    else:
                        values = f[key][:100]
                        
                    # Count unique values
                    if dtype.kind in 'iuf':  # integer, unsigned int, or float
                        unique_vals = np.unique(values)
                        n_unique = len(unique_vals)
                        
                        # If there are few unique values, it's likely a label field
                        if n_unique <= 10:
                            print(f"  - {key}: Shape {shape}, Type {dtype}")
                            print(f"    Unique values: {unique_vals}")
                            print(f"    Likely a label field")
                        else:
                            print(f"  - {key}: Shape {shape}, Type {dtype}, {n_unique} unique values")
                    else:
                        print(f"  - {key}: Shape {shape}, Type {dtype}")
                else:
                    print(f"  - {key}: Shape {shape}, Type {dtype}")
            # Handle groups (folders in the HDF5 hierarchy)
            elif isinstance(f[key], h5py.Group):
                print(f"  - {key}: Group containing {len(list(f[key].keys()))} items")

def view_image_label(h5_path, image_index):
    """
    View the label vector for a specific image in an HDF5 file.
    
    Args:
        h5_path: Path to the HDF5 file
        image_index: Index of the image to view the label for
    """
    with h5py.File(h5_path, "r") as f:
        if image_index >= f["images"].shape[0]:
            print(f"Error: Index {image_index} out of bounds. File contains {f['images'].shape[0]} images.")
            return
            
        # Get the label vector
        label = f["multilabel_stb"][image_index]
        print(f"Label vector for image {image_index}: {label}")
        
        # Optionally display the image
        image = f["images"][image_index]
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title(f"Image {image_index} with label: {label}")
        plt.axis('off')
        plt.savefig("label_vector_image.png")

def plot_normal_cases(dataset, num_samples=4, save_path="normal_cases.png"):
    """
    Find and plot samples with no silicosis and no TB (label=0).
    
    Args:
        dataset: HDF5Dataset2 instance with multiclass_stb labels
        num_samples: Number of normal cases to display
        save_path: Path to save the output figure
    """
    normal_indices = []
    
    # Search through dataset to find normal cases (label = 0)
    for idx in range(len(dataset)):
        try:
            _, label, _ = dataset[idx]
            
            # Convert to scalar if tensor
            if isinstance(label, torch.Tensor):
                if label.dim() == 0 or (label.dim() == 1 and label.size(0) == 1):
                    # Single value tensor
                    label_val = label.item()
                    if label_val == 0:  # No silicosis (0), No TB (< 4)
                        normal_indices.append(idx)
                else:
                    # Skip multilabel tensors in this version
                    continue
            else:
                # Handle scalar case
                if label == 0:
                    normal_indices.append(idx)
            
            if len(normal_indices) >= num_samples:
                break
                
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
    
    # Report how many normal cases were found
    print(f"Found {len(normal_indices)} cases with label=0 (no silicosis, no TB)")
    
    if not normal_indices:
        print("No normal cases found.")
        return
    
    # Create a grid of images
    fig, axes = plt.subplots(1, len(normal_indices), figsize=(4*len(normal_indices), 4))
    if len(normal_indices) == 1:
        axes = [axes]  # Make iterable if only one plot
    
    # Plot each normal case
    for i, idx in enumerate(normal_indices):
        image, label, filename = dataset[idx]
        # Convert from tensor if needed
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        
        # Display image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Normal Case\nID: {idx}\nLabel: {label.item() if isinstance(label, torch.Tensor) else label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved normal cases to {save_path}")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")
    
    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        

        check_empty_study_ids(hdf5_file_path)

        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]


        def normalize_to_hu_range(img_tensor):
            """Normalize image tensor to Hounsfield Unit range (-1024, 1024)"""
            min_val = img_tensor.min()
            max_val = img_tensor.max()
            
            # Scale to [0,1]
            normalized = (img_tensor - min_val) / (max_val - min_val)
            
            # Scale to [-1024, 1024]
            return normalized * 2048 - 1024
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(normalize_to_hu_range)  # Scale [0,1] to [-1024,1024]
        ])

        # Create an HDF5SilicosisDataset instance
        mbod_dataset_merged = HDF5Dataset2(
            hdf5_path=hdf5_file_path,
            labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )
        rand_dataset_v1 = HDF5Dataset(
            hdf5_path = config["rand_output"]["hdf5_file"],
            labels_key="multilabel_stb",
            images_key="images",
            augmentations=None,
            preprocess=preprocess,
        )

        mbod_857 = HDF5Dataset2(
            hdf5_path = config["mbod_857_silicosis_output"]["hdf5_file"],
            labels_key="profusion_score",
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )

        # Path to Kaggle TB dataset
        kaggle_tb_path = config["kaggle_TB"]["outputpath"]  # Ensure this is set in config.yaml
        mc_sz_dataset_path = config["MC_SZ_TB"]["outputpath"]  # Path to the MC-SZ dataset

        # Create an instance of KaggleTBDataset
        kaggle_tb_dataset = HDF5Dataset(
            hdf5_path = kaggle_tb_path,
            labels_key="tuberculosis",
            preprocess = preprocess,
            augmentations=None,
        )

        # Create an instance of MC_SZ_TBDataset
        mc_sz_dataset = HDF5Dataset2(
            hdf5_path = mc_sz_dataset_path,
            labels_key="tuberculosis",  # Assuming this is the correct key for MC-SZ labels
            preprocess = preprocess,
            augmentations=None,
        )


        ilo_dataset = HDF5Dataset2(
            hdf5_path=ilo_hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )

        print(rand_dataset_v1[120][1])
        print(mbod_dataset_merged[120][1])
        plot_normal_cases(mbod_dataset_merged, num_samples=3)



    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error loading model or generating visualizations: {e}")
        raise
