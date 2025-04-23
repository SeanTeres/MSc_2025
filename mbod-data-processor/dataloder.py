import os
import json
import torch
from datasets.hdf_dataset import HDF5Dataset
from torch.utils.data import DataLoader, Subset


def save_split_indices(indices, file_path):
    """Save split indices to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(indices, f)


def load_split_indices(file_path):
    """Load split indices from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_dataloaders(
    hdf5_path,
    preprocess,
    train_split=0.8,
    batch_size=16,
    labels_key="tb_labels",
    split_file=None,
    augmentations=None
):
    # Initialize dataset
    dataset = HDF5Dataset(hdf5_path, preprocess, labels_key=labels_key, augmentations=augmentations)

    # Check if split file exists
    if split_file and os.path.exists(split_file):
        # Load saved split indices
        split_indices = load_split_indices(split_file)
        train_indices = split_indices["train"]
        val_indices = split_indices["val"]
        test_indices = split_indices["test"]
    else:
        # Generate random split
        train_size = int(train_split * len(dataset))
        test_size = len(dataset) - train_size
        val_size = test_size // 2
        test_size = test_size - val_size

        # Get random indices
        indices = torch.randperm(len(dataset)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Save split indices if split_file is provided
        if split_file:
            save_split_indices(
                {"train": train_indices,
                 "val": val_indices,
                 "test": test_indices},
                split_file,
            )

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create dataloaders
    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = None

    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_loader = None

    if len(test_dataset,) > 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader

