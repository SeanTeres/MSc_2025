import h5py
import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import yaml
import glob
import torchvision.transforms as transforms

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, preprocess, labels_key="tb_labels", images_key="images", augmentations=None):
        self.hdf5_path = hdf5_path
        self.preprocess = preprocess
        self.augmentations = augmentations
        self.labels_key = labels_key
        self.images_key = images_key

        # Open HDF5 file to get dataset sizes
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            self.data_size = len(hdf5_file[labels_key])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            image = hdf5_file[self.images_key][idx]
            label = hdf5_file[self.labels_key][idx]

        if image.dtype == np.float16:
            image = image.astype(np.float32)

        image = Image.fromarray(image)

        if self.augmentations:
            image = self.augmentations(image)

        if self.preprocess:
            image = self.preprocess(image)

        # Half precision
        # image = image.half()

        return image, torch.tensor(label, dtype=torch.long)


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
    augmentations=None,
    oversample=True, 
    clf_task_labels_key="tuberculosis",
):
    # Initialize dataset
    dataset = HDF5Dataset(hdf5_path, preprocess, labels_key=labels_key, augmentations=augmentations)

    # Check if split file exists
    if split_file and os.path.exists(split_file):
        print("Using split file...")
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

    if oversample and len(train_dataset) > 0:
        # Precompute labels for all samples in the dataset
        # With this code that handles both cases:
        all_labels = []
        for idx in range(len(dataset)):
            label = dataset[idx][1]
            if label.dim() > 0 and label.size(0) > 1:
                # For multilabel, use the first element (or choose based on clf_task_labels_key)
                # This should match your selection in train/evaluate_model
                if clf_task_labels_key == "silicosis":
                    all_labels.append(label[0].item())  # Use silicosis element
                elif clf_task_labels_key == "tuberculosis":
                    all_labels.append(label[1].item())  # Use TB element
                else:
                    all_labels.append(label[0].item())  # Default to first element
            else:
                # For single-label tensors
                all_labels.append(label.item())
        # Get labels for the training set
        train_labels = [all_labels[idx] for idx in train_indices]

        # Count label frequencies
        label_counts = Counter(train_labels)

        # Ensure all possible labels are included in label_counts
        all_possible_labels = set(all_labels)
        for label in all_possible_labels:
            if label not in label_counts:
                label_counts[label] = 0.0  # Assign a count of 0 for missing classes

        # Compute weights for each sample
        weights = [1.0 / label_counts[label] for label in train_labels]

        # Apply a power transformation to reduce the effect of weights
        scaling_factor = 0.5  # Adjust this value to control the effect
        weights = [w ** scaling_factor for w in weights]

        # Create a WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        # Use the sampler in the DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if len(train_dataset) > 0 else None

    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    if len(test_dataset,) > 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
        

if __name__ == "__main__":
    with open("bin_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_path_mbod = config["DATA_PATH_MBOD"]

    preprocess = transforms.Compose([
    # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    augmentations = transforms.Compose([
        transforms.RandomRotation(degrees=10, expand=False, fill=0),
        # T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # T.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
    ])

    x1,_,_ = get_dataloaders(
        config["DATA_PATH_MBOD"],
        preprocess=preprocess,
        oversample=False,
        labels_key="silicosis",
        augmentations=augmentations,
        train_split=0.7,
        split_file="data_splits/new_split.json"
    )

    x2,_,_ = get_dataloaders(
        config["DATA_PATH_MBOD"],
        preprocess=preprocess,
        oversample=False,
        labels_key="silicosis",
        augmentations=augmentations,
        split_file=config["SPLIT_FILE_MBOD"],
    )

    config_files = glob.glob("configs/*.yaml")

    for cfg_path in config_files:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            print(cfg)
