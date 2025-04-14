from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from datasets.hdf_dataset import HDF5SilicosisDataset
from utils import LABEL_SCHEMES
from data_splits import stratify
import numpy as np
import torch


def get_dataloaders(hdf5_path, train_split=0.6, batch_size=16, labels_key="tb_labels", image_key="images"):
    dataset = HDF5SilicosisDataset(hdf5_path, labels_key=labels_key, image_key=image_key, label_metadata=LABEL_SCHEMES)

    train_size = int(train_split * len(dataset))
    other_size = len(dataset) - train_size
    val_size = other_size//2
    test_size = other_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_stratified_dataloaders(hdf5_path, batch_size=16, labels_key="profusion_score", image_key="images", oversample=False):
    """
    Creates dataloaders using stratified splits that maintain class distribution.
    
    Args:
        hdf5_path: Path to the HDF5 dataset file
        batch_size: Batch size for the data loaders
        labels_key: Key for the labels in the HDF5 file
        image_key: Key for the images in the HDF5 file
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create the full dataset
    dataset = HDF5SilicosisDataset(
        hdf5_file_path=hdf5_path,
        labels_key=labels_key,
        image_key=image_key,
        label_metadata=LABEL_SCHEMES
    )
    
    # Generate stratified splits
    train_indices, val_indices, test_indices = stratify(dataset)
    
    print(f"Split sizes: Train={len(train_indices)}, Validation={len(val_indices)}, Test={len(test_indices)}")
    
    # Create dataset subsets using torch's built-in Subset class
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders with appropriate sampling strategy
    if oversample and labels_key == "profusion_score":  # Only apply oversampling for multi-class
        # Get all labels to compute class weights
        all_labels = [dataset[i]["lab"] for i in train_indices]
        
        # Convert to tensor/numpy array if needed
        if isinstance(all_labels[0], torch.Tensor):
            all_labels = torch.stack(all_labels).numpy()
        else:
            all_labels = np.array(all_labels)
        
        # Convert labels to integers before using bincount
        all_labels_int = all_labels.astype(np.int64)
        print(f"Label data type before conversion: {all_labels.dtype}")
        print(f"Label data type after conversion: {all_labels_int.dtype}")
        
        # Count samples per class
        class_counts = np.bincount(all_labels_int)
        print(f"Class distribution in training set: {class_counts}")
        
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.max()
        
        # FIX: Create a mapping from original index to relative position in the train set
        train_indices_set = set(train_indices)
        train_indices_map = {idx: pos for pos, idx in enumerate(train_indices)}
        
        # Assign weight to each sample based on its class using the index mapping
        sample_weights = []
        for i in train_indices:
            label_idx = int(all_labels[train_indices_map[i]])
            sample_weights.append(class_weights[label_idx])
        
        # Create sampler with computed weights
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True
        )
        
        print(f"Using weighted sampling with class weights: {class_weights}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        print("NOT OVERSAMPLING")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader