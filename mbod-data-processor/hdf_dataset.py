from .dataset import Dataset
import torch
import numpy as np
import h5py
from .utils import ILO_CLASSIFICATION_DICTIONARY


def normalize_for_xrv(img_tensor):
    """
    Normalize image tensor to the range expected by TorchXRayVision models.
    """
    # First, scale to 0-1 range
    img_min = img_tensor.min()
    img_max = img_tensor.max()
    img_normalized = (img_tensor - img_min) / (img_max - img_min)
    
    # Then optionally scale to [-1024, 1024] range expected by some XRV models
    img_normalized = img_normalized * 2048 - 1024
    
    return img_normalized


class HDF5SilicosisDataset(Dataset):
    def __init__(
        self,
        hdf5_file_path,
        labels_key,
        image_key,
        label_metadata,
        transform=None,
        data_aug=None,
    ):
        super(HDF5SilicosisDataset, self).__init__()

        self.hdf5_file = h5py.File(hdf5_file_path, "r")
        self.images = self.hdf5_file[image_key]  # Assuming the dataset has 'images'
        self.labels = self.hdf5_file[labels_key]  # Assuming 'labels'
        self.image_ids = [x for x in self.hdf5_file["study_id"]]

        self.ilo_classification_dictionary = ILO_CLASSIFICATION_DICTIONARY

        self.pathologies = sorted(self.ilo_classification_dictionary.keys())
        self.label_metadata = label_metadata or {}
        self.labels_key = labels_key

        self.transform = transform
        self.data_aug = data_aug

    def __len__(self):
        return len(self.labels)

    def get_label_names(self):
        return self.label_metadata.get(
            self.labels_key,
            [f"Class {i}" for i in range(self.hdf5_file[self.labels_key].shape[-1])],
        )

    def __getitem__(self, idx):
        image = self.images[idx]
        label_vector = self.labels[idx]
        image_id = self.image_ids[idx]

        # Convert image to float32 and normalize
        image = torch.tensor(image, dtype=torch.float32) / 255.
        
        # For SEAN for xrv
        image = normalize_for_xrv(image)
        

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.data_aug:
            image = self.data_aug(image)

        sample = {"img": image,
                  "lab": torch.tensor(label_vector, dtype=torch.float32),
                  "image_id": image_id}
        return sample

    def __str__(self):
        """
        Generate a summary of the dataset, including the number of samples,
        data augmentation status, and label distribution for the current labels_key.
        """
        num_samples = len(self)
        label_names = self.get_label_names()  # Dynamically fetch label names
        summary = "HDF5 Dataset Summary:\n"
        summary += f"Number of samples: {num_samples}\n"
        summary += f"Labels key: {self.labels_key}\n"
        summary += (
            f"Data augmentation: {'Enabled' if self.data_aug else 'Disabled'}\n\n"
        )
        summary += "Label distribution:\n"

        # Handle label distribution based on label shape (1D or multi-column)
        if self.labels.ndim == 1 or self.labels.shape[1] == 1:
            # Single-class labels
            label_indices = np.array(self.labels).flatten()
            label_counts = {
                label_names[i]: np.sum(label_indices == i)
                for i in range(len(label_names))
            }
        else:
            # Multi-class or multi-label representation
            labels_array = np.array(self.labels)
            label_counts = {
                label_names[i]: int(np.sum(labels_array[:, i]))
                for i in range(len(label_names))
            }

        # Calculate ratios and build the output summary
        for label, count in label_counts.items():
            ratio = count / num_samples
            summary += f"- {label}: {count} samples ({ratio:.2%})\n"

        return summary
