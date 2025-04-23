import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import pydicom
import numpy as np
import torchvision.transforms as transforms
from skimage.color import rgb2gray
from skimage.transform import resize
from torchxrayvision.datasets import XRayCenterCrop
import torchxrayvision as xrv
from torch.utils.data import Dataset
import os
from train_utils import helpers


class MultiClassBaseClassifier(nn.Module):
    def __init__(self, in_features, num_classes=4):
        """
        Multi-class classifier for pneumoconiosis profusion scoring.

        Args:
            in_features (int): Number of input features from the backbone model
            num_classes (int): Number of classes for classification (default: 4 for profusion levels 0-3)
        """
        super(MultiClassBaseClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)  # Output layer

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function for logits
        return x
    

class ILOImagesDataset(Dataset):
    def __init__(self, ilo_images_dir, target_size=512, transform=None,
                 filter_one_per_label=False, augment_label_0=False, augmentation_transforms=None):
        self.ilo_images_dir = ilo_images_dir
        self.target_size = target_size
        self.transform = transform
        self.filter_one_per_label = filter_one_per_label
        self.augment_label_0 = augment_label_0
        self.augmentation_transforms = augmentation_transforms or []
        
        self.files = [f for f in os.listdir(self.ilo_images_dir) if f.endswith('.dcm')]
        self.valid_files = [f for f in self.files if f[0].isdigit() and (f[0] == f[1])]
        self.label_to_files = self._build_label_to_files()

        if self.filter_one_per_label:
            self.filtered_files = self._filter_one_per_label()
        else:
            self.filtered_files = self.valid_files

        self.extra_augmented_samples = []

        if self.augment_label_0 and 0 in self.label_to_files and len(self.label_to_files[0]) >= 2:
            chosen_files = random.sample(self.label_to_files[0], 2)
            for i in range(3):  # We want 3 augmented samples
                base_file = chosen_files[i % 2]
                self.extra_augmented_samples.append((base_file, i))

        # Organize the filenames by label (the first character of the filename)
        self.label_to_files = self._build_label_to_files()

        if self.filter_one_per_label:
            # If filter_one_per_label is True, keep only one file per label (4 in total)
            self.filtered_files = self._filter_one_per_label()
        else:
            # Otherwise, use the valid files as is
            self.filtered_files = self.valid_files

    def _build_label_to_files(self):
        """Organize the filenames by label, using the first character of the filename as the label."""
        label_to_files = {}

        for file in self.valid_files:
            label = int(file[0])  # First character as label (assuming it's a number)
            if label not in label_to_files:
                label_to_files[label] = []
            label_to_files[label].append(file)

        return label_to_files

    def _filter_one_per_label(self):
        """Select only one anchor per label."""
        filtered_files = []
        for label, files in self.label_to_files.items():
            # Randomly select one file for each label (you can change this logic if needed)
            filtered_files.append(random.choice(files))
        return filtered_files

    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, idx):
        # Get filename from the filtered list (if filtering is enabled)
        filename = self.filtered_files[idx]
        filepath = os.path.join(self.ilo_images_dir, filename)

        # Read the image and preprocess
        pixel_tensor, _ = helpers.read_and_normalize_xray(filepath, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])

        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Label is the first character of the filename (converted to int)
        label = int(filename[0])

        if self.transform:
            pixel_tensor = self.transform(pixel_tensor)

        return pixel_tensor, label, filename