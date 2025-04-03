import sys
sys.path.append('/home/sean/MSc/code')
import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torchxrayvision as xrv
from skimage.color import rgb2gray
from skimage.transform import resize
import pydicom
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import random
import utils.helpers as helpers




class DICOMDataset1(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=224):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = None
        
        # Define valid target labels
        self.valid_targets = {
            "Profusion": "Profusion Label",
            "TBA-TBU": "TBA-TBU Label",
            "Profusion or TBA-TBU": "Profusion or TBA-TBU Label",
            "Profusion and TBA-TBU": "Profusion and TBA-TBU Label",
        }
    
    def set_target(self, target_label, target_size):
        if target_label not in self.valid_targets:
            raise ValueError(f"Invalid target_label. Must be one of {list(self.valid_targets.keys())}")
        
        self.target_label = target_label
        self.target_size = target_size
        
        # Pre-compute all labels
        self._assign_labels()
    
    def _assign_labels(self):
        for idx in range(len(self.metadata_df)):
            prof_label = self.metadata_df.iloc[idx]['Profusion']
            tba_1 = self.metadata_df.iloc[idx]['strFindingsSimplified1']
            tba_2 = self.metadata_df.iloc[idx]['strFindingsSimplified2']
            
            tba_1_bool = helpers.contains_tba_or_tbu(tba_1)
            tba_2_bool = helpers.contains_tba_or_tbu(tba_2)
            
            # Profusion Label
            self.metadata_df.loc[idx, 'Profusion Label'] = 1 if prof_label in ['1/0', '1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3'] else 0
            
            # TBA/TBU Label
            self.metadata_df.loc[idx, 'TBA-TBU Label'] = 1 if (tba_1_bool and tba_2_bool) else 0
            
            # Profusion or TBA/TBU Label
            prof_positive = prof_label in ['1/0', '1/1', '1/2', '2/1', '2/2', '2/3', '3/2', '3/3']
            self.metadata_df.loc[idx, 'Profusion or TBA-TBU Label'] = 1 if (prof_positive or (tba_1_bool and tba_2_bool)) else 0
            
            # Profusion and TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion and TBA-TBU Label'] = 1 if (prof_positive and (tba_1_bool and tba_2_bool)) else 0

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['(0020,000d) UI Study Instance UID']
        dicom_file = os.path.join(self.dicom_dir, dicom_filename + '.dcm')

        # Process image
        pixel_tensor, pixel_array = helpers.read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])
        
        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Get target based on target_label
        target = int(self.metadata_df.iloc[idx][self.valid_targets[self.target_label]])

        
        if self.transform:
            pixel_tensor = self.transform(pixel_tensor)

        return pixel_tensor, target

class DICOMDataset2(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=224):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = None
        
        # Define valid target labels
        self.valid_targets = {
            "Profusion": "Profusion Label",
            "TBA-TBU": "TBA-TBU Label",
            "Profusion or TBA-TBU": "Profusion or TBA-TBU Label",
            "Profusion and TBA-TBU": "Profusion and TBA-TBU Label"
        }
    
    def set_target(self, target_label, target_size):
        if target_label not in self.valid_targets:
            raise ValueError(f"Invalid target_label. Must be one of {list(self.valid_targets.keys())}")
        
        self.target_label = target_label
        self.target_size = target_size
        
        # Pre-compute all labels
        self._assign_labels()
    
    def _assign_labels(self):
        # print("ASSIGNING LABELS to D2")
        for idx in range(len(self.metadata_df)):
            profusion = self.metadata_df.iloc[idx]['Radiologist: ILO Profusion']
            profusion = int(profusion[0])
            prof_bool = profusion in [1, 2, 3]
            tba = self.metadata_df.iloc[idx]['Radiologist: TB (TBA or TBU)']
            
            # Profusion Label
            self.metadata_df.loc[idx, 'Profusion Label'] = 1 if prof_bool else 0
            
            # TBA/TBU Label
            self.metadata_df.loc[idx, 'TBA-TBU Label'] = 1 if tba else 0
            
            # Profusion or TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion or TBA-TBU Label'] = 1 if (profusion or tba) else 0
            
            # Profusion and TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion and TBA-TBU Label'] = 1 if (profusion and tba) else 0

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['Anonymized Filename']
        dicom_file = os.path.join(self.dicom_dir, dicom_filename)

        # Process image
        pixel_tensor, pixel_array = helpers.read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])
        
        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Get target based on target_label
        target = int(self.metadata_df.iloc[idx][self.valid_targets[self.target_label]])

        if self.transform:
            pixel_tensor = self.transform(pixel_tensor)

        return pixel_tensor, target
    
class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, augmentations_list):
        """
        base_dataset: Original dataset.
        augmentations_list: List of transformations to apply individually to each sample.
        """
        self.base_dataset = base_dataset
        self.augmentations_list = augmentations_list

    def __len__(self):
        # Each original sample is repeated once for each augmentation plus the original
        return len(self.base_dataset) * (len(self.augmentations_list) + 1)

    def __getitem__(self, idx):
        # Identify the original sample and which augmentation (if any) to apply
        original_idx = idx // (len(self.augmentations_list) + 1)
        augment_idx = idx % (len(self.augmentations_list) + 1)

        pixel_tensor, label = self.base_dataset[original_idx]

        if augment_idx > 0:  # Apply the specific augmentation
            augmentation = self.augmentations_list[augment_idx - 1]
            pixel_tensor = augmentation(pixel_tensor)

        pixel_tensor = (pixel_tensor - pixel_tensor.min()) / (pixel_tensor.max() - pixel_tensor.min())
        # Rescale to [-1024, 1024] if needed for xrv models
        pixel_tensor = pixel_tensor * (1024 - (-1024)) + (-1024)

        return pixel_tensor, label


class BaseClassifier(nn.Module):
    def __init__(self, in_features):
        super(BaseClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 512)  # Input size is 1024
        self.fc2 = nn.Linear(512, 256)            # Additional hidden layer
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)  # Output layer
        return x
    
class BaseClassifierWithDropout(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5):
        super(BaseClassifierWithDropout, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 512)  # Input size is 1024
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)  # Additional hidden layer
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = F.relu(self.fc1(x))  # First hidden layer
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = self.dropout2(x)
        x = self.fc3(x)  # Output layer
        return x

class BaseClassifier512(nn.Module):
    def __init__(self, in_features):
        super(BaseClassifier512, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 1024)  # Input size is 1024
        self.fc2 = nn.Linear(1024, 512)            # Additional hidden layer
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
    

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer
        return x
    


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

class DICOMDataset1_CL(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=512):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = None

        # Define valid target labels
        self.valid_targets = {
            "Profusion": "Profusion Label",
            "TBA-TBU": "TBA-TBU Label",
            "Profusion or TBA-TBU": "Profusion or TBA-TBU Label",
            "Profusion and TBA-TBU": "Profusion and TBA-TBU Label",
        }

    def set_target(self, target_label, target_size):
        if target_label not in self.valid_targets:
            raise ValueError(f"Invalid target_label. Must be one of {list(self.valid_targets.keys())}")

        self.target_label = target_label
        self.target_size = target_size

        # Pre-compute all labels
        self._assign_labels()

    def _assign_labels(self):
        for idx in range(len(self.metadata_df)):
            prof_label = str(self.metadata_df.iloc[idx]['Profusion']).strip()  # Ensure it's a string and strip extra spaces
            tba_1 = self.metadata_df.iloc[idx]['strFindingsSimplified1']
            tba_2 = self.metadata_df.iloc[idx]['strFindingsSimplified2']

            tba_1_bool = helpers.contains_tba_or_tbu(tba_1)
            tba_2_bool = helpers.contains_tba_or_tbu(tba_2)

            # Extract first digit from prof_label if valid, else assign 0
            if prof_label == "nan" or prof_label == "" or '/' not in prof_label:
                prof_first_digit = 0  # Treat as 0 when invalid or missing
            else:
                try:
                    # Split by '/' and get the first part, then convert to integer
                    prof_first_digit = int(prof_label.split('/')[0]) if prof_label else 0
                except ValueError:
                    prof_first_digit = 0  # In case of malformed values (e.g., '3/3x')

            # Profusion Label: 1 if the first digit of the profusion score is within a certain set
            self.metadata_df.loc[idx, 'Profusion Label'] = int(np.round(prof_first_digit, 0))

            # TBA/TBU Label
            self.metadata_df.loc[idx, 'TBA-TBU Label'] = 1 if (tba_1_bool and tba_2_bool) else 0

            # Profusion or TBA/TBU Label
            prof_positive = prof_first_digit in [1, 2, 3]
            self.metadata_df.loc[idx, 'Profusion or TBA-TBU Label'] = 1 if (prof_positive or (tba_1_bool and tba_2_bool)) else 0

            # Profusion and TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion and TBA-TBU Label'] = 1 if (prof_positive and (tba_1_bool and tba_2_bool)) else 0
        self.metadata_df["Profusion Label"] = pd.to_numeric(self.metadata_df["Profusion Label"], downcast="integer")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['(0020,000d) UI Study Instance UID']
        dicom_file = os.path.join(self.dicom_dir, dicom_filename + '.dcm')

        # Process image
        pixel_tensor, pixel_array = helpers.read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])

        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Get target based on target_label
        target = int(self.metadata_df.iloc[idx][self.valid_targets[self.target_label]])

        if self.transform:
            pixel_tensor = self.transform(pixel_tensor)

        return pixel_tensor, target
    

class DICOMDataset2_CL(Dataset):
    def __init__(self, dicom_dir, metadata_df, transform=None, target_size=512):
        self.dicom_dir = dicom_dir
        self.metadata_df = metadata_df
        self.transform = transform
        self.target_size = target_size
        self.target_label = None

        # Define valid target labels
        self.valid_targets = {
            "Profusion": "Profusion Label",
            "TBA-TBU": "TBA-TBU Label",
            "Profusion or TBA-TBU": "Profusion or TBA-TBU Label",
            "Profusion and TBA-TBU": "Profusion and TBA-TBU Label"
        }

    def set_target(self, target_label, target_size):
        if target_label not in self.valid_targets:
            raise ValueError(f"Invalid target_label. Must be one of {list(self.valid_targets.keys())}")

        self.target_label = target_label
        self.target_size = target_size

        # Pre-compute all labels
        self._assign_labels()

    def _assign_labels(self):
        for idx in range(len(self.metadata_df)):
            profusion = self.metadata_df.iloc[idx]['Radiologist: ILO Profusion']
            tba = self.metadata_df.iloc[idx]['Radiologist: TB (TBA or TBU)']

            # Profusion Label
            self.metadata_df.loc[idx, 'Profusion Label'] = profusion[0]

            # TBA/TBU Label
            self.metadata_df.loc[idx, 'TBA-TBU Label'] = 1 if tba else 0

            # Profusion or TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion or TBA-TBU Label'] = 1 if ((profusion != 0) or tba) else 0

            # Profusion and TBA/TBU Label
            self.metadata_df.loc[idx, 'Profusion and TBA-TBU Label'] = 1 if ((profusion != 0) and tba) else 0

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        dicom_filename = self.metadata_df.iloc[idx]['Anonymized Filename']
        dicom_file = os.path.join(self.dicom_dir, dicom_filename)

        # Process image
        pixel_tensor, pixel_array = helpers.read_and_normalize_xray(dicom_file, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True)
        resize_transform = transforms.Compose([xrv.datasets.XRayResizer(self.target_size)])

        pixel_tensor = resize_transform(pixel_tensor.numpy())
        pixel_tensor = pixel_tensor.squeeze(0)
        pixel_tensor = transforms.ToTensor()(pixel_tensor)

        # Get target based on target_label
        target = int(self.metadata_df.iloc[idx][self.valid_targets[self.target_label]])

        if self.transform:
            pixel_tensor = self.transform(pixel_tensor)

        return pixel_tensor, target


class ILOImagesDataset(Dataset):
    def __init__(self, ilo_images_dir, target_size=512, transform=None, filter_one_per_label=False):
        self.ilo_images_dir = ilo_images_dir
        self.target_size = target_size
        self.transform = transform
        self.filter_one_per_label = filter_one_per_label  # New parameter to control filtering

        # List all files in the directory
        self.files = [f for f in os.listdir(self.ilo_images_dir) if f.endswith('.dcm')]

        # Filter out files where the first character is not a number
        self.valid_files = [
            f for f in self.files if f[0].isdigit() and (f[0] == f[1])
        ]

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

class TripletDataset(Dataset):
    def __init__(self, target_dataset, anchor_dataset, max_triplets=None):
        """
        Args:
            target_dataset: A PyTorch Dataset (single dataset or ConcatDataset)
            anchor_dataset: A PyTorch Dataset (e.g., ILOImagesDataset) for anchors
            max_triplets (int, optional): Maximum number of triplets to generate
        """
        self.target_dataset = target_dataset
        self.anchor_dataset = anchor_dataset
        self.max_triplets = max_triplets
        self.is_concat_dataset = hasattr(target_dataset, 'datasets')
        
        # Extract labels from metadata_df of the target dataset
        self.class_to_indices = self._build_class_indices(target_dataset)

    def _build_class_indices(self, dataset):
        """Builds a dictionary mapping each class label to a list of indices."""
        class_to_indices = {}
        
        if self.is_concat_dataset:  # For ConcatDataset
            for i, subdataset in enumerate(dataset.datasets):
                metadata_df = subdataset.metadata_df
                labels = metadata_df['Profusion Label'].values
                
                for idx, label in enumerate(labels):
                    label = int(label)  # Ensure consistent integer labeling
                    if label not in class_to_indices:
                        class_to_indices[label] = []
                    class_to_indices[label].append((i, idx))  # (subdataset_idx, image_idx)
        else:  # For single dataset
            metadata_df = dataset.metadata_df
            labels = metadata_df['Profusion Label'].values
            
            for idx, label in enumerate(labels):
                label = int(label)
                if label not in class_to_indices:
                    class_to_indices[label] = []
                class_to_indices[label].append(idx)  # Direct index
        
        return class_to_indices

    def __len__(self):
        """Return the number of triplets."""
        if self.max_triplets:
            return self.max_triplets
        return len(self.anchor_dataset) * len(self.target_dataset)

    def __getitem__(self, index):
        """Returns (anchor, positive, negative) triplet along with labels."""
        if self.max_triplets and index >= self.max_triplets:
            raise IndexError("Exceeded maximum number of triplets")

        # Find the index in the anchor and target datasets
        anchor_index = random.randint(0, len(self.anchor_dataset) - 1)
        target_index = index % len(self.target_dataset)
        
        # Get the anchor sample
        anchor_img, anchor_label, anchor_filename = self.anchor_dataset[anchor_index]

        # Get the positive sample (same class as the anchor)
        if self.is_concat_dataset:
            positive_img, positive_label = self.target_dataset[target_index]
        else:
            positive_img, positive_label = self.target_dataset[target_index]
        
        # Ensure the positive sample has the same label as the anchor
        max_attempts = 50  # Prevent infinite loops
        attempts = 0
        
        while positive_label != anchor_label and attempts < max_attempts:
            target_index = random.randint(0, len(self.target_dataset) - 1)
            if self.is_concat_dataset:
                positive_img, positive_label = self.target_dataset[target_index]
            else:
                positive_img, positive_label = self.target_dataset[target_index]
            attempts += 1
            
        if attempts >= max_attempts:
            print(f"Warning: Could not find matching positive for label {anchor_label} after {max_attempts} attempts")
        
        # Negative sample: Pick a sample from a different class
        negative_label = random.choice([lbl for lbl in self.class_to_indices.keys() if lbl != anchor_label])
        
        if self.is_concat_dataset:
            negative_idx = random.choice(self.class_to_indices[negative_label])
            subdataset_idx, image_idx = negative_idx
            negative_img, _ = self.target_dataset.datasets[subdataset_idx][image_idx]
        else:
            negative_idx = random.choice(self.class_to_indices[negative_label])
            negative_img, _ = self.target_dataset[negative_idx]

        return anchor_img, positive_img, negative_img, anchor_label, positive_label, negative_label, anchor_filename
    
