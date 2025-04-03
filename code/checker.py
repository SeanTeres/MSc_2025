import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, Subset, ConcatDataset
import torchvision.transforms as transforms
import os
import torchxrayvision as xrv
from skimage.color import rgb2gray
from skimage.transform import resize
import pydicom
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
import wandb
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import sys
sys.path.append('/home/sean/MSc/code')
import utils.helpers as helpers
import utils.classes as classes
import utils.train_utils as train_utils
import torch.nn as nn
import torch.optim as optim


wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
# Initialize wandb before using it in other parts of the code
wandb.init(project='MBOD-New', name='TESTTT', config={
    "batch_size": 8
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dicom_dir_1 = '/home/sean/MSc/data/MBOD_Datasets/Dataset-1'
dicom_dir_2 = '/home/sean/MSc/data/MBOD_Datasets/Dataset-2'
metadata_1 = pd.read_excel("/home/sean/MSc/data/MBOD_Datasets/Dataset-1/FileDatabaseWithRadiology.xlsx")

metadata_2 = pd.read_excel("/home/sean/MSc/data/MBOD_Datasets/Dataset-2/Database_Training-2024.08.28.xlsx")

ILO_imgs = '/home/sean/MSc/data/ilo-radiographs-dicom'

d1_cl = classes.DICOMDataset1_CL(dicom_dir=dicom_dir_1, metadata_df=metadata_1)
d1_cl._assign_labels()
d1_cl.set_target("Profusion", 512)  # Add this line, choose appropriate target and size

d2_cl = classes.DICOMDataset2_CL(dicom_dir=dicom_dir_2, metadata_df=metadata_2)
d2_cl._assign_labels()
d2_cl.set_target("Profusion", 512)  # Add this line, choose appropriate target and size

combined_dataset = ConcatDataset([d1_cl, d2_cl])
# Convert 'Profusion Label' to integer for both datasets
d1_cl.metadata_df['Profusion Label'] = d1_cl.metadata_df['Profusion Label'].astype(int)
d2_cl.metadata_df['Profusion Label'] = d2_cl.metadata_df['Profusion Label'].astype(int)

# Now concatenate the metadata frames
combined_metadata = pd.concat([d1_cl.metadata_df, d2_cl.metadata_df], ignore_index=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

combined_dataset = ConcatDataset([d1_cl, d2_cl])
# Convert 'Profusion Label' to integer for both datasets
d1_cl.metadata_df['Profusion Label'] = d1_cl.metadata_df['Profusion Label'].astype(int)
d2_cl.metadata_df['Profusion Label'] = d2_cl.metadata_df['Profusion Label'].astype(int)

# Now concatenate the metadata frames
combined_metadata = pd.concat([d1_cl.metadata_df, d2_cl.metadata_df], ignore_index=True)

ilo_dataset = classes.ILOImagesDataset(ilo_images_dir=ILO_imgs, filter_one_per_label=False)

triplet_dataset = classes.TripletDataset(d2_cl, ilo_dataset, max_triplets=80)

model_1 = xrv.models.ResNet(weights="resnet50-res512-all")
raw_model = xrv.models.ResNet(weights="resnet50-res512-all")

raw_model = raw_model.to(device)
model_1 = model_1.to(device)

checkpoint_path = '/home/sean/MSc/code/contrastive/checkpoints/checkpoint-new_epoch_10.pth'
# Initialize checkpoint model
checkpoint_model = xrv.models.ResNet(weights="resnet50-res512-all")
checkpoint_model.to(device)

try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if 'model_state_dict' is present, else assume direct state_dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Handle potential "module." prefix issue (if checkpoint was saved with DataParallel)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove 'module.' if necessary
        new_state_dict[new_key] = value

    checkpoint_model.load_state_dict(new_state_dict, strict=False)

    print("Checkpoint successfully loaded.")

except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    raise

checkpoint_model = checkpoint_model.to(device)

# Optionally print metadata from the checkpoint
if "epoch" in checkpoint:
    print(f"Loaded model from epoch: {checkpoint['epoch']}")
if "loss" in checkpoint:
    print(f"Checkpoint loss: {checkpoint['loss']}")
    print(f"Loaded model from epoch: {checkpoint['epoch']}")


train_indices, val_indices, test_indices = helpers.split_dataset(combined_dataset)
train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)
test_dataset = torch.utils.data.Subset(combined_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

train_indices_d2, val_indices_d2, test_indices_d2 = helpers.split_dataset(d2_cl)
train_dataset_d2 = torch.utils.data.Subset(d2_cl, train_indices_d2)
val_dataset_d2 = torch.utils.data.Subset(d2_cl, val_indices_d2)
test_dataset_d2 = torch.utils.data.Subset(d2_cl, test_indices_d2)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

train_loader_d2 = DataLoader(train_dataset_d2, batch_size=8, shuffle=True)
val_loader_d2 = DataLoader(val_dataset_d2, batch_size=8, shuffle=False)
test_loader_d2 = DataLoader(test_dataset_d2, batch_size=8, shuffle=False)


anchors, anc_labels, targets, target_labels = train_utils.extract_features_2(triplet_dataset, raw_model, device, anchors_only=False)

train_utils.plot_tsne_2(anchors, anc_labels, targets, target_labels, 4, "MSc/code/2D_tsne.png", False, None, False, "Pre-training t-SNE")

train_utils.plot_tsne_3d_interactive(anchors, anc_labels, targets, target_labels, 4, "MSc/code/3D_tsne_INT.html", False, None, False, "Pre-training t-SNE 3D")

anchors_trained, anc_labels_trained, targets_trained, target_labels_trained = train_utils.extract_features_2(triplet_dataset, checkpoint_model, device, anchors_only=False)

train_utils.plot_tsne_2(anchors_trained, anc_labels_trained, targets_trained, target_labels_trained, 4, "MSc/code/trained-2D_tsne.png", False, None, False, "Trained t-SNE")

train_utils.plot_tsne_3d_interactive(anchors_trained, anc_labels_trained, targets_trained, target_labels_trained, 4, "MSc/code/trained-3D_tsne_INT.html", False, None, False, "Trained t-SNE 3D")



