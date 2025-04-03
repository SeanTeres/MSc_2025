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
# Initialize wandb before using it in other parts of the code
wandb.init(project='MBOD-New', name='multi-all-metrics', config={
    "batch_size": 8,
    "augmentation": False,
    "lr": 0.001,
    "model": "resnet50-res512-all",
    "epochs": 20,
    "oversampling": True,
    "train_dataset": "MBOD 2"
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

model_1 = xrv.models.ResNet(weights="resnet50-res512-all")
model_2 = xrv.models.ResNet(weights="resnet50-res512-all")
model_1 = model_1.to(device)
model_2 = model_2.to(device)


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


    
# model_1, model_2 = train_both_models(train_loader=train_loader_d2, val_loader=val_loader_d2, model_1=model_1, model_2=model_2, n_epochs=2, lr=0.001, device=device, pos_weight_1=None, pos_weight_2=None, experiment_name="Test")

train_labels = []
for idx in train_indices_d2:
    _, label = train_dataset_d2.dataset[idx]  # Access the underlying dataset through Subset
    train_labels.append(label)
train_labels = np.array(train_labels)

# Calculate binary labels (0 for normal, 1 for any abnormal)
binary_labels = (train_labels > 0).astype(int)

# Calculate class weights and convert to float32
binary_counts = np.bincount(binary_labels)
pos_weight_1 = torch.tensor([binary_counts[0] / binary_counts[1]], dtype=torch.float32).to(device)

# Calculate multi-class weights and convert to float32
multi_counts = np.bincount(train_labels)
total_samples = len(train_labels)
weights_multi = torch.tensor(total_samples / (len(multi_counts) * multi_counts), dtype=torch.float32).to(device)


print("Class distribution:")
print(f"Binary - Normal: {binary_counts[0]}, Abnormal: {binary_counts[1]}")
print(f"Multi - Classes: {dict(enumerate(multi_counts))}")
print("\nWeights:")
print(f"Binary positive weight: {pos_weight_1.item():.4f}")
print(f"Multi-class weights: {weights_multi.tolist()}")

# First train a baseline model without weights
print("Training models with balanced class weights...")
model_1, model_2 = train_utils.train_both_models(
    train_loader=train_loader_d2,
    val_loader=val_loader_d2,
    model_1=model_1,
    model_2=model_2,
    n_epochs=20,
    lr=0.001,
    device=device,
    pos_weight_1=pos_weight_1,  # Binary classification weight
    pos_weight_2=weights_multi,  # Multi-class weights
    experiment_name="Weighted Training TEST"
)


# Usage example:
test_results = train_utils.test_both_models(
    test_loader=test_loader_d2,
    model_1=model_1,
    model_2=model_2,
    device=device
)

# Access the results
binary_preds = test_results['binary_preds']
binary_labels = test_results['binary_labels']
multi_preds = test_results['multi_preds']
multi_labels = test_results['multi_labels']

helpers.plot_combined_conf_mat("Profusion", d2_cl, binary_preds, test_indices_d2, True, "MBOD 2")

# Binary Classification Confusion Matrix
plt.figure(figsize=(8, 6))
binary_cm = confusion_matrix(binary_labels, binary_preds)
sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal'])
plt.title('Binary Classification Confusion Matrix - MBOD 2')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('binary_confusion_matrix.png')
plt.close()

# Multi-class Classification Confusion Matrix
plt.figure(figsize=(10, 8))
multi_cm = confusion_matrix(multi_labels, multi_preds)
sns.heatmap(multi_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Low', 'Medium', 'High'],
            yticklabels=['Normal', 'Low', 'Medium', 'High'])
plt.title('Multi-class Classification Confusion Matrix - MBOD 2')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('multi_confusion_matrix.png')
plt.close()

# Print accuracy scores
binary_accuracy = (binary_cm.diagonal().sum() / binary_cm.sum()) * 100
multi_accuracy = (multi_cm.diagonal().sum() / multi_cm.sum()) * 100

print(f"\nBinary Classification Accuracy: {binary_accuracy:.2f}%")
print(f"Multi-class Classification Accuracy: {multi_accuracy:.2f}%")

# Also save the confusion matrices to wandb
wandb.log({
    "binary_confusion_matrix": wandb.Image('binary_confusion_matrix.png'),
    "multi_confusion_matrix": wandb.Image('multi_confusion_matrix.png')
})



