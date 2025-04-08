import os
import sys
sys.path.append('/home/sean/MSc/code')
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import utils.helpers as helpers
import utils.classes as classes
import utils.train_utils as train_utils
import random
import torchxrayvision as xrv
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import torch
import wandb
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns



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

ilo_dataset = classes.ILOImagesDataset(ilo_images_dir=ILO_imgs, filter_one_per_label=False)

triplet_dataset = classes.TripletDataset(combined_dataset, ilo_dataset, max_triplets=160)
triplet_dataset_d2 = classes.TripletDataset(d2_cl, ilo_dataset, max_triplets=160)
triplet_dataset_d1 = classes.TripletDataset(d1_cl, ilo_dataset, max_triplets=160)

train_dataset, val_dataset, test_dataset = helpers.split_triplet_dataset(triplet_dataset)
train_dataset_d2, val_dataset_d2, test_dataset_d2 = helpers.split_triplet_dataset(triplet_dataset_d2)
train_dataset_d1, val_dataset_d1, test_dataset_d1 = helpers.split_triplet_dataset(triplet_dataset_d1)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

train_loader_d2 = DataLoader(train_dataset_d2, batch_size=8, shuffle=True)
val_loader_d2 = DataLoader(val_dataset_d2, batch_size=8, shuffle=False)
test_loader_d2 = DataLoader(test_dataset_d2, batch_size=8, shuffle=False)

raw_model = xrv.models.ResNet(weights="resnet50-res512-all")
raw_model = raw_model.to(device)

checkpoint_model = xrv.models.ResNet(weights="resnet50-res512-all")
checkpoint_model = checkpoint_model.to(device)

checkpoint_path = '/home/sean/MSc/code/contrastive/checkpoints/trip240-m_1-comb-SHN_20.pth'  # Path to your checkpoint file (adjust as needed)
checkpoint = torch.load(checkpoint_path)

# Restore model state_dict
checkpoint_model.load_state_dict(checkpoint['model_state_dict'])

experiment_name = "trip240-m_1-comb-SHN_20"

# all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives = train_utils.extract_features_2(triplet_dataset, raw_model, device, False)

# train_utils.plot_tsne_2(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives, 4, f"/home/sean/MSc/code/contrastive/t-SNE/{experiment_name}-raw.png", False, None, False, "Pre-training t-SNE")
# train_utils.plot_tsne_3d_interactive(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives,
                                     # num_classes=4, save_path=f"/home/sean/MSc/code/contrastive/t-SNE/{experiment_name}-raw-3d.png", trained=False, epoch=None, log_wandb=False,
                                     # plot_name="Pre-training 3D t-SNE")

# all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives = train_utils.extract_features_2(triplet_dataset, checkpoint_model, device, False)

# train_utils.plot_tsne_2(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives, 4, f"/home/sean/MSc/code/contrastive/t-SNE/{experiment_name}-trained.png", True, 30, False, "Trained t-SNE")
# train_utils.plot_tsne_3d_interactive(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives,
                                     # num_classes=4, save_path=f"/home/sean/MSc/code/contrastive/t-SNE/{experiment_name}-trained-3d.html", trained=True, epoch=30, log_wandb=False,
                                     # plot_name="Trained 3D t-SNE")


train_anchors, train_anchor_labels, train_embeddings, train_labels = train_utils.extract_features_2(train_dataset, checkpoint_model, device, False)

test_anchors, test_anchor_labels, test_embeddings, test_labels = train_utils.extract_features_2(test_dataset, checkpoint_model, device, False)


knn = KNeighborsClassifier(n_neighbors=4, metric='cosine')
knn.fit(train_embeddings, train_labels)


test_preds = knn.predict(test_embeddings)

train_preds = knn.predict(train_embeddings)


train_accuracy = accuracy_score(train_labels, train_preds)
train_recall = recall_score(train_labels, train_preds, average='macro')
train_precision = precision_score(train_labels, train_preds, average='macro')
train_f1 = f1_score(train_labels, train_preds, average='macro')
train_kappa = cohen_kappa_score(train_labels, train_preds)


test_accuracy = accuracy_score(test_labels, test_preds)
test_recall = recall_score(test_labels, test_preds, average='macro')
test_precision = precision_score(test_labels, test_preds, average='macro')
test_f1 = f1_score(test_labels, test_preds, average='macro')
test_kappa = cohen_kappa_score(test_labels, test_preds)

print("Train Set Metrics:")
print(f"Accuracy: {train_accuracy * 100:.2f}%")
print(f"Recall: {train_recall * 100:.2f}%")
print(f"Precision: {train_precision * 100:.2f}%")
print(f"F1 Score: {train_f1 * 100:.2f}%")
print(f"Kappa Score: {train_kappa}%")

print("\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(f"Recall: {test_recall * 100:.2f}%")
print(f"Precision: {test_precision * 100:.2f}%")
print(f"F1 Score: {test_f1 * 100:.2f}%")
print(f"Kappa Score: {test_kappa}%")


# 1. t-SNE Visualization for Train and Test Set
tsne = TSNE(n_components=2, random_state=42)

# Reduce dimensions for train and test embeddings
train_embeddings_2d = tsne.fit_transform(train_embeddings)
test_embeddings_2d = tsne.fit_transform(test_embeddings)

# 2. Confusion Matrix for Train and Test
train_cm = confusion_matrix(train_labels, train_preds)
test_cm = confusion_matrix(test_labels, test_preds)

# Create the figure with 2 rows and 2 columns for the plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1st Row: t-SNE for Train and Test
axes[0, 0].scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_labels, cmap='viridis', alpha=0.7)
axes[0, 0].set_title("t-SNE of Train Set")
axes[0, 0].set_xlabel("t-SNE Component 1")
axes[0, 0].set_ylabel("t-SNE Component 2")

axes[0, 1].scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c=test_labels, cmap='viridis', alpha=0.7)
axes[0, 1].set_title("t-SNE of Test Set")
axes[0, 1].set_xlabel("t-SNE Component 1")
axes[0, 1].set_ylabel("t-SNE Component 2")

# 2nd Row: Confusion Matrix for Train and Test
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], xticklabels=np.unique(train_labels), yticklabels=np.unique(train_labels))
axes[1, 0].set_title('Confusion Matrix - Train Set')
axes[1, 0].set_xlabel('Predicted Labels')
axes[1, 0].set_ylabel('True Labels')

sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
axes[1, 1].set_title('Confusion Matrix - Test Set')
axes[1, 1].set_xlabel('Predicted Labels')
axes[1, 1].set_ylabel('True Labels')

plt.tight_layout()
plt.savefig("CL_eval.png")
plt.show()