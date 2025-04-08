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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
import seaborn as sns
import csv



dicom_dir_1 = 'data/MBOD_Datasets/Dataset-1'
dicom_dir_2 = 'data/MBOD_Datasets/Dataset-2'
metadata_1 = pd.read_excel("data/MBOD_Datasets/Dataset-1/FileDatabaseWithRadiology.xlsx")

metadata_2 = pd.read_excel("data/MBOD_Datasets/Dataset-2/Database_Training-2024.08.28.xlsx")

ILO_imgs = 'data/ilo-radiographs-dicom'

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
num_triplets = 240
triplet_dataset = classes.TripletDataset(combined_dataset, ilo_dataset, max_triplets=num_triplets)
triplet_dataset_d2 = classes.TripletDataset(d2_cl, ilo_dataset, max_triplets=num_triplets)
triplet_dataset_d1 = classes.TripletDataset(d1_cl, ilo_dataset, max_triplets=num_triplets)

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

checkpoint_path = 'code/contrastive/checkpoints/trip160-m_05-comb-SHN_25.pth'  # Path to your checkpoint file (adjust as needed)
checkpoint = torch.load(checkpoint_path)

# Restore model state_dict
checkpoint_model.load_state_dict(checkpoint['model_state_dict'])

experiment_name = "trip240-m_1-comb-SHN_30"

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


# Fit KNN classifier
knn = KNeighborsClassifier(n_neighbors=4, metric='cosine')
knn.fit(train_embeddings, train_labels)

# Evaluate on Train Set
train_precision_at_k, train_recall_at_k, train_f1_at_k = helpers.compute_top_k_metrics(train_embeddings, train_labels, knn, k=4)

# Evaluate on Test Set
test_precision_at_k, test_recall_at_k, test_f1_at_k = helpers.compute_top_k_metrics(test_embeddings, test_labels, knn, k=4)

# Print results
print("Train Set Metrics:")
print(f"Precision@k: {train_precision_at_k * 100:.2f}%")
print(f"Recall@k: {train_recall_at_k * 100:.2f}%")
print(f"F1@k: {train_f1_at_k * 100:.2f}%")

print("\nTest Set Metrics:")
print(f"Precision@k: {test_precision_at_k * 100:.2f}%")
print(f"Recall@k: {test_recall_at_k * 100:.2f}%")
print(f"F1@k: {test_f1_at_k * 100:.2f}%")

# Now print the general classification metrics
train_preds_knn = knn.predict(train_embeddings)
test_preds_knn = knn.predict(test_embeddings)

train_accuracy = accuracy_score(train_labels, train_preds_knn)
train_recall = recall_score(train_labels, train_preds_knn, average='macro')
train_precision = precision_score(train_labels, train_preds_knn, average='macro')
train_f1 = f1_score(train_labels, train_preds_knn, average='macro')
train_kappa = cohen_kappa_score(train_labels, train_preds_knn)

test_accuracy = accuracy_score(test_labels, test_preds_knn)
test_recall = recall_score(test_labels, test_preds_knn, average='macro')
test_precision = precision_score(test_labels, test_preds_knn, average='macro')
test_f1 = f1_score(test_labels, test_preds_knn, average='macro')
test_kappa = cohen_kappa_score(test_labels, test_preds_knn)

print("\nTrain Set General Metrics:")
print(f"Accuracy: {train_accuracy * 100:.2f}%")
print(f"Recall: {train_recall * 100:.2f}%")
print(f"Precision: {train_precision * 100:.2f}%")
print(f"F1 Score: {train_f1 * 100:.2f}%")
print(f"Kappa Score: {train_kappa}%")

print("\nTest Set General Metrics:")
print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(f"Recall: {test_recall * 100:.2f}%")
print(f"Precision: {test_precision * 100:.2f}%")
print(f"F1 Score: {test_f1 * 100:.2f}%")
print(f"Kappa Score: {test_kappa}%")

# Save metrics to CSV file
import os
import csv

# Create results directory if it doesn't exist
os.makedirs('/home/sean/MSc/code/contrastive/results', exist_ok=True)

# Define the csv file path
results_file = '/home/sean/MSc/code/contrastive/results/experiment_log.csv'

# Check if file exists to determine if we need to write headers
file_exists = os.path.isfile(results_file)

# Prepare the data row
results_data = {
    'experiment_name': experiment_name,
    'train_precision_at_k': train_precision_at_k * 100,
    'train_recall_at_k': train_recall_at_k * 100,
    'train_f1_at_k': train_f1_at_k * 100,
    'test_precision_at_k': test_precision_at_k * 100,
    'test_recall_at_k': test_recall_at_k * 100,
    'test_f1_at_k': test_f1_at_k * 100,
    'train_accuracy': train_accuracy * 100,
    'train_recall': train_recall * 100,
    'train_precision': train_precision * 100,
    'train_f1': train_f1 * 100,
    'train_kappa': train_kappa,
    'test_accuracy': test_accuracy * 100,
    'test_recall': test_recall * 100,
    'test_precision': test_precision * 100,
    'test_f1': test_f1 * 100,
    'test_kappa': test_kappa,
    'checkpoint_path': checkpoint_path,
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Write to the CSV file
with open(results_file, mode='a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results_data.keys())
    
    # Write header if file doesn't exist
    if not file_exists:
        writer.writeheader()
    
    writer.writerow(results_data)

print(f"\nResults saved to {results_file}")

# 1. t-SNE Visualization for Train and Test Set
tsne = TSNE(n_components=2, random_state=42)

# 2. Confusion Matrix for Train and Test
train_cm = confusion_matrix(train_labels, train_preds_knn)
test_cm = confusion_matrix(test_labels, test_preds_knn)

print(np.unique(train_anchor_labels), len(train_anchor_labels))
print(np.unique(train_labels), len(train_labels))

# Reduce dimensions for train and test embeddings
train_embeddings = np.concatenate([train_anchors, train_embeddings], axis=0)
test_embeddings = np.concatenate([test_anchors, test_embeddings], axis=0)
train_labels = np.concatenate([train_anchor_labels, train_labels], axis=0)
test_labels = np.concatenate([test_anchor_labels, test_labels], axis=0)

tsne_train_embeddings_2d = tsne.fit_transform(train_embeddings)
tsne_test_embeddings_2d = tsne.fit_transform(test_embeddings)

anchor_train_indices = np.arange(len(train_anchors))
other_train_indices = np.arange(len(train_anchors), len(train_embeddings))

anchor_test_indices = np.arange(len(test_anchors))
other_test_indices = np.arange(len(test_anchors), len(test_embeddings))

train_norm = Normalize(vmin=np.min(train_labels), vmax=np.max(train_labels))
test_norm = Normalize(vmin=np.min(test_labels), vmax=np.max(test_labels))

# Create the figure with 2 rows and 2 columns for the plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot anchors directly without trying to index through train_embeddings_2d
axes[0,0].scatter(tsne_train_embeddings_2d[anchor_train_indices, 0], tsne_train_embeddings_2d[anchor_train_indices, 1],
                              c=train_anchor_labels, cmap='viridis', marker='*', s=50, norm=train_norm, label="Anchors")
x = axes[0,0].scatter(tsne_train_embeddings_2d[other_train_indices, 0], tsne_train_embeddings_2d[other_train_indices, 1],
                              c=train_labels[other_train_indices], cmap='viridis', alpha=0.7, norm=train_norm)
axes[0, 0].set_title("t-SNE of Train Set")
axes[0, 0].set_xlabel("t-SNE Component 1")
axes[0, 0].set_ylabel("t-SNE Component 2")


axes[0,1].scatter(tsne_test_embeddings_2d[anchor_test_indices, 0], tsne_test_embeddings_2d[anchor_test_indices, 1],
                              c=test_anchor_labels, cmap='viridis', marker='*', s=50, norm=test_norm, label="Anchors")
y = axes[0,1].scatter(tsne_test_embeddings_2d[other_test_indices, 0], tsne_test_embeddings_2d[other_test_indices, 1],
                              c=test_labels[other_test_indices], cmap='viridis', alpha=0.7, norm=test_norm)
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

# Add a legend for the anchor points
axes[0, 0].legend(loc="upper right")
axes[0, 1].legend(loc="upper right")

fig.colorbar(x, ax=axes[0, 0], label='Class Label')
fig.colorbar(y, ax=axes[0, 1], label='Class Label')

plt.tight_layout()
plt.savefig("KNN_eval_trip160-m_05-comb-SHN_25.png")
plt.show()
