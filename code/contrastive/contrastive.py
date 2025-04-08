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
import torch
import wandb
import torch.optim as optim
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import Normalize
import csv


wandb.login()
experiment_name = 'trip240-m_05-comb-SHN_50' 
# Initialize wandb before using it in other parts of the code
wandb.init(project='CL-New', name=experiment_name, config={
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 1e-3,
    "margin": 1,
    "optimizer": "Adam",
    "loss_function": "TripletMarginLoss",
    "anchor_dataset": "ILO only",
    "target_dataset": "MBOD-combined",
    "num_triplets": 240,
    "pretrained_MBOD": "False",
    "triplet_selection": "Batch SHN",
    "augmentations": "False",
    "distance_metric": "L2"
})


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

triplet_dataset = classes.TripletDataset(combined_dataset, ilo_dataset, max_triplets=wandb.config.num_triplets)

train_dataset, val_dataset, test_dataset = helpers.split_triplet_dataset(triplet_dataset)

model = xrv.models.ResNet(weights="resnet50-res512-all")
raw_model = xrv.models.ResNet(weights="resnet50-res512-all")
model = model.to(device)
raw_model = raw_model.to(device)

all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives = train_utils.extract_features_2(triplet_dataset, raw_model, device)
train_utils.plot_tsne_2(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives, 
          save_path=None, trained=False, epoch=None, log_wandb=True, plot_name="Pre-training t-SNE")
# train_utils.plot_tsne_3d_interactive(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives,
                                     # num_classes=4, save_path=None, trained=False, epoch=None, log_wandb=True,
                                     # plot_name="Pre-training 3D t-SNE")


train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
triplet_loss = torch.nn.TripletMarginLoss(margin=wandb.config.margin, p=2)  # Or any custom contrastive loss function



# Training loop
epochs = wandb.config.epochs

# Now start the training loop
for epoch in range(epochs):
    # print(f'EPOCH: {epoch + 1}/{epochs}')
    # Training phase
    model.train()
    epoch_loss = 0.0
    all_train_embeddings = []
    all_train_labels = []
    anchor_embeddings = {}  # Dictionary to store unique anchor embeddings
    anchor_labels = {}      # Dictionary to store unique anchor labels

    for step, (anchor, positive, negative, anchor_label, positive_label, negative_label, anchor_filename) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_label, positive_label, negative_label = (
            anchor_label.to(device), positive_label.to(device), negative_label.to(device)
        )
        print(f"Batch: {step + 1}/{len(train_loader)}")
        
        optimizer.zero_grad()

        # Forward pass: Compute features for anchor, positive, and negative
        anchor_feats = model.features(anchor)
        positive_feats = model.features(positive)
        negative_feats = model.features(negative)

        semi_hard_negative_feats = helpers.get_semi_hard_negative(anchor_feats, negative_feats, positive_feats, wandb.config.margin)

        # Compute loss with semi-hard negatives if found
        if semi_hard_negative_feats.size(0) > 0:
            loss = triplet_loss(anchor_feats, positive_feats, semi_hard_negative_feats)
        else:
            # Fallback to original negatives if no semi-hard ones found
            print("No semi-hard found - defaulting back to normal")
            loss = triplet_loss(anchor_feats, positive_feats, negative_feats)
            
        loss.backward()
        optimizer.step()

        # Update loss tracking
        epoch_loss += loss.item()

        # Collect embeddings for mAP computation (for positive and negative samples)
        all_train_embeddings.append(positive_feats.detach().cpu().numpy())
        all_train_embeddings.append(negative_feats.detach().cpu().numpy())
        all_train_labels.append(positive_label.cpu().numpy())
        all_train_labels.append(negative_label.cpu().numpy())

        # Collect anchor embeddings separately (only once per anchor)
        for i, label in enumerate(anchor_label.cpu().numpy()):
            if label not in anchor_embeddings:
                anchor_embeddings[label] = anchor_feats[i].detach().cpu().numpy()
                anchor_labels[label] = label

        wandb.log({
            'batch_loss': loss.item(),
            'batch': step + 1
        })

        # Delete tensors and clear GPU memory after each batch
        del anchor, anchor_label, anchor_feats, negative, negative_label, negative_feats, positive, positive_label, positive_feats
        torch.cuda.empty_cache()

    # Compute epoch-level train loss
    epoch_loss /= len(train_loader)
    
    # Compute train mAP
    if all_train_embeddings and all_train_labels:
        all_train_embeddings = np.concatenate(all_train_embeddings, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)
        train_map = helpers.compute_map(all_train_embeddings, all_train_labels)
    else:
        train_map = 0.0
    
    # Inside the training loop (after validation phase)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_embeddings = []
    val_labels = []
    val_anchor_embeddings = {}  # Dictionary to store unique validation anchor embeddings
    val_anchor_labels = {}      # Dictionary to store unique validation anchor labels
    
    with torch.no_grad():
        for step, (anchor, positive, negative, anchor_label, positive_label, negative_label, anchor_filename) in enumerate(val_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_label, positive_label, negative_label = (
                anchor_label.to(device), positive_label.to(device), negative_label.to(device)
            )
            
            # Forward pass: Compute features for anchor, positive, and negative
            anchor_feats = model.features(anchor)
            positive_feats = model.features(positive)
            negative_feats = model.features(negative)
            
            # Compute triplet loss
            loss = triplet_loss(anchor_feats, positive_feats, negative_feats)
            val_loss += loss.item()
            
            # Collect ALL validation embeddings for mAP computation
            val_embeddings.append(anchor_feats.detach().cpu().numpy())
            val_embeddings.append(positive_feats.detach().cpu().numpy())
            val_embeddings.append(negative_feats.detach().cpu().numpy())
            val_labels.append(anchor_label.cpu().numpy())
            val_labels.append(positive_label.cpu().numpy())
            val_labels.append(negative_label.cpu().numpy())

            wandb.log({
            'val_batch_loss': loss.item()
            })
            
            # Collect UNIQUE anchor embeddings for t-SNE (only once per anchor)
            for i, label in enumerate(anchor_label.cpu().numpy()):
                if label not in val_anchor_embeddings:
                    val_anchor_embeddings[label] = anchor_feats[i].detach().cpu().numpy()
                    val_anchor_labels[label] = label
            
            # Delete tensors and clear GPU memory after each batch
            del anchor, anchor_label, anchor_feats, negative, negative_label, negative_feats, positive, positive_label, positive_feats
            torch.cuda.empty_cache()
    
    val_loss /= len(val_loader)
    
    # Compute validation mAP if we have embeddings
    if val_embeddings and val_labels:
        val_embeddings_concat = np.concatenate(val_embeddings, axis=0)
        val_labels_concat = np.concatenate(val_labels, axis=0)
        val_map = helpers.compute_map(val_embeddings_concat, val_labels_concat)
    else:
        val_map = 0.0

    # Convert anchor embeddings and labels from dictionary to arrays
    train_anchor_embeddings = np.array(list(anchor_embeddings.values()))
    train_anchor_labels = np.array(list(anchor_labels.values()))
    
    if (epoch + 1) % 5 == 0:
        all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives = train_utils.extract_features_2(triplet_dataset, model, device)
        train_utils.plot_tsne_2(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives, 
                save_path=None, trained=True, epoch=(epoch+1), log_wandb=True, plot_name="Training t-SNE")
        
        val_anchor_embeddings, val_anchor_labels, val_embeddings, val_labels = train_utils.extract_features_2(val_dataset, model, device, anchors_only=False)
    # Create validation t-SNE plot
        if len(val_embeddings) > 0:
            train_utils.plot_tsne_2(
                val_anchor_embeddings,    # Only unique anchors from validation
                val_anchor_labels,        # Validation anchor labels
                val_embeddings,          # All validation embeddings
                val_labels,              # All validation labels
                save_path=None,
                num_classes=4,                  # Number of classes
                trained=True,                   # Set to True if the model is trained
                epoch=epoch + 1,                # Include epoch number for context in the plot title
                log_wandb=True,                 # Log the plot to WandB
                plot_name="Validation t-SNE"    # Name for the plot
            )

            # train_utils.plot_tsne_3d_interactive(
                # val_anchor_embeddings_array,
                # val_anchor_labels_array,
                # val_embeddings_concat,
                # val_labels_concat,
                # save_path=None,
                # trained=True,
                # epoch = epoch+1,
                # log_wandb=True,
                # plot_name="Validation 3D t-SNE"
            # )
        
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': epoch_loss,
        'train_mAP': train_map,
        'val_loss': val_loss,
        'val_mAP': val_map,
    })

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train mAP: {train_map:.4f}, Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}")
        # Save checkpoint

    if (epoch+1) % 5 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, f'/home/sean/MSc/code/contrastive/checkpoints/{experiment_name}_{epoch + 1}.pth')



train_anchors, train_anchor_labels, train_embeddings, train_labels = train_utils.extract_features_2(train_dataset, model, device, False)

test_anchors, test_anchor_labels, test_embeddings, test_labels = train_utils.extract_features_2(test_dataset, model, device, False)


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
    'checkpoint_path': experiment_name + "",
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

# After computing t-SNE embeddings and before creating the plot
# Find misclassified points in train set
train_misclassified = train_labels != train_preds_knn
train_misclassified_indices_anchors = anchor_train_indices[train_misclassified[anchor_train_indices]]
train_misclassified_indices_others = other_train_indices[train_misclassified[other_train_indices]]

# Find misclassified points in test set
test_misclassified = test_labels != test_preds_knn
test_misclassified_indices_anchors = anchor_test_indices[test_misclassified[anchor_test_indices]]
test_misclassified_indices_others = other_test_indices[test_misclassified[other_test_indices]]

# Create the figure with 2 rows and 2 columns for the plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot train set
# Correctly classified points
axes[0,0].scatter(tsne_train_embeddings_2d[anchor_train_indices, 0], tsne_train_embeddings_2d[anchor_train_indices, 1],
                  c=train_anchor_labels, cmap='viridis', marker='*', s=80, norm=train_norm, label="Anchors")
x = axes[0,0].scatter(tsne_train_embeddings_2d[other_train_indices, 0], tsne_train_embeddings_2d[other_train_indices, 1],
                      c=train_labels[other_train_indices], cmap='viridis', alpha=0.7, norm=train_norm)

# Highlight misclassified points with red outlines
axes[0,0].scatter(tsne_train_embeddings_2d[train_misclassified_indices_anchors, 0], tsne_train_embeddings_2d[train_misclassified_indices_anchors, 1],
                  facecolors='none', edgecolors='red', marker='*', s=100, linewidth=1.5, label="Misclassified Anchors")
axes[0,0].scatter(tsne_train_embeddings_2d[train_misclassified_indices_others, 0], tsne_train_embeddings_2d[train_misclassified_indices_others, 1],
                  facecolors='none', edgecolors='red', marker='o', s=80, linewidth=1.5, label="Misclassified")



# After computing t-SNE embeddings and before creating the plot
# Find misclassified points in train set
train_misclassified = train_labels != train_preds_knn
train_misclassified_indices_anchors = anchor_train_indices[train_misclassified[anchor_train_indices]]
train_misclassified_indices_others = other_train_indices[train_misclassified[other_train_indices]]

# Find misclassified points in test set
test_misclassified = test_labels != test_preds_knn
test_misclassified_indices_anchors = anchor_test_indices[test_misclassified[anchor_test_indices]]
test_misclassified_indices_others = other_test_indices[test_misclassified[other_test_indices]]

# Create the figure with 2 rows and 2 columns for the plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot test set
# Correctly classified points
axes[0,1].scatter(tsne_test_embeddings_2d[anchor_test_indices, 0], tsne_test_embeddings_2d[anchor_test_indices, 1],
                  c=test_anchor_labels, cmap='viridis', marker='*', s=80, norm=test_norm, label="Anchors")
y = axes[0,1].scatter(tsne_test_embeddings_2d[other_train_indices, 0], tsne_test_embeddings_2d[other_test_indices, 1],
                      c=test_labels[other_test_indices], cmap='viridis', alpha=0.7, norm=test_norm)

# Highlight misclassified points with red outlines
axes[0,1].scatter(tsne_test_embeddings_2d[test_misclassified_indices_anchors, 0], tsne_test_embeddings_2d[test_misclassified_indices_anchors, 1],
                  facecolors='none', edgecolors='red', marker='*', s=100, linewidth=1.5, label="Misclassified Anchors")
axes[0,1].scatter(tsne_test_embeddings_2d[test_misclassified_indices_others, 0], tsne_test_embeddings_2d[test_misclassified_indices_others, 1],
                  facecolors='none', edgecolors='red', marker='o', s=80, linewidth=1.5, label="Misclassified")


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

# Add a figure-level title with experiment name
fig.suptitle(f"Model Evaluation: {experiment_name}", fontsize=16, y=0.98)
# Adjust layout to make room for title
plt.subplots_adjust(top=0.9)


plt.tight_layout()
plt.savefig(f"code/contrastive/results/KNN_eval_{experiment_name}.png")

# Log the figure to wandb
if wandb.run is not None:
    wandb.log({"KNN_evaluation": wandb.Image(fig)})

wandb.finish()  # Close the wandb run