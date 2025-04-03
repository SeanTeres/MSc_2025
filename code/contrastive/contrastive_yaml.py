import yaml
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import helpers, classes, train_utils
import random
import torchxrayvision as xrv
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader
import wandb
import numpy as np

import torch
import wandb
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import numpy as np

import yaml
import argparse
import wandb

# Load config.yaml
config_path = "/home/sean/MSc/code/cl_config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Parse command-line argument for experiment selection
parser = argparse.ArgumentParser(description="Select an experiment configuration.")
parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment (e.g., contrastive_exp1)")
args = parser.parse_args()

# Load the selected experiment
if args.experiment not in config["experiments"]:
    raise ValueError(f"Experiment '{args.experiment}' not found in config.yaml.")

experiment_config = config["experiments"][args.experiment]

# Initialize Weights & Biases
wandb.init(project=experiment_config["project_name"], name=args.experiment, config=experiment_config["parameters"])

print(f"Running experiment: {args.experiment}")
print(f"Parameters: {experiment_config['parameters']}")


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
ilo_dataset_filt = classes.ILOImagesDataset(ilo_images_dir=ILO_imgs, filter_one_per_label=True)

triplet_dataset = classes.TripletDataset(combined_dataset, ilo_dataset, max_triplets=wandb.config.num_triplets)
triplet_dataset_filt = classes.TripletDataset(combined_dataset, ilo_dataset_filt, max_triplets=wandb.config.num_triplets)

train_dataset, val_dataset, test_dataset = helpers.split_triplet_dataset(triplet_dataset)

model = xrv.models.ResNet(weights="resnet50-res512-all")
raw_model = xrv.models.ResNet(weights="resnet50-res512-all")
model = model.to(device)
raw_model = raw_model.to(device)

all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives = train_utils.extract_features_2(triplet_dataset, raw_model, device)
train_utils.plot_tsne_2(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives, 
          save_path=None, trained=False, epoch=None, log_wandb=True, plot_name="Pre-training t-SNE")


train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
triplet_loss = torch.nn.TripletMarginLoss(margin=wandb.config.margin, p=2)  # Or any custom contrastive loss function

import numpy as np
from sklearn.metrics import average_precision_score

def compute_map(embeddings, labels):
    print("Computing mAP...")
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Print min, max, mean for the first embedding before normalization
    print("Before normalization (first embedding):")
    print(f"Min: {np.min(embeddings[0])}, Max: {np.max(embeddings[0])}, Mean: {np.mean(embeddings[0])}")
    
    norms_before = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Normalize embeddings to unit vectors
    epsilon = 1e-8
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + epsilon)
    
    # Print min, max, mean for the first embedding after normalization
    print("After normalization (first embedding):")
    print(f"Min: {np.min(embeddings[0])}, Max: {np.max(embeddings[0])}, Mean: {np.mean(embeddings[0])}")
    
    # Verify norms after normalization
    norms_after = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    unique_labels = np.unique(labels)
    
    ap_scores = []
    
    for i in range(len(embeddings)):
        # Compute cosine similarity (dot product on normalized embeddings)
        similarity = embeddings @ embeddings[i]
        sorted_indices = np.argsort(-similarity)[1:]  # Exclude self-match
        
        # Compute relevance labels
        relevant_labels = (labels[sorted_indices] == labels[i]).astype(int)
        
        # Compute AP if there are any positives
        if relevant_labels.sum() > 0:
            ap_score = average_precision_score(relevant_labels, similarity[sorted_indices])
            ap_scores.append(ap_score)
        else:
            print(f"No positives for anchor {i}, skipping AP calculation.")
    
    return np.mean(ap_scores) if ap_scores else 0.0



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

        # Compute triplet loss
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
        train_map = compute_map(all_train_embeddings, all_train_labels)
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
        val_map = compute_map(val_embeddings_concat, val_labels_concat)
    else:
        val_map = 0.0

    # Convert anchor embeddings and labels from dictionary to arrays
    train_anchor_embeddings = np.array(list(anchor_embeddings.values()))
    train_anchor_labels = np.array(list(anchor_labels.values()))
    val_anchor_embeddings_array = np.array(list(val_anchor_embeddings.values()))
    val_anchor_labels_array = np.array(list(val_anchor_labels.values()))
    
    print(f"Number of unique training anchors: {len(train_anchor_embeddings)}")
    print(f"Number of unique validation anchors: {len(val_anchor_embeddings_array)}")
    # all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives = train_utils.extract_features_2(triplet_dataset, raw_model, device)
    # train_utils.plot_tsne_2(all_anchors, all_labels, all_positives_negatives, all_labels_positives_negatives, 
            # save_path=None, trained=False, epoch=None, log_wandb=True, plot_name="Pre-training t-SNE")
    
    if (epoch + 1) % 5 == 0:
        train_anchor_embeddings, train_anchor_labels = train_utils.extract_features_2(train_dataset, model, device, anchors_only=True)
        # Call the plot_tsne_2 function to log the training t-SNE plot to WandB
        train_utils.plot_tsne_2(
            train_anchor_embeddings,     # Only unique anchors are plotted once
            train_anchor_labels,         # Anchor labels
            all_train_embeddings,        # Use positive and negative embeddings
            all_train_labels,            # Use positive and negative labels
            save_path=None,
            num_classes=4,               # Number of classes
            trained=True,                # Set to True if the model is trained
            epoch=epoch + 1,             # Include epoch number for context in the plot title
            log_wandb=True,              # Log the plot to WandB
            plot_name="Training t-SNE"   # Name for the plot
        )
        
        val_anchor_embeddings, val_anchor_labels = train_utils.extract_features_2(val_dataset, model, device, anchors_only=True)    
    # Create validation t-SNE plot
        if len(val_embeddings) > 0:
            train_utils.plot_tsne_2(
                val_anchor_embeddings_array,    # Only unique anchors from validation
                val_anchor_labels_array,        # Validation anchor labels
                val_embeddings_concat,          # All validation embeddings
                val_labels_concat,              # All validation labels
                save_path=None,
                num_classes=4,                  # Number of classes
                trained=True,                   # Set to True if the model is trained
                epoch=epoch + 1,                # Include epoch number for context in the plot title
                log_wandb=True,                 # Log the plot to WandB
                plot_name="Validation t-SNE"    # Name for the plot
            )
        
    # ADD THIS CODE HERE - Log epoch-level metrics to wandb
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': epoch_loss,
        'train_mAP': train_map,
        'val_loss': val_loss,
        'val_mAP': val_map,
    })

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train mAP: {train_map:.4f}, Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}")
        # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_loss,
        'train_mAP': train_map,
        'val_loss': val_loss,
        'val_mAP': val_map,
    }
    torch.save(checkpoint, f'checkpoint-2_epoch_{epoch + 1}.pth')

wandb.finish()  # Close the wandb run
