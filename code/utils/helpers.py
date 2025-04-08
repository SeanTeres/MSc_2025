import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, ConcatDataset
import torchvision.transforms as transforms
import os
import torchxrayvision as xrv
from skimage.color import rgb2gray
from skimage.transform import resize
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
import utils.classes as classes
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier


def read_and_normalize_xray(dicom_name, voi_lut=False, fix_monochrome=True, transforms=None, normalize=True):
    """Reads a DICOM file, normalizes it, and returns the tensor and pixel array."""
    ds = pydicom.dcmread(dicom_name)

    if voi_lut:
        pixel_array = pydicom.apply_voi_lut(ds.pixel_array.astype(np.float32), ds)
    else:
        pixel_array = ds.pixel_array.astype(np.float32)

    if ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']:
        pixel_array = rgb2gray(pixel_array)

    if fix_monochrome and ds.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = np.amax(pixel_array) - pixel_array

    pixel_array = pixel_array.astype(np.float32)
    # Convert to tensor (1, H, W) and apply transforms (resize, crop)
    pixel_tensor = torch.from_numpy(pixel_array).unsqueeze(0)  # Add channel dimension
    if transforms:
        pixel_tensor = transforms(pixel_tensor)

    # Normalize if specified
    if normalize:
        pixel_tensor = (pixel_tensor - pixel_tensor.min()) / (pixel_tensor.max() - pixel_tensor.min())
        # Rescale to [-1024, 1024] if needed for xrv models
        pixel_tensor = pixel_tensor * (1024 - (-1024)) + (-1024)

    pixel_array = pixel_tensor.numpy()

    return pixel_tensor, pixel_array


def split_with_indices(dataset_class, train_size, to_augment=False):
    """Takes a dataset class and the training set size, validation and test splits are even for the remaining portion."""
    test_set_size = 1 - train_size
    indices_d =  list(range(len(dataset_class)))

    train_indices, test_indices = train_test_split(indices_d, test_size=test_set_size, random_state=42)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)

    train_dataset = Subset(dataset_class, train_indices)
    val_dataset = Subset(dataset_class, val_indices)
    test_dataset = Subset(dataset_class, test_indices)

    return train_dataset, val_dataset, test_dataset

def split_dataset(dataset, train_size=0.7, random_state=42):
    """Split dataset into train, validation, and test sets with fixed indices."""
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=1-train_size, random_state=random_state)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=random_state)
    
    return train_indices, val_indices, test_indices

def create_dataloaders(train, aug_train, val, test, batch_size, target):
    """Function to create dataloaders with optional oversampling.
    target: full string of label."""
    # print("Creating dataloaders with optional oversampling...")
    print(f"LENGTHS: {len(train), len(aug_train)}")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    aug_train_loader = DataLoader(aug_train, batch_size=batch_size, shuffle=True)
    
    # Other loaders remain the same
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    print(f"LENGTHS: {len(train_loader), len(aug_train_loader)}")
    
    return train_loader, aug_train_loader, val_loader, test_loader

def contains_tba_or_tbu(string):
    if isinstance(string, float):
        return False
    return 'tba' in string.lower() or 'tbu' in string.lower()


# Function to calculate sample weights for WeightedRandomSampler
def calculate_sample_weights(labels):
    """
    Calculate sample weights for a dataset to enable oversampling using WeightedRandomSampler.
    Args:
        labels (pd.Series): Class labels for the dataset.
    Returns:
        np.ndarray: Sample weights for each data point.
    """
    class_counts = labels.value_counts().to_dict()  # Count occurrences of each class
    total_samples = len(labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    sample_weights = labels.map(class_weights).to_numpy()  # Map weights to each sample
    return sample_weights

def calc_label_dist(dataset, subset, disease_label):
    """Calculates label distribution for a dataset or subset at specified disease label."""
    if len(dataset) == len(subset):
        # Full dataset case
        labels = dataset.metadata_df[disease_label]
    else:
        # Subset case
        labels = dataset.metadata_df.loc[subset.indices, disease_label]
    
    # Convert labels to integers
    labels = labels.astype(int)
    
    return Counter(labels)

from tqdm import tqdm

def extract_pixel_intensities(dataloader):
    pixel_intensities = []
    for images, _ in tqdm(dataloader, desc="Extracting pixel intensities"):
        for image in images:
            pixel_intensities.extend(image.numpy().flatten())
    return pixel_intensities

import numpy as np

def get_alpha_FLoss(train, target_label):
    """
    Compute a single alpha value for binary focal loss.
    
    """
    class_counts = train.dataset.metadata_df[target_label].value_counts()
    class_counts = class_counts.sort_index()

    # Ensure class counts exist
    if len(class_counts) < 2:
        raise ValueError("Dataset must contain both positive (1) and negative (0) samples.")

    minority = class_counts[0]  # Assuming class 0 is the minority
    majority = class_counts[1]  # Assuming class 1 is the majority

    # Compute alpha as the proportion of the negative class
    alpha = minority / (majority + minority)

    return alpha


def compute_pos_weight(train, target_label):
    """Compute pos_weight for BCEWithLogitsLoss."""
    class_counts = train.dataset.metadata_df[target_label].value_counts()
    class_counts = class_counts.sort_index()

    if len(class_counts) < 2:
        raise ValueError("Dataset must contain both positive (1) and negative (0) samples.")

    N_pos = class_counts[1]  # Positive class count
    N_neg = class_counts[0]  # Negative class count

    pos_weight = torch.tensor([ N_neg / N_pos ], dtype=torch.float32)

    return pos_weight

def salt_and_pepper_noise_tensor(image, prob=0.02):
    """
    Apply salt-and-pepper noise to a PyTorch tensor image.
    
    :param image: PyTorch tensor of shape (C, H, W), values in [0,1].
    :param prob: Probability of a pixel being affected.
    :return: Noisy image tensor.
    """
    assert image.dim() == 3, "Input must be a 3D tensor (C, H, W)"
    
    noisy_image = image.clone()  # Clone to avoid modifying original image
    
    # Generate random noise mask
    rand_tensor = torch.rand_like(image)  # Random values between [0,1]

    # Apply Salt (white pixels)
    noisy_image[rand_tensor < prob / 2] = 1.0  # If image is in [0,1], use 255.0 for [0,255]

    # Apply Pepper (black pixels)
    noisy_image[rand_tensor > 1 - prob / 2] = 0.0

    return noisy_image


def split_triplet_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits a triplet dataset into train, validation, and test sets.
    
    Args:
        dataset: The TripletDataset instance.
        train_ratio: The ratio of the dataset to be used for training.
        val_ratio: The ratio of the dataset to be used for validation.
        test_ratio: The ratio of the dataset to be used for testing.
        
    Returns:
        train_dataset, val_dataset, test_dataset: Three subsets of the original dataset.
    """
    # Get the total number of triplets
    total_samples = len(dataset)
    
    # Calculate the number of samples for each split
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size  # Ensure all data is used
    
    # Randomly shuffle indices
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # Split the indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subsets using the shuffled indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

def compute_map(embeddings, labels):
    print("Computing mAP...")
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Print first 2 embeddings before normalization and their norms
    print("Before normalization:")
    print("Embeddings[0:1]:", embeddings[:1])
    norms_before = np.linalg.norm(embeddings, axis=1, keepdims=True)
    print("Norms[0:1]:", norms_before[:1])
    
    # Normalize each embedding vector (L2 norm)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Print first 2 embeddings after normalization and their norms
    print("After normalization:")
    print("Embeddings[0:1]:", embeddings[:1])
    norms_after = np.linalg.norm(embeddings, axis=1, keepdims=True)
    print("Norms[0:1]:", norms_after[:1])
    
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    
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
    
    return np.mean(ap_scores) if ap_scores else 0.0

def check_multi_class_split_integrity(dataset, split_indices, dataset_name):
    """
    Check the integrity of a dataset split, including class distribution.
    Works with both regular and concatenated datasets.
    
    Args:
        dataset: The dataset (can be a regular Dataset or ConcatDataset)
        split_indices: Dictionary with 'train', 'val', 'test' keys containing indices
        dataset_name: Name of the dataset for reporting
    """
    total_samples = len(dataset)
    train_size = len(split_indices['train'])
    val_size = len(split_indices['val'])
    test_size = len(split_indices['test'])
    is_concat_dataset = hasattr(dataset, 'datasets')
    
    print(f"\n===== {dataset_name} Split Analysis =====")
    print(f"Dataset type: {'Concatenated' if is_concat_dataset else 'Regular'}")
    print(f"Total samples: {total_samples}")
    print(f"Train split: {train_size} samples ({train_size/total_samples*100:.1f}%)")
    print(f"Val split: {val_size} samples ({val_size/total_samples*100:.1f}%)")
    print(f"Test split: {test_size} samples ({test_size/total_samples*100:.1f}%)")
    
    # Check if all indices are accounted for
    total_in_splits = train_size + val_size + test_size
    print(f"Total in splits: {total_in_splits} (should equal {total_samples})")
    if total_in_splits != total_samples:
        print(f"WARNING: Missing {total_samples - total_in_splits} samples!")
    
    # Check for overlap between splits
    train_set = set(split_indices['train'])
    val_set = set(split_indices['val'])
    test_set = set(split_indices['test'])
    
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    if train_val_overlap:
        print(f"ERROR: {len(train_val_overlap)} samples overlap between train and validation!")
    if train_test_overlap:
        print(f"ERROR: {len(train_test_overlap)} samples overlap between train and test!")
    if val_test_overlap:
        print(f"ERROR: {len(val_test_overlap)} samples overlap between validation and test!")
    
    # Check class distribution
    class_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # Different handling for concatenated vs regular datasets
    if is_concat_dataset:
        # Handle concatenated dataset
        for split_name, indices in split_indices.items():
            split_class_dist = {0: 0, 1: 0, 2: 0, 3: 0}
            
            for idx in indices:
                # For concatenated datasets, we need to map global index to dataset and local index
                dataset_idx, sample_idx = map_global_index_to_local(dataset, idx)
                subdataset = dataset.datasets[dataset_idx]
                
                if hasattr(subdataset, 'metadata_df'):
                    label = subdataset.metadata_df.iloc[sample_idx]['Profusion Label']
                    split_class_dist[label] += 1
                else:
                    print(f"WARNING: Subdataset {dataset_idx} does not have metadata_df attribute")
                    
            print(f"\n{split_name.capitalize()} split class distribution:")
            total_in_split = sum(split_class_dist.values())
            for class_label, count in split_class_dist.items():
                percentage = (count / total_in_split) * 100 if total_in_split > 0 else 0
                print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")
    
    elif hasattr(dataset, 'metadata_df'):
        # Handle regular dataset with metadata_df
        for split_name, indices in split_indices.items():
            split_class_dist = {0: 0, 1: 0, 2: 0, 3: 0}
            
            for idx in indices:
                label = dataset.metadata_df.iloc[idx]['Profusion Label']
                split_class_dist[label] += 1
                
            print(f"\n{split_name.capitalize()} split class distribution:")
            total_in_split = sum(split_class_dist.values())
            for class_label, count in split_class_dist.items():
                percentage = (count / total_in_split) * 100 if total_in_split > 0 else 0
                print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")
    else:
        print("Cannot analyze class distribution - dataset does not have metadata_df attribute")

def map_global_index_to_local(concat_dataset, global_idx):
    """
    Maps a global index in a ConcatDataset to (dataset_idx, local_idx)
    
    Args:
        concat_dataset: The ConcatDataset
        global_idx: The global index
        
    Returns:
        tuple: (dataset_idx, local_idx)
    """
    if not hasattr(concat_dataset, 'datasets'):
        raise ValueError("Input must be a ConcatDataset")
    
    dataset_idx = 0
    for dataset in concat_dataset.datasets:
        if global_idx < len(dataset):
            return dataset_idx, global_idx
        global_idx -= len(dataset)
        dataset_idx += 1
    
    raise IndexError("Global index out of range")
# Extract TBA-TBU labels for test sets
def get_tba_tbu_labels(dataset, indices):
    """Extract TBA-TBU labels for given indices"""
    tba_tbu_labels = []
    for idx in indices:
        label = dataset.metadata_df.iloc[idx]['TBA-TBU Label']
        tba_tbu_labels.append(label)
    return np.array(tba_tbu_labels)

def get_prof_labels(dataset, indices):
    """Extract TBA-TBU labels for given indices"""
    prof_labels = []
    for idx in indices:
        label = dataset.metadata_df.iloc[idx]['Profusion Label']
        prof_labels.append(label)
    return np.array(prof_labels)


def plot_combined_conf_mat(predicted_label_name, dataset, predicted_labels, relevant_indices, log_to_wandb=False, dataset_name=""):
    """
    Function to generate and plot a confusion matrix that shows the relationship between
    the predicted profusion labels, true labels, and TB status.

    Parameters:
    - predicted_label_name (str): The name of the predicted label (e.g., 'Profusion').
    - dataset (object): The dataset from which labels are retrieved (d1, d2).
    - predicted_labels (ndarray): The predicted labels (0 or 1).
    - relevant_indices (list or ndarray): The indices of the samples to be used for testing.
    - log_to_wandb (bool): Whether to log the plot to W&B.
    - dataset_name (str): Name of the dataset for logging purposes (e.g., "MBOD 1").
    """
    predicted_labels = np.array(predicted_labels).flatten().astype(int)

    # Get TBA-TBU labels for test sets
    test_tba_tbu = get_tba_tbu_labels(dataset, relevant_indices)

    # Get profusion labels
    test_prof = get_prof_labels(dataset, relevant_indices)

    # Initialize a confusion matrix to store the results (4x2: True/False Positives and Negatives)
    conf_matrix = np.zeros((4, 2), dtype=int)

    # Loop through the different combinations of true labels and TB status
    for true_label in [0, 1]:  # True labels for profusion (0 or 1)
        for tb_status in [0, 1]:  # TB status (TB− or TB+)
            
            # For each TB status, we will classify as TP, FP, TN, FN
            mask_true = (test_prof == true_label) & (test_tba_tbu == tb_status)
            
            for pred_label in [0, 1]:  # Predicted labels for profusion (0 or 1)
                mask_pred = (predicted_labels == pred_label)
                
                # Combine the true and predicted masks
                mask_combined = mask_true & mask_pred
                
                # Increment the appropriate cell in the confusion matrix
                if true_label == 0 and pred_label == 0:
                    conf_matrix[0, tb_status] += np.sum(mask_combined)  # TN
                elif true_label == 0 and pred_label == 1:
                    conf_matrix[1, tb_status] += np.sum(mask_combined)  # FP
                elif true_label == 1 and pred_label == 0:
                    conf_matrix[2, tb_status] += np.sum(mask_combined)  # FN
                elif true_label == 1 and pred_label == 1:
                    conf_matrix[3, tb_status] += np.sum(mask_combined)  # TP

    # Convert the confusion matrix to a DataFrame for easier visualization
    index_labels = ["TN (True 0, Pred 0)", "FP (True 0, Pred 1)", "FN (True 1, Pred 0)", "TP (True 1, Pred 1)"]
    columns = ["TB−", "TB+"]
    df_cm = pd.DataFrame(conf_matrix, index=index_labels, columns=columns)

    # Plot the confusion matrix with the TB status indicated at the top
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True)

    # Increase font size for labels and title
    plt.xlabel("TB Status", fontsize=12)
    plt.ylabel("Profusion ≥ 1/0 Prediction", fontsize=12, labelpad=8)
    plt.title(f"Confusion Matrix with {predicted_label_name} and TB Status - {dataset_name}", fontsize=18)

    # Rotate y-axis labels to read left to right
    plt.yticks(rotation=0)
    
    # Create a filename for saving
    file_name = f"combined_cm_{dataset_name.replace(' ', '_').lower()}.png"
    

    
    # Log to WandB if requested
    if log_to_wandb:
        try:
            import wandb
            wandb.log({
                f"combined_confusion_matrix": wandb.Image(plt)
                # f"{dataset_name}_tb_neg_accuracy": (conf_matrix[0, 0] + conf_matrix[3, 0]) / np.sum(conf_matrix[:, 0]),
                # f"{dataset_name}_tb_pos_accuracy": (conf_matrix[0, 1] + conf_matrix[3, 1]) / np.sum(conf_matrix[:, 1])
            })
        except ImportError:
            print("WandB not available for logging.")
        except Exception as e:
            print(f"Error logging to WandB: {e}")

    # Display the plot
    plt.show()
    
    # Return the confusion matrix for further analysis
    return conf_matrix


def compute_map(embeddings, labels):
    print("Computing mAP...")
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    

    norms_before = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Normalize embeddings to unit vectors
    epsilon = 1e-8
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + epsilon)
    
  
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

def get_semi_hard_negative(anchor_feats, negative_feats, positive_feats, margin):
    """
    Find semi-hard negatives that are farther from the anchor than the positive,
    but still within the margin.
    
    Args:
        anchor_feats: Features of anchor samples (N x D)
        negative_feats: Features of negative samples (N x D)
        positive_feats: Features of positive samples (N x D)
        margin: Margin value from triplet loss
    
    Returns:
        Selected semi-hard negative features with same batch size as input
    """
    with torch.no_grad():
        # Calculate distances
        positive_dist = torch.norm(anchor_feats - positive_feats, dim=1)
        negative_dist = torch.norm(anchor_feats - negative_feats, dim=1)
        
        # Find semi-hard negatives
        semi_hard_mask = (negative_dist > positive_dist) & (negative_dist < positive_dist + margin)
        
        if semi_hard_mask.any():
            # Get indices of semi-hard negatives
            semi_hard_indices = torch.where(semi_hard_mask)[0]
            
            # If we don't have enough semi-hard negatives, pad with original negatives
            if len(semi_hard_indices) < len(anchor_feats):
                num_needed = len(anchor_feats) - len(semi_hard_indices)
                # Get indices of non-semi-hard negatives
                non_semi_hard_indices = torch.where(~semi_hard_mask)[0]
                # Randomly select from non-semi-hard if available
                if len(non_semi_hard_indices) > 0:
                    padding_indices = non_semi_hard_indices[torch.randperm(len(non_semi_hard_indices))[:num_needed]]
                    selected_indices = torch.cat([semi_hard_indices, padding_indices])
                else:
                    # If no non-semi-hard negatives, repeat semi-hard ones
                    padding_indices = semi_hard_indices[torch.randperm(len(semi_hard_indices))[:num_needed]]
                    selected_indices = torch.cat([semi_hard_indices, padding_indices])
            else:
                # If we have more semi-hard negatives than needed, randomly select subset
                selected_indices = semi_hard_indices[torch.randperm(len(semi_hard_indices))[:len(anchor_feats)]]
            
            return negative_feats[selected_indices]
        else:
            # If no semi-hard negatives found, return original negatives
            return negative_feats
        


# Function to compute Precision@k, Recall@k, F1@k
def compute_top_k_metrics(embeddings, labels, knn_model, k=4):
    """
    Compute precision, recall, and F1 at k for KNN embeddings with standard IR definitions.
    """
    distances, indices = knn_model.kneighbors(embeddings, n_neighbors=k)
    training_labels = knn_model._y
    
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    for i in range(len(embeddings)):
        true_label = labels[i]
        neighbor_labels = training_labels[indices[i]]
        
        # Count correct predictions among top k
        correct = np.sum(neighbor_labels == true_label)
        
        # Standard precision@k
        precision = correct / k
        
        # Standard binary recall@k (was the relevant item found?)
        # For classification with 1 correct answer per query, this is the standard definition
        recall = 1.0 if correct > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
    
    return (
        np.mean(all_precisions),
        np.mean(all_recalls),
        np.mean(all_f1s)
    )
def plot_decision_boundaries(ax, embeddings_2d, labels, preds, title):
    # Define mesh grid boundaries
    x_min, x_max = embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1
    y_min, y_max = embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1
    h = 0.2  # Grid step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create KNN classifier on the 2D t-SNE space and predict mesh grid
    knn = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
    knn_viz = knn(n_neighbors=4, metric='euclidean')
    knn_viz.fit(embeddings_2d, labels)
    Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    
    # Plot data points
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        c=labels, edgecolors='k', cmap='viridis', s=40, alpha=0.8)
    
    # Mark incorrect predictions
    misclassified = labels != preds
    ax.scatter(embeddings_2d[misclassified, 0], embeddings_2d[misclassified, 1],
              marker='x', c='red', s=100, label='Misclassified')
    
    ax.set_title(title)
    return scatter