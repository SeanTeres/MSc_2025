import pydicom
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import torch.nn.functional as F
from sklearn import metrics

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


def compute_pairwise_distances(embeddings, metric='cosine'):
    """
    Compute pairwise distances between a set of embeddings.

    Parameters:
    - embeddings: Tensor of shape (batch_size, embedding_dim)
    - metric: Distance metric to use ('cosine' or 'euclidean')

    Returns:
    - dist_matrix: Pairwise distance matrix of shape (batch_size, batch_size)
    """
    batch_size = embeddings.size(0)

    if metric == 'cosine':
        # Normalize the embeddings to unit vectors
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        dist_matrix = 1 - torch.mm(embeddings_normalized, embeddings_normalized.t())  # Cosine distance is 1 - similarity
    elif metric == 'euclidean':
        dist_matrix = torch.cdist(embeddings, embeddings)  # Euclidean distance
    else:
        raise ValueError("Metric must be 'cosine' or 'euclidean'.")

    return dist_matrix


def compute_map_per_class(embeddings, labels):
    from sklearn.metrics import average_precision_score
    import numpy as np
    from collections import defaultdict

    # Normalize embeddings (redundant if already normalized)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T)  # [N, N]
    labels = labels.numpy()
    sim_matrix = sim_matrix.numpy()

    class_to_aps = defaultdict(list)

    for i in range(len(labels)):
        current_label = labels[i]
        true = (labels == current_label).astype(np.int32)
        pred = sim_matrix[i]

        # Remove self-comparison
        true = np.delete(true, i)
        pred = np.delete(pred, i)

        if true.sum() == 0:
            continue

        ap = average_precision_score(true, pred)
        class_to_aps[current_label].append(ap)

    # Average APs per class
    class_map = {}
    for class_id, aps in class_to_aps.items():
        if len(aps) > 0:
            class_map[int(class_id)] = np.mean(aps)

    overall_map = np.mean(list(class_map.values()))
    
    return overall_map, class_map

# After initializing your dataset and before creating the model
def calculate_class_weights(train_loader, num_classes=4, dampening_factor=0.5):
    """
    Calculate class weights for light oversampling.
    
    Args:
        train_loader: DataLoader for training data
        num_classes: Number of classes in the dataset
        dampening_factor: Controls the strength of oversampling (0-1)
                          0 = equal weights, 1 = fully inverse weights
    
    Returns:
        torch.Tensor: Tensor of class weights
    """
    # Count instances of each class
    class_counts = torch.zeros(num_classes)
    
    print("Counting class distribution...")
    for batch in train_loader:
        labels = batch[1]
        for i in range(num_classes):
            class_counts[i] += (labels == i).sum().item()
    
    total_samples = class_counts.sum().item()
    class_frequencies = class_counts / total_samples
    
    # Calculate weights (inversely proportional to frequency)
    raw_weights = 1.0 / class_frequencies
    
    # Normalize weights to sum to num_classes
    normalized_weights = raw_weights * (num_classes / raw_weights.sum())
    
    # Apply dampening factor for lighter oversampling
    dampened_weights = 1.0 + dampening_factor * (normalized_weights - 1.0)
    
    print("Class distribution:")
    for i in range(num_classes):
        print(f"Class {i}: {class_counts[i]} samples ({class_frequencies[i]:.2%})")
    
    print("\nCalculated weights:")
    for i in range(num_classes):
        print(f"Class {i}: raw={raw_weights[i]:.4f}, dampened={dampened_weights[i]:.4f}")
    
    return dampened_weights

def compute_sensitivity_at_specificity(embeddings, labels, specificity_target=0.95):
    """
    Compute sensitivity at a given specificity threshold for each class.
    
    Args:
        embeddings: Normalized embeddings tensor
        labels: Ground truth labels
        specificity_target: Target specificity value (default: 0.95)
    
    Returns:
        overall_sensitivity: Average sensitivity across all classes
        class_sensitivities: Dictionary with sensitivity for each class
    """
    # Convert to numpy for sklearn
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Get unique class labels
    unique_labels = np.unique(labels_np)
    n_classes = len(unique_labels)
    
    # Calculate similarity matrix (cosine similarity)
    sim_matrix = np.matmul(embeddings_np, embeddings_np.T)
    
    class_sensitivities = {}
    class_counts = []
    
    # For each class
    for class_label in unique_labels:
        # Identify samples of this class
        class_mask = (labels_np == class_label)
        class_indices = np.where(class_mask)[0]
        
        # Skip if too few samples
        if len(class_indices) < 2:
            class_sensitivities[int(class_label)] = 0.0
            continue
            
        # Calculate sensitivities for each sample in this class
        sensitivities = []
        
        for idx in class_indices:
            # Create binary labels (1 for same class, 0 for different class)
            # Exclude the current sample itself
            other_indices = np.arange(len(labels_np))
            other_indices = other_indices[other_indices != idx]
            
            binary_labels = (labels_np[other_indices] == class_label).astype(int)
            scores = sim_matrix[idx, other_indices]
            
            # Skip if no positive or negative samples
            if np.sum(binary_labels) == 0 or np.sum(binary_labels) == len(binary_labels):
                continue
                
            # Calculate ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(binary_labels, scores)
            
            # Get specificity = 1 - fpr
            specificity = 1 - fpr
            
            # Find threshold closest to target specificity without going below it
            valid_indices = np.where(specificity >= specificity_target)[0]
            if len(valid_indices) > 0:
                # Get index where specificity is closest to target
                closest_idx = valid_indices[np.argmin(np.abs(specificity[valid_indices] - specificity_target))]
                sample_sensitivity = tpr[closest_idx]
                sensitivities.append(sample_sensitivity)
        
        # Calculate mean sensitivity for this class
        if sensitivities:
            class_sensitivities[int(class_label)] = float(np.mean(sensitivities))
            class_counts.append(len(sensitivities))
        else:
            class_sensitivities[int(class_label)] = 0.0
    
    # Calculate overall sensitivity, weighted by number of samples per class
    valid_sensitivities = [sens for cls, sens in class_sensitivities.items() if sens > 0]
    if valid_sensitivities:
        overall_sensitivity = float(np.mean(valid_sensitivities))
    else:
        overall_sensitivity = 0.0
    
    return overall_sensitivity, class_sensitivities

def compute_specificity_at_sensitivity(embeddings, labels, sensitivity_target=0.90):
    """
    Compute specificity at a given sensitivity threshold for each class.
    
    Args:
        embeddings: Normalized embeddings tensor
        labels: Ground truth labels
        sensitivity_target: Target sensitivity value (default: 0.90)
    
    Returns:
        overall_specificity: Average specificity across all classes
        class_specificities: Dictionary with specificity for each class
    """
    # Convert to numpy for sklearn
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Get unique class labels
    unique_labels = np.unique(labels_np)
    
    # Calculate similarity matrix (cosine similarity)
    sim_matrix = np.matmul(embeddings_np, embeddings_np.T)
    
    class_specificities = {}
    
    # For each class
    for class_label in unique_labels:
        # Identify samples of this class
        class_mask = (labels_np == class_label)
        class_indices = np.where(class_mask)[0]
        
        # Skip if too few samples
        if len(class_indices) < 2:
            class_specificities[int(class_label)] = 0.0
            continue
            
        # Calculate specificities for each sample in this class
        specificities = []
        
        for idx in class_indices:
            # Create binary labels (1 for same class, 0 for different class)
            # Exclude the current sample itself
            other_indices = np.arange(len(labels_np))
            other_indices = other_indices[other_indices != idx]
            
            binary_labels = (labels_np[other_indices] == class_label).astype(int)
            scores = sim_matrix[idx, other_indices]
            
            # Skip if no positive or negative samples
            if np.sum(binary_labels) == 0 or np.sum(binary_labels) == len(binary_labels):
                continue
                
            # Calculate ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(binary_labels, scores)
            
            # Get specificity = 1 - fpr
            specificity = 1 - fpr
            
            # Find threshold closest to target sensitivity without going below it
            valid_indices = np.where(tpr >= sensitivity_target)[0]
            if len(valid_indices) > 0:
                # Get index where sensitivity (tpr) is closest to target
                closest_idx = valid_indices[np.argmin(np.abs(tpr[valid_indices] - sensitivity_target))]
                sample_specificity = specificity[closest_idx]
                specificities.append(sample_specificity)
        
        # Calculate mean specificity for this class
        if specificities:
            class_specificities[int(class_label)] = float(np.mean(specificities))
        else:
            class_specificities[int(class_label)] = 0.0
    
    # Calculate overall specificity, unweighted average across classes with valid measurements
    valid_specificities = [spec for cls, spec in class_specificities.items() if spec > 0]
    if valid_specificities:
        overall_specificity = float(np.mean(valid_specificities))
    else:
        overall_specificity = 0.0
    
    return overall_specificity, class_specificities