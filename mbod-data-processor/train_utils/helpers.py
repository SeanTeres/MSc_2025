import pydicom
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import torch.nn.functional as F

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

def select_all_negative_samples(img, lab):
    """
    For each image in the batch, select all non-matching samples (images with a different label).
    
    Parameters:
    - img: Tensor of shape (batch_size, C, H, W), batch of images
    - lab: Tensor of shape (batch_size,), labels corresponding to the images
    
    Returns:
    - all_negative_images: List of tensors where each element is a tensor of non-matching images for each image in the batch
    - all_negative_labels: List of tensors where each element is a tensor of labels corresponding to the non-matching images
    """
    all_negative_images = []
    all_negative_labels = []
    
    # Iterate over the images and labels
    for i, batch_label in enumerate(lab):
        # Get indices of all images that do not have the same label as the current image
        non_matching_indices = torch.nonzero(lab != batch_label).squeeze(1).to(img.device)

        # Get all non-matching images and their corresponding labels
        negative_images = img[non_matching_indices]
        negative_labels = lab[non_matching_indices]
        
        # Append to the result lists
        all_negative_images.append(negative_images)
        all_negative_labels.append(negative_labels)
    
    return non_matching_indices

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