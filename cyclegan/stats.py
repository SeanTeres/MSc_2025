import numpy as np
import h5py
from scipy import ndimage
import torch

def compute_dataset_stats(hdf5_path, images_key="images"):
    """Compute dataset-wide statistics for normalization"""
    with h5py.File(hdf5_path, "r") as f:
        # Calculate on full dataset or sample for very large datasets
        imgs = f[images_key][:]
        min_val = float(imgs.min())
        max_val = float(imgs.max())
        mean_val = float(imgs.mean())
        std_val = float(imgs.std())
    
    print(f"Dataset {hdf5_path} stats:")
    print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")
    
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val
    }

def create_dataset_normalizer(stats):
    """Create a normalization function using pre-computed dataset statistics"""
    def normalize_with_dataset_stats(img_tensor):
        # Min-max normalization using dataset-wide statistics
        normalized = (img_tensor - stats['min']) / (stats['max'] - stats['min'])
        return normalized
    return normalize_with_dataset_stats

def create_dataset_standardizer(stats):
    """Create a standardization function using pre-computed dataset statistics"""
    def standardize_with_dataset_stats(img_tensor):
        # Z-score standardization using dataset-wide statistics
        standardized = (img_tensor - stats['mean']) / (stats['std'] + 1e-8)  # Add epsilon to avoid division by zero
        return standardized
    return standardize_with_dataset_stats

def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight for interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    
    # Interpolated images
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    # Get discriminator output
    d_interpolates = D(interpolates)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def normalize_to_hu_range(img_tensor):
    """Normalize image tensor to Hounsfield Unit range (-1024, 1024)"""
    # Scale to [-1024, 1024]
    return img_tensor * 2048 - 1024

def normalize_to_range_0_1(img_tensor):
    """Normalize image tensor to range [0,1] based on its min and max values"""
    batch_min = img_tensor.min()
    batch_max = img_tensor.max()
    
    # Avoid division by zero if the image is constant
    if batch_max - batch_min == 0:
        return torch.zeros_like(img_tensor)
    
    normalized = (img_tensor - batch_min) / (batch_max - batch_min)
    return normalized

def apply_log_filter(img_tensor):
    """Apply Laplacian of Gaussian filter to the image tensor"""
    # Convert to numpy for processing
    img_np = img_tensor.squeeze(0).numpy()
    
    # Apply LoG filter (sigma controls the amount of blurring)
    log_img = ndimage.gaussian_laplace(img_np, sigma=0.2)
    
    # Normalize the LoG output to [0,1]
    log_min, log_max = log_img.min(), log_img.max()
    if log_max > log_min:
        log_img = (log_img - log_min) / (log_max - log_min)
    
    # Convert back to tensor with channel dimension
    return torch.from_numpy(log_img).float().unsqueeze(0)


def plot_normalization_comparison(mbod_path, kaggle_path, mbod_stats, kaggle_stats, dataset_names, log_to_wandb=False):
    """
    Plot histograms of pixel value distributions before and after normalization.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load sample images from each dataset
    with h5py.File(mbod_path, "r") as f:
        # Sample 100 images or fewer if dataset is smaller
        sample_size = min(100, f["images"].shape[0])
        mbod_samples = f["images"][:sample_size].astype(np.float32)
    
    with h5py.File(kaggle_path, "r") as f:
        sample_size = min(100, f["images"].shape[0])
        kaggle_samples = f["images"][:sample_size].astype(np.float32)
    
    # Create normalizer functions
    normalize_mbod = create_dataset_normalizer(mbod_stats)
    normalize_kaggle = create_dataset_normalizer(kaggle_stats)
    
    # Create figure for comparison
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot original distributions
    axs[0, 0].hist(mbod_samples.flatten(), bins=100, alpha=0.6, density=True, label='MBOD')
    axs[0, 0].hist(kaggle_samples.flatten(), bins=100, alpha=0.6, density=True, label='Kaggle')
    axs[0, 0].set_title('Original Pixel Distributions', fontsize=14)
    axs[0, 0].set_xlabel('Pixel Values')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].legend()
    
    # Apply dataset-specific normalization
    mbod_tensors = [torch.from_numpy(img) for img in mbod_samples]
    kaggle_tensors = [torch.from_numpy(img) for img in kaggle_samples]
    
    mbod_normalized = np.array([normalize_mbod(img).numpy() for img in mbod_tensors])
    kaggle_normalized = np.array([normalize_kaggle(img).numpy() for img in kaggle_tensors])
    
    # Plot after dataset normalization
    axs[0, 1].hist(mbod_normalized.flatten(), bins=100, alpha=0.6, density=True, label='MBOD Normalized')
    axs[0, 1].hist(kaggle_normalized.flatten(), bins=100, alpha=0.6, density=True, label='Kaggle Normalized')
    axs[0, 1].set_title('After Dataset-Wide Normalization', fontsize=14)
    axs[0, 1].set_xlabel('Normalized Values (0-1)')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()
    
    # Apply per-image normalization for comparison
    def per_image_normalize(img):
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min == 0:
            return np.zeros_like(img)
        return (img - img_min) / (img_max - img_min)
    
    mbod_per_img = np.array([per_image_normalize(img) for img in mbod_samples])
    kaggle_per_img = np.array([per_image_normalize(img) for img in kaggle_samples])
    
    # Plot after per-image normalization
    axs[1, 0].hist(mbod_per_img.flatten(), bins=100, alpha=0.6, density=True, label='MBOD Per-image')
    axs[1, 0].hist(kaggle_per_img.flatten(), bins=100, alpha=0.6, density=True, label='Kaggle Per-image')
    axs[1, 0].set_title('After Per-image Normalization', fontsize=14)
    axs[1, 0].set_xlabel('Normalized Values (0-1)')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].legend()
    
    # Apply standardization
    standardize_mbod = create_dataset_standardizer(mbod_stats)
    standardize_kaggle = create_dataset_standardizer(kaggle_stats)
    
    mbod_std = np.array([standardize_mbod(img).numpy() for img in mbod_tensors])
    kaggle_std = np.array([standardize_kaggle(img).numpy() for img in kaggle_tensors])
    
    # Plot after standardization
    axs[1, 1].hist(mbod_std.flatten(), bins=100, alpha=0.6, density=True, label='MBOD Standardized')
    axs[1, 1].hist(kaggle_std.flatten(), bins=100, alpha=0.6, density=True, label='Kaggle Standardized')
    axs[1, 1].set_title('After Z-score Standardization', fontsize=14)
    axs[1, 1].set_xlabel('Standardized Values')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].legend()

    if log_to_wandb:
        import wandb
        wandb.log({f"{dataset_names}-dist_comparison": wandb.Image(fig)})
        print("Logged normalization comparison to Weights & Biases.")
    
    plt.tight_layout()
    plt.savefig(f'{dataset_names}-normalization_comparison.png', dpi=300)
    plt.show()

    print(f"Normalization comparison saved to '{dataset_names}-normalization_comparison.png'")

