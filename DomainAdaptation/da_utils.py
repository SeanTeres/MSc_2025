
import sys
import os
import yaml
# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
import numpy as np
import matplotlib.pyplot as plt
from datasets.dataloader import get_dataloaders, get_dataloaders_with_files
import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import seaborn as sns
import os
from sklearn.manifold import TSNE
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity




multiclass_stb_mapping = {
    0: "Profusion 0, No TB",
    1: "Profusion 1, No TB",
    2: "Profusion 2, No TB",
    3: "Profusion 3, No TB",
    4: "Profusion 0, With TB",
    5: "Profusion 1, With TB",
    6: "Profusion 2, With TB",
    7: "Profusion 3, With TB"
}

multilabel_stb_mapping = {
    (0,0,0,0,1): "No Finding",
    (1,0,0,0,0): "Profusion 1, No TB",
    (1,1,0,0,0): "Profusion 2, No TB",
    (1,1,1,0,0): "Profusion 3, No TB",
    (0,0,0,1,0): "Profusion 0, With TB",
    (1,0,0,1,0): "Profusion 1, With TB",
    (1,1,0,1,0): "Profusion 2, With TB",
    (1,1,1,1,0): "Profusion 3, With TB"
}

multilabel_to_multiclass = {
    (0,0,0,0,1): 0,  # No Finding (Profusion 0, No TB)
    (1,0,0,0,0): 1,  # Profusion 1, No TB
    (1,1,0,0,0): 2,  # Profusion 2, No TB
    (1,1,1,0,0): 3,  # Profusion 3, No TB
    (0,0,0,1,0): 4,  # Profusion 0, With TB
    (1,0,0,1,0): 5,  # Profusion 1, With TB
    (1,1,0,1,0): 6,  # Profusion 2, With TB
    (1,1,1,1,0): 7,  # Profusion 3, With TB
}

def load_config(config_path="config.yaml"):
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
    Returns:
        A dictionary containing the configuration settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error reading the configuration file: {e}")

# Add mining statistics
def analyze_mined_triplets(anchors, positives, negatives, labels):
    anchor_labels = labels[anchors]
    
    tb_pos_anchors = (anchor_labels == 1).sum()
    tb_neg_anchors = (anchor_labels == 0).sum()
    
    print(f"TB+ anchors: {tb_pos_anchors} ({tb_pos_anchors/len(anchors)*100:.1f}%)")
    print(f"TB- anchors: {tb_neg_anchors} ({tb_neg_anchors/len(anchors)*100:.1f}%)")
    
    return {
        'tb_pos_anchor_ratio': tb_pos_anchors.item() / len(anchors),
        'tb_neg_anchor_ratio': tb_neg_anchors.item() / len(anchors)
    }

class ForeverDataIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(data_loader)

    def __next__(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            batch = next(self.iter)
        return batch
    
def reinitialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)



def multilabel_stb_to_multiclass(label):
    """Convert a multilabel_stb vector to multiclass_stb integer."""

    # multiclass_labels = [multilabel_stb_to_multiclass(label) for label in multilabels.tolist()] ----> USAGE

    return multilabel_to_multiclass.get(tuple(label), -1)  # -1 for unknown/invalid

def visualize_tsne_with_kaggle_tb(model, device, experiment_name, ilo_dataset, mbod_loader, tb_loader=None,
                trained=False, log_to_wandb=False, n_epochs=0, set_name="Training", entire_dataset=False, special_filename=None, tb_dataset_name="TB-Net", convert_multilabel=False):
    print("Starting t-SNE visualization generation...")
    model.eval()

    mbod_feats, mbod_labels = [], []
    print("Processing MBOD batches...")
    for batch in mbod_loader:
        imgs, labels = batch[0].to(device), batch[1].cpu().numpy()

        if convert_multilabel:
            # Convert multilabel to multiclass (0-3)
            labels = [multilabel_stb_to_multiclass(label) for label in labels.tolist()]
            labels = np.array(labels)
        
        with torch.no_grad():
            feats = model.features(imgs)
            feats = torch.flatten(feats, start_dim=1)
        mbod_feats.append(feats.cpu().numpy())
        mbod_labels.append(labels)
    mbod_feats = np.concatenate(mbod_feats, axis=0)
    mbod_labels = np.concatenate(mbod_labels, axis=0)
    print(f"Processed {len(mbod_feats)} MBOD images.")

    ilo_feats, ilo_labels = [], []
    if ilo_dataset is not None:
        print(f"\nProcessing {len(ilo_dataset)} ILO images...\n")
        for idx in range(len(ilo_dataset)):
            sample = ilo_dataset[idx]
            img = sample[0].unsqueeze(0).to(device)
            label = sample[1]
            label = label.item() if isinstance(label, torch.Tensor) else float(label)

            with torch.no_grad():
                feats = model.features(img)
                feats = torch.flatten(feats, start_dim=1)

            ilo_feats.append(feats.cpu().numpy())
            ilo_labels.append(label)

        ilo_feats = np.concatenate(ilo_feats, axis=0)
        ilo_labels = np.array(ilo_labels)
        print(f"Processed {len(ilo_feats)} ILO images.")



    tb_feats, tb_labels = [], []
    if tb_loader is not None:
        print("Processing TB batches...")
        for batch in tb_loader:
            imgs, labels = batch[0].to(device), batch[1].cpu().numpy()
            with torch.no_grad():
                feats = model.features(imgs)
                feats = torch.flatten(feats, start_dim=1)
            tb_feats.append(feats.cpu().numpy())
            tb_labels.append(labels)
        tb_feats = np.concatenate(tb_feats, axis=0)
        tb_labels = np.concatenate(tb_labels, axis=0)
        print(f"Processed {len(tb_feats)} TB images.")

    # Combine all features and labels
    all_feats = np.concatenate([ilo_feats, mbod_feats] + ([tb_feats] if tb_loader else []), axis=0)

    # Track sources separately
    sources = (
        ['ILO'] * len(ilo_feats) +
        ['MBOD'] * len(mbod_feats) +
        (['TB'] * len(tb_feats) if tb_loader else [])
    )

    # Save labels separately for MBOD/ILO and TB
    all_labels = np.concatenate([ilo_labels, mbod_labels] + ([tb_labels] if tb_loader else []), axis=0)
    tb_indices = [i for i, s in enumerate(sources) if s == 'TB']
    non_tb_indices = [i for i, s in enumerate(sources) if s != 'TB']

    # t-SNE
    print("Fitting t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, verbose=0)
    all_feats_2d = tsne.fit_transform(all_feats)

    os.makedirs('visualizations', exist_ok=True)

    # Profusion score colormap (e.g. tab10)
    profusion_labels = np.unique([l for i, l in enumerate(all_labels) if sources[i] != 'TB'])
    profusion_cmap = plt.cm.get_cmap('tab10', max(len(profusion_labels), 10))
    profusion_colors = {label: profusion_cmap(i % 10) for i, label in enumerate(profusion_labels)}

    # TB label colormap (use different colormap like Set1)
    tb_cmap = plt.cm.get_cmap('Pastel2', 4)
    tb_colors = {0: tb_cmap(0), 1: tb_cmap(3)}

    tb_net_colors = {
        0: '#9467bd',  # TB negative (purple)
        1: '#e377c2',  # TB positive (pink)
    }


    tab10_colors = [
        '#1f77b4',  # Profusion 0 (blue)
        '#ff7f0e',  # Profusion 1 (orange)
        '#2ca02c',  # Profusion 2 (green)
        '#d62728',  # Profusion 3 (red)
    ]

    plt.figure(figsize=(14, 10))

    # Plot ILO & MBOD
    for label in profusion_labels:
        prof_label = label % 4
        for source, marker, size, alpha in [('ILO', '*', 120, 0.8), ('MBOD', 'o', 40, 0.6)]:
            idx = [
                i for i in range(len(all_labels))
                if all_labels[i] == label and sources[i] == source
            ]
            if idx:
                
                coords = all_feats_2d[idx]
                if source == 'MBOD':
                    plt.scatter(
                        coords[:, 0], coords[:, 1],
                        c=[profusion_colors[prof_label]],
                        marker=marker if label >= 4 else 'x',
                        s=size,
                        label=f'{source} - Profusion {int(prof_label)} (TB+)' if label >= 4 else f'{source} - Profusion {int(prof_label)} (TB-)',
                        alpha=alpha,
                        edgecolors='black' if source == 'ILO' else 'white',
                        linewidths=0.5
                    )
                elif source == 'ILO':
                    plt.scatter(
                        coords[:, 0], coords[:, 1],
                        c=[profusion_colors[prof_label]],
                        marker=marker,
                        s=size,
                        label=f'{source} - Profusion {int(prof_label)} (TB-)',
                        alpha=alpha,
                        edgecolors='black',
                        linewidths=0.5
                    )

    # Plot TB
    if tb_loader is not None:
        for tb_label in [0, 1]:
            idx = [i for i in tb_indices if all_labels[i] == tb_label]
            if idx:
                coords = all_feats_2d[idx]
                plt.scatter(
                    coords[:, 0], coords[:, 1],
                    c=[tb_net_colors[tb_label]],
                    marker='x' if tb_label == 0 else 'o',  # X for TB negative, O for TB positive
                    s=40,
                    label=f'{tb_dataset_name} (TB-)' if tb_label == 0 else f'{tb_dataset_name} (TB+)',
                    alpha=1.0,
                    edgecolors='white',
                    linewidths=0.7
                )

    title_to_add = "Entire Dataset" if entire_dataset else ""
    plt.title(f"t-SNE Visualization {'(Trained Model)' if trained else '(Untrained Model)'}\n{title_to_add} - Epoch {n_epochs}\n {experiment_name}", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.grid(True, alpha=0.3)

    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)

    file_name = f"visualizations/tsne_all_sources{'_trained' if trained else '_untrained'}.png"
    plt.tight_layout()

    if special_filename is not None:
        file_name = f"visualizations/tsne_all_sources_{special_filename}{'_trained' if trained else '_untrained'}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE visualizations to {file_name}")

    if log_to_wandb:
        print("Logging t-SNE visualizations to wandb...")
        wandb.log({
            f"{set_name} tsne": wandb.Image(file_name)
        })
        print("Logged visualizations to wandb successfully")

    return file_name

# Add this function to collect features
def collect_domain_features(model, dataloader, device, max_batches=10):
    """Collect features from a dataloader"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(dataloader):
            if i >= max_batches:
                break
                
            imgs = imgs.to(device)
            feats = model.features(imgs)
            features.append(feats.cpu())
            labels.append(lbls)
    
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)




def plot_interdomain_distances(src_embs, tgt_embs, distance_obj=None, distance_type="L2", 
                              normalize=True, bins=100, epoch=None, 
                              title="Inter-Domain Distance Distribution",
                              log_to_wandb=True, max_samples=10000, fig_size=(10, 6)):
    """
    Plot and analyze the distribution of distances between source and target embeddings.
    
    Args:
        src_embs (torch.Tensor): Source domain embeddings
        tgt_embs (torch.Tensor): Target domain embeddings
        distance_obj: Distance metric object (like LpDistance or CosineSimilarity)
        distance_type (str): Type of distance when no object provided ('L2', 'cosine')
        normalize (bool): Whether to normalize embeddings
        bins (int): Number of bins for histogram
        epoch (int): Current epoch number for logging
        title (str): Plot title
        log_to_wandb (bool): Whether to log plot to wandb
        max_samples (int): Max number of pairs to compute (prevents OOM)
        fig_size (tuple): Figure size
    
    Returns:
        dict: Dictionary with distance statistics
    """
    with torch.no_grad():
        # Move to CPU and normalize if needed
        src_embs = src_embs.detach().cpu()
        tgt_embs = tgt_embs.detach().cpu()
        
        if normalize:
            src_embs = F.normalize(src_embs, p=2, dim=1)
            tgt_embs = F.normalize(tgt_embs, p=2, dim=1)
        
        # Sample if needed to prevent memory issues
        if src_embs.size(0) * tgt_embs.size(0) > max_samples:
            src_idx = torch.randperm(src_embs.size(0))[:int(np.sqrt(max_samples))]
            tgt_idx = torch.randperm(tgt_embs.size(0))[:int(np.sqrt(max_samples))]
            src_embs = src_embs[src_idx]
            tgt_embs = tgt_embs[tgt_idx]
            
        # Calculate distances
        if distance_obj is not None:
            # Use the provided distance object
            if isinstance(distance_obj, CosineSimilarity):
                # For cosine similarity, convert to distance (1 - similarity)
                distances = 1 - distance_obj(src_embs, tgt_embs)
                metric_name = "Cosine Distance"
            else:
                distances = distance_obj(src_embs, tgt_embs)
                if isinstance(distance_obj, LpDistance):
                    if distance_obj.p == 2:
                        if distance_obj.power == 2:
                            metric_name = "Squared L2 Distance"
                        else:
                            metric_name = "L2 Distance"
                    else:
                        metric_name = f"Lp Distance (p={distance_obj.p})"
                else:
                    metric_name = "Distance"
        else:
            # Calculate distances directly
            if distance_type == "cosine":
                similarities = torch.mm(src_embs, tgt_embs.t())
                distances = 1 - similarities
                metric_name = "Cosine Distance"
            else:
                distances = torch.cdist(src_embs, tgt_embs, p=2)
                metric_name = "L2 Distance"
        
        # Flatten and convert to numpy
        dist_values = distances.flatten().numpy()
        
        # Calculate statistics
        stats = {
            "min": float(np.min(dist_values)),
            "max": float(np.max(dist_values)),
            "mean": float(np.mean(dist_values)),
            "median": float(np.median(dist_values)),
            "std": float(np.std(dist_values)),
            "q1": float(np.percentile(dist_values, 25)),
            "q3": float(np.percentile(dist_values, 75))
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Plot histogram with KDE
        sns.histplot(
            dist_values, bins=bins, kde=True, 
            color='#3498db', alpha=0.7, 
            edgecolor='black', linewidth=0.5,
            ax=ax
        )
        
        # Add vertical lines for key statistics
        ax.axvline(stats["mean"], color='red', linestyle='-', linewidth=2, alpha=0.7, 
                  label=f'Mean: {stats["mean"]:.4f}')
        ax.axvline(stats["median"], color='green', linestyle='--', linewidth=2, alpha=0.7, 
                  label=f'Median: {stats["median"]:.4f}')
        
        # Add stats box
        stats_text = (f"Min: {stats['min']:.4f}\n"
                     f"Max: {stats['max']:.4f}\n"
                     f"Mean: {stats['mean']:.4f}\n"
                     f"Median: {stats['median']:.4f}\n"
                     f"Std: {stats['std']:.4f}\n"
                     f"Q1: {stats['q1']:.4f}\n"
                     f"Q3: {stats['q3']:.4f}")
        
        # Add text box with stats
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Set labels and title
        epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title}{epoch_str}")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Log to wandb if requested
        if log_to_wandb:
            if epoch is not None:
                wandb.log({
                    f"distance_distribution/epoch_{epoch}": wandb.Image(fig),
                    "distance_stats/mean": stats["mean"],
                    "distance_stats/median": stats["median"],
                    "distance_stats/std": stats["std"]
                }, step=epoch)
            else:
                wandb.log({
                    "distance_distribution": wandb.Image(fig),
                    "distance_stats/mean": stats["mean"],
                    "distance_stats/median": stats["median"],
                    "distance_stats/std": stats["std"]
                })
        
        plt.tight_layout()
        return fig, stats