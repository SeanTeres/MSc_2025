import sys
import os

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

import torch.utils
import torch.utils.data
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datasets.dataloader import get_dataloaders, get_dataloaders_with_files
import torchxrayvision as xrv
import torch
from train_utils import classes, helpers
import torch.nn.functional as F
import torch.nn as nn
import wandb
import seaborn as sns
import io
import torchvision.transforms as transforms
import os
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MultiClassBaseClassifier(nn.Module):
    def __init__(self, in_features, num_classes=4):
        """
        Multi-class classifier for pneumoconiosis profusion scoring.

        Args:
            in_features (int): Number of input features from the backbone model
            num_classes (int): Number of classes for classification (default: 4 for profusion levels 0-3)
        """
        super(MultiClassBaseClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)  # Output layer

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function for logits
        return x





def visualize_tsne(model, device, ilo_dataset, mbod_loader, trained=False, log_to_wandb=False, n_epochs=0, set_name="Training", entire_dataset=False, is_mstb=False, color_by_profusion=False):
    """
    Generate t-SNE visualization for comparing embeddings from ILO and MBOD datasets
    
    Args:
        model: The model to extract features with
        device: Device to run computations on (cpu/cuda)
        ilo_dataset: Dataset of ILO images (same format as MBOD but no DataLoader)
        mbod_loader: DataLoader for MBOD dataset
        trained: Whether the model is trained or not (for filename)
        log_to_wandb: Whether to log results to W&B
        n_epochs: Number of epochs the model was trained (for title)
        color_by_profusion: Whether to color by profusion score (0-3) instead of full class labels
    """
    print("Starting t-SNE visualization generation...")
    model.eval()

    # Profusion score mapping
    multiclass_stb_mapping = {
        0: "Profusion 0, No TB",
        1: "Profusion 1, No TB",
        2: "Profusion 2, No TB",
        3: "Profusion 3, No TB",
        4: "Profusion 0, With TB",
        5: "Profusion 1, With TB",
        6: "Profusion 2, With TB",
        7: "Profusion 3, With TB",
    }
    
    # Fixed colors for profusion scores 0-3 using the first 4 colors of tab10
    profusion_colors = {
        0: '#1f77b4',  # Profusion 0 (blue)
        1: '#ff7f0e',  # Profusion 1 (orange)
        2: '#2ca02c',  # Profusion 2 (green)
        3: '#d62728',  # Profusion 3 (red)
    }

    ilo_feats = []
    ilo_labels = []

    # Process ILO dataset
    if ilo_dataset is not None:
        print(f"\nProcessing {len(ilo_dataset)} ILO images...\n")
        for idx in range(len(ilo_dataset)):
            sample = ilo_dataset[idx]
            img = sample[0].unsqueeze(0).to(device)  # Add batch dimension
            label = sample[1]
            # print(f"label: {label}")

            if isinstance(label, (torch.Tensor, np.ndarray)):
                label = label.item() if hasattr(label, 'item') else float(label)

            with torch.no_grad():
                feats = model.features(img)
                feats = torch.flatten(feats, start_dim=1)

            ilo_feats.append(feats.cpu().numpy())
            ilo_labels.append(label)

        ilo_feats = np.concatenate(ilo_feats, axis=0)
        ilo_labels = np.array(ilo_labels)
        print(f"Processed {len(ilo_feats)} ILO images.")

    mbod_feats = []
    mbod_labels = []

    # Process MBOD DataLoader
    print("Processing MBOD batches...")
    for batch in mbod_loader:
        imgs = batch[0].to(device)  # Add channel dim if missing
        labels = batch[1]
        


        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        if is_mstb:
            labels = labels % 4

            
        with torch.no_grad():
            feats = model.features(imgs)
            feats = torch.flatten(feats, start_dim=1)

        mbod_feats.append(feats.cpu().numpy())
        mbod_labels.append(labels)

    mbod_feats = np.concatenate(mbod_feats, axis=0)
    mbod_labels = np.concatenate(mbod_labels, axis=0)
    print(f"Processed {len(mbod_feats)} MBOD images.")

    # Combine ILO and MBOD features and labels
    if ilo_dataset is not None:
        all_feats = np.concatenate([ilo_feats, mbod_feats], axis=0)
        all_labels = np.concatenate([ilo_labels, mbod_labels], axis=0)
    else:
        all_feats = mbod_feats
        all_labels = mbod_labels

    print("Fitting t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, verbose=1)
    all_feats_2d = tsne.fit_transform(all_feats)

    # Create directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Get unique class labels
    unique_labels = np.unique(all_labels)
    
    # Create profusion score version of labels (0-3)
    profusion_scores = all_labels % 4
    
    print(f"Found {len(unique_labels)} unique classes: {unique_labels}")
    print(f"Mapping to {len(np.unique(profusion_scores))} profusion scores: {np.unique(profusion_scores)}")
    
    # Separate ILO and MBOD features
    n_ilo = len(ilo_feats)
    ilo_coords = all_feats_2d[:n_ilo]
    mbod_coords = all_feats_2d[n_ilo:]
    ilo_labels_subset = all_labels[:n_ilo]
    mbod_labels_subset = all_labels[n_ilo:]
    
    # Create file names with profusion coloring in the name
    file_name_by_class = f"visualizations/tsne_profusion_colored{'_trained' if trained else '_untrained'}.png"
    
    # Visualization by class and source
    plt.figure(figsize=(14, 10))
    
    # Plot each unique label
    for label in unique_labels:
        # Get profusion score (0-3) for this label
        prof_score = int(label) % 4
        # Get TB status (False for labels 0-3, True for labels 4-7)
        is_tb_positive = int(label) >= 4
        # Get the color based on profusion score only
        color = profusion_colors[prof_score]
        
        # ILO points for this class (always star marker)
        idx_ilo = np.where(ilo_labels_subset == label)[0]
        if len(idx_ilo) > 0:
            plt.scatter(
                ilo_coords[idx_ilo, 0], 
                ilo_coords[idx_ilo, 1],
                c=color,  # Use consistent color for profusion score
                marker='*',  # Stars for ILO
                s=120, 
                label=f'ILO - Profusion {prof_score}',
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )
        
        # MBOD points for this class (circle for TB-, triangle for TB+)
        idx_mbod = np.where(mbod_labels_subset == label)[0]

        if is_mstb:
            if len(idx_mbod) > 0:
                plt.scatter(
                    mbod_coords[idx_mbod, 0], 
                    mbod_coords[idx_mbod, 1],
                    c=color,  # Use consistent color for profusion score
                    marker='X' if not is_tb_positive else 'o',  # Circle for TB+, triangle for TB-
                    s=40, 
                    label=f'MBOD - Profusion {prof_score} {"(TB+)" if is_tb_positive else "(TB-)"}',
                    alpha=0.6,
                    edgecolors='white',
                    linewidths=0.2
                )
        else:
            if len(idx_mbod) > 0:
                plt.scatter(
                    mbod_coords[idx_mbod, 0], 
                    mbod_coords[idx_mbod, 1],
                    c=color,  # Use consistent color for profusion score
                    marker='o',  # Circle 
                    s=40, 
                    label=f'MBOD - Profusion {prof_score}',
                    alpha=0.6,
                    edgecolors='white',
                    linewidths=0.2
                )
    
    if entire_dataset:
        title_to_add = "Entire Dataset"
    else:
        title_to_add = ""

    plt.title(f"t-SNE Visualization (Profusion Score Coloring) {'(Trained Model)' if trained else '(Untrained Model)'}\n{title_to_add} - Epoch {n_epochs}", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create a custom legend with one entry per class and source combination
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(file_name_by_class, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE visualization to {file_name_by_class}")
    
    # Log to wandb if requested
    if log_to_wandb:
        print("Logging t-SNE visualizations to wandb...")
        wandb.log({
            f"{set_name} tsne": wandb.Image(file_name_by_class)
        })
        print("Logged visualizations to wandb successfully")

def visualize_multiple_tsne_3d_with_ilo2(model, device, data_loaders, loader_names, ilo_dataset, 
                                        experiment_name, output_filename=None):
    """
    Generate a single HTML with multiple 3D interactive t-SNE plots for different data loaders,
    including ILO reference images with distinct markers in each plot.
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.manifold import TSNE
    import os
    import numpy as np
    import torch  # You forgot to import torch in your snippet!
    
    # Tab10 color scheme
    tab10_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    model.to(device)
    model.eval()
    
    # Function to extract features from a dataset or dataloader
    def extract_feats(dataset_or_loader, is_loader=True):
        feats, labels, filenames = [], [], []
        with torch.no_grad():
            if is_loader:
                for x, y, filename in dataset_or_loader:
                    x = x.to(device)
                    f = model.features(x)
                    f = torch.flatten(f, start_dim=1).cpu().numpy()
                    feats.append(f)
                    labels.append(y.numpy())
                    filenames.extend(filename)  # Assuming filename is a list or batch
            else:
                for i in range(len(dataset_or_loader)):
                    x, y, filename = dataset_or_loader[i]
                    x = x.unsqueeze(0).to(device)
                    f = model.features(x)
                    f = torch.flatten(f, start_dim=1).cpu().numpy()
                    feats.append(f)
                    labels.append([y])
                    filenames.append(filename)  # Assuming filename is a single string
        return np.concatenate(feats), np.concatenate(labels), filenames
    
    print("Extracting features from ILO reference dataset...")
    ilo_feats, ilo_labels, ilo_filenames = extract_feats(ilo_dataset, is_loader=False)
    
    n_plots = len(data_loaders)
    cols = min(2, n_plots)
    rows = (n_plots + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scene'}] * cols] * rows,
        subplot_titles=loader_names,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    for i, (loader_key, loader_name) in enumerate(zip(data_loaders.keys(), loader_names)):
        loader_or_dataset = data_loaders[loader_key]
        is_loader = not hasattr(loader_or_dataset, '__getitem__')
        
        print(f"Extracting features for {loader_name}...")
        mbod_feats, mbod_labels, mbod_filenames = extract_feats(loader_or_dataset, is_loader=is_loader)
        
        all_feats = np.concatenate([ilo_feats, mbod_feats], axis=0)
        all_labels = np.concatenate([ilo_labels, mbod_labels], axis=0)
        all_filenames = ilo_filenames + mbod_filenames
        
        sources = ['ILO'] * len(ilo_labels) + ['MBOD'] * len(mbod_labels)
        
        print(f"Running t-SNE for {loader_name}...")
        tsne = TSNE(n_components=3, perplexity=min(30, len(all_feats)-1),
                    random_state=42, n_iter=1000, verbose=0)
        tsne_feats = tsne.fit_transform(all_feats)
        
        ilo_tsne = tsne_feats[:len(ilo_labels)]
        mbod_tsne = tsne_feats[len(ilo_labels):]
        
        row = i // cols + 1
        col = i % cols + 1
        
        unique_labels = sorted(np.unique(all_labels))
        
        for j, label in enumerate(unique_labels):
            color_idx = int(label) % len(tab10_colors)
            color = tab10_colors[color_idx]
            
            mbod_indices = np.where(mbod_labels == label)[0]
            if len(mbod_indices) > 0:
                hovertexts_mbod = [
                    f"Filename: {mbod_filenames[idx]}<br>X: {mbod_tsne[idx, 0]:.2f}<br>Y: {mbod_tsne[idx, 1]:.2f}<br>Z: {mbod_tsne[idx, 2]:.2f}"
                    for idx in mbod_indices
                ]
                scatter_mbod = go.Scatter3d(
                    x=mbod_tsne[mbod_indices, 0],
                    y=mbod_tsne[mbod_indices, 1],
                    z=mbod_tsne[mbod_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=color,
                        opacity=0.7,
                        symbol='circle',
                        line=dict(width=0.5, color='DarkSlateGrey'),
                    ),
                    text=hovertexts_mbod,
                    hoverinfo='text',
                    name=f'MBOD Class {int(label)}',
                    legendgroup=f'Class {int(label)}',
                    showlegend=(i == 0)
                )
                fig.add_trace(scatter_mbod, row=row, col=col)

            # For ILO samples
            ilo_indices = np.where(ilo_labels == label)[0]
            if len(ilo_indices) > 0:
                hovertexts_ilo = [
                    f"Filename: {ilo_filenames[idx]}<br>X: {ilo_tsne[idx, 0]:.2f}<br>Y: {ilo_tsne[idx, 1]:.2f}<br>Z: {ilo_tsne[idx, 2]:.2f}"
                    for idx in ilo_indices
                ]
                scatter_ilo = go.Scatter3d(
                    x=ilo_tsne[ilo_indices, 0],
                    y=ilo_tsne[ilo_indices, 1],
                    z=ilo_tsne[ilo_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=color,
                        opacity=1.0,
                        symbol='square-open',
                        line=dict(width=1.0, color='black'),
                    ),
                    text=hovertexts_ilo,
                    hoverinfo='text',
                    name=f'ILO Class {int(label)}',
                    legendgroup=f'ILO Class {int(label)}',
                    showlegend=(i == 0)
                )
                fig.add_trace(scatter_ilo, row=row, col=col)
        
        fig.update_scenes(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3',
            aspectmode='cube',
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis=dict(showgrid=True, gridcolor='lightgrey'),
            zaxis=dict(showgrid=True, gridcolor='lightgrey'),
            row=row, col=col,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        )
    
    fig.update_layout(
        height=600*rows,
        width=1200,
        title_text=f"{experiment_name}",
        margin=dict(l=0, r=0, b=0, t=80),
        template="plotly_white",
        legend=dict(
            groupclick="toggleitem",
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
        )
    )
    
    if output_filename is None:
        output_filename = f"{experiment_name}_multi_tsne3d.html"
        
    output_dir = os.path.dirname(os.path.join("tsne_html", output_filename))
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join("tsne_html", output_filename)
    fig.write_html(html_path, include_plotlyjs='cdn',
                   config={
                       'displayModeBar': True,
                       'scrollZoom': True,
                       'displaylogo': False,
                       'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                   })
    
    print(f"Saved multi t-SNE visualization to {html_path}")
    return html_path

def check_empty_study_ids(hdf5_path):
    """
    Check how many samples have an empty study_id column
    
    Args:
        hdf5_path: Path to the HDF5 file
    """
    with h5py.File(hdf5_path, "r") as f:
        total_samples = f["study_id"].shape[0]
        empty_count = 0
        problematic_indices = []
        
        print(f"Checking {total_samples} study IDs for empty values...")
        
        for idx in range(total_samples):
            study_id = f["study_id"][idx]
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            
            # Check if study_id is empty or just whitespace
            if not study_id or study_id.strip() == '':
                empty_count += 1
                problematic_indices.append(idx)
                
        print(f"\nFound {empty_count} empty study IDs out of {total_samples} samples ({empty_count/total_samples:.2%})")
        
        if empty_count > 0 and empty_count <= 10:
            print("\nIndices with empty study IDs:")
            for idx in problematic_indices:
                print(f"  Sample {idx}")
        elif empty_count > 10:
            print("\nFirst 10 indices with empty study IDs:")
            for idx in problematic_indices[:10]:
                print(f"  Sample {idx}")
                
        # Also check for study IDs that don't have enough parts when split by '.'
        problem_format_count = 0
        for idx in range(total_samples):
            study_id = f["study_id"][idx]
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            
            parts = study_id.split('.')
            if len(parts) < 3:
                problem_format_count += 1
                
        print(f"\nFound {problem_format_count} study IDs without at least 3 parts when split by '.' ({problem_format_count/total_samples:.2%})")

# Extract "best" or "final" from checkpoint path
def extract_model_type(checkpoint_path):
    # Using string split approach
    filename = os.path.basename(checkpoint_path)  # Gets "best_model.pth" or "final_model.pth"
    model_type = filename.split('_')[0]  # Gets "best" or "final"
    return model_type

def visualize_multiple_tsne_3d_with_ilo2_and_tb(model, device, data_loaders, loader_names, ilo_dataset, 
                                        experiment_name, output_filename=None):
    """
    Generate a single HTML with multiple 3D interactive t-SNE plots for different data loaders,
    including ILO reference images with distinct markers in each plot.
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.manifold import TSNE
    import os
    import numpy as np
    import torch
    
    # Tab10 color scheme for 4 profusion scores
    tab10_colors = [
        '#1f77b4',  # Profusion 0 (blue)
        '#ff7f0e',  # Profusion 1 (orange)
        '#2ca02c',  # Profusion 2 (green)
        '#d62728',  # Profusion 3 (red)
    ]
    
    # Map multiclass_stb to profusion scores (0–3)
    def get_profusion_score(label):
        return label % 4  # Map 0–3 to 0–3, and 4–7 to 0–3

    model.to(device)
    model.eval()
    
    # Function to extract features from a dataset or dataloader
    def extract_feats(dataset_or_loader, is_loader=True):
        feats, labels, filenames = [], [], []
        with torch.no_grad():
            if is_loader:
                for x, y, filename in dataset_or_loader:
                    x = x.to(device)
                    f = model.features(x)
                    f = torch.flatten(f, start_dim=1).cpu().numpy()
                    feats.append(f)
                    labels.append(y.numpy())
                    filenames.extend(filename)  # Assuming filename is a list or batch
            else:
                for i in range(len(dataset_or_loader)):
                    x, y, filename = dataset_or_loader[i]
                    x = x.unsqueeze(0).to(device)
                    f = model.features(x)
                    f = torch.flatten(f, start_dim=1).cpu().numpy()
                    feats.append(f)
                    labels.append([y])
                    filenames.append(filename)  # Assuming filename is a single string
        return np.concatenate(feats), np.concatenate(labels), filenames
    
    print("Extracting features from ILO reference dataset...")
    ilo_feats, ilo_labels, ilo_filenames = extract_feats(ilo_dataset, is_loader=False)
    
    n_plots = len(data_loaders)
    cols = min(2, n_plots)
    rows = (n_plots + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scene'}] * cols] * rows,
        subplot_titles=loader_names,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    for i, (loader_key, loader_name) in enumerate(zip(data_loaders.keys(), loader_names)):
        loader_or_dataset = data_loaders[loader_key]
        is_loader = not hasattr(loader_or_dataset, '__getitem__')
        
        print(f"Extracting features for {loader_name}...")
        mbod_feats, mbod_labels, mbod_filenames = extract_feats(loader_or_dataset, is_loader=is_loader)
        
        all_feats = np.concatenate([ilo_feats, mbod_feats], axis=0)
        all_labels = np.concatenate([ilo_labels, mbod_labels], axis=0)
        all_filenames = ilo_filenames + mbod_filenames
        
        sources = ['ILO'] * len(ilo_labels) + ['MBOD'] * len(mbod_labels)
        
        print(f"Running t-SNE for {loader_name}...")
        tsne = TSNE(n_components=3, perplexity=min(30, len(all_feats)-1),
                    random_state=42, n_iter=1000, verbose=0)
        tsne_feats = tsne.fit_transform(all_feats)
        
        ilo_tsne = tsne_feats[:len(ilo_labels)]
        mbod_tsne = tsne_feats[len(ilo_labels):]
        
        row = i // cols + 1
        col = i % cols + 1
        
        unique_labels = sorted(np.unique(all_labels))
        
        for j, label in enumerate(unique_labels):
            # Map the label to its corresponding profusion score
            profusion_score = get_profusion_score(label)
            color = tab10_colors[profusion_score]
            
            # Determine the shape based on TB status
            is_tb_positive = label >= 4
            shape = 'circle' if is_tb_positive else 'x'  # Circle for TB+, cross for TB-
            
            # Plot MBOD samples
            mbod_indices = np.where(mbod_labels == label)[0]
            if len(mbod_indices) > 0:
                hovertexts_mbod = [
                    f"Filename: {mbod_filenames[idx]}<br>X: {mbod_tsne[idx, 0]:.2f}<br>Y: {mbod_tsne[idx, 1]:.2f}<br>Z: {mbod_tsne[idx, 2]:.2f}"
                    for idx in mbod_indices
                ]
                scatter_mbod = go.Scatter3d(
                    x=mbod_tsne[mbod_indices, 0],
                    y=mbod_tsne[mbod_indices, 1],
                    z=mbod_tsne[mbod_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=4 if is_tb_positive else 2,
                        color=color,
                        opacity=0.7,
                        symbol=shape,  # Use different shapes for TB-positive
                        line=dict(width=0.5, color='DarkSlateGrey'),
                    ),
                    text=hovertexts_mbod,
                    hoverinfo='text',
                    name=f'MBOD Profusion {profusion_score} {"(TB+)" if is_tb_positive else "(TB-)"}',
                    legendgroup=f'Profusion {profusion_score}',
                    showlegend=(i == 0)
                )
                fig.add_trace(scatter_mbod, row=row, col=col)

            # Plot ILO samples
            ilo_indices = np.where(ilo_labels == label)[0]
            if len(ilo_indices) > 0:
                hovertexts_ilo = [
                    f"Filename: {ilo_filenames[idx]}<br>X: {ilo_tsne[idx, 0]:.2f}<br>Y: {ilo_tsne[idx, 1]:.2f}<br>Z: {ilo_tsne[idx, 2]:.2f}"
                    for idx in ilo_indices
                ]
                scatter_ilo = go.Scatter3d(
                    x=ilo_tsne[ilo_indices, 0],
                    y=ilo_tsne[ilo_indices, 1],
                    z=ilo_tsne[ilo_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=color,
                        opacity=1.0,
                        symbol='diamond',  # Use diamond shape for ILO
                        line=dict(width=1.0, color='black'),
                    ),
                    text=hovertexts_ilo,
                    hoverinfo='text',
                    name=f'ILO Profusion {profusion_score} {"(TB+)" if is_tb_positive else "(TB-)"}',
                    legendgroup=f'Profusion {profusion_score}',
                    showlegend=(i == 0)
                )
                fig.add_trace(scatter_ilo, row=row, col=col)
        
        fig.update_scenes(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3',
            aspectmode='cube',
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis=dict(showgrid=True, gridcolor='lightgrey'),
            zaxis=dict(showgrid=True, gridcolor='lightgrey'),
            row=row, col=col,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        )
    
    fig.update_layout(
        height=600*rows,
        width=1200,
        title_text=f"{experiment_name}",
        margin=dict(l=0, r=0, b=0, t=80),
        template="plotly_white",
        legend=dict(
            groupclick="toggleitem",
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
            itemsizing='constant',  # This makes legend markers a consistent size

        )
    )
    
    if output_filename is None:
        output_filename = f"{experiment_name}_multi_tsne3d.html"
        
    output_dir = os.path.dirname(os.path.join("tsne_html", output_filename))
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join("tsne_html", output_filename)
    fig.write_html(html_path, include_plotlyjs='cdn',
                   config={
                       'displayModeBar': True,
                       'scrollZoom': True,
                       'displaylogo': False,
                       'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                   })
    
    print(f"Saved multi t-SNE visualization to {html_path}")
    return html_path



def visualize_tsne_with_kaggle_tb(model, device, experiment_name, ilo_dataset, mbod_loader, tb_loader=None,
                trained=False, log_to_wandb=False, n_epochs=0, set_name="Training", entire_dataset=False):
    print("Starting t-SNE visualization generation...")
    model.eval()

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

    mbod_feats, mbod_labels = [], []
    print("Processing MBOD batches...")
    for batch in mbod_loader:
        imgs, labels = batch[0].to(device), batch[1].cpu().numpy()
        with torch.no_grad():
            feats = model.features(imgs)
            feats = torch.flatten(feats, start_dim=1)
        mbod_feats.append(feats.cpu().numpy())
        mbod_labels.append(labels)
    mbod_feats = np.concatenate(mbod_feats, axis=0)
    mbod_labels = np.concatenate(mbod_labels, axis=0)
    print(f"Processed {len(mbod_feats)} MBOD images.")

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
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, verbose=1)
    all_feats_2d = tsne.fit_transform(all_feats)

    os.makedirs('visualizations', exist_ok=True)

    # Profusion score colormap (e.g. tab10)
    profusion_labels = np.unique([l for i, l in enumerate(all_labels) if sources[i] != 'TB'])
    profusion_cmap = plt.cm.get_cmap('tab10', max(len(profusion_labels), 10))
    profusion_colors = {label: profusion_cmap(i % 10) for i, label in enumerate(profusion_labels)}

    # TB label colormap (use different colormap like Set1)
    tb_cmap = plt.cm.get_cmap('Pastel2', 4)
    tb_colors = {0: tb_cmap(0), 1: tb_cmap(3)}

    plt.figure(figsize=(14, 10))

    # Plot ILO & MBOD
    for label in profusion_labels:
        for source, marker, size, alpha in [('ILO', '*', 120, 0.8), ('MBOD', 'o', 40, 0.6)]:
            idx = [
                i for i in range(len(all_labels))
                if all_labels[i] == label and sources[i] == source
            ]
            if idx:
                coords = all_feats_2d[idx]
                plt.scatter(
                    coords[:, 0], coords[:, 1],
                    c=[profusion_colors[label]],
                    marker=marker,
                    s=size,
                    label=f'{source} - {int(label)}',
                    alpha=alpha,
                    edgecolors='black' if source == 'ILO' else 'white',
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
                    c=[tb_colors[tb_label]],
                    marker='x',
                    s=40,
                    label=f'TB - {tb_label}',
                    alpha=1.0,
                    edgecolors='black',
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
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE visualizations to {file_name}")

    if log_to_wandb:
        print("Logging t-SNE visualizations to wandb...")
        wandb.log({
            f"{set_name} tsne": wandb.Image(file_name)
        })
        print("Logged visualizations to wandb successfully")


def visualize_separate_tsne_3d_with_ilo2_and_tb(model, device, data_loaders, loader_names, ilo_dataset, 
                                        experiment_name, output_dir=None):
    """
    Generate multiple individual 3D interactive t-SNE plot HTML files for different data loaders,
    including ILO reference images with distinct markers in each plot.
    Saves each visualization to its own HTML file named by the loader name.
    """
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.manifold import TSNE
    import os
    import numpy as np
    import torch
    
    # Tab10 color scheme for 4 profusion scores
    tab10_colors = [
        '#1f77b4',  # Profusion 0 (blue)
        '#ff7f0e',  # Profusion 1 (orange)
        '#2ca02c',  # Profusion 2 (green)
        '#d62728',  # Profusion 3 (red)
    ]
    
    # Map multiclass_stb to profusion scores (0–3)
    def get_profusion_score(label):
        return label % 4  # Map 0–3 to 0–3, and 4–7 to 0–3

    model.to(device)
    model.eval()
    
    # Function to extract features from a dataset or dataloader
    def extract_feats(dataset_or_loader, is_loader=True):
        feats, labels, filenames = [], [], []
        with torch.no_grad():
            if is_loader:
                for x, y, filename in dataset_or_loader:
                    x = x.to(device)
                    f = model.features(x)
                    f = torch.flatten(f, start_dim=1).cpu().numpy()
                    feats.append(f)
                    labels.append(y.numpy())
                    filenames.extend(filename)  # Assuming filename is a list or batch
            else:
                for i in range(len(dataset_or_loader)):
                    x, y, filename = dataset_or_loader[i]
                    x = x.unsqueeze(0).to(device)
                    f = model.features(x)
                    f = torch.flatten(f, start_dim=1).cpu().numpy()
                    feats.append(f)
                    labels.append([y])
                    filenames.append(filename)  # Assuming filename is a single string
        return np.concatenate(feats), np.concatenate(labels), filenames
    
    print("Extracting features from ILO reference dataset...")
    ilo_feats, ilo_labels, ilo_filenames = extract_feats(ilo_dataset, is_loader=False)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join("tsne_html", experiment_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    html_paths = []
    
    for i, (loader_key, loader_name) in enumerate(zip(data_loaders.keys(), loader_names)):
        loader_or_dataset = data_loaders[loader_key]
        is_loader = not hasattr(loader_or_dataset, '__getitem__')
        
        print(f"Extracting features for {loader_name}...")
        mbod_feats, mbod_labels, mbod_filenames = extract_feats(loader_or_dataset, is_loader=is_loader)
        
        all_feats = np.concatenate([ilo_feats, mbod_feats], axis=0)
        all_labels = np.concatenate([ilo_labels, mbod_labels], axis=0)
        all_filenames = ilo_filenames + mbod_filenames
        
        sources = ['ILO'] * len(ilo_labels) + ['MBOD'] * len(mbod_labels)
        
        print(f"Running t-SNE for {loader_name}...")
        tsne = TSNE(n_components=3, perplexity=min(30, len(all_feats)-1),
                    random_state=42, n_iter=1000, verbose=0)
        tsne_feats = tsne.fit_transform(all_feats)
        
        ilo_tsne = tsne_feats[:len(ilo_labels)]
        mbod_tsne = tsne_feats[len(ilo_labels):]
        
        # Create a new figure for each dataset
        fig = go.Figure()
        
        unique_labels = sorted(np.unique(all_labels))
        
        for j, label in enumerate(unique_labels):
            # Map the label to its corresponding profusion score
            profusion_score = get_profusion_score(label)
            color = tab10_colors[profusion_score]
            
            # Determine the shape based on TB status
            is_tb_positive = label >= 4
            shape = 'circle' if is_tb_positive else 'x'  # Circle for TB+, cross for TB-
            
            # Plot MBOD samples
            mbod_indices = np.where(mbod_labels == label)[0]
            if len(mbod_indices) > 0:
                hovertexts_mbod = [
                    f"Filename: {mbod_filenames[idx]}<br>X: {mbod_tsne[idx, 0]:.2f}<br>Y: {mbod_tsne[idx, 1]:.2f}<br>Z: {mbod_tsne[idx, 2]:.2f}"
                    for idx in mbod_indices
                ]
                scatter_mbod = go.Scatter3d(
                    x=mbod_tsne[mbod_indices, 0],
                    y=mbod_tsne[mbod_indices, 1],
                    z=mbod_tsne[mbod_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=4 if is_tb_positive else 2,
                        color=color,
                        opacity=0.7,
                        symbol=shape,  # Use different shapes for TB-positive
                        line=dict(width=0.5, color='DarkSlateGrey'),
                    ),
                    text=hovertexts_mbod,
                    hoverinfo='text',
                    name=f'MBOD Profusion {profusion_score} {"(TB+)" if is_tb_positive else "(TB-)"}',
                    legendgroup=f'Profusion {profusion_score}',
                )
                fig.add_trace(scatter_mbod)

            # Plot ILO samples
            ilo_indices = np.where(ilo_labels == label)[0]
            if len(ilo_indices) > 0:
                hovertexts_ilo = [
                    f"Filename: {ilo_filenames[idx]}<br>X: {ilo_tsne[idx, 0]:.2f}<br>Y: {ilo_tsne[idx, 1]:.2f}<br>Z: {ilo_tsne[idx, 2]:.2f}"
                    for idx in ilo_indices
                ]
                scatter_ilo = go.Scatter3d(
                    x=ilo_tsne[ilo_indices, 0],
                    y=ilo_tsne[ilo_indices, 1],
                    z=ilo_tsne[ilo_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=color,
                        opacity=1.0,
                        symbol='diamond',  # Use diamond shape for ILO
                        line=dict(width=1.0, color='black'),
                    ),
                    text=hovertexts_ilo,
                    hoverinfo='text',
                    name=f'ILO Profusion {profusion_score} {"(TB+)" if is_tb_positive else "(TB-)"}',
                    legendgroup=f'Profusion {profusion_score}',
                )
                fig.add_trace(scatter_ilo)
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3',
                aspectmode='cube',
                xaxis=dict(showgrid=True, gridcolor='lightgrey'),
                yaxis=dict(showgrid=True, gridcolor='lightgrey'),
                zaxis=dict(showgrid=True, gridcolor='lightgrey'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            height=800,
            width=1000,
            title_text=f"{experiment_name} - {loader_name}",
            margin=dict(l=0, r=0, b=0, t=80),
            template="plotly_white",
            legend=dict(
                groupclick="toggleitem",
                orientation="h",
                yanchor="bottom",
                y=0.95,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="lightgrey",
                borderwidth=1,
                itemsizing='constant'  # Makes legend markers a consistent size
            )
        )
        
        
        # Create a clean filename based on the loader name
        clean_loader_name = loader_name.lower().replace(" ", "_")
        html_filename = f"{experiment_name}_{clean_loader_name}_tsne3d.html"
        html_path = os.path.join(output_dir, html_filename)
        
        fig.write_html(html_path, include_plotlyjs='cdn',
                       config={
                           'displayModeBar': True,
                           'scrollZoom': True,
                           'displaylogo': False,
                           'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                       })
        
        html_paths.append(html_path)
        print(f"Saved t-SNE visualization for {loader_name} to {html_path}")
    
    return html_paths

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")
    
    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        

        check_empty_study_ids(hdf5_file_path)

        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]

        preprocess = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # Create an HDF5SilicosisDataset instance
        mbod_dataset_merged = HDF5Dataset2(
            hdf5_path=hdf5_file_path,
            labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )

        # Path to Kaggle TB dataset
        kaggle_tb_path = config["kaggle_TB"]["outputpath"]  # Ensure this is set in config.yaml

        # Create an instance of KaggleTBDataset
        kaggle_tb_dataset = HDF5Dataset(
            hdf5_path = kaggle_tb_path,
            labels_key="tuberculosis",
            preprocess = preprocess
        )


        ilo_dataset = HDF5Dataset2(
            hdf5_path=ilo_hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )

        train_loader, val_loader, test_loader = get_dataloaders_with_files(
            hdf5_path=hdf5_file_path,
            preprocess=preprocess,
            batch_size=16,
            labels_key="multiclass_stb",
            split_file="stratified_split_filt.json",
            augmentations=None,
            oversample=None
        )

        # wandb.login()
        # wandb.init(project='MBOD-cl', name='img_test')
        
        # Load the saved model checkpoint

        # Define the mapping for multiclass_stb
        multiclass_stb_mapping = {
            0: "Profusion 0, No TB",
            1: "Profusion 1, No TB",
            2: "Profusion 2, No TB",
            3: "Profusion 3, No TB",
            4: "Profusion 0, With TB",
            5: "Profusion 1, With TB",
            6: "Profusion 2, With TB",
            7: "Profusion 3, With TB",
        }
        
        experiment_name = "mstb_v2-sin_m_01_05"
            
            
        checkpoint_path = f"/home/sean/MSc_2025/codev2/checkpoints/{experiment_name}/final_model.pth"
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Initialize model architecture (same as used during training)
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = model.to(device)

        if(experiment_name.startswith("clf")):
            model.classifier = MultiClassBaseClassifier(in_features=2048, num_classes=4).to(device)

        
        # Initialize optimizer (needed for loading state)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Load checkpoint with model state, optimizer state, and epoch
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
                
        raw_model = xrv.models.ResNet(weights="resnet50-res512-all")
        raw_model = raw_model.to(device)
        
        print(f"Successfully loaded model from epoch {epoch}")

        # wandb.init(project="tsne-visualization")
        #
        mbod_merged_loader = torch.utils.data.DataLoader(
            mbod_dataset_merged,
            batch_size=16,
            shuffle=False
        )

        tb_loader = torch.utils.data.DataLoader(
            kaggle_tb_dataset,
            batch_size=16,
            shuffle=False
        )

            # Create dictionary of data loaders/datasets to visualize
        data_loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
            "entire": mbod_dataset_merged
        }
        
        # Display names for each loader
        loader_names = [
            "Training Set",
            "Validation Set", 
            "Test Set", 
            "Entire Dataset"
        ]
        
        model_type = extract_model_type(checkpoint_path)

        # Call the function to create individual HTML files
        html_paths = visualize_separate_tsne_3d_with_ilo2_and_tb(
            model=model, 
            device=device, 
            data_loaders=data_loaders,
            loader_names=loader_names,
            ilo_dataset=ilo_dataset,
            experiment_name=experiment_name,
            output_dir=f"tsne_html/{experiment_name}/individual_{model_type}"  # Custom output directory
        )

        # Print the paths to all generated HTML files
        print(f"Generated {len(html_paths)} individual t-SNE visualizations:")
        for path in html_paths:
            print(f"  - {path}")
        
        # visualize_multiple_tsne_3d_with_ilo2_and_tb(
        #     model=model, 
        #     device=device, 
        #     data_loaders=data_loaders,
        #     loader_names=loader_names,
        #     ilo_dataset=ilo_dataset,
        #     experiment_name=experiment_name,
        #     output_filename=f"{experiment_name}/TEST-multi_tsne3d_v3_{model_type}.html"
        # )

        # visualize_tsne_with_kaggle_tb(model, device, experiment_name, ilo_dataset, mbod_merged_loader, tb_loader, trained=True,
        #           log_to_wandb=False, n_epochs=epoch, set_name="Trained Model", entire_dataset=True)
        
        # visualize_tsne_with_kaggle_tb(raw_model, device, experiment_name, ilo_dataset, mbod_merged_loader, tb_loader, trained=False,
        #           log_to_wandb=False, n_epochs=None, set_name="Raw Model", entire_dataset=True)



        # visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, trained=True, log_to_wandb=False,
        #                n_epochs=epoch, set_name="Trained Model", entire_dataset=True, is_mstb=False)

    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error loading model or generating visualizations: {e}")
        raise
