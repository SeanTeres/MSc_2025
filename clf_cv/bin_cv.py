from datetime import datetime
import glob
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchxrayvision as xrv
import wandb
import numpy as np
import yaml
import torch.nn.functional as F
import sklearn.metrics as skmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import get_dataloaders


# Add codev2 and DomainAdaptation to path
sys.path.append(os.path.abspath("../DomainAdaptation"))
from da_utils import reinitialize_weights

sys.path.append(os.path.abspath("../classification"))
import metrics
from cross_validation import plot_combined_conf_mat, plot_tb_stratified_binary_cm
from clf_manager import BinaryClassifier, XRVBasedClassifier
import scipy.stats
sys.path.append(os.path.abspath("../simclr"))
from simclr import SimCLR  # Import your SimCLR class

def load_simclr_encoder(model, checkpoint_path):
    """
    Load pretrained SimCLR encoder weights into a torchxrayvision ResNet model.
    
    Args:
        model: The target ResNet model (without classifier)
        checkpoint_path: Path to the SimCLR checkpoint file
    """
    print(f"Loading SimCLR checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    simclr_state_dict = checkpoint['model_state_dict']
    
    # Create a temporary SimCLR model to properly load the weights
    temp_xrv_model = xrv.models.ResNet(weights="resnet50-res512-all")
    temp_simclr_model = SimCLR(temp_xrv_model, out_dim=128)
    
    # Load the full SimCLR state dict
    temp_simclr_model.load_state_dict(simclr_state_dict)
    
    # Extract only the encoder weights (the underlying ResNet model)
    encoder_state_dict = temp_simclr_model.model.state_dict()
    
    # Load weights into the target model
    missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {unexpected_keys}")
    
    print("✅ SimCLR encoder weights loaded successfully")
    return model


def plot_combined_with_stratified_cm(confusion_matrix, all_labels, all_preds, original_labels, epoch=None, task="tuberculosis", use_multilabel=False):
    """
    Plots both standard confusion matrix and stratified confusion matrix on the same figure.
    
    Args:
        confusion_matrix: Standard 2x2 confusion matrix
        all_labels: Ground truth labels (0 or 1)
        all_preds: Predicted labels (0 or 1)
        original_labels: Original multiclass_stb labels (0-7)
        epoch: Current epoch for title
        task: Either "tuberculosis" or "silicosis" to determine stratification
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Plot standard confusion matrix on the left
    if confusion_matrix.shape == (2, 2):
        # Binary case
        TN, FP, FN, TP = confusion_matrix.ravel()

        # Calculate metrics
        accuracy  = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        # Plot standard CM
        cm_display = skmetrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0, 1])
        cm_display.plot(ax=ax1, cmap='Blues', colorbar=False)

        # Title with epoch + metrics
        title = f"Confusion Matrix (Epoch {epoch})" if epoch is not None else "Confusion Matrix"
        ax1.set_title(title, fontsize=14)

        # Add text box with metrics
        metrics_text = (f"Accuracy = {accuracy:.3f}\n"
                        f"Precision = {precision:.3f}\n"
                        f"Recall = {recall:.3f}\n"
                        f"F1 = {f1:.3f}\n"
                        f"Specificity = {specificity:.3f}")
        ax1.text(1.05, 0.5, metrics_text, transform=ax1.transAxes,
                 fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1))
        
    if isinstance(original_labels, list):
        original_labels = np.array(original_labels)
    
        
    
    # Plot stratified confusion matrix on the right
    if task == "tuberculosis":

        if use_multilabel:
            # For multilabel, original_labels is expected to be a 2D array where
            # the first column is TB status (0 or 1) and the second column is silicosis profusion score (0-3)
            tb_status = original_labels[:, 1]
            silicosis_status = original_labels[:, 0]
        else:
            # Determine silicosis status (profusion > 0 means silicosis+)   
            silicosis_status = ((original_labels % 4) > 0).astype(int)    
        # Create empty matrix for counts
        stratified_matrix = np.zeros((4, 2), dtype=int)
        
        # Fill the matrix with counts
        # TP (TB+ / TB+)
        tp_mask = (all_labels == 1) & (all_preds == 1)
        stratified_matrix[0, 0] = np.sum(tp_mask & (silicosis_status == 1))  # Silicosis+
        stratified_matrix[0, 1] = np.sum(tp_mask & (silicosis_status == 0))  # Silicosis-
        
        # FP (TB+ / TB-)
        fp_mask = (all_labels == 0) & (all_preds == 1)
        stratified_matrix[1, 0] = np.sum(fp_mask & (silicosis_status == 1))  # Silicosis+
        stratified_matrix[1, 1] = np.sum(fp_mask & (silicosis_status == 0))  # Silicosis-
        
        # FN (TB- / TB+)
        fn_mask = (all_labels == 1) & (all_preds == 0)
        stratified_matrix[2, 0] = np.sum(fn_mask & (silicosis_status == 1))  # Silicosis+
        stratified_matrix[2, 1] = np.sum(fn_mask & (silicosis_status == 0))  # Silicosis-
        
        # TN (TB- / TB-)
        tn_mask = (all_labels == 0) & (all_preds == 0)
        stratified_matrix[3, 0] = np.sum(tn_mask & (silicosis_status == 1))  # Silicosis+
        stratified_matrix[3, 1] = np.sum(tn_mask & (silicosis_status == 0))  # Silicosis-
        
        # Define labels
        row_labels = [
            "TP (TB+ / TB+)",
            "FP (TB+ / TB-)",
            "FN (TB- / TB+)", 
            "TN (TB- / TB-)"
        ]
        col_labels = ["Silicosis+", "Silicosis-"]
        
        # Create heatmap
        im = ax2.imshow(stratified_matrix, cmap="PuBu")  
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax2)
        cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
        
        # Configure axes
        ax2.set_xticks(np.arange(len(col_labels)))
        ax2.set_yticks(np.arange(len(row_labels)))
        ax2.set_xticklabels(col_labels)
        ax2.set_yticklabels(row_labels)
        
        ax2.set_xlabel("Silicosis Status")
        ax2.set_ylabel("Predicted vs True TB")
        ax2.set_title("Silicosis-stratified TB Detection", fontsize=14)
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax2.text(j, i, stratified_matrix[i, j], 
                       ha="center", va="center", color="black")
                       
    elif task == "silicosis":
        # Implement TB-stratified silicosis display
        # This would be similar to plot_tb_stratified_binary_cm function
        if use_multilabel:
            # For multilabel, original_labels is expected to be a 2D array where
            # the first column is TB status (0 or 1) and the second column is silicosis profusion score (0-3)
            tb_status = original_labels[:, 1]
            silicosis_status = original_labels[:, 0]
        else:
            # Determine silicosis status (profusion > 0 means silicosis+)   
            tb_status = (original_labels >= 4).astype(int)
        # Create empty matrix
        stratified_matrix = np.zeros((4, 2), dtype=int)
        
        # Fill matrix with counts
        # TP (Silicosis+ / Silicosis+)
        tp_mask = (all_labels == 1) & (all_preds == 1)
        stratified_matrix[0, 0] = np.sum(tp_mask & (tb_status == 1))  # TB+
        stratified_matrix[0, 1] = np.sum(tp_mask & (tb_status == 0))  # TB-
        
        # FP (Silicosis+ / Silicosis-)
        fp_mask = (all_labels == 0) & (all_preds == 1)
        stratified_matrix[1, 0] = np.sum(fp_mask & (tb_status == 1))  # TB+
        stratified_matrix[1, 1] = np.sum(fp_mask & (tb_status == 0))  # TB-
        
        # FN (Silicosis- / Silicosis+)
        fn_mask = (all_labels == 1) & (all_preds == 0)
        stratified_matrix[2, 0] = np.sum(fn_mask & (tb_status == 1))  # TB+
        stratified_matrix[2, 1] = np.sum(fn_mask & (tb_status == 0))  # TB-
        
        # TN (Silicosis- / Silicosis-)
        tn_mask = (all_labels == 0) & (all_preds == 0)
        stratified_matrix[3, 0] = np.sum(tn_mask & (tb_status == 1))  # TB+
        stratified_matrix[3, 1] = np.sum(tn_mask & (tb_status == 0))  # TB-
        
        # Define labels
        row_labels = [
            "TP (Silicosis+ / Silicosis+)",
            "FP (Silicosis+ / Silicosis-)",
            "FN (Silicosis- / Silicosis+)", 
            "TN (Silicosis- / Silicosis-)"
        ]
        col_labels = ["TB+", "TB-"]
        
        # Create heatmap
        im = ax2.imshow(stratified_matrix, cmap="PuBu")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax2)
        cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
        
        # Configure axes
        ax2.set_xticks(np.arange(len(col_labels)))
        ax2.set_yticks(np.arange(len(row_labels)))
        ax2.set_xticklabels(col_labels)
        ax2.set_yticklabels(row_labels)
        
        ax2.set_xlabel("TB Status")
        ax2.set_ylabel("Predicted vs True Silicosis")
        ax2.set_title("TB-stratified Silicosis Detection", fontsize=14)
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                ax2.text(j, i, stratified_matrix[i, j], 
                       ha="center", va="center", color="black")
    
    plt.tight_layout()
    return fig

def plot_silicosis_stratified_tb_cm(all_labels, all_preds, original_labels):
    """
    Create a silicosis-stratified binary confusion matrix for TB detection.
    
    Args:
        all_labels: Ground truth TB labels (0 or 1)
        all_preds: Predicted TB labels (0 or 1)
        original_labels: Original multiclass_stb labels (0-7)
    """
    # Determine silicosis status (profusion > 0 means silicosis+)   
    silicosis_status = ((original_labels % 4) > 0).astype(int)    
    # Create empty matrix for counts
    stratified_matrix = np.zeros((4, 2), dtype=int)
    
    # Fill the matrix with counts
    # TP (TB+ / TB+)
    tp_mask = (all_labels == 1) & (all_preds == 1)
    stratified_matrix[0, 0] = np.sum(tp_mask & (silicosis_status == 1))  # Silicosis+
    stratified_matrix[0, 1] = np.sum(tp_mask & (silicosis_status == 0))  # Silicosis-
    
    # FP (TB+ / TB-)
    fp_mask = (all_labels == 0) & (all_preds == 1)
    stratified_matrix[1, 0] = np.sum(fp_mask & (silicosis_status == 1))  # Silicosis+
    stratified_matrix[1, 1] = np.sum(fp_mask & (silicosis_status == 0))  # Silicosis-
    
    # FN (TB- / TB+)
    fn_mask = (all_labels == 1) & (all_preds == 0)
    stratified_matrix[2, 0] = np.sum(fn_mask & (silicosis_status == 1))  # Silicosis+
    stratified_matrix[2, 1] = np.sum(fn_mask & (silicosis_status == 0))  # Silicosis-
    
    # TN (TB- / TB-)
    tn_mask = (all_labels == 0) & (all_preds == 0)
    stratified_matrix[3, 0] = np.sum(tn_mask & (silicosis_status == 1))  # Silicosis+
    stratified_matrix[3, 1] = np.sum(tn_mask & (silicosis_status == 0))  # Silicosis-
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define labels
    row_labels = [
        "TP (TB+ / TB+)",
        "FP (TB+ / TB-)",
        "FN (TB- / TB+)", 
        "TN (TB- / TB-)"
    ]
    col_labels = ["Silicosis+", "Silicosis-"]
    
    # Create heatmap
    im = ax.imshow(stratified_matrix, cmap="PuBu")  # GnBu, GnBu, BuPu
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Label the axes
    ax.set_xlabel("Silicosis Status")
    ax.set_ylabel("Predicted vs True TB")
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations in each cell
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, stratified_matrix[i, j], 
                   ha="center", va="center", color="black")
    
    plt.title("Silicosis-stratified TB Detection Confusion Matrix")
    plt.tight_layout()
    
    return fig

def mean_std_var_ci(metric_list):
    arr = np.array(metric_list)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    var = np.var(arr, ddof=1)
    n = len(arr)
    # 95% confidence interval for the mean (t-distribution)
    ci95 = scipy.stats.t.interval(
        0.95, n-1, loc=mean, scale=std/np.sqrt(n)
    ) if n > 1 else (mean, mean)
    return {
        "mean": mean,
        "std": std,
        "var": var,
        "ci95_low": ci95[0],
        "ci95_high": ci95[1]
    }

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries"""
    import random
    import numpy as np
    import torch
    import os
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For sklearn (if used)
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass
    
    print(f"🌱 Random seeds set to {seed} for reproducibility")
    return seed

def plot_combined_conf_mat(confusion_matrix, epoch=None):
    """
    Plots confusion matrix and adds metrics + epoch to figure.
    Works for binary, multilabel (avg), and multiclass.
    """
    if confusion_matrix.shape == (2, 2):
        # Binary case
        TN, FP, FN, TP = confusion_matrix.ravel()

        # Calculate metrics manually
        accuracy  = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display = skmetrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0, 1])
        cm_display.plot(ax=ax, cmap='Blues', colorbar=False)

        # Title with epoch + metrics
        title = f"Confusion Matrix (Epoch {epoch})" if epoch is not None else "Confusion Matrix"
        ax.set_title(title, fontsize=14)

        # Add text box with metrics
        metrics_text = (f"Accuracy = {accuracy:.3f}\n"
                        f"Precision = {precision:.3f}\n"
                        f"Recall = {recall:.3f}\n"
                        f"F1 = {f1:.3f}\n"
                        f"Specificity = {specificity:.3f}")
        ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1))
        plt.tight_layout()
        return fig

    elif len(confusion_matrix.shape) == 3:
        # Multilabel: average + normalize
        avg_cm = np.mean(confusion_matrix, axis=0)
        norm_cm = avg_cm / (np.sum(avg_cm, axis=1, keepdims=True))
        fig = plt.figure(figsize=(8, 6))
        cm_display = skmetrics.ConfusionMatrixDisplay(norm_cm, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca(), cmap='Blues')
        title = f"Normalised Multilabel Confusion Matrix (Epoch {epoch})" if epoch else "Normalised Multilabel Confusion Matrix"
        plt.title(title)
        return fig

    else:
        # Multiclass case
        display_labels = [i for i in range(confusion_matrix.shape[0])]
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display = skmetrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=display_labels)
        cm_display.plot(ax=ax, cmap='Blues')
        title = f"Multiclass Confusion Matrix (Epoch {epoch})" if epoch else "Multiclass Confusion Matrix"
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
def normalize_to_hu_range(img_tensor):
    """Normalize image tensor to Hounsfield Unit range (-1024, 1024)"""
    min_val = img_tensor.min()
    max_val = img_tensor.max()
    
    # Scale to [0,1]
    normalized = (img_tensor - min_val) / (max_val - min_val)
    
    # Scale to [-1024, 1024]
    return normalized * 2048 - 1024
        

class BinaryMonteCarloCV:
    def __init__(self, model, cfg, labels_key, num_classes, device, loss_fn, hdf5_path, batch_size, epochs, optimizer_type, learning_rate, weight_decay, use_oversampling, checkpoint_save_target, exp_name, split_file, train_set_name, test_set_name, clf_task_labels_key):
        self.model = model
        self.cfg = cfg
        self.labels_key = labels_key
        self.num_classes = num_classes
        self.device = device
        self.loss_fn = loss_fn
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_oversampling = use_oversampling
        self.checkpoint_save_target = checkpoint_save_target
        self.exp_name = exp_name
        self.split_file = split_file
        self.train_set_name = train_set_name
        self.clf_task_labels_key = clf_task_labels_key
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(normalize_to_hu_range)  # Scale [0,1] to [-1024,1024]
        ])
        
        self.checkpoint_save_target = f"{self.checkpoint_save_target}/{self.exp_name}"

        self.augmentations = transforms.Compose([
        transforms.RandomRotation(degrees=10, expand=False, fill=0),
        # T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # T.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
        ])

        if not (os.path.isdir(self.checkpoint_save_target)):
            os.makedirs(self.checkpoint_save_target)

        self._validate_task_key()
        print(f"[INIT] Active task = {self.clf_task_labels_key}  (USE_MULTILABEL={self.cfg['USE_MULTILABEL']})")

    
    def _validate_task_key(self):
        allowed = {"tuberculosis", "silicosis"}
        if self.clf_task_labels_key not in allowed:
            raise ValueError(f"Invalid CLF_TASK_LABELS_KEY={self.clf_task_labels_key}. Allowed={allowed}")

    def _extract_task_labels(self, batch_labels):
        # batch_labels: (B,) or (B,2)
        if self.cfg['USE_MULTILABEL']:
            if batch_labels.dim() != 2 or batch_labels.size(1) < 2:
                raise ValueError("Expected multilabel tensor with 2 columns.")
            if self.clf_task_labels_key == "tuberculosis":
                return batch_labels[:, 1].long()
            elif self.clf_task_labels_key == "silicosis":
                return batch_labels[:, 0].long()
        else:
            # Single multiclass → derive binary
            if self.train_set_name == "MBOD":
                if self.clf_task_labels_key == "tuberculosis":
                    return (batch_labels >= 4).long()
                elif self.clf_task_labels_key == "silicosis":
                    return ((batch_labels % 4) > 0).long()
        raise ValueError("Unsupported combination in _extract_task_labels.")

    def debug_check_labels(self, loader, n=64):
        pairs = []
        for i, (_, labels) in enumerate(loader):
            # Expect shape (B,2) for multilabel
            if labels.dim() == 2 and labels.size(1) >= 2:
                for row in labels.tolist():
                    pairs.append(row[:2])
            if len(pairs) >= n:
                break
        pairs = pairs[:n]
        arr = np.array(pairs)
        print(f"[DEBUG] First {len(arr)} multilabel rows (cols 0,1):")
        print(arr)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            col0_counts = np.bincount(arr[:,0].astype(int))
            col1_counts = np.bincount(arr[:,1].astype(int))
            print(f"[DEBUG] Col0 bincount = {col0_counts}")
            print(f"[DEBUG] Col1 bincount = {col1_counts}")
            if arr[:,0].max() > 1 or arr[:,1].max() > 1:
                print("[DEBUG] Non-binary values detected. Verify that the silicosis column is binary.")
            same = np.all(arr[:,0] == arr[:,1])
            print(f"[DEBUG] Columns identical? {same}")
            corr = np.corrcoef(arr[:,0], arr[:,1])[0,1] if not same else 1.0
            print(f"[DEBUG] Correlation col0 vs col1: {corr:.4f}")

    def local_get_dataloaders(self, train_split=None, iteration=None):
        """Get dataloaders with consistent splits"""
        now = datetime.now()
        currentTime = now.strftime("%Y-%m-%d_%H-%M-%S")

        if self.split_file:
            print(f"Loading from split file: {self.split_file}")
            split_file = self.split_file
        else:
            # Create new split file name for this iteration
            print("Creating new split file...")
            split_file = f"data_splits/{self.exp_name}_{currentTime}.json"
            if not os.path.exists("data_splits"):
                os.makedirs("data_splits")

        # Get train loader with augmentations
        train_loader, _, _ = get_dataloaders(
            hdf5_path=self.hdf5_path,
            preprocess=self.preprocess,
            train_split=train_split if train_split else 0.7,
            batch_size=self.batch_size,
            labels_key=self.labels_key,
            split_file=split_file,  # This will create and save the split file
            augmentations=self.augmentations,
            oversample=self.use_oversampling,
            clf_task_labels_key=self.cfg["CLF_TASK_LABELS_KEY"],
        )

        # Get val/test loaders using same split file but without augmentations
        _, val_loader, test_loader = get_dataloaders(
            hdf5_path=self.hdf5_path,
            preprocess=self.preprocess,
            train_split=train_split if train_split else 0.7,
            batch_size=1,
            labels_key=self.labels_key,
            split_file=split_file,  # This will use the saved split file
            augmentations=None,
            oversample=False,
            clf_task_labels_key=self.cfg["CLF_TASK_LABELS_KEY"],
        )



        if self.cfg['USE_MULTILABEL']:
            self.debug_check_labels(train_loader)

        return train_loader, val_loader, test_loader
    
    def evaluate_model(self, dataloader, epoch, loss_fn, name="", fold=None):
        print(f"\nEVALUATING on {name}\n")
        metrics_dict = {}
        self.model.eval()

        all_labels = []
        all_probs = []
        all_feats = []
        all_logits = []

        all_original_labels = []

        total_loss = 0.0

        with torch.no_grad():

            for batch_imgs, batch_labels in tqdm(dataloader, desc=f"Evaluating {name} {fold}"):
                all_original_labels.extend(batch_labels.cpu().numpy())
                task_labels = self._extract_task_labels(batch_labels)
                batch_imgs = batch_imgs.to(self.device)
                task_labels = task_labels.to(self.device)

                feats = self.model.features(batch_imgs)
                logits = self.model.classifier(feats)
                probs = torch.sigmoid(logits)

                loss_val = loss_fn(logits, task_labels.float().unsqueeze(1))
                total_loss += loss_val.item()

                all_labels.extend(task_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

            avg_loss = total_loss / len(dataloader)
            print(f"Average Loss - {name} {fold}: {avg_loss}")

            all_labels = np.array(all_labels).flatten()
            all_probs = np.array(all_probs).flatten()
            all_logits = np.array(all_logits).flatten()

            # prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)
            # bins = list(range(len(prob_true)))
            # wandb.log({f"{name}/calibration_curve": wandb.plot.line_series(
            #     xs=[bins, bins],
            #     ys=[prob_true, prob_pred],
            #     keys=["True", "Predicted"],
            #     title="Calibration Curve",
            #     xname="Bin"
            # )}, step=epoch)

            # print(all_labels)
            # print(all_probs)

            # preds_05 = all_probs > 0.5
            
            # Compute metrics
            # cm_05 = skmetrics.confusion_matrix(all_labels, preds_05)

            spec_at_09_sens, threshold = metrics.specificity_at_sensitivity(all_labels, all_probs, 0.9)
            preds_opt = all_probs > threshold
            cm_opt = skmetrics.confusion_matrix(all_labels, preds_opt)

            # acc_05 = metrics.get_accuracy(cm_05)
            # f1_05 = metrics.get_f1_score(cm_05)
            # sens_05 = metrics.get_sensitivity(cm_05)
            # spec_05 = metrics.get_specificity(cm_05)
            # kappa_05 = metrics.get_cohens_kappa(cm_05)


            acc_opt = metrics.get_accuracy(cm_opt)
            sens_opt = metrics.get_sensitivity(cm_opt)
            spec_opt = metrics.get_specificity(cm_opt)
            f1_opt = metrics.get_f1_score(cm_opt)
            kappa_opt = metrics.get_cohens_kappa(cm_opt)

            if self.train_set_name == "MBOD":
                if(epoch) % 5 == 0 or name == "test":
                    if self.clf_task_labels_key == "tuberculosis":
                        if self.cfg['USE_MULTILABEL']:
                            # Don't flatten for multilabel case
                            original_labels_array = np.array(all_original_labels)
                        else:
                            # Flatten for single label case
                            original_labels_array = np.array(all_original_labels).flatten()
                            
                        comb_strat_cm_fig = plot_combined_with_stratified_cm(
                            cm_opt, all_labels, preds_opt, original_labels_array, 
                            epoch=epoch, task=self.clf_task_labels_key, 
                            use_multilabel=self.cfg['USE_MULTILABEL']
                        )
                        wandb.log({f"cm/{name}_combined_stratified_cm": wandb.Image(comb_strat_cm_fig)}, step=epoch)
                    elif self.clf_task_labels_key == "silicosis":
                        if self.cfg['USE_MULTILABEL']:
                            # Don't flatten for multilabel case
                            original_labels_array = np.array(all_original_labels)
                        else:
                            # Flatten for single label case
                            original_labels_array = np.array(all_original_labels).flatten()
                            
                        comb_strat_cm_fig = plot_combined_with_stratified_cm(
                            cm_opt, all_labels, preds_opt, original_labels_array, 
                            epoch=epoch, task=self.clf_task_labels_key, 
                            use_multilabel=self.cfg['USE_MULTILABEL']
                        )
                        wandb.log({f"cm/{name}_combined_stratified_cm": wandb.Image(comb_strat_cm_fig)}, step=epoch)
                    else:
                        raise ValueError("Unsupported clf_task_labels_key for stratified CM(s)")

            try:
                auc_score = skmetrics.roc_auc_score(all_labels, all_probs)
                fpr, tpr, _ = skmetrics.roc_curve(all_labels, all_probs)

            except ValueError as e:
                print(f"Warning: Could not calculate auc")
                auc_score = None
                fpr, tpr = None, None

            

            metrics_dict["auc"] = auc_score
            metrics_dict["fpr"] = fpr
            metrics_dict["tpr"] = tpr
            metrics_dict["loss"] = total_loss

            # metrics_dict["cm_05"] = cm_05
            # metrics_dict["acc_05"] = acc_05
            # metrics_dict["f1_05"] = f1_05
            # metrics_dict["sens_05"] = sens_05
            # metrics_dict["spec_05"] = spec_05
            # metrics_dict["kappa_05"] = kappa_05

            metrics_dict["spec_at_09_sens"] = spec_at_09_sens
            metrics_dict["threshold"] = threshold
            metrics_dict["cm_opt"] = cm_opt
            metrics_dict["acc_opt"] = acc_opt
            metrics_dict["f1_opt"] = f1_opt
            metrics_dict["sens_opt"] = sens_opt
            metrics_dict["spec_opt"] = spec_opt
            metrics_dict["kappa_opt"] = kappa_opt

            metrics_dict["all_labels"] = all_labels
            metrics_dict["all_preds_opt"] = preds_opt
            metrics_dict["all_original_labels"] = all_original_labels

            cm_opt_img = plot_combined_conf_mat(cm_opt, epoch=epoch+1)
            # cm_05_img = plot_combined_conf_mat(cm_05, epoch=epoch+1)


        # print(f"labels: {all_labels} \n logits: {all_logits} \n probs: {all_probs}")
        

        return metrics_dict
    
    def train(self, cfg, optimizer, fold):

        best_spec_at_09_sens = -np.inf
        best_model_path = None
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

        train_loader, val_loader, test_loader = self.local_get_dataloaders(train_split=0.7, iteration=fold)

        pos_weight = calculate_pos_weight(train_loader=train_loader, task_key=self.clf_task_labels_key) \
                     if self.cfg["WEIGHTED_LOSS"] else torch.tensor(1.0)

        if self.cfg["LOSS_FUNC"] == "BCEWithLogitsLoss":
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight if self.cfg["WEIGHTED_LOSS"] else None)
        elif self.cfg["LOSS_FUNC"] == "FocalLoss":
            self.loss_fn = FocalLoss(alpha=1.0, gamma=2.0,
                                     pos_weight=pos_weight if self.cfg["WEIGHTED_LOSS"] else None,
                                     reduction="mean")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )

        for epoch in tqdm(range(cfg["EPOCHS"]), desc=f"Training on {cfg['TRAIN_SET_NAME']}"):
            self.model.train()
            total_loss = 0
            epoch_task_labels = []
            for batch_imgs, batch_labels in train_loader:
                task_labels = self._extract_task_labels(batch_labels)
                epoch_task_labels.append(task_labels.cpu().numpy())

                batch_imgs = batch_imgs.to(self.device)
                task_labels = task_labels.to(self.device)

                optimizer.zero_grad()
                feats = self.model.features(batch_imgs)
                logits = self.model.classifier(feats)
                loss = self.loss_fn(logits, task_labels.float().unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_task_labels = np.concatenate(epoch_task_labels)
            pos_count = (epoch_task_labels == 1).sum()
            neg_count = (epoch_task_labels == 0).sum()
            print(f"[EPOCH {epoch+1}] Task={self.clf_task_labels_key} label balance: pos={pos_count} neg={neg_count} ratio={(pos_count/(neg_count+1e-6)):.3f}")
            print(f"Epoch {epoch+1}/{cfg['EPOCHS']}, Loss: {total_loss/len(train_loader)}")

            train_dict = self.evaluate_model(train_loader, epoch, self.loss_fn, name="train", fold=fold)

           # cm_05_img = plot_combined_conf_mat(train_dict["cm_05"], epoch=epoch+1)
            cm_opt_img = plot_combined_conf_mat(train_dict["cm_opt"], epoch=epoch+1)

            
            wandb.log({
                "train/loss": train_dict["loss"],
                "train/auc": train_dict["auc"],
                "train/fpr": train_dict["fpr"],
                "train/tpr": train_dict["tpr"],
                # "train/acc_05": train_dict["acc_05"],
                # "train/f1_05": train_dict["f1_05"],
                # "train/sens_05": train_dict["sens_05"],
                # "train/spec_05": train_dict["spec_05"],
                # "train/kappa_05": train_dict["kappa_05"],
                "train/spec_at_09_sens": train_dict["spec_at_09_sens"],
                "train/threshold": train_dict["threshold"],
                "cm/train_cm_opt": wandb.Image(cm_opt_img) if epoch % 5 == 0 else None,
                # "cm/train_cm_05": wandb.Image(cm_05_img) if epoch % 5 == 0 else None,
                "train/acc_opt": train_dict["acc_opt"],
                "train/f1_opt": train_dict["f1_opt"],
                "train/sens_opt": train_dict["sens_opt"],
                "train/spec_opt": train_dict["spec_opt"],
                "train/kappa_opt": train_dict["kappa_opt"],

                "pos_weight": pos_weight.item()
            }, step=epoch)

            val_dict = self.evaluate_model(val_loader, epoch, self.loss_fn, name="val", fold=fold)
            # Checkpoint logic
            if val_dict["spec_at_09_sens"] > best_spec_at_09_sens:
                best_spec_at_09_sens = val_dict["spec_at_09_sens"]
                # Save model and optimizer state
                model_filename = f"{self.checkpoint_save_target}/model_{self.cfg['RUN_NAME']}_fold{fold}_best.pth"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'spec_at_09_sens': best_spec_at_09_sens
                }, model_filename)
                best_model_path = model_filename
                print(f"Checkpointed model at {model_filename} (spec@0.9sens={best_spec_at_09_sens:.4f})")

                # Create both standard and combined confusion matrix plots
                best_cm_standard = plot_combined_conf_mat(val_dict["cm_opt"], epoch=epoch+1)
                best_cm_combined = plot_combined_with_stratified_cm(
                    val_dict["cm_opt"], 
                    val_dict["all_labels"], 
                    val_dict["all_preds_opt"], 
                    val_dict["all_original_labels"],
                    epoch=epoch+1,
                    task=self.clf_task_labels_key,
                    use_multilabel=self.cfg['USE_MULTILABEL']
                )
                
                # Log both visualizations
                wandb.log({
                    "cm/best_val_spec_at_sens": wandb.Image(best_cm_standard),
                    "cm/best_val_combined": wandb.Image(best_cm_combined),
                    "val/best_spec_at_09_sens": best_spec_at_09_sens
                }, step=epoch)
            
            # val_cm_05_img = plot_combined_conf_mat(val_dict["cm_05"], epoch=epoch+1)
            val_cm_opt_img = plot_combined_conf_mat(val_dict["cm_opt"], epoch=epoch+1)

            scheduler.step(val_dict["spec_opt"])

            wandb.log({
                "val/auc": val_dict["auc"],
                "val/fpr": val_dict["fpr"],
                "val/tpr": val_dict["tpr"],
                "val/loss": val_dict["loss"],
                # "val/acc_05": val_dict["acc_05"],
                # "val/f1_05": val_dict["f1_05"],
                # "val/sens_05": val_dict["sens_05"],
                # "val/spec_05": val_dict["spec_05"],
                # "val/kappa_05": val_dict["kappa_05"],
                "val/spec_at_09_sens": val_dict["spec_at_09_sens"],
                "val/threshold": val_dict["threshold"],
                "cm/val_cm_opt": wandb.Image(val_cm_opt_img) if (epoch % 5 == 0) else None,
                # "cm/val_cm_05": wandb.Image(val_cm_05_img) if (epoch % 5 == 0) else None,
                "val/acc_opt": val_dict["acc_opt"],
                "val/f1_opt": val_dict["f1_opt"],
                "val/sens_opt": val_dict["sens_opt"],
                "val/spec_opt": val_dict["spec_opt"],
                "val/kappa_opt": val_dict["kappa_opt"],
            }, step=epoch)

        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate best model on test set
        print(f"\nEvaluating BEST model on test set (fold {fold})\n")
        best_test_dict = self.evaluate_model(test_loader, epoch, self.loss_fn, name="best_test", fold=fold)
        
        # Create confusion matrix images
        best_test_cm_opt_img = plot_combined_conf_mat(best_test_dict["cm_opt"], epoch=None)
        
        # Create combined confusion matrix with stratified visualization
        best_test_comb_strat_cm_fig = plot_combined_with_stratified_cm(
            best_test_dict["cm_opt"], 
            best_test_dict["all_labels"], 
            best_test_dict["all_preds_opt"], 
            best_test_dict["all_original_labels"],
            task=self.clf_task_labels_key,
            use_multilabel=self.cfg['USE_MULTILABEL']
        )

        wandb.log({
        "best_test/auc": best_test_dict["auc"],
        "best_test/spec_at_09_sens": best_test_dict["spec_at_09_sens"],
        "best_test/threshold": best_test_dict["threshold"],
        "cm/best_test_cm_opt": wandb.Image(best_test_cm_opt_img),
        "cm/best_test_combined_stratified": wandb.Image(best_test_comb_strat_cm_fig),
        "best_test/acc_opt": best_test_dict["acc_opt"],
        "best_test/f1_opt": best_test_dict["f1_opt"],
        "best_test/sens_opt": best_test_dict["sens_opt"],
        "best_test/spec_opt": best_test_dict["spec_opt"],
        "best_test/kappa_opt": best_test_dict["kappa_opt"],
        "best_model_epoch": checkpoint['epoch']
    })
        

        test_dict = self.evaluate_model(test_loader, epoch, self.loss_fn, name="test", fold=fold)

        return test_dict
    
    def save_results_to_excel(experiment_name, fold_metrics, summary_metrics, excel_path="experiment_results.xlsx"):
        import pandas as pd
        """
        Save per-fold and summary metrics to an Excel file.
        """
        # Per-fold DataFrame
        df_folds = pd.DataFrame(fold_metrics)
        df_folds.insert(0, "fold", range(1, len(df_folds)+1))
        df_folds["experiment"] = experiment_name

        # Summary DataFrame (mean, std, var, ci95)
        summary_rows = []
        for metric, stats in summary_metrics.items():
            row = {"metric": metric}
            row.update(stats)
            row["experiment"] = experiment_name
            summary_rows.append(row)
        df_summary = pd.DataFrame(summary_rows)

        # Write to Excel (append if exists)
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a" if os.path.exists(excel_path) else "w") as writer:
            df_folds.to_excel(writer, sheet_name=f"{experiment_name}_folds", index=False)
            df_summary.to_excel(writer, sheet_name=f"{experiment_name}_summary", index=False)
    

    def run_k_folds(self):
        project_name = self.cfg["PROJECT_NAME"]
        exp_name = self.cfg["RUN_NAME"]

        accuracies = []
        sensitivities = []
        kappas = []
        specificities = []
        fold_metrics = []


       # wandb.login()

        group_name = f"{self.exp_name}-{wandb.util.generate_id()}"

        for i in range(self.cfg["NUM_FOLDS"]):

            if self.cfg["RESOLUTION"] == 512:
                model = xrv.models.ResNet(weights="resnet50-res512-all")

                if not self.cfg["PRETRAINED"]:
                    # raise ValueError("need to still fix random init.")
                    reinitialize_weights(model)
                elif self.cfg["USE_RAND_V3_WEIGHTS"]:
                    model = load_simclr_encoder(model, "/home/sean/MSc_2025/simclr/checkpoints/simclr_phoenix-mpt-t_05-grad_acc2_final_simclr_model.pth")
            else:
                raise ValueError(f"Unsupported resolution: {self.cfg['RESOLUTION']}")

            if(self.cfg["CLF_TYPE"] == "Linear"):
                model.classifier = XRVBasedClassifier(input_dim=2048, num_classes=1, name="bin_XRV-Base")
            elif(self.cfg["CLF_TYPE"] == "MLP"):
                model.classifier = BinaryClassifier(input_dim=2048, name="bin_mlp-Base")
            elif(self.cfg["CLF_TYPE"] == "MLP2"):
                model.classifier = BinaryClassifier(input_dim=2048, dropout_rate=0.1, name="bin_mlp-dout_01")
            else:
                raise ValueError(f"Unsupported classifier type: {self.cfg['CLF_TYPE']}")
            
            model = model.to(self.device)
            self.model = model

            if self.cfg["FREEZE_ENC"]:
                for name, param in self.model.named_parameters():
                # Only freeze parameters that are NOT in the classifier
                    if not name.startswith("classifier"):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                # Only optimize classifier parameters
                optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=config["LEARNING_RATE"], weight_decay=self.cfg["WEIGHT_DECAY"])
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=config["LEARNING_RATE"], weight_decay=self.cfg["WEIGHT_DECAY"])



            wandb.init(
                project=project_name,
                name=f"{exp_name}-fold-{i+1}",
                group=group_name,
                config={
                    "loss_fn": self.loss_fn.__class__.__name__,
                    "optimizer": self.optimizer_type.__name__,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "labels_key": self.labels_key,
                    "batch_size":self.batch_size,
                    "epochs": self.epochs,
                    "save_dir": self.checkpoint_save_target,
                    "fold": i
                },
            )

            wandb.config.update(self.cfg)

            test_dict = self.train(self.cfg, optimizer, fold=i)


            accuracies.append(test_dict["acc_opt"])
            sensitivities.append(test_dict["sens_opt"])
            kappas.append(test_dict["kappa_opt"])
            specificities.append(test_dict["spec_opt"])

                    # NEW: Collect detailed metrics for this fold
            fold_metric = {
                "acc": test_dict["acc_opt"],
                "sens": test_dict["sens_opt"],
                "spec": test_dict["spec_opt"],
                "kappa": test_dict["kappa_opt"],
                "f1": test_dict["f1_opt"],
                "auc": test_dict["auc"],
                "spec_at_09_sens": test_dict["spec_at_09_sens"],
                "threshold": test_dict["threshold"]
            }
            fold_metrics.append(fold_metric)


           # test_cm_05_img = plot_combined_conf_mat(test_dict["cm_05"], epoch=None)
            
            test_cm_opt_img = plot_combined_conf_mat(test_dict["cm_opt"], epoch=None)


            wandb.log({
                "test/auc": test_dict["auc"],
                "test/fpr": test_dict["fpr"],
                "test/tpr": test_dict["tpr"],
                "test/loss": test_dict["loss"],
                #"test/acc_05": test_dict["acc_05"],
                #"test/f1_05": test_dict["f1_05"],
                #"test/sens_05": test_dict["sens_05"],
                #"test/spec_05": test_dict["spec_05"],
                #"test/kappa_05": test_dict["kappa_05"],
                "test/spec_at_09_sens": test_dict["spec_at_09_sens"],
                "test/threshold": test_dict["threshold"],
                "cm/test_cm_opt": wandb.Image(test_cm_opt_img),
                #"cm/test_cm_05": wandb.Image(test_cm_05_img),
                "test/acc_opt": test_dict["acc_opt"],
                "test/f1_opt": test_dict["f1_opt"],
                "test/sens_opt": test_dict["sens_opt"],
                "test/spec_opt": test_dict["spec_opt"],
                "test/kappa_opt": test_dict["kappa_opt"],
            })

            wandb.finish()

        results = {
            "accuracy": mean_std_var_ci(accuracies),
            "sensitivity": mean_std_var_ci(sensitivities),
            "specificity": mean_std_var_ci(specificities),
            "kappa": mean_std_var_ci(kappas)
        }
        

        return results


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


class FocalLoss(nn.Module):
    """
    Binary Focal Loss with optional class balancing via pos_weight.
    Equivalent to BCEWithLogitsLoss + focal term.
    """

    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, reduction="mean"):
        """
        Args:
            alpha (float): Scaling factor to balance overall loss (like BCE alpha).
            gamma (float): Focusing parameter; higher = more focus on hard examples.
            pos_weight (float, optional): Same meaning as in BCEWithLogitsLoss.
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Raw logits, shape (N,).
            targets (Tensor): Binary targets {0,1}, shape (N,).
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction="none"  # we'll handle reduction after focal scaling
        )

        # Probabilities
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal scaling
        focal_factor = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def calculate_pos_weight(train_loader, task_key=None):
    pos = 0
    neg = 0
    for _, labels in train_loader:
        if labels.dim() == 2:
            if task_key == "silicosis":
                labels = labels[:, 0]
            elif task_key == "tuberculosis":
                labels = labels[:, 1]
            else:
                raise ValueError("Unknown task_key in calculate_pos_weight")
        labels = labels.cpu().numpy()
        pos += (labels == 1).sum()
        neg += (labels == 0).sum()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)

if __name__ == "__main__":

    config_files = glob.glob("configs/*.yaml")

    for cfg_path in config_files:
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
            print(config)


        # Initialize CUDA device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model.classifier = XRVBasedClassifier(input_dim=2048, num_classes=1, name="bin_XRV-Base")
        model = model.to(device)

        if(config["TRAIN_SET_NAME"] == "TBNET"):
            hdf5_path = config["DATA_PATH_TBNET"]
        elif(config["TRAIN_SET_NAME"] == "MBOD"):
            hdf5_path = config["DATA_PATH_MBOD"]
        elif(config["TRAIN_SET_NAME"] == "RAND"):
            hdf5_path = config["DATA_PATH_RAND"]
        else:
            raise ValueError("Unsupported TRAIN_SET_NAME in config")



        cv = BinaryMonteCarloCV(
            model=model,
            cfg=config,
            labels_key=config["LABELS_KEY"],  # Specify your target label
            num_classes=config["NUM_CLASSES"],  # Binary classification
            device=device,
            loss_fn=None,
            hdf5_path=hdf5_path,
            batch_size=config["BATCH_SIZE"],
            epochs=config["EPOCHS"],
            optimizer_type=torch.optim.Adam,
            learning_rate=config["LEARNING_RATE"],
            weight_decay=config["WEIGHT_DECAY"],
            use_oversampling=config["OVERSAMPLE"],
            checkpoint_save_target=config["CHECKPOINT_SAVE_DIR"],
            exp_name=config["RUN_NAME"],
            split_file=None,
            train_set_name=config["TRAIN_SET_NAME"],
            test_set_name=config["TEST_SET_NAME"],
            clf_task_labels_key=config["CLF_TASK_LABELS_KEY"]
        )

        results = cv.run_k_folds()
        print(f"Results for {config['RUN_NAME']}: {results}")

