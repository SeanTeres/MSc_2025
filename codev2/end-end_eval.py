import sys
import os


# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

sys.path.append(os.path.abspath("/home/sean/MSc_2025/codev2"))

from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datasets.dataloader import get_dataloaders
import torchxrayvision as xrv
import torch
from train_utils import classes, helpers
import torch.nn.functional as F
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score
import seaborn as sns
from sklearn.calibration import calibration_curve
import io
import torchvision.transforms as transforms
import os
import math
import random
import pandas as pd
from datetime import datetime


from datasets.utils import ILO_CLASSIFICATION_DICTIONARY
from tsne import visualize_multiple_tsne_3d_with_ilo2, visualize_tsne, MultiClassBaseClassifier, extract_model_type

from medvae.medvae_main import MVAE
from datasets.kaggle_tb import KaggleTBDataset  # Import the KaggleTBDataset class


def evaluate_classifier_performance(model, data_loader, device):
    """
    Evaluate model classifier performance on a dataset.

    Args:
        model: The PyTorch model used for evaluation.
        data_loader: DataLoader for the dataset.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        Dictionary containing predictions, true labels, and features.
    """
    model.eval()
    all_features = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Extract features
            features = model.features(images)
            embeddings = F.normalize(features, p=2, dim=1)  # Normalize features
            
            # Get predictions
            predictions = model.classifier(features)
            _, predicted_classes = torch.max(predictions, 1)

            all_features.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(predicted_classes.cpu())

    # Concatenate all features and labels
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()

    return {
        "features": all_features,
        "true_labels": all_labels,
        "predictions": all_preds
    }


def calculate_metrics(true_labels, predictions):
    """
    Calculate multi-class and binary metrics for model evaluation.

    Args:
        true_labels: Ground truth labels.
        predictions: Model predictions.

    Returns:
        A dictionary with various performance metrics.
    """
    unique_classes = np.unique(true_labels)
    total_samples = len(true_labels)

    print("=== Sklearn Classification Report (Reference) ===")
    print(classification_report(true_labels, predictions))

    # Manual overall accuracy
    overall_accuracy = (predictions == true_labels).sum() / total_samples
    print(f"\nManual Overall Accuracy: {overall_accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=unique_classes)

    # Lists for macro and weighted aggregation
    precision_list = []
    recall_list = []
    f1_list = []
    specificity_list = []
    accuracy_list = []
    kappa_list = []
    support_list = []

    for idx, cls in enumerate(unique_classes):
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        support = cm[idx, :].sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0.0

        # Kappa for the class (Cohen's kappa as binary for class vs rest)
        expected_acc = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total_samples ** 2)
        kappa = (accuracy - expected_acc) / (1 - expected_acc) if (1 - expected_acc) > 0 else 0.0

        # Append metrics for aggregation
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        specificity_list.append(specificity)
        accuracy_list.append(accuracy)
        kappa_list.append(kappa)
        support_list.append(support)

    support_array = np.array(support_list)
    total_support = support_array.sum()

    def weighted_avg(values):
        return np.average(values, weights=support_array)

    # Macro averages (unweighted)
    macro_avg = {
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1_score": np.mean(f1_list),
        "specificity": np.mean(specificity_list),
        "accuracy": np.mean(accuracy_list),
        "kappa": np.mean(kappa_list),
        "support": total_support
    }

    # Weighted averages (by support)
    weighted_avg_metrics = {
        "precision": weighted_avg(precision_list),
        "recall": weighted_avg(recall_list),
        "f1_score": weighted_avg(f1_list),
        "specificity": weighted_avg(specificity_list),
        "accuracy": weighted_avg(accuracy_list),
        "kappa": weighted_avg(kappa_list),
        "support": total_support
    }

    # Calculate binary metrics (class 0 vs all others combined)
    binary_labels = (true_labels > 0).astype(int)
    binary_preds = (predictions > 0).astype(int)
    
    # Binary confusion matrix
    binary_cm = confusion_matrix(binary_labels, binary_preds)
    
    # Extract values from binary confusion matrix
    if binary_cm.shape == (2, 2):
        binary_tn, binary_fp, binary_fn, binary_tp = binary_cm.ravel()
    else:
        # Handle cases where not all classes are present
        binary_tp = binary_tn = binary_fp = binary_fn = 0
        for i in range(len(binary_labels)):
            if binary_labels[i] == 1 and binary_preds[i] == 1:
                binary_tp += 1
            elif binary_labels[i] == 0 and binary_preds[i] == 0:
                binary_tn += 1
            elif binary_labels[i] == 0 and binary_preds[i] == 1:
                binary_fp += 1
            elif binary_labels[i] == 1 and binary_preds[i] == 0:
                binary_fn += 1
    
    # Calculate binary metrics
    binary_accuracy = (binary_tp + binary_tn) / total_samples
    binary_precision = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) > 0 else 0.0
    binary_recall = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) > 0 else 0.0
    binary_specificity = binary_tn / (binary_tn + binary_fp) if (binary_tn + binary_fp) > 0 else 0.0
    binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0.0
    
    # Cohen's kappa for binary case
    binary_expected_acc = ((binary_tp + binary_fp) * (binary_tp + binary_fn) + (binary_fn + binary_tn) * (binary_fp + binary_tn)) / (total_samples ** 2)
    binary_kappa = (binary_accuracy - binary_expected_acc) / (1 - binary_expected_acc) if (1 - binary_expected_acc) > 0 else 0.0
    
    binary_metrics = {
        "accuracy": binary_accuracy,
        "precision": binary_precision,
        "recall": binary_recall,
        "specificity": binary_specificity,
        "f1_score": binary_f1,
        "kappa": binary_kappa,
        "tp": binary_tp,
        "fp": binary_fp,
        "fn": binary_fn,
        "tn": binary_tn
    }
    
    print("\n=== Binary Metrics (Class 0 vs. Others) ===")
    print(f"Accuracy: {binary_accuracy:.4f}")
    print(f"Precision: {binary_precision:.4f}")
    print(f"Recall/Sensitivity: {binary_recall:.4f}")
    print(f"Specificity: {binary_specificity:.4f}")
    print(f"F1 Score: {binary_f1:.4f}")
    print(f"Kappa: {binary_kappa:.4f}")
    print(f"TP: {binary_tp}, FP: {binary_fp}, FN: {binary_fn}, TN: {binary_tn}")

    print("\n=== Macro Average Metrics ===")
    for k, v in macro_avg.items():
        print(f"{k.capitalize()}: {v:.4f}" if k != "support" else f"{k.capitalize()}: {v}")

    print("\n=== Weighted Average Metrics ===")
    for k, v in weighted_avg_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}" if k != "support" else f"{k.capitalize()}: {v}")

    return {
        "overall_accuracy": overall_accuracy,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg_metrics,
        "binary": binary_metrics,
        "unique_classes": unique_classes.tolist()
    }


def save_experiment_to_excel(experiment_name, checkpoint_path, results_val, results_test, results_all, excel_path="end-end_clf_logs.xlsx"):
    """
    Save experiment results to an Excel file.
    
    Args:
        experiment_name: Name of the experiment
        checkpoint_path: Path to the model checkpoint
        results_val: Dictionary containing validation results
        results_test: Dictionary containing test results
        results_all: Dictionary containing results from the entire dataset
        excel_path: Path to the Excel file
    """
    # Create a row with all metrics
    data = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': checkpoint_path,
        # Validation set metrics
        'val_overall_accuracy': results_val['overall_accuracy'],
        'val_macro_precision': results_val['macro_avg']['precision'],
        'val_macro_recall': results_val['macro_avg']['recall'],
        'val_macro_f1': results_val['macro_avg']['f1_score'],
        'val_macro_specificity': results_val['macro_avg']['specificity'],
        'val_weighted_precision': results_val['weighted_avg']['precision'],
        'val_weighted_recall': results_val['weighted_avg']['recall'],
        'val_weighted_f1': results_val['weighted_avg']['f1_score'],
        'val_weighted_specificity': results_val['weighted_avg']['specificity'],
        
        # Binary metrics for validation
        'val_binary_accuracy': results_val['binary']['accuracy'],
        'val_binary_precision': results_val['binary']['precision'],
        'val_binary_recall': results_val['binary']['recall'],
        'val_binary_specificity': results_val['binary']['specificity'],
        'val_binary_f1': results_val['binary']['f1_score'],
        'val_binary_kappa': results_val['binary']['kappa'],
        'val_binary_tp': results_val['binary']['tp'],
        'val_binary_fp': results_val['binary']['fp'],
        'val_binary_fn': results_val['binary']['fn'],
        'val_binary_tn': results_val['binary']['tn'],
        
        # Test set metrics
        'test_overall_accuracy': results_test['overall_accuracy'],
        'test_macro_precision': results_test['macro_avg']['precision'],
        'test_macro_recall': results_test['macro_avg']['recall'],
        'test_macro_f1': results_test['macro_avg']['f1_score'],
        'test_macro_specificity': results_test['macro_avg']['specificity'],
        'test_weighted_precision': results_test['weighted_avg']['precision'],
        'test_weighted_recall': results_test['weighted_avg']['recall'],
        'test_weighted_f1': results_test['weighted_avg']['f1_score'],
        'test_weighted_specificity': results_test['weighted_avg']['specificity'],
        
        # Binary metrics for test
        'test_binary_accuracy': results_test['binary']['accuracy'],
        'test_binary_precision': results_test['binary']['precision'],
        'test_binary_recall': results_test['binary']['recall'],
        'test_binary_specificity': results_test['binary']['specificity'],
        'test_binary_f1': results_test['binary']['f1_score'],
        'test_binary_kappa': results_test['binary']['kappa'],
        'test_binary_tp': results_test['binary']['tp'],
        'test_binary_fp': results_test['binary']['fp'],
        'test_binary_fn': results_test['binary']['fn'],
        'test_binary_tn': results_test['binary']['tn'],
        
        # Entire set metrics
        'all_overall_accuracy': results_all['overall_accuracy'],
        'all_macro_precision': results_all['macro_avg']['precision'],
        'all_macro_recall': results_all['macro_avg']['recall'],
        'all_macro_f1': results_all['macro_avg']['f1_score'],
        'all_macro_specificity': results_all['macro_avg']['specificity'],
        'all_weighted_precision': results_all['weighted_avg']['precision'],
        'all_weighted_recall': results_all['weighted_avg']['recall'],
        'all_weighted_f1': results_all['weighted_avg']['f1_score'],
        'all_weighted_specificity': results_all['weighted_avg']['specificity'],
        
        # Binary metrics for all
        'all_binary_accuracy': results_all['binary']['accuracy'],
        'all_binary_precision': results_all['binary']['precision'],
        'all_binary_recall': results_all['binary']['recall'],
        'all_binary_specificity': results_all['binary']['specificity'],
        'all_binary_f1': results_all['binary']['f1_score'],
        'all_binary_kappa': results_all['binary']['kappa'],
        'all_binary_tp': results_all['binary']['tp'],
        'all_binary_fp': results_all['binary']['fp'],
        'all_binary_fn': results_all['binary']['fn'],
        'all_binary_tn': results_all['binary']['tn'],
    }
    
    # Convert to DataFrame
    df_new = pd.DataFrame([data])
    
    # Check if file exists
    if os.path.exists(excel_path):
        # Read existing data and append new row
        df_existing = pd.read_excel(excel_path)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new
    
    # Save to Excel
    df_updated.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")


def plot_and_save_confusion_matrices(true_labels, predictions, train_labels, train_preds, val_labels, val_preds, experiment_name, save_path="confusion_matrices.png"):
    """
    Plot and save confusion matrices for training, validation, and test sets.

    Args:
        true_labels: Test set true labels.
        predictions: Test set predictions.
        train_labels: Training set true labels.
        train_preds: Training set predictions.
        val_labels: Validation set true labels.
        val_preds: Validation set predictions.
        experiment_name: Name of the experiment for the plot title.
        save_path: Path to save the confusion matrix figure.
    """
    # Compute confusion matrices
    test_cm = confusion_matrix(true_labels, predictions)
    train_cm = confusion_matrix(train_labels, train_preds)
    val_cm = confusion_matrix(val_labels, val_preds)

    # Normalize confusion matrices
    train_cm_normalized = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]
    val_cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]
    test_cm_normalized = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"{experiment_name} -- Confusion Matrices", fontsize=20)
    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0][0])
    axes[0][0].set_title("Training Set Confusion Matrix")
    axes[0][0].set_xlabel("Predicted Labels")
    axes[0][0].set_ylabel("True Labels")

    sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0][1])
    axes[0][1].set_title("Validation Set Confusion Matrix")
    axes[0][1].set_xlabel("Predicted Labels")
    axes[0][1].set_ylabel("True Labels")

    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0][2])
    axes[0][2].set_title("Test Set Confusion Matrix")
    axes[0][2].set_xlabel("Predicted Labels")
    axes[0][2].set_ylabel("True Labels")

    sns.heatmap(train_cm_normalized, annot=True, cmap="Blues", ax=axes[1][0])
    axes[1][0].set_title("Normalized Training Set Confusion Matrix")
    axes[1][0].set_xlabel("Predicted Labels")
    axes[1][0].set_ylabel("True Labels")

    sns.heatmap(val_cm_normalized, annot=True, cmap="Blues", ax=axes[1][1])
    axes[1][1].set_title("Normalized Validation Set Confusion Matrix")
    axes[1][1].set_xlabel("Predicted Labels")
    axes[1][1].set_ylabel("True Labels")

    sns.heatmap(test_cm_normalized, annot=True, cmap="Blues", ax=axes[1][2])
    axes[1][2].set_title("Normalized Test Set Confusion Matrix")
    axes[1][2].set_xlabel("Predicted Labels")
    axes[1][2].set_ylabel("True Labels")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrices saved to {save_path}")
    plt.close()
    
    # Create binarized confusion matrices (class 0 vs all other classes)
    # Binarize the labels and predictions (0 remains 0, all other classes become 1)
    train_bin_labels = (train_labels > 0).astype(int)
    train_bin_preds = (train_preds > 0).astype(int)
    
    val_bin_labels = (val_labels > 0).astype(int)
    val_bin_preds = (val_preds > 0).astype(int)
    
    test_bin_labels = (true_labels > 0).astype(int)
    test_bin_preds = (predictions > 0).astype(int)
    
    # Compute binary confusion matrices
    train_bin_cm = confusion_matrix(train_bin_labels, train_bin_preds)
    val_bin_cm = confusion_matrix(val_bin_labels, val_bin_preds)
    test_bin_cm = confusion_matrix(test_bin_labels, test_bin_preds)
    
    # Plot binarized confusion matrices
    fig_bin, axes_bin = plt.subplots(1, 3, figsize=(18, 6))
    fig_bin.suptitle(f"{experiment_name} -- Binarized Confusion Matrices (Class 0 vs Others)", fontsize=20)
    
    labels = ["Class 0", "Other Classes"]
    
    # Training set
    sns.heatmap(train_bin_cm, annot=True, fmt="d", cmap="Blues", 
               xticklabels=labels, yticklabels=labels, ax=axes_bin[0])
    axes_bin[0].set_title("Training Set - Binary Confusion Matrix")
    axes_bin[0].set_xlabel("Predicted Labels")
    axes_bin[0].set_ylabel("True Labels")
    
    # Validation set
    sns.heatmap(val_bin_cm, annot=True, fmt="d", cmap="Blues", 
               xticklabels=labels, yticklabels=labels, ax=axes_bin[1])
    axes_bin[1].set_title("Validation Set - Binary Confusion Matrix")
    axes_bin[1].set_xlabel("Predicted Labels")
    axes_bin[1].set_ylabel("True Labels")
    
    # Test set
    sns.heatmap(test_bin_cm, annot=True, fmt="d", cmap="Blues", 
               xticklabels=labels, yticklabels=labels, ax=axes_bin[2])
    axes_bin[2].set_title("Test Set - Binary Confusion Matrix")
    axes_bin[2].set_xlabel("Predicted Labels")
    axes_bin[2].set_ylabel("True Labels")
    
    # Adjust layout and save the figure
    binary_save_path = save_path.replace('.png', '_binary.png')
    if binary_save_path == save_path:
        binary_save_path = f"{save_path.split('.')[0]}_binary.png"
    
    plt.tight_layout()
    plt.savefig(binary_save_path)
    print(f"Binary confusion matrices saved to {binary_save_path}")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")
    
    preprocess = transforms.Compose([
    # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]
     

        # Create an HDF5SilicosisDataset instance
        mbod_dataset_merged = HDF5Dataset(
            hdf5_path=hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )
        ilo_dataset = HDF5Dataset2(
            hdf5_path=ilo_hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )

        # Path to Kaggle TB dataset
        kaggle_tb_path = config["kaggle_TB"]["outputpath"]  # Ensure this is set in config.yaml

        # Create an instance of KaggleTBDataset
        kaggle_tb_dataset_original = HDF5Dataset(
            hdf5_path=kaggle_tb_path,
            labels_key="tuberculosis",
            preprocess=preprocess
        )

        vae_model = MVAE(
        model_name='medvae_4_3_2d',
        modality='xray',
        ).to(device)
        vae_model.requires_grad_(False)
        vae_model.eval()

        ae = vae_model.model

        # Getting the transform and applying it
        transform = vae_model.get_transform()

        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = model.to(device)


        # Retrieve the labels
        labels = mbod_dataset_merged.get_labels()

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

        oversample = True
        batch_size = 16



        # Print the first 10 labels with their descriptions
        print("Multiclass STB Labels and Descriptions:")
        for i, label in enumerate(labels[:10]):
            description = multiclass_stb_mapping[label]
            print(f"Sample {i}: Label {label} - {description}")


        augmentations_list = transforms.Compose([
            transforms.RandomRotation(degrees=10, expand=False, fill=0),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
        ])

        # Get the dataloaders
        train_loader, _, _ = get_dataloaders(
            hdf5_path=hdf5_file_path,
            preprocess=preprocess,
            batch_size=16,
            labels_key="profusion_score",
            split_file="stratified_split_filt.json",
            augmentations=augmentations_list,
            oversample=None,
            balanced_batches=False
        )

        _, val_loader, test_loader = get_dataloaders(
            hdf5_path=hdf5_file_path,
            preprocess=preprocess,
            batch_size=16,
            labels_key="profusion_score",
            split_file="stratified_split_filt.json",
            augmentations=None,
            oversample=None,
            balanced_batches=False
        )
        
        
        # Initialize model architecture (same as used during training)
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = model.to(device)
        # # Print label distributions 
        # print("\n===== TRAIN DATALOADER =====")
        # print_dataloader_label_distribution(train_loader, multiclass_stb_mapping)

        # print("\n===== VALIDATION DATALOADER =====")
        # print_dataloader_label_distribution(val_loader, multiclass_stb_mapping) 

        # print("\n===== TEST DATALOADER =====")
        # print_dataloader_label_distribution(test_loader, multiclass_stb_mapping)


        experiments_list = ["clf_01-BSHN-p_ilo_50-OS_aug-sin_m_01_04", "clf_02-BSHN-p_ilo_50-OS_aug-sin_m_01_05", "clf_025-BSHN-p_ilo_50-OS_aug-sin_m_01_05",
                            "clf_03-BSHN-p_ilo_50-OS_aug-sin_m_01_04"]
        
        model_checkpoints = ["final", "best"]
        
        # Create the directory for end-to-end classifier logs if it doesn't exist
        results_dir = "end-end_clf_logs"
        os.makedirs(results_dir, exist_ok=True)
        
        for experiment_name in experiments_list:

            for checkpoint_path in model_checkpoints:
                full_checkpoint_path = f"/home/sean/MSc_2025/codev2/checkpoints/{experiment_name}/{checkpoint_path}_model.pth"
                print(f"Loading model from checkpoint: {full_checkpoint_path}")

                # Create experiment-specific directory if it doesn't exist
                exp_results_dir = os.path.join(results_dir, experiment_name)
                os.makedirs(exp_results_dir, exist_ok=True)
                
                # Initialize model architecture
                model = xrv.models.ResNet(weights="resnet50-res512-all")
                model = model.to(device)
                
                if experiment_name.startswith("clf"):
                    model.classifier = MultiClassBaseClassifier(in_features=2048, num_classes=4).to(device)
            
                # Initialize optimizer (needed for loading state)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                
                # Load checkpoint with model state, optimizer state, and epoch
                checkpoint = torch.load(full_checkpoint_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                
                print(f"Successfully loaded model from epoch {epoch}")

                # Create dataloader for the full dataset
                mbod_merged_loader = torch.utils.data.DataLoader(
                    mbod_dataset_merged,
                    batch_size=16,
                    shuffle=False
                )

                # Evaluate model performance on different datasets
                print("\nEvaluating model on datasets...")
                print("\nTRAIN SET:")
                train_results = evaluate_classifier_performance(model, train_loader, device)
                
                print("\nVALIDATION SET:")
                val_results = evaluate_classifier_performance(model, val_loader, device)
                
                print("\nTEST SET:")  
                test_results = evaluate_classifier_performance(model, test_loader, device)
                
                print("\nALL DATA:")
                all_results = evaluate_classifier_performance(model, mbod_merged_loader, device)

                # Calculate metrics for each dataset
                print("\nCalculating metrics for Train set:")
                train_metrics = calculate_metrics(train_results["true_labels"], train_results["predictions"])
                
                print("\nCalculating metrics for Validation set:")
                val_metrics = calculate_metrics(val_results["true_labels"], val_results["predictions"])
                
                print("\nCalculating metrics for Test set:")
                test_metrics = calculate_metrics(test_results["true_labels"], test_results["predictions"])
                
                print("\nCalculating metrics for All data:")
                all_metrics = calculate_metrics(all_results["true_labels"], all_results["predictions"])

                # Save experiment results to Excel
                excel_path = os.path.join(results_dir, "end-end_clf_logs.xlsx")
                save_experiment_to_excel(experiment_name, full_checkpoint_path, val_metrics, test_metrics, all_metrics, excel_path)

                # Plot and save confusion matrices
                cm_save_path = os.path.join(exp_results_dir, f"{experiment_name}_{checkpoint_path}_cm.png")
                plot_and_save_confusion_matrices(
                    test_results["true_labels"], 
                    test_results["predictions"],
                    train_results["true_labels"], 
                    train_results["predictions"],
                    val_results["true_labels"], 
                    val_results["predictions"],
                    f"{experiment_name} ({checkpoint_path})",
                    cm_save_path
                )

                # Generate and save confusion matrix for all data
                all_cm = confusion_matrix(all_results["true_labels"], all_results["predictions"])
                plt.figure(figsize=(10, 8))
                sns.heatmap(all_cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"{experiment_name} ({checkpoint_path}) - All Data Confusion Matrix")
                plt.xlabel("Predicted Labels")
                plt.ylabel("True Labels")
                plt.tight_layout()
                all_cm_path = os.path.join(exp_results_dir, f"{experiment_name}_{checkpoint_path}_all_cm.png")
                plt.savefig(all_cm_path)
                plt.close()
                print(f"All data confusion matrix saved to {all_cm_path}")

                # Clean up to free memory
                del model
                torch.cuda.empty_cache()

    except KeyError as e:
        print(f"Missing configuration: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
