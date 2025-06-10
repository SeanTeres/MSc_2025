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
import pandas as pd
from datetime import datetime
from tsne import MultiClassBaseClassifier, extract_model_type
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


def plot_and_save_confusion_matrices(knn, train_features, train_labels, val_features, val_labels, test_features, test_labels, save_path="confusion_matrices.png"):
    """
    Plot and save confusion matrices for training, validation, and test sets.

    Args:
        knn: Trained KNN classifier.
        train_features: Features for the training set.
        train_labels: Labels for the training set.
        val_features: Features for the validation set.
        val_labels: Labels for the validation set.
        test_features: Features for the test set.
        test_labels: Labels for the test set.
        save_path: Path to save the confusion matrix figure.
    """
    # Predict labels for each set
    train_predictions = knn.predict(train_features)
    val_predictions = knn.predict(val_features)
    test_predictions = knn.predict(test_features)

    # Compute confusion matrices
    train_cm = confusion_matrix(train_labels, train_predictions)
    val_cm = confusion_matrix(val_labels, val_predictions)
    test_cm = confusion_matrix(test_labels, test_predictions)

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
    plt.show()
    
    # Create binarized confusion matrices (class 0 vs all other classes)
    # Binarize the labels and predictions (0 remains 0, all other classes become 1)
    train_bin_labels = (train_labels > 0).astype(int)
    train_bin_preds = (train_predictions > 0).astype(int)
    
    val_bin_labels = (val_labels > 0).astype(int)
    val_bin_preds = (val_predictions > 0).astype(int)
    
    test_bin_labels = (test_labels > 0).astype(int)
    test_bin_preds = (test_predictions > 0).astype(int)
    
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
    plt.show()



def extract_features_and_labels(model, data_loader, device):
    """
    Extract features and labels from a dataset using the model.

    Args:
        model: The PyTorch model used for feature extraction.
        data_loader: DataLoader for the dataset.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        features: A numpy array of extracted features.
        labels: A numpy array of corresponding labels.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Extract features
            features = model.features(images)
            features = F.normalize(features, p=2, dim=1)  # Normalize features

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all features and labels
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_features, all_labels

def train_knn_classifier(train_features, train_labels, n_neighbors=5):
    """
    Train a KNN classifier.

    Args:
        train_features: Features for training.
        train_labels: Labels for training.
        n_neighbors: Number of neighbors for KNN.

    Returns:
        knn: Trained KNN classifier.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_features, train_labels)
    return knn



def evaluate_knn_classifier(knn, val_features, val_labels):
    """
    Evaluate a KNN classifier on multi-class data with macro and weighted average metrics.

    Args:
        knn: Trained KNN classifier.
        val_features: Features for validation.
        val_labels: Ground truth labels for validation.

    Returns:
        A dictionary with overall accuracy, macro and weighted average metrics.
    """
    predictions = knn.predict(val_features)
    unique_classes = np.unique(val_labels)
    total_samples = len(val_labels)

    print("=== Sklearn Classification Report (Reference) ===")
    print(classification_report(val_labels, predictions))

    # Manual overall accuracy
    overall_accuracy = (predictions == val_labels).sum() / total_samples
    print(f"\nManual Overall Accuracy: {overall_accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(val_labels, predictions, labels=unique_classes)

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
    binary_labels = (val_labels > 0).astype(int)
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

def save_experiment_to_excel(experiment_name, checkpoint_path, results_val, results_test, results_all, excel_path="knn_experiment_results.xlsx"):
    """
    Save experiment results to an Excel file.
    
    Args:
        experiment_name: Name of the experiment
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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")
    
    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        

        # check_empty_study_ids(hdf5_file_path)

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
        mbod_dataset_merged = HDF5Dataset(
            hdf5_path=hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )


        ilo_dataset = HDF5Dataset(
            hdf5_path=ilo_hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )

        train_loader, val_loader, test_loader = get_dataloaders(
            hdf5_path=hdf5_file_path,
            preprocess=preprocess,
            batch_size=16,
            labels_key="profusion_score",
            split_file="stratified_split_filt.json",
            augmentations=None,
            oversample=True
        )

        # wandb.login()
        # wandb.init(project='MBOD-cl', name='img_test')
        
        # Load the saved model checkpoint
        
        experiments_list = ["NEW_mdp-BSHN-p_ilo_50-OS_aug-sin_m_01_05", "NEW_mdp-BSHN-p_ilo_50-OS_aug-sin_m_02_06", "NEW_mdp-BSHN-p_ilo_50-OS_aug-sin_m_01_05-fast",
                            "b24-BSHN-p_ilo_50-OS_aug-sin_m_01_04-slow", "b24-BSHN-p_ilo_50-OS_aug-sin_m_005_02", "clf_01-BSHN-p_ilo_50-OS_aug-sin_m_01_04",
                            "clf_02-BSHN-p_ilo_50-OS_aug-sin_m_01_05", "clf_025-BSHN-p_ilo_50-OS_aug-sin_m_01_05","clf_03-BSHN-p_ilo_50-OS_aug-sin_m_01_04"]
        
        model_checkpoints = ["final", "best"]
        
        for experiment_name in experiments_list:

            for checkpoint_path in model_checkpoints:
                checkpoint_path = f"/home/sean/MSc_2025/codev2/checkpoints/{experiment_name}/{checkpoint_path}_model.pth"
                print(f"Loading model from checkpoint: {checkpoint_path}")

                results_dir = "cl-logs"
                cl_dir = results_dir
                results_dir = os.path.join(results_dir, experiment_name)
                # Create checkpoint directory if it doesn't exist
                os.makedirs(results_dir, exist_ok=True)
                
                # Initialize model architecture (same as used during training)
                model = xrv.models.ResNet(weights="resnet50-res512-all")
                model = model.to(device)

                        
                raw_model = xrv.models.ResNet(weights="resnet50-res512-all")
                raw_model = raw_model.to(device)

                if(experiment_name.startswith("clf")):
                    model.classifier = MultiClassBaseClassifier(in_features=2048, num_classes=4).to(device)
                    raw_model.classifier = MultiClassBaseClassifier(in_features=2048, num_classes=4).to(device)
            
                
                # Initialize optimizer (needed for loading state)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                
                # Load checkpoint with model state, optimizer state, and epoch
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']

                
                print(f"Successfully loaded model from epoch {epoch}")

                # wandb.init(project="tsne-visualization")
                #
                mbod_merged_loader = torch.utils.data.DataLoader(
                    mbod_dataset_merged,
                    batch_size=16,
                    shuffle=False
                )


                # Extract features and labels for training and validation datasets
                train_features, train_labels = extract_features_and_labels(model, train_loader, device)
                val_features, val_labels = extract_features_and_labels(model, val_loader, device)
                test_features, test_labels = extract_features_and_labels(model, test_loader, device)

                all_features, all_labels = extract_features_and_labels(model, mbod_merged_loader, device)

                # Train the KNN classifier
                knn = train_knn_classifier(train_features, train_labels, n_neighbors=5)

                # Evaluate the KNN classifier
                print("\nVALIDATION SET:")
                results_val = evaluate_knn_classifier(knn, val_features, val_labels)

                print("\nTEST SET:")
                results_test = evaluate_knn_classifier(knn, test_features, test_labels)

                print("\nENTIRE SET:")
                results_all = evaluate_knn_classifier(knn, all_features, all_labels)

                # Save experiment results to Excel
                excel_path = os.path.join(cl_dir, "knn_experiment_results_new.xlsx")
                save_experiment_to_excel(experiment_name, checkpoint_path, results_val, results_test, results_all, excel_path)

                plot_and_save_confusion_matrices(knn, train_features, train_labels, val_features, val_labels, test_features, test_labels, save_path=f"{results_dir}/{experiment_name}_knn_cm")


                all_predictions = knn.predict(all_features)

                all_cm = confusion_matrix(all_labels, all_predictions)

                all_cm_normalized = all_cm.astype('float') / all_cm.sum(axis=1)[:, np.newaxis]

                fig, axes = plt.subplots(1, 1, figsize=(16, 6))

                sns.heatmap(all_cm, annot=True, fmt="d", cmap="Blues")

                axes.set_title("Entire Set Confusion Matrix")

                axes.set_xlabel("Predicted Labels")

                axes.set_ylabel("True Labels")

                plt.tight_layout()
                save_path = f"{experiment_name}_entire_set_confusion_matrix.png"
                plt.savefig(save_path)
                print(f"Confusion matrices saved to {save_path}")
                plt.show()

                del model, raw_model, knn, train_features, train_labels, val_features, val_labels, test_features, test_labels
                torch.cuda.empty_cache()







    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error loading model or generating visualizations: {e}")
        raise