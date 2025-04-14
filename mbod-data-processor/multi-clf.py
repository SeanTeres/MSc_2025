from datasets.hdf_dataset import HDF5SilicosisDataset
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from dataloader import get_dataloaders, get_stratified_dataloaders
import torchxrayvision as xrv
import torch
from train_utils import classes
import torch.nn.functional as F
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score
import seaborn as sns
from sklearn.calibration import calibration_curve
import io
import os


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


def explore_hdf5_dataset(hdf5_path, num_samples=5):
    """
    Explore and display various fields from an HDF5 dataset
    
    Args:
        hdf5_path: Path to the HDF5 file
        num_samples: Number of samples to display (default: 5)
    """
    with h5py.File(hdf5_path, "r") as f:
        # Print available fields
        print("Available fields in the dataset:")
        for key in f.keys():
            print(f"- {key}: shape {f[key].shape}, dtype {f[key].dtype}")
        
        # Get total number of samples
        total_samples = f["images"].shape[0]
        print(f"\nTotal number of samples: {total_samples}")
        
        # Select random indices to explore
        indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
        
        print(f"\nExploring {len(indices)} random samples:")
        
        for idx in indices:
            print(f"\nSample {idx}:")
            
            # Study ID
            study_id = f["study_id"][idx]
            if isinstance(study_id, bytes):
                study_id = study_id.decode('utf-8')
            print(f"  Study ID: {study_id}")
            
            # Labels
            print(f"  Lab: {f['lab'][idx]}")
            
            # Disease flags
            print(f"  Tuberculosis: {f['tuberculosis'][idx]}")
            print(f"  Silicosis: {f['silicosis'][idx]}")
            print(f"  Silicosis+TB: {f['silicosis_tuberculosis'][idx]}")
            print(f"  Active TB: {f['active_tuberculosis'][idx]}")
            print(f"  Full TB: {f['full_tuberculosis'][idx]}")
            print(f"  Profusion Score: {f['profusion_score'][idx]}")

            print(f"Shape: {f['images'][idx].shape}")
            print(f"min, max: {f['images'][idx].min()}, {f['images'][idx].max()}")
            
            # Display the image
            # plt.figure(figsize=(5, 5))
            # plt.imshow(f["images"][idx], cmap='gray')
            # plt.title(f"Sample {idx} - Profusion: {f['profusion_score'][idx]}")
            # plt.axis('off')
            # plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config()
    
    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        

        check_empty_study_ids(hdf5_file_path)


        # Create an HDF5SilicosisDataset instance
        mbod_dataset_merged = HDF5SilicosisDataset(
            hdf5_file_path=hdf5_file_path,
            labels_key="lab",  # Main pathology labels, 'lab' for all labels
            image_key="images",
            label_metadata=LABEL_SCHEMES,
            data_aug=None,
            transform=None
        )

       # prof_labels = get_label_scheme_supports(d_857, 'slicosis')

        # explore_hdf5_dataset(hdf5_file_path, num_samples=10)
        wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
        experiment_name = "mb_clf-comb-OS-th_04-70"
        wandb.init(project='MBOD-multi', name=experiment_name,
           config={
               "batch_size": 16,
               "augmentation": False,
               "lr": 0.001,
               "model": "resnet50-res512-all",
               "epochs": 70,
               "oversampling": True,
               "train_dataset": "MBOD Combined",
               "binary_thresh": 0.4
           })
    
        # Get the dataloaders
        train_loader, val_loader, test_loader = get_stratified_dataloaders(
            hdf5_file_path,
            batch_size=wandb.config.batch_size,
            labels_key="profusion_score",
            image_key="images",
            oversample=wandb.config.oversampling
        )


        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model.classifier = classes.MultiClassBaseClassifier(in_features=2048, num_classes=4).to(device)
        model = model.to(device)

        multi_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
        binary_criterion = nn.BCELoss()

        best_val_multi_loss = float('inf')  # Start with a very high loss
        best_val_multi_f1 = 0.0  # Start with a very low F1 score

        # Directory to save the best model
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(wandb.config.epochs):
            print(f"Epoch {epoch+1}/{wandb.config.epochs}")
            print("=" * 50)
            model.train()

            multi_true_labels = []  # For multi-class
            multi_pred_labels = []  # For multi-class (argmax of logits)
            multi_pred_probs = []   # For multi-class (all probabilities)
            
            # For binary task
            binary_true_labels = []
            binary_preds = []
            binary_probs = []

            train_b_loss = 0.0
            train_m_loss = 0.0


            for idx, sample in enumerate(train_loader):
                
                # Get the batch data
                img = sample["img"]  # Shape: [batch_size, channels, height, width]
                img = img.unsqueeze(1)
                lab = sample["lab"].to(device)
                lab = lab.long()
                study_id = sample["image_id"]



                binary_labels = (lab >= 1).long().to(device)  # or .int() if you want integer 0/1

                img = img.to(device)

                feats = model.features(img)
                logits = model.classifier(feats)


                # calculate multi-class loss
                m_loss = multi_criterion(logits, lab)

                probs = F.softmax(logits, dim=1)
                #print(f"probs: {probs}")
                p_abnormal = torch.sum(probs[:, 1:], dim=1)  # Sum of all abnormal classes
                #print(f"p_abnormal: {p_abnormal}")            
                p_normal = 1 - p_abnormal
                #print(f"p_normal: {p_normal}")

                # print(f"p_abnormal: {p_abnormal}")
                # Calculate binary loss
                p_abnormal_clipped = torch.clamp(p_abnormal, 0.0, 1.0)
                b_loss = binary_criterion(p_abnormal_clipped, binary_labels.float())
                binary_preds_tensor = (p_normal <= wandb.config.binary_thresh).long()


                if((idx + 1) % 30 == 0):
                    print(f"Batch {idx+1}/{len(train_loader)} - Binary Loss: {b_loss.item():.4f} | Multi Loss: {m_loss.item():.4f}")
                    print(f"Binary Labels: {binary_labels.cpu().numpy()}")
                    print(f"Binary Predictions: {binary_preds_tensor.cpu().numpy()}")
                    print(f"binary_preds_tensor: {binary_preds_tensor}")
                    print(f"p_abnormal: {p_abnormal}")
                    print(f"p_normal: {p_normal}")
                    
                    print("\n" + "=" * 50 + "\n")
                    
                    print(f"Multi-class Labels: {lab.cpu().numpy()}")
                    print(f"Multi-class Predictions: {torch.argmax(probs, dim=1).cpu().numpy()}")
                    
                    print("\n" + "=" * 50 + "\n")
                    
                   # print(f"PROBS: {probs}")
                    
                    print("\n" + "=" * 50 + "\n")

                    thresholded_result = (p_normal < wandb.config.binary_thresh)
                    print(f"Threshold: {wandb.config.binary_thresh}")
                    print(f"p_normal: {p_normal}")
                    print(f"Comparison result: {thresholded_result}")
                    print(f"Resulting predictions: {thresholded_result.long()}")                   



                with torch.no_grad():
                    # Append true labels and predicted labels for multi-class
                    multi_true_labels.extend(lab.cpu().numpy())
                    multi_pred_labels.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    multi_pred_probs.extend(probs.cpu().numpy())

                    # Append true labels and predicted probabilities for binary task
                    binary_true_labels.extend(binary_labels.cpu().numpy())
                    binary_preds.extend(binary_preds_tensor.cpu().numpy())
                    binary_probs.extend(p_abnormal_clipped.cpu().numpy())

                    train_b_loss += b_loss.item()
                    train_m_loss += m_loss.item()

                wandb.log({
                    "batch": idx,
                    "batch_loss_binary": b_loss.item(),
                    "batch_loss_multi": m_loss.item(),
                })

                optimizer.zero_grad()
                m_loss.backward()
                optimizer.step()


                del img, lab, binary_labels, study_id, feats, logits, probs, p_abnormal, p_normal, m_loss, b_loss
                torch.cuda.empty_cache()
            
            # Calculate average losses for the epoch
            train_b_loss /= len(train_loader)
            train_m_loss /= len(train_loader)

            multi_true_labels = np.array(multi_true_labels)
            multi_pred_labels = np.array(multi_pred_labels)


            binary_probs = np.array(binary_probs)  # Predicted probabilities
            binary_preds = np.array(binary_preds)  # Binary predictions
            binary_true_labels = np.array(binary_true_labels)   
            
                     
            # Calculate metrics for multi-class classification
            multi_accuracy = accuracy_score(multi_true_labels, multi_pred_labels)
            multi_f1 = f1_score(multi_true_labels, multi_pred_labels, average='weighted')
            multi_kappa = cohen_kappa_score(multi_true_labels, multi_pred_labels)
            multi_recall = recall_score(multi_true_labels, multi_pred_labels, average='weighted')
            multi_precision = precision_score(multi_true_labels, multi_pred_labels, average='weighted')

            
            # Calculate metrics for binary classification
            binary_accuracy = accuracy_score(binary_true_labels, binary_preds)
            binary_f1 = f1_score(binary_true_labels, binary_preds, average='binary')
            binary_kappa = cohen_kappa_score(binary_true_labels, binary_preds)
            binary_recall = recall_score(binary_true_labels, binary_preds, average='binary')
            binary_precision = precision_score(binary_true_labels, binary_preds, average='binary')
            binary_auc_roc = roc_auc_score(binary_true_labels, binary_probs)

            
            print(f"Train Multi-class - Accuracy: {multi_accuracy:.4f}, F1: {multi_f1:.4f}, Kappa: {multi_kappa:.4f}")
            print(f"Train Binary - Accuracy: {binary_accuracy:.4f}, F1: {binary_f1:.4f}, Kappa: {binary_kappa:.4f}")


            
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch+1,
                "train_binary_loss": train_b_loss,
                "train_multi_loss": train_m_loss,
                "train_multi_accuracy": multi_accuracy,
                "train_multi_f1": multi_f1,
                "train_multi_kappa": multi_kappa,
                "train_binary_accuracy": binary_accuracy,
                "train_binary_f1": binary_f1,
                "train_binary_kappa": binary_kappa,
                "train_binary_recall": binary_recall,
                "train_binary_precision": binary_precision,
                "train_multi_recall": multi_recall,
                "train_multi_precision": multi_precision,
                "train_binary_auc_roc": binary_auc_roc
            })

            cm_binary = confusion_matrix(binary_true_labels, binary_preds)
            cm_multi = confusion_matrix(multi_true_labels, multi_pred_labels)

            print("TRAIN - MULTI")
            print(cm_multi)
            print("TRAIN - BINARY")
            print(cm_binary)

            if (epoch + 1) % 5 == 0:
                # Create a side-by-side figure for both confusion matrices
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Plot binary confusion matrix on the left
                sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=["0", "1"], 
                            yticklabels=["0", "1"],
                            ax=ax1, cbar=False)
                ax1.set_xlabel("Predicted")
                ax1.set_ylabel("True")
                ax1.set_title("Binary Confusion Matrix")

                # Plot multi-class confusion matrix on the right
                sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=["0", "1", "2", "3"], 
                            yticklabels=["0", "1", "2", "3"],
                            ax=ax2)
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("True")
                ax2.set_title("Multi-Class Confusion Matrix")

                # Add a main title for the entire figure
                plt.suptitle(f"Confusion Matrices ({epoch + 1} epochs)", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the suptitle

                # Save the combined figure
                combined_cm_plot_path = "combined_confusion_matrices.png"
                plt.savefig(combined_cm_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                # Log the combined image to Wandb
                wandb.log({"Train Confusion Matrices": wandb.Image(combined_cm_plot_path)})


                # Create a probability distribution plot by true class
                # Create a side-by-side figure for both probability distributions
                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                
                abnormal_count = np.sum(binary_true_labels == 1)
                normal_count = np.sum(binary_true_labels == 0)
                # === LEFT PLOT: Abnormal Probability ===
                axs[0].hist(binary_probs[binary_true_labels == 1], bins=20, alpha=0.5,
                            label=f'True Abnormal', color='red')
                axs[0].hist(binary_probs[binary_true_labels == 0], bins=20, alpha=0.5,
                            label=f'True Normal', color='green')
                axs[0].axvline(x=(1 - wandb.config.binary_thresh), color='black', linestyle='--',
                            label=f'Threshold ({(1 - wandb.config.binary_thresh)})')

                # axs[0].text(0.8, axs[0].get_ylim()[1] * 0.9, f"Abnormal: {abnormal_count}", color='red', fontsize=12)

                axs[0].set_xlabel('Abnormal Probability')
                axs[0].set_ylabel('Count')
                axs[0].set_title(f'Distribution of Abnormal Probabilities by True Class ({epoch+1} epochs)')
                axs[0].legend()
                axs[0].grid(True, alpha=0.3)

                # === RIGHT PLOT: Normal Probability ===
                normal_probs = 1 - binary_probs
                axs[1].hist(normal_probs[binary_true_labels == 1], bins=20, alpha=0.5,
                            label=f'True Abnormal', color='red')
                axs[1].hist(normal_probs[binary_true_labels == 0], bins=20, alpha=0.5,
                            label=f'True Normal', color='green')
                axs[1].axvline(x=wandb.config.binary_thresh, color='black', linestyle='--',
                            label=f'Threshold ({wandb.config.binary_thresh})')

            # axs[1].text(0.8, axs[1].get_ylim()[1] * 0.9, f"Normal: {normal_count}", color='green', fontsize=12)

                axs[1].set_xlabel('Normal Probability')
                axs[1].set_ylabel('Count')
                axs[1].set_title(f'Distribution of Normal Probabilities by True Class ({epoch+1} epochs)')
                axs[1].legend()
                axs[1].grid(True, alpha=0.3)

                plt.tight_layout()

                # Save and log the combined plot
                combined_plot_path = "combined_probability_distributions.png"
                plt.savefig(combined_plot_path)
                plt.close()

                wandb.log({"Train Probability Distributions": wandb.Image(combined_plot_path)})


            # Validation phase
            model.eval()

            val_b_loss = 0.0
            val_m_loss = 0.0
            val_multi_true_labels = []
            val_multi_pred_labels = []
            val_multi_pred_probs = []
            val_binary_true_labels = []
            val_binary_preds = []
            val_binary_probs = []

            # Disable gradient computation for validation
            with torch.no_grad():
                for idx, sample in enumerate(val_loader):
                    
                    # Get the batch data
                    img = sample["img"]  # Expected shape: [batch_size, channels, height, width]
                    img = img.unsqueeze(1)  # Add channel dimension if necessary
                    lab = sample["lab"].to(device)
                    lab = lab.long()  # Ensure labels are long for multi-class loss
                    study_id = sample["image_id"]

                    # Create binary labels: 0 for normal, 1 for abnormal (lab>=1)
                    binary_labels = (lab >= 1).long().to(device)

                    # Transfer image to device
                    img = img.to(device)

                    # Forward pass through the model
                    feats = model.features(img)
                    logits = model.classifier(feats)

                    # Calculate multi-class loss
                    m_loss = multi_criterion(logits, lab)

                    # Convert logits to probabilities with softmax
                    probs = F.softmax(logits, dim=1)
                    # Compute abnormal probability as the sum of the probabilities for all non-normal classes
                    p_abnormal = torch.sum(probs[:, 1:], dim=1)
                    p_normal = 1 - p_abnormal  # Derived normal probability (not used for loss)

                    p_nrm_check = probs[:, 0]
                    p_ab_check = torch.sum(probs[:, 1:], dim=1)

                    if not torch.allclose(p_nrm_check, p_normal, atol=1e-6):
                        print(f"p_normal: {p_normal}, p_nrm_check: {p_nrm_check}")
                    if not torch.allclose(p_nrm_check, p_normal, atol=1e-6):
                        print(f"p_abnormal: {p_abnormal}, p_abnrm_check: {p_ab_check}")
                    

                    # Calculate binary loss using the abnormal probability
                    p_abnormal_clipped = torch.clamp(p_abnormal, 0.0, 1.0)
                    b_loss = binary_criterion(p_abnormal_clipped, binary_labels.float())
                    # Determine binary predictions using a threshold from your wandb configuration
                    binary_preds_tensor = (p_normal < wandb.config.binary_thresh).long()

                    # Log the multi-class and binary true labels, predictions, and probabilities for evaluation purposes
                    val_multi_true_labels.extend(lab.cpu().numpy())
                    val_multi_pred_labels.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    val_multi_pred_probs.extend(probs.cpu().numpy())
                    val_binary_true_labels.extend(binary_labels.cpu().numpy())
                    val_binary_preds.extend(binary_preds_tensor.cpu().numpy())
                    val_binary_probs.extend(p_abnormal_clipped.cpu().numpy())

                    # Accumulate losses over batches
                    val_b_loss += b_loss.item()
                    val_m_loss += m_loss.item()

                    # Clean up to free memory
                    del img, lab, binary_labels, study_id, feats, logits, probs, p_abnormal, p_normal, m_loss, b_loss, binary_preds_tensor
                    torch.cuda.empty_cache()

            # Calculate average losses for the validation phase
            val_b_loss /= len(val_loader)
            val_m_loss /= len(val_loader)

            # Convert validation lists to numpy arrays for metric calculation
            val_multi_true_labels = np.array(val_multi_true_labels)
            val_multi_pred_labels = np.array(val_multi_pred_labels)


            val_binary_true_labels = np.array(val_binary_true_labels)
            val_binary_preds = np.array(val_binary_preds)
            val_binary_probs = np.array(val_binary_probs)
            
            # Calculate metrics for validation multi-class classification
            val_multi_accuracy = accuracy_score(val_multi_true_labels, val_multi_pred_labels)
            val_multi_f1 = f1_score(val_multi_true_labels, val_multi_pred_labels, average='weighted')
            val_multi_kappa = cohen_kappa_score(val_multi_true_labels, val_multi_pred_labels)
            val_multi_recall = recall_score(val_multi_true_labels, val_multi_pred_labels, average='weighted')
            val_multi_precision = precision_score(val_multi_true_labels, val_multi_pred_labels, average='weighted')
            
            # Calculate metrics for validation binary classification
            val_binary_accuracy = accuracy_score(val_binary_true_labels, val_binary_preds)
            val_binary_f1 = f1_score(val_binary_true_labels, val_binary_preds, average='binary')
            val_binary_kappa = cohen_kappa_score(val_binary_true_labels, val_binary_preds)
            val_binary_recall = recall_score(val_binary_true_labels, val_binary_preds, average='binary')
            val_binary_precision = precision_score(val_binary_true_labels, val_binary_preds, average='binary')
            val_binary_auc_roc = roc_auc_score(val_binary_true_labels, val_binary_probs)
            
            print(f"Validation Multi-class - Accuracy: {val_multi_accuracy:.4f}, F1: {val_multi_f1:.4f}, Kappa: {val_multi_kappa:.4f}")
            print(f"Validation Binary - Accuracy: {val_binary_accuracy:.4f}, F1: {val_binary_f1:.4f}, Kappa: {val_binary_kappa:.4f}")

            cm_binary_val = confusion_matrix(val_binary_true_labels, val_binary_preds)
            cm_multi_val = confusion_matrix(val_multi_true_labels, val_multi_pred_labels)


            confidence_values = np.where(val_binary_preds == 1, val_binary_probs, 1 - val_binary_probs)
            # Determine whether predictions are correct
            correct_predictions = (val_binary_preds == val_binary_true_labels)


            print("VAL - MULTI")
            print(cm_multi_val)
            print("VAL - BINARY")
            print(cm_binary_val)

            if((epoch + 1) % 5 == 0):
                # Create a side-by-side figure for both confusion matrices
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Plot binary confusion matrix on the left
                sns.heatmap(cm_binary_val, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=["0", "1"], 
                            yticklabels=["0", "1"],
                            ax=ax1, cbar=False)
                ax1.set_xlabel("Predicted")
                ax1.set_ylabel("True")
                ax1.set_title("Binary Confusion Matrix")

                # Plot multi-class confusion matrix on the right
                sns.heatmap(cm_multi_val, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=["0", "1", "2", "3"], 
                            yticklabels=["0", "1", "2", "3"],
                            ax=ax2)
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("True")
                ax2.set_title("Multi-Class Confusion Matrix")

                # Add a main title for the entire figure
                plt.suptitle("Confusion Matrices", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the suptitle

                # Save the combined figure
                combined_cm_plot_path = "combined_confusion_matrices.png"
                plt.savefig(combined_cm_plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                # Log the combined image to Wandb
                wandb.log({"Validation Confusion Matrices": wandb.Image(combined_cm_plot_path)})



                fig, axs = plt.subplots(1, 2, figsize=(12, 4))

                # === LEFT PLOT: Abnormal Probability ===
                axs[0].hist(val_binary_probs[val_binary_true_labels == 1], bins=20, alpha=0.5,
                            label=f'True Abnormal', color='red')
                axs[0].hist(val_binary_probs[val_binary_true_labels == 0], bins=20, alpha=0.5,
                            label=f'True Normal', color='green')
                axs[0].axvline(x=(1 - wandb.config.binary_thresh), color='black', linestyle='--',
                            label=f'Threshold ({(1 - wandb.config.binary_thresh)})')

                abnormal_count = np.sum(val_binary_true_labels == 1)
                normal_count = np.sum(val_binary_true_labels == 0)

                axs[0].set_xlabel('Abnormal Probability')
                axs[0].set_ylabel('Count')
                axs[0].set_title('Distribution of Abnormal Probabilities by True Class')
                axs[0].legend()
                axs[0].grid(True, alpha=0.3)

                # === RIGHT PLOT: Normal Probability ===
                normal_probs = 1 - val_binary_probs
                axs[1].hist(normal_probs[val_binary_true_labels == 1], bins=20, alpha=0.5,
                            label=f'True Abnormal', color='red')
                axs[1].hist(normal_probs[val_binary_true_labels == 0], bins=20, alpha=0.5,
                            label=f'True Normal', color='green')
                axs[1].axvline(x=wandb.config.binary_thresh, color='black', linestyle='--',
                            label=f'Threshold ({wandb.config.binary_thresh})')


                axs[1].set_xlabel('Normal Probability')
                axs[1].set_ylabel('Count')
                axs[1].set_title('Distribution of Normal Probabilities by True Class')
                axs[1].legend()
                axs[1].grid(True, alpha=0.3)

                plt.tight_layout()

                # Save and log the combined plot
                combined_plot_path = "combined_probability_distributions.png"
                plt.savefig(combined_plot_path)
                plt.close()

                wandb.log({"Val Probability Distributions": wandb.Image(combined_plot_path)})

            print(f"Validation Loss (Binary): {val_b_loss:.4f} | Validation Loss (Multi): {val_m_loss:.4f}")
            wandb.log({
                "epoch": epoch+1,
                "val_binary_loss": val_b_loss,
                "val_multi_loss": val_m_loss,
                "val_multi_accuracy": val_multi_accuracy,
                "val_multi_f1": val_multi_f1,
                "val_multi_kappa": val_multi_kappa,
                "val_binary_accuracy": val_binary_accuracy,
                "val_binary_f1": val_binary_f1,
                "val_binary_kappa": val_binary_kappa,
                "val_binary_recall": val_binary_recall,
                "val_binary_precision": val_binary_precision,
                "val_multi_recall": val_multi_recall,
                "val_multi_precision": val_multi_precision,
                "val_binary_auc_roc": val_binary_auc_roc
            })


             # Check if the current model is the best based on val_multi_loss
            if val_m_loss < best_val_multi_loss:
                best_val_multi_loss = val_m_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(checkpoint_dir, f"{experiment_name}-best_model_loss.pth"))
                print(f"Saved best model based on val_multi_loss: {val_m_loss:.4f}")

            # Check if the current model is the best based on val_multi_f1
            if val_multi_f1 > best_val_multi_f1:
                best_val_multi_f1 = val_multi_f1
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()

                }, os.path.join(checkpoint_dir, f"{experiment_name}-best_model_f1.pth"))
                print(f"Saved best model based on val_multi_f1: {val_multi_f1:.4f}")

            
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(checkpoint_dir, f"{experiment_name}-final_model.pth"))
        print(f"Saved best model based on val_multi_loss: {val_m_loss:.4f}")
        # test phase
        model.eval()

        test_b_loss = 0.0
        test_m_loss = 0.0
        test_multi_true_labels = []
        test_multi_pred_labels = []
        test_multi_pred_probs = []
        test_binary_true_labels = []
        test_binary_preds = []
        test_binary_probs = []

        # Disable gradient computation for test
        with torch.no_grad():
            for idx, sample in enumerate(test_loader):

                # Optionally print progress every 10 batches or on the first batch
                if ((idx + 1) % 10 == 0 or idx == 0):
                    print(f"Test Batch {idx+1}/{len(test_loader)}")

                # Get the batch data
                img = sample["img"]  # Expected shape: [batch_size, channels, height, width]
                img = img.unsqueeze(1)  # Add channel dimension if necessary
                lab = sample["lab"].to(device)
                lab = lab.long()  # Ensure labels are long for multi-class loss
                study_id = sample["image_id"]

                # Create binary labels: 0 for normal, 1 for abnormal (lab>=1)
                binary_labels = (lab >= 1).long().to(device)

                # Transfer image to device
                img = img.to(device)

                # Forward pass through the model
                feats = model.features(img)
                logits = model.classifier(feats)

                # Calculate multi-class loss
                m_loss = multi_criterion(logits, lab)

                # Convert logits to probabilities with softmax
                probs = F.softmax(logits, dim=1)
                # Compute abnormal probability as the sum of the probabilities for all non-normal classes
                p_abnormal = torch.sum(probs[:, 1:], dim=1)
                p_normal = 1 - p_abnormal  # Derived normal probability (not used for loss)

                p_normal_2 = probs[:, 0]

                # Calculate binary loss using the abnormal probability
                p_abnormal_clipped = torch.clamp(p_abnormal, 0.0, 1.0)

                test_binary_probs.extend(p_abnormal_clipped.cpu().numpy())

                b_loss = binary_criterion(p_abnormal_clipped, binary_labels.float())
                # Determine binary predictions using a threshold from your wandb configuration
                binary_preds_tensor = (p_normal < wandb.config.binary_thresh).long()

                # Log the multi-class and binary true labels, predictions, and probabilities for evaluation purposes
                test_multi_true_labels.extend(lab.cpu().numpy())
                test_multi_pred_labels.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_multi_pred_probs.extend(probs.cpu().numpy())
                test_binary_true_labels.extend(binary_labels.cpu().numpy())
                test_binary_preds.extend(binary_preds_tensor.cpu().numpy())

                # Accumulate losses over batches
                test_b_loss += b_loss.item()
                test_m_loss += m_loss.item()

                # Clean up to free memory
                del img, lab, binary_labels, study_id, feats, logits, probs, p_abnormal, p_normal, m_loss, b_loss, binary_preds_tensor
                torch.cuda.empty_cache()

        # Calculate average losses for the test phase
        test_b_loss /= len(test_loader)
        test_m_loss /= len(test_loader)

        # Convert test lists to numpy arrays for metric calculation
        test_multi_true_labels = np.array(test_multi_true_labels)
        test_multi_pred_labels = np.array(test_multi_pred_labels)

        test_binary_true_labels = np.array(test_binary_true_labels)
        test_binary_preds = np.array(test_binary_preds)
        test_binary_probs = np.array(test_binary_probs)  # Predicted probabilities


        # Calculate metrics for test multi-class classification
        test_multi_accuracy = accuracy_score(test_multi_true_labels, test_multi_pred_labels)
        test_multi_f1 = f1_score(test_multi_true_labels, test_multi_pred_labels, average='weighted')
        test_multi_kappa = cohen_kappa_score(test_multi_true_labels, test_multi_pred_labels)
        test_multi_recall = recall_score(test_multi_true_labels, test_multi_pred_labels, average='weighted')
        test_multi_precision = precision_score(test_multi_true_labels, test_multi_pred_labels, average='weighted')


        # Calculate metrics for test binary classification
        test_binary_accuracy = accuracy_score(test_binary_true_labels, test_binary_preds)
        test_binary_f1 = f1_score(test_binary_true_labels, test_binary_preds, average='binary')
        test_binary_kappa = cohen_kappa_score(test_binary_true_labels, test_binary_preds)
        test_binary_recall = recall_score(test_binary_true_labels, test_binary_preds, average='binary')
        test_binary_precision = precision_score(test_binary_true_labels, test_binary_preds, average='binary')
        test_binary_auc_roc = roc_auc_score(test_binary_true_labels, test_binary_probs)

        print(f"Test Multi-class - Accuracy: {test_multi_accuracy:.4f}, F1: {test_multi_f1:.4f}, Kappa: {test_multi_kappa:.4f}")
        print(f"Test Binary - Accuracy: {test_binary_accuracy:.4f}, F1: {test_binary_f1:.4f}, Kappa: {test_binary_kappa:.4f}")

        print(f"Test Loss (Binary): {test_b_loss:.4f} | test Loss (Multi): {test_m_loss:.4f}")
        wandb.log({
            "epoch": epoch+1,
            "test_binary_loss": test_b_loss,
            "test_multi_loss": test_m_loss,
            "test_multi_accuracy": test_multi_accuracy,
            "test_multi_f1": test_multi_f1,
            "test_multi_kappa": test_multi_kappa,
            "test_binary_accuracy": test_binary_accuracy,
            "test_binary_f1": test_binary_f1,
            "test_binary_kappa": test_binary_kappa,
            "test_binary_recall": test_binary_recall,
            "test_binary_precision": test_binary_precision,
            "test_multi_recall": test_multi_recall,
            "test_multi_precision": test_multi_precision,
            "test_binary_auc_roc": test_binary_auc_roc
        })

        cm_binary = confusion_matrix(test_binary_true_labels, test_binary_preds)
        cm_multi = confusion_matrix(test_multi_true_labels, test_multi_pred_labels)


        # Create a side-by-side figure for both confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot binary confusion matrix on the left
        sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["0", "1"], 
                    yticklabels=["0", "1"],
                    ax=ax1, cbar=False)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        ax1.set_title("Binary Confusion Matrix")

        # Plot multi-class confusion matrix on the right
        sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["0", "1", "2", "3"], 
                    yticklabels=["0", "1", "2", "3"],
                    ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title("Multi-Class Confusion Matrix")

        # Add a main title for the entire figure
        plt.suptitle("Confusion Matrices", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the suptitle

        # Save the combined figure
        combined_cm_plot_path = "combined_confusion_matrices.png"
        plt.savefig(combined_cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Log the combined image to Wandb
        wandb.log({"Test Confusion Matrices": wandb.Image(combined_cm_plot_path)})


        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # === LEFT PLOT: Abnormal Probability ===
        axs[0].hist(test_binary_probs[test_binary_true_labels == 1], bins=20, alpha=0.5,
                    label=f'True Abnormal', color='red')
        axs[0].hist(test_binary_probs[test_binary_true_labels == 0], bins=20, alpha=0.5,
                    label=f'True Normal', color='green')
        axs[0].axvline(x=(1 - wandb.config.binary_thresh), color='black', linestyle='--',
                    label=f'Threshold ({(1 - wandb.config.binary_thresh)})')

        abnormal_count = np.sum(test_binary_true_labels == 1)
        normal_count = np.sum(test_binary_true_labels == 0)

        axs[0].set_xlabel('Abnormal Probability')
        axs[0].set_ylabel('Count')
        axs[0].set_title('Distribution of Abnormal Probabilities by True Class')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # === RIGHT PLOT: Normal Probability ===
        normal_probs = 1 - test_binary_probs
        axs[1].hist(normal_probs[test_binary_true_labels == 1], bins=20, alpha=0.5,
                    label=f'True Abnormal', color='red')
        axs[1].hist(normal_probs[test_binary_true_labels == 0], bins=20, alpha=0.5,
                    label=f'True Normal', color='green')
        axs[1].axvline(x=wandb.config.binary_thresh, color='black', linestyle='--',
                    label=f'Threshold ({wandb.config.binary_thresh})')


        axs[1].set_xlabel('Normal Probability')
        axs[1].set_ylabel('Count')
        axs[1].set_title('Distribution of Normal Probabilities by True Class')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()



    except KeyError as e:
        print(f"Missing configuration: {e}")
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")