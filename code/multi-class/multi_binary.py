import torch
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, Subset, ConcatDataset
import torchvision.transforms as transforms
import os
import torchxrayvision as xrv
from skimage.color import rgb2gray
from skimage.transform import resize
import pydicom
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
import wandb
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix, accuracy_score
import sys
sys.path.append('c:/Users/user-pc/Masters/MSc_2025/code')
import utils.helpers as helpers
import utils.classes as classes
import utils.train_utils as train_utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gc
from sklearn.calibration import calibration_curve


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dicom_dir_1 = 'C:/Users/user-pc/Masters/MSc_2025/data/MBOD_Datasets/Dataset-1'
dicom_dir_2 = 'C:/Users/user-pc/Masters/MSc_2025/data/MBOD_Datasets/Dataset-2'

metadata_1 = pd.read_excel("C:/Users/user-pc/Masters/MSc_2025/data/MBOD_Datasets/Dataset-1/FileDatabaseWithRadiology.xlsx")

metadata_2 = pd.read_excel("C:/Users/user-pc/Masters/MSc_2025/data/MBOD_Datasets/Dataset-2/Database_Training-2024.08.28.xlsx")

ILO_imgs = 'C:/Users/user-pc/Masters/MSc_2025/data/ilo-radiographs-dicom'

d1_cl = classes.DICOMDataset1_CL(dicom_dir=dicom_dir_1, metadata_df=metadata_1)
d1_cl._assign_labels()
d1_cl.set_target("Profusion", 512)  # Add this line, choose appropriate target and size

d2_cl = classes.DICOMDataset2_CL(dicom_dir=dicom_dir_2, metadata_df=metadata_2)
d2_cl._assign_labels()
d2_cl.set_target("Profusion", 512)  # Add this line, choose appropriate target and size

combined_dataset = ConcatDataset([d1_cl, d2_cl])
# Convert 'Profusion Label' to integer for both datasets
d1_cl.metadata_df['Profusion Label'] = d1_cl.metadata_df['Profusion Label'].astype(int)
d2_cl.metadata_df['Profusion Label'] = d2_cl.metadata_df['Profusion Label'].astype(int)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert 'Profusion Label' to integer for both datasets
d1_cl.metadata_df['Profusion Label'] = d1_cl.metadata_df['Profusion Label'].astype(int)
d2_cl.metadata_df['Profusion Label'] = d2_cl.metadata_df['Profusion Label'].astype(int)


train_indices_d2, val_indices_d2, test_indices_d2 = helpers.split_dataset(d2_cl)
train_dataset_d2 = torch.utils.data.Subset(d2_cl, train_indices_d2)
val_dataset_d2 = torch.utils.data.Subset(d2_cl, val_indices_d2)
test_dataset_d2 = torch.utils.data.Subset(d2_cl, test_indices_d2)

train_indices_d1, val_indices_d1, test_indices_d1 = helpers.split_dataset(d1_cl)
train_dataset_d1 = torch.utils.data.Subset(d1_cl, train_indices_d1)
val_dataset_d1 = torch.utils.data.Subset(d1_cl, val_indices_d1)
test_dataset_d1 = torch.utils.data.Subset(d1_cl, test_indices_d1)

torch.cuda.empty_cache()

with open('C:/Users/user-pc/Masters/MSc_2025/code/multi-class/multi_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

for experiment_name, experiment in config['experiments'].items():
    torch.cuda.empty_cache()
    print(f"Running experiment: {experiment_name}")
    lr = experiment['lr']
    n_epochs = experiment['n_epochs']
    batch_size = experiment['batch_size']
    train_dataset = experiment['train_dataset']
    model_name = experiment['model']
    oversampling = experiment['oversampling']
    loss_function = experiment['loss_function']
    augmentations = experiment['augmentation']
    model_resolution = experiment['model_resolution']
    binary_thresh = experiment['binary_thresh']
    optimizer = experiment['optimizer']


    wandb.init(project='MBOD-New', name=experiment_name, config=experiment)


    model = xrv.models.ResNet(weights="resnet50-res512-all")
    model.classifier = classes.MultiClassBaseClassifier(in_features=2048, num_classes=4).to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    if(train_dataset == "MBOD 857"):
        train_loader = DataLoader(train_dataset_d2, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_d2, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset_d2, batch_size=batch_size, shuffle=False)

    
    multi_criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.BCEWithLogitsLoss()

    # EPOCHS START HERE
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        model.train()

        # Lists for storing predictions and true labels for the epoch
        multi_preds_train = []
        multi_labels_train = []
        multi_probs_train = []


        binary_preds_train = []
        binary_labels_train = []
        binary_probs_train = []

        # For loss tracking
        train_loss_multi = 0.0
        train_loss_binary = 0.0
        train_total = 0

        for idx, (img, labels) in enumerate(train_loader):

            if((idx + 1 ) % 5 == 0):
                print(f"Batch {idx+1}/{len(train_loader)}")

            optimizer.zero_grad()

            img = img.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            train_total += batch_size

            # Forward pass
            feats = model.features(img)
            output = model.classifier(feats)
            probs = F.softmax(output, dim=1)

            # --- Multi-class predictions ---
            # Get predictions from model outputs (argmax of softmax probabilities)
            _, m_preds = torch.max(probs, 1)
            
            # --- Binary predictions ---
            # Create binary labels: class 0 = normal (0), all other classes = abnormal (1)
            # BCEWithLogitsLoss expects float targets
            binary_labels = (labels != 0).float()  
            # If probability of class 0 is less than or equal to the threshold, predict abnormal (1), else normal (0)
            b_preds = (probs[:, 0] <= binary_thresh).long()  
            
            # Calculate losses
            multi_loss = multi_criterion(output, labels)
            binary_loss = binary_criterion(output[:, 0], binary_labels)
            
            # Backpropagation using the multi-class loss for training purposes
            multi_loss.backward()
            optimizer.step()

            # Track losses
            train_loss_multi += multi_loss.item() * batch_size
            train_loss_binary += binary_loss.item() * batch_size

            # Convert predictions and labels to CPU numpy arrays for sklearn
            multi_preds_train.extend(m_preds.cpu().numpy())
            multi_labels_train.extend(labels.cpu().numpy())

            binary_preds_train.extend(b_preds.cpu().numpy())
            # For binary labels, we need to cast the tensor to long before converting to numpy
            binary_labels_train.extend(binary_labels.long().cpu().numpy())

            wandb.log({
                "batch": idx + 1,
                "batch_loss_multi": multi_loss.item(),
                "batch_loss_binary": binary_loss.item()
            })

            del img, labels, feats, output, probs, m_preds, b_preds, multi_loss, binary_loss, binary_labels
            torch.cuda.empty_cache()

        # End of epoch: compute average losses
        avg_loss_multi = train_loss_multi / train_total
        avg_loss_binary = train_loss_binary / train_total

        # Compute epoch-level metrics using scikit-learn
        multi_accuracy = accuracy_score(multi_labels_train, multi_preds_train)
        multi_f1 = f1_score(multi_labels_train, multi_preds_train, average='macro')
        multi_precision = precision_score(multi_labels_train, multi_preds_train, average='macro')
        multi_recall = recall_score(multi_labels_train, multi_preds_train, average='macro')
        multi_kappa = cohen_kappa_score(multi_labels_train, multi_preds_train)

        binary_accuracy = accuracy_score(binary_labels_train, binary_preds_train)
        binary_f1 = f1_score(binary_labels_train, binary_preds_train)
        binary_precision = precision_score(binary_labels_train, binary_preds_train)
        binary_recall = recall_score(binary_labels_train, binary_preds_train)
        binary_kappa = cohen_kappa_score(binary_labels_train, binary_preds_train)


        multi_preds_train.clear()
        multi_labels_train.clear()
        binary_preds_train.clear()
        binary_labels_train.clear()

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Multi-class Loss: {avg_loss_multi:.4f} | Accuracy: {multi_accuracy:.4f} | F1: {multi_f1:.4f} | Precision: {multi_precision:.4f} | Recall: {multi_recall:.4f}")
        print(f"  Binary Loss: {avg_loss_binary:.4f} | Accuracy: {binary_accuracy:.4f} | F1: {binary_f1:.4f} | Precision: {binary_precision:.4f} | Recall: {binary_recall:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss_multi": avg_loss_multi,
            "train_loss_binary": avg_loss_binary,
            "train_accuracy_multi": multi_accuracy,
            "train_f1_multi": multi_f1,
            "train_precision_multi": multi_precision,
            "train_recall_multi": multi_recall,
            "train_accuracy_binary": binary_accuracy,
            "train_f1_binary": binary_f1,
            "train_precision_binary": binary_precision,
            "train_recall_binary": binary_recall,
            "train_kappa_binary": binary_kappa,
            "train_kappa_multi": multi_kappa
        })

        # Save model checkpoint every 5 epochs
        if (epoch+1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'C:/Users/user-pc/Masters/MSc_2025/code/multi-class/checkpoints/{experiment_name}_{epoch + 1}.pth')
        
        # VALIDATION PHASE
        print("Validating...")
        model.eval()  # Set model to evaluation mode
        # Initialize lists for storing validation predictions and labels
        val_multi_preds = []
        val_multi_labels = []
        val_binary_preds = []
        val_binary_labels = []
        val_loss_multi = 0.0
        val_loss_binary = 0.0
        val_total = 0

        with torch.no_grad():
            for idx, (img, labels) in enumerate(val_loader):  # Assume you have a "val_loader" defined
                img = img.to(device)
                labels = labels.to(device)
                batch_size = labels.size(0)
                val_total += batch_size

                # Forward pass for validation
                feats = model.features(img)
                output = model.classifier(feats)
                probs = F.softmax(output, dim=1)

                # Multi-class predictions
                _, m_preds = torch.max(probs, 1)
                # Binary predictions and labels
                binary_labels = (labels != 0).float()
                b_preds = (probs[:, 0] <= binary_thresh).long()

                # Compute losses for validation
                multi_loss = multi_criterion(output, labels)
                binary_loss = binary_criterion(output[:, 0], binary_labels)

                val_loss_multi += multi_loss.item() * batch_size
                val_loss_binary += binary_loss.item() * batch_size

                # Aggregate predictions and labels for metrics computation
                val_multi_preds.extend(m_preds.cpu().numpy())
                val_multi_labels.extend(labels.cpu().numpy())
                val_binary_preds.extend(b_preds.cpu().numpy())
                val_binary_labels.extend(binary_labels.long().cpu().numpy())

        avg_val_loss_multi = val_loss_multi / val_total
        avg_val_loss_binary = val_loss_binary / val_total

        val_multi_accuracy = accuracy_score(val_multi_labels, val_multi_preds)
        val_multi_f1 = f1_score(val_multi_labels, val_multi_preds, average='macro')
        val_multi_precision = precision_score(val_multi_labels, val_multi_preds, average='macro')
        val_multi_recall = recall_score(val_multi_labels, val_multi_preds, average='macro')
        val_multi_kappa = cohen_kappa_score(val_multi_labels, val_multi_preds)

        val_binary_accuracy = accuracy_score(val_binary_labels, val_binary_preds)
        val_binary_f1 = f1_score(val_binary_labels, val_binary_preds, average='macro')
        val_binary_precision = precision_score(val_binary_labels, val_binary_preds, average='macro')
        val_binary_recall = recall_score(val_binary_labels, val_binary_preds, average='macro')
        val_binary_kappa = cohen_kappa_score(val_binary_labels, val_binary_preds)

        val_multi_preds.clear()
        val_multi_labels.clear()
        val_binary_preds.clear()
        val_binary_labels.clear()


        print(f"Validation Epoch {epoch+1} Summary:")
        print(f"  Multi-class Loss: {avg_val_loss_multi:.4f} | Accuracy: {val_multi_accuracy:.4f} | F1: {val_multi_f1:.4f}, Precision: {val_multi_precision:.4f}, Recall: {val_multi_recall:.4f}, Kappa: {val_multi_kappa:.4f}")
        print(f"  Binary Loss: {avg_val_loss_binary:.4f} | Accuracy: {val_binary_accuracy:.4f} | F1: {val_binary_f1:.4f}, Precision: {val_binary_precision:.4f}, Recall: {val_binary_recall:.4f}, Kappa: {val_binary_kappa:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "val_loss_multi": avg_val_loss_multi,
            "val_loss_binary": avg_val_loss_binary,
            "val_accuracy_multi": val_multi_accuracy,
            "val_f1_multi": val_multi_f1,
            "val_accuracy_binary": val_binary_accuracy,
            "val_f1_binary": val_binary_f1,
            "val_precision_multi": val_multi_precision,
            "val_recall_multi": val_multi_recall,
            "val_precision_binary": val_binary_precision,
            "val_recall_binary": val_binary_recall,
            "val_kappa_multi": val_multi_kappa,
            "val_kappa_binary": val_binary_kappa
        })

        gc.collect()

    # TESTING PHASE
    print("Testing...")
    model.eval()
    test_multi_preds = []
    test_multi_labels = []
    test_binary_preds = []
    test_binary_labels = []

    test_loss_multi = 0.0
    test_loss_binary = 0.0
    test_total = 0

    with torch.no_grad():
        for idx, (img, labels) in enumerate(test_loader):
            img = img.to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)
            test_total += batch_size

            # Forward pass through the model
            feats = model.features(img)
            output = model.classifier(feats)
            probs = F.softmax(output, dim=1)

            # Multi-class predictions: take the class with the highest probability
            _, m_preds = torch.max(probs, 1)
            # Binary predictions: use the probability of class 0 and threshold it
            # Here, class 0 is considered "normal" (0), and any probability <= binary_thresh is abnormal (1)
            binary_labels = (labels != 0).float()  # Convert true labels to binary targets (0 for class 0, 1 otherwise)
            b_preds = (probs[:, 0] <= binary_thresh).long()

            # Accumulate losses (weighted by batch size)
            test_loss_multi += multi_criterion(output, labels).item() * batch_size
            test_loss_binary += binary_criterion(output[:, 0], binary_labels).item() * batch_size

            # Aggregate predictions and labels for evaluation using sklearn
            test_multi_preds.extend(m_preds.cpu().numpy())
            test_multi_labels.extend(labels.cpu().numpy())
            test_binary_preds.extend(b_preds.cpu().numpy())
            test_binary_labels.extend(binary_labels.long().cpu().numpy())

    # Compute average losses over all test samples
    avg_test_loss_multi = test_loss_multi / test_total
    avg_test_loss_binary = test_loss_binary / test_total

    # Compute metrics using sklearn
    test_multi_accuracy = accuracy_score(test_multi_labels, test_multi_preds)
    test_multi_f1 = f1_score(test_multi_labels, test_multi_preds, average='macro')
    test_multi_precision = precision_score(test_multi_labels, test_multi_preds, average='macro')
    test_multi_recall = recall_score(test_multi_labels, test_multi_preds, average='macro')
    test_multi_kappa = cohen_kappa_score(test_multi_labels, test_multi_preds)

    test_binary_accuracy = accuracy_score(test_binary_labels, test_binary_preds)
    test_binary_f1 = f1_score(test_binary_labels, test_binary_preds)
    test_binary_precision = precision_score(test_binary_labels, test_binary_preds)
    test_binary_recall = recall_score(test_binary_labels, test_binary_preds)
    test_binary_kappa = cohen_kappa_score(test_binary_labels, test_binary_preds)

    print("Test Summary:")
    print(f"  Multi-class Loss: {avg_test_loss_multi:.4f}, Accuracy: {test_multi_accuracy:.4f}, F1: {test_multi_f1:.4f}, Precision: {test_multi_precision:.4f}, Recall: {test_multi_recall:.4f}, Kappa: {test_multi_kappa:.4f}")
    print(f"  Binary Loss: {avg_test_loss_binary:.4f}, Accuracy: {test_binary_accuracy:.4f}, F1: {test_binary_f1:.4f}, Precision: {test_binary_precision:.4f}, Recall: {test_binary_recall:.4f}, Kappa: {test_binary_kappa:.4f}")

    cm_binary_d2 = confusion_matrix(test_binary_labels, test_binary_preds)
    cm_multi_d2 = confusion_matrix(test_multi_labels, test_multi_preds)

    # converting to binary version for combined confusion matrix
    d2_cl.metadata_df['Profusion Label'] = (d2_cl.metadata_df['Profusion Label'] > 0).astype(int)
    
    combined_cm_d2 = helpers.plot_combined_conf_mat('Profusion', d2_cl, test_binary_preds, test_indices_d2, True, "MBOD 857")

    # HERE
    # After test phase and logging matrices
    print(f"Binary Classification Confusion Matrix:\n{cm_binary_d2}")
    print(f"Multi-class Classification Confusion Matrix:\n{cm_multi_d2}")
    
    # Add logging for confusion matrices
    class_names = ['Normal (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)']
    
    # 1. Binary Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_binary_d2, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Abnormal'],
               yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Binary Classification Confusion Matrix')
    wandb.log({"Binary Confusion Matrix": wandb.Image(plt)})
    plt.close()
    
    # 2. Multi-class Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_multi_d2, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Multi-class Classification Confusion Matrix')
    wandb.log({"Multi-class Confusion Matrix": wandb.Image(plt)})
    plt.close()
    
    # Close the wandb run before starting the next experiment
    wandb.finish()
