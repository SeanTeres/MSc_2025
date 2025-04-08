import torch
import sys
sys.path.append('/home/sean/MSc/code')
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torchxrayvision as xrv
from skimage.color import rgb2gray
from skimage.transform import resize
from torchxrayvision.datasets import XRayCenterCrop
import pandas as pd
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import time
import random
from tqdm import tqdm
from sklearn.manifold import TSNE
import io
from PIL import Image as PILImage
from utils.helpers import read_and_normalize_xray, split_with_indices
from utils.classes import DICOMDataset1, DICOMDataset2, AugmentedDataset, BaseClassifier
import plotly.express as px
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import utils.classes as classes
from sklearn.preprocessing import normalize


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
    
def train_model(train_loader, val_loader, model, n_epochs, lr, device, pos_weight, experiment_name):
    """Function to train a model on a given training dataloader."""
    patience = 3
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))  # Use BCEWithLogitsLoss for binary classification
    optim_1 = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()  # Start timing

    best_val_f1 = 0.0
    best_val_epoch_loss = float('inf')
    best_val_f1 = float('-inf')
    best_val_kappa = float('-inf')
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')

        model.train()  # set to training mode

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        # Training Phase
        for idx, (imgs, labels) in enumerate(train_loader):
            # print(f"Batch: {idx+1}/{len(train_loader)}")
            corr, tot = 0, 0
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            optim_1.zero_grad()

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            loss.backward()
            optim_1.step()

            running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
            corr += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            batch_acc = corr / tot

        # Log metrics every batch

            wandb.log({
                "batch": (idx + 1),
                "batch_loss": loss.item(),
                "batch_accuracy": batch_acc
            })

            del imgs, labels, output, features
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        epoch_kappa = cohen_kappa_score(all_labels, all_preds)

                # Compute confusion matrix components
        tp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
        tn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
        fp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
        fn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))

        # Calculate TPR, TNR, FPR, FNR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0


        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "train_precision": epoch_precision,
            "train_recall": epoch_recall,
            "train_f1": epoch_f1,
            "train_kappa": epoch_kappa,
            "train_tpr": tpr,
            "train_tnr": tnr,
            "train_fpr": fpr,
            "train_fnr": fnr
        })

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

                features = model.features2(imgs)
                output = model.classifier(features)

                loss = criterion(output, labels)

                val_running_loss += loss.item()

                preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                del imgs, labels, output, features
                torch.cuda.empty_cache()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_epoch_precision = precision_score(val_labels, val_preds, average='weighted')
        val_epoch_recall = recall_score(val_labels, val_preds, average='weighted')
        val_epoch_f1 = f1_score(val_labels, val_preds, average='weighted')

        val_kappa = cohen_kappa_score(val_labels, val_preds)

                        # Compute confusion matrix components
        tp = np.sum((np.array(val_preds) == 1) & (np.array(val_labels) == 1))
        tn = np.sum((np.array(val_preds) == 0) & (np.array(val_labels) == 0))
        fp = np.sum((np.array(val_preds) == 1) & (np.array(val_labels) == 0))
        fn = np.sum((np.array(val_preds) == 0) & (np.array(val_labels) == 1))

        # Calculate TPR, TNR, FPR, FNR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Log validation metrics
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "val_precision": val_epoch_precision,
            "val_recall": val_epoch_recall,
            "val_f1": val_epoch_f1,
            "val_kappa": val_kappa,
            "val_tpr": tpr,
            "val_tnr": tnr,
            "val_fpr": fpr,
            "val_fnr": fnr
        })

        # Print results per epoch
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}, "
              f"Training Precision: {epoch_precision:.4f}, Training Recall: {epoch_recall:.4f}, Training F1: {epoch_f1:.4f}, Training Kappa: {epoch_kappa}")

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}, "
              f"Validation Precision: {val_epoch_precision:.4f}, Validation Recall: {val_epoch_recall:.4f}, Validation F1: {val_epoch_f1:.4f}, Validation Kappa: {val_kappa}")
        
        epoch_train_cm = confusion_matrix(all_labels, all_preds)
        epoch_val_cm = confusion_matrix(val_labels, val_preds)

        print(f"Training Confusion Matrix:\n{epoch_train_cm}")
        print(f"Validation Confusion Matrix:\n{epoch_val_cm}")
        print("****"*25 + "\n")

        # Update best validation metrics
        if val_epoch_loss < best_val_epoch_loss:
            best_val_epoch_loss = val_epoch_loss
            # Optionally save the model state
            torch.save({'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optim_1.state_dict(),
                       'epoch': epoch+1,
                       }, f'/home/sean/MSc/code/binary/checkpoints/{experiment_name}_best_model_val_loss.pth' )

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            # Optionally save the model state
            torch.save({'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optim_1.state_dict(),
                       'epoch': epoch+1,
                       }, f'/home/sean/MSc/code/binary/checkpoints/{experiment_name}_best_model_val_kappa.pth' )
        
        if val_epoch_f1 > best_val_f1:
            best_val_f1 = val_epoch_f1
            # Optionally save the model state
            torch.save({'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optim_1.state_dict(),
                       'epoch': epoch+1,
                       }, f'/home/sean/MSc/code/binary/checkpoints/{experiment_name}_best_model_val_f1.pth' )
            
        # Early stopping logic
        # if val_epoch_f1 > best_val_f1:
          #  best_val_f1 = val_epoch_f1
           # epochs_since_improvement = 0
       #  else:
           # epochs_since_improvement += 1

       #  if epochs_since_improvement >= patience:
            #  print(f"Early stopping triggered after {epoch+1} epochs.")
             # break

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim_1.state_dict(),
                'epoch': epoch+1,
                }, f'/home/sean/MSc/code/binary/checkpoints/{experiment_name}_final_model.pth' )
    return model

def train_model_with_focal_loss(train_loader, val_loader, model, n_epochs, lr, device, alpha, gamma, experiment_name):
    """Function to train a model on a given training dataloader using Binary Focal Loss."""
    patience = 3
    
    model = model.to(device)

    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
    optim_1 = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()  # Start timing

    best_val_f1 = 0.0
    best_val_epoch_loss = float('inf')
    best_val_kappa = float('-inf')
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')

        model.train()  # set to training mode

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        # Training Phase
        for idx, (imgs, labels) in enumerate(train_loader):
            # print(f"Batch: {idx+1}/{len(train_loader)}")
            corr, tot = 0, 0
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            optim_1.zero_grad()

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            loss.backward()
            optim_1.step()

            running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
            corr += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            batch_acc = corr / tot

        # Log metrics every batch

            wandb.log({
                "batch": idx + 1,
                "batch_loss": loss.item(),
                "batch_accuracy": batch_acc
            })

            del imgs, labels, output, features
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        epoch_kappa = cohen_kappa_score(all_labels, all_preds)

        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "train_precision": epoch_precision,
            "train_recall": epoch_recall,
            "train_f1": epoch_f1,
            "train_kappa": epoch_kappa
        })

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

                features = model.features2(imgs)
                output = model.classifier(features)

                loss = criterion(output, labels)

                val_running_loss += loss.item()

                preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                del imgs, labels, output, features
                torch.cuda.empty_cache()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_epoch_precision = precision_score(val_labels, val_preds, average='weighted')
        val_epoch_recall = recall_score(val_labels, val_preds, average='weighted')
        val_epoch_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_kappa = cohen_kappa_score(val_labels, val_preds)

        # Log validation metrics
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "val_precision": val_epoch_precision,
            "val_recall": val_epoch_recall,
            "val_f1": val_epoch_f1,
            "val_kappa": val_kappa
        })

        # Print results per epoch
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}, "
              f"Training Precision: {epoch_precision:.4f}, Training Recall: {epoch_recall:.4f}, Training F1: {epoch_f1:.4f}")

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}, "
              f"Validation Precision: {val_epoch_precision:.4f}, Validation Recall: {val_epoch_recall:.4f}, Validation F1: {val_epoch_f1:.4f}, "
              f"Validation Kappa: {val_kappa}")
        
        epoch_train_cm = confusion_matrix(all_labels, all_preds)
        epoch_val_cm = confusion_matrix(val_labels, val_preds)

        print(f"Training Confusion Matrix:\n{epoch_train_cm}")
        print(f"Validation Confusion Matrix:\n{epoch_val_cm}")
        print("****"*25 + "\n")

        # Update best validation metrics
        if val_epoch_loss < best_val_epoch_loss:
            print(f"Prev. best val loss: {best_val_epoch_loss:.4f}, New best val loss: {val_epoch_loss:.4f}\n")
            best_val_epoch_loss = val_epoch_loss
            # Optionally save the model state
            torch.save({'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optim_1.state_dict(),
                       'epoch': epoch+1,
                       }, f'{experiment_name}_best_model_val_loss.pth' )


        if val_kappa > best_val_kappa:
            print(f"Prev. best val kappa: {best_val_kappa:.4f}, New best val kappa: {val_kappa:.4f}\n")
            best_val_kappa = val_kappa
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim_1.state_dict(),
                'epoch': epoch + 1,
            }, f'{experiment_name}_best_model_val_kappa.pth')


        # Early stopping logic
        # if val_epoch_f1 > best_val_f1:
        #     best_val_f1 = val_epoch_f1
        #     epochs_since_improvement = 0
        # else:
        #     epochs_since_improvement += 1

        # if epochs_since_improvement >= patience:
        #     print(f"Early stopping triggered after {epoch+1} epochs.")
        #     break

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return model

def train_multiclass_model(train_loader, val_loader, model, n_epochs, lr, device, class_weights, experiment_name):
    """
    Function to train a multi-class classification model with CrossEntropyLoss.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: Model to be trained
        n_epochs: Number of epochs to train for
        lr: Learning rate
        device: Device to train on (cuda or cpu)
        class_weights: Weights for each class in the loss function (tensor)
        experiment_name: Name of the experiment for logging
    """
    patience = 3
    model = model.to(device)

    # Use CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optim_1 = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()  # Start timing

    best_val_f1 = float('-inf')
    best_val_epoch_loss = float('inf')
    best_val_kappa = float('-inf')
    epochs_since_improvement = 0

    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')

        model.train()  # set to training mode

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        # Training Phase
        for idx, (imgs, labels) in enumerate(train_loader):
            # print(f"Batch: {idx+1}/{len(train_loader)}")
            corr, tot = 0, 0
            imgs = imgs.to(device)
            labels = labels.to(device)
            optim_1.zero_grad()

            features = model.features(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            loss.backward()
            optim_1.step()

            running_loss += loss.item()

            _, preds = torch.max(output, 1)  # Get the predicted class
            print(f"Preds: ", preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            tot += labels.size(0)
            corr += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            if(idx % 10 == 0):
                print(np.array(labels.cpu().numpy()))
                print(np.array(preds.cpu().numpy()))
                print(f"Batch Accuracy: {corr / tot}")

            batch_acc = corr / tot

            # Log metrics every batch
            wandb.log({
                "batch": (idx + 1) + epoch * len(train_loader),
                "batch_loss": loss.item(),
                "batch_accuracy": batch_acc
            })

            del imgs, labels, output, features
            torch.cuda.empty_cache()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        # Calculate multi-class metrics
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_kappa = cohen_kappa_score(all_labels, all_preds)

        # Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "train_precision": epoch_precision,
            "train_recall": epoch_recall,
            "train_f1": epoch_f1,
            "train_kappa": epoch_kappa
        })

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).long()  # Ensure labels are long for CrossEntropyLoss

                features = model.features(imgs)
                output = model.classifier(features)

                loss = criterion(output, labels)

                val_running_loss += loss.item()

                _, preds = torch.max(output, 1)  # Get the predicted class
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                del imgs, labels, output, features
                torch.cuda.empty_cache()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        
        # Calculate multi-class metrics
        val_epoch_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_epoch_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_epoch_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_kappa = cohen_kappa_score(val_labels, val_preds)

        # Log validation metrics
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "val_precision": val_epoch_precision,
            "val_recall": val_epoch_recall,
            "val_f1": val_epoch_f1,
            "val_kappa": val_kappa
        })

        # Print results per epoch
        print(f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}, "
              f"Training Precision: {epoch_precision:.4f}, Training Recall: {epoch_recall:.4f}, Training F1: {epoch_f1:.4f}, Training Kappa: {epoch_kappa}")

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}, "
              f"Validation Precision: {val_epoch_precision:.4f}, Validation Recall: {val_epoch_recall:.4f}, "
              f"Validation F1: {val_epoch_f1:.4f}, Validation Kappa: {val_kappa}")
        
        cm_epoch = confusion_matrix(all_labels, all_preds)
        cm_val_epoch = confusion_matrix(val_labels, val_preds)

        print(f"Training Confusion Matrix:\n{cm_epoch}")
        print(f"Validation Confusion Matrix:\n{cm_val_epoch}")
        

        # Update best validation metrics
        if val_epoch_loss < best_val_epoch_loss:
            print(f"Prev. best val loss: {best_val_epoch_loss:.4f}, New best val loss: {val_epoch_loss:.4f}\n")
            best_val_epoch_loss = val_epoch_loss
            # Save the model state
            torch.save({'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optim_1.state_dict(),
                       'epoch': epoch+1,
                       }, f'/home/sean/MSc/code/multi-class/{experiment_name}_best_model_val_loss.pth')

        if val_kappa > best_val_kappa:
            print(f"Prev. best val kappa: {best_val_kappa:.4f}, New best val kappa: {val_kappa:.4f}\n")
            best_val_kappa = val_kappa
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim_1.state_dict(),
                'epoch': epoch + 1,
            }, f'/home/sean/MSc/code/multi-class/{experiment_name}_best_model_val_kappa.pth')
            
        if val_epoch_f1 > best_val_f1:
            print(f"Prev. best val F1: {best_val_f1:.4f}, New best val F1: {val_epoch_f1:.4f}\n")
            best_val_f1 = val_epoch_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim_1.state_dict(),
                'epoch': epoch + 1,
            }, f'/home/sean/MSc/code/multi-class/{experiment_name}_best_model_val_f1.pth')

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Save final model
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim_1.state_dict(),
                'epoch': n_epochs,
                }, f'/home/sean/MSc/code/multi-class/{experiment_name}_{n_epochs}_final_model.pth')
    
    return model



def test_model(test_loader, model, device, test_dataset_name):
    print("TESTING...")
    """Function to evaluate a trained model on a specific test loader.
    Returns the true labels and predicted labels for further analysis."""
    criterion = nn.BCEWithLogitsLoss()

    label_mapping = {0: "None", 1: "Profusion ≥ 1/0"}

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            test_running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

            del imgs, labels, output, features
            torch.cuda.empty_cache()

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_kappa = cohen_kappa_score(test_labels, test_preds)

    # Create confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_mapping.values()), 
                yticklabels=list(label_mapping.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {test_dataset_name} Test Set')
    
    # Save the confusion matrix plot
    cm_path = f'confusion_matrix_{test_dataset_name.replace(" ", "_").lower()}.png'
    # plt.savefig(cm_path)
    # print(f"Confusion matrix saved to {cm_path}")
    
    # Log the confusion matrix image to wandb
    wandb.log({
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_kappa": test_kappa
    })
    
    # Add classification report
    report = classification_report(test_labels, test_preds, 
                                   target_names=list(label_mapping.values()),
                                   output_dict=True)
    

    # Print test results
    print(f"Test Results for {test_dataset_name} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}, "
        f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}, "
        f"Cohen's Kappa: {test_kappa:.4f}")
    
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(test_labels, test_preds, target_names=list(label_mapping.values()))}")

    print("****"*25)

    return test_labels, test_preds

def test_model_with_focal_loss(test_loader, model, device, test_dataset_name, alpha, gamma):
    print("TESTING with Focal Loss...")
    """Function to evaluate a trained model on a specific test loader.
    Returns the true labels and predicted labels for further analysis."""
    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)

    label_mapping = {0: "None", 1: "Profusion ≥ 1/0"}

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            test_running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

            del imgs, labels, output, features
            torch.cuda.empty_cache()

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_kappa = cohen_kappa_score(test_labels, test_preds)

    # Create confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label_mapping.values()), 
                yticklabels=list(label_mapping.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {test_dataset_name} Test Set')
    
    # Save the confusion matrix plot
    cm_path = f'confusion_matrix_{test_dataset_name.replace(" ", "_").lower()}.png'
    # plt.savefig(cm_path)
    # print(f"Confusion matrix saved to {cm_path}")
    
    # Log the confusion matrix image to wandb
    wandb.log({
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_kappa": test_kappa
    })
    
    # Add classification report
    report = classification_report(test_labels, test_preds, 
                                   target_names=list(label_mapping.values()),
                                   output_dict=True)
    

    # Print test results
    print(f"Test Results for {test_dataset_name} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}, "
        f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}, "
        f"Cohen's Kappa: {test_kappa:.4f}")
    
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(test_labels, test_preds, target_names=list(label_mapping.values()))}")

    print("****"*25)

    return test_labels, test_preds

def validate_model(val_loader, model, device, val_dataset_name):
    """Function to evaluate a trained model on a specific validation loader.
    Returns the validation loss, accuracy, precision, recall, F1 score, and Cohen's Kappa."""
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are float and match output size

            features = model.features2(imgs)
            output = model.classifier(features)

            loss = criterion(output, labels)

            val_running_loss += loss.item()

            preds = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

            del imgs, labels, output, features
            torch.cuda.empty_cache()

    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_precision = precision_score(val_labels, val_preds)
    val_recall = recall_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    val_kappa = cohen_kappa_score(val_labels, val_preds)

    # Log validation metrics
    wandb.log({
        f"{val_dataset_name}_val_loss": val_loss,
        f"{val_dataset_name}_val_accuracy": val_acc,
        f"{val_dataset_name}_val_precision": val_precision,
        f"{val_dataset_name}_val_recall": val_recall,
        f"{val_dataset_name}_val_f1": val_f1,
        f"{val_dataset_name}_val_kappa": val_kappa
    })

    # Print validation results
    print(f"Validation Results for {val_dataset_name} - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}, "
          f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, "
          f"Cohen's Kappa: {val_kappa:.4f}")

    print("****" * 25)

    return val_loss, val_acc, val_precision, val_recall, val_f1, val_kappa



def plot_tsne_2(anchors, anchor_labels, positives_negatives, target_labels, num_classes=4, 
              save_path=None, trained=False, epoch=None, log_wandb=False, plot_name=None):
    # Combine anchors and targets for t-SNE
    embeddings = np.concatenate([anchors, positives_negatives], axis=0)
    labels = np.concatenate([anchor_labels, target_labels], axis=0)
    print(f"Anchor labels for t-SNE plot: {anchor_labels}")  # Add this line for debugging

    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Normalize the labels for continuous color scale
    norm = Normalize(vmin=np.min(labels), vmax=np.max(labels))

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot anchors with a distinct marker and color by label
    anchor_indices = np.arange(len(anchors))  # The indices for anchor embeddings
    scatter_anchors = plt.scatter(tsne_embeddings[anchor_indices, 0], tsne_embeddings[anchor_indices, 1], 
                                  c=anchor_labels, cmap='viridis', marker='*', s=100, alpha=0.7, norm=norm)
    
    # Plot all other points (positives and negatives) with colors based on labels
    other_indices = np.arange(len(anchors), len(anchors) + len(positives_negatives))  # Indices for positives and negatives
    scatter = plt.scatter(tsne_embeddings[other_indices, 0], tsne_embeddings[other_indices, 1], 
                          c=labels[other_indices], cmap='viridis', s=50, alpha=0.6, norm=norm)
    
    # Add color bar for labels
    plt.colorbar(scatter, label='Class Label')
    
    # Add legend for anchors
    handles, labels = scatter_anchors.legend_elements()
    plt.legend(handles, labels, title="Anchor Class")

    if trained:
        plt.title(f'{plot_name} Visualization - Epoch {epoch}')
    else:
        plt.title(f'{plot_name} Visualization')

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Save the image if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")

    if log_wandb:
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Log to wandb
        wandb.log({
            plot_name: wandb.Image(PILImage.open(buf))  # Open from buffer and log to wandb
        })
        
        buf.close()
        
        plt.close()

    else:
        plt.show()


def plot_tsne_3d_interactive(anchors, anchor_labels, positives_negatives, target_labels, num_classes=4, 
                              save_path=None, trained=False, epoch=None, log_wandb=False, plot_name=None):
    # Combine anchors and targets for t-SNE
    embeddings = np.concatenate([anchors, positives_negatives], axis=0)
    labels = np.concatenate([anchor_labels, target_labels], axis=0)
    print(f"Anchor labels for t-SNE plot: {anchor_labels}")  # Add this line for debugging

    # Perform t-SNE with 3 components (3D)
    tsne = TSNE(n_components=3, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Create a DataFrame for Plotly
    tsne_df = pd.DataFrame(tsne_embeddings, columns=["x", "y", "z"])
    tsne_df['label'] = labels  # Add the labels to the DataFrame
    tsne_df['type'] = ['anchor'] * len(anchors) + ['positive/negative'] * len(positives_negatives)

    # Create the 3D scatter plot with Plotly
    fig = px.scatter_3d(tsne_df, x='x', y='y', z='z', color='label', 
                        color_continuous_scale='viridis',  # Using a valid colorscale
                        symbol='type',  # Differentiates anchors from positives/negatives
                        labels={'label': 'Class Label', 'type': 'Type'},
                        title=f'{plot_name} - 3D t-SNE Visualization')
    
    # If you want to save the plot as an interactive HTML file
    if save_path:
        fig.write_html(save_path)
        print(f"t-SNE plot saved to {save_path}")

    # Log the plot to wandb if needed
    if log_wandb:
        # Log the plot to wandb as an interactive HTML
        wandb.log({plot_name: wandb.Html(fig.to_html())})

    # Show the plot
    fig.show()





def extract_features_2(triplet_dataset, model, device, anchors_only=False):
    all_anchors = []
    all_labels_anchors = []
    all_pos_neg = []
    all_labels_pos_neg = []

    # First determine how to access anchor_dataset
    if hasattr(triplet_dataset, 'dataset') and hasattr(triplet_dataset.dataset, 'anchor_dataset'):
        anchor_dataset = triplet_dataset.dataset.anchor_dataset
    elif hasattr(triplet_dataset, 'anchor_dataset'):
        anchor_dataset = triplet_dataset.anchor_dataset
    else:
        raise AttributeError("Cannot find anchor_dataset in the provided dataset")

    if anchors_only:
        # Process anchors separately (only once per anchor)
        print("Extracting anchor embeddings ONLY...")
        for anchor_idx in tqdm(range(len(anchor_dataset))):
            anchor, anchor_label, anchor_filename = anchor_dataset[anchor_idx]
            anchor_embedding = model.features(anchor.unsqueeze(0).to(device)).cpu().detach().numpy()
            all_anchors.append(anchor_embedding)
            all_labels_anchors.append(np.array([anchor_label]))

        # Convert lists to arrays
        all_anchors = np.concatenate(all_anchors, axis=0)
        all_labels_anchors = np.concatenate(all_labels_anchors, axis=0)

        # Normalize the embeddings using L2 normalization
        all_anchors = normalize(all_anchors, axis=1, norm='l2')

        return all_anchors, all_labels_anchors
    else:
        # Process anchors separately (only once per anchor)
        print("Extracting anchor embeddings...")
        for anchor_idx in tqdm(range(len(anchor_dataset))):
            anchor, anchor_label, anchor_filename = anchor_dataset[anchor_idx]
            anchor_embedding = model.features(anchor.unsqueeze(0).to(device)).cpu().detach().numpy()
            all_anchors.append(anchor_embedding)
            all_labels_anchors.append(np.array([anchor_label]))  # Ensure consistent shape

        # Convert lists to arrays
        all_anchors = np.concatenate(all_anchors, axis=0)
        all_labels_anchors = np.concatenate(all_labels_anchors, axis=0)

        # Process the target dataset (positives and negatives)
        print("Extracting target dataset embeddings...")
        for _, positive, negative, _, positive_label, negative_label, _ in tqdm(triplet_dataset, total=len(triplet_dataset)):
            positive_embedding = model.features(positive.unsqueeze(0).to(device)).cpu().detach().numpy()
            negative_embedding = model.features(negative.unsqueeze(0).to(device)).cpu().detach().numpy()

            all_pos_neg.append(np.concatenate([positive_embedding, negative_embedding], axis=0))
            all_labels_pos_neg.append(np.array([positive_label, negative_label]))

        # Convert lists to arrays
        all_pos_neg = np.concatenate(all_pos_neg, axis=0)
        all_labels_pos_neg = np.concatenate(all_labels_pos_neg, axis=0)

        # Normalize the embeddings using L2 normalization
        all_anchors = normalize(all_anchors, axis=1, norm='l2')
        all_pos_neg = normalize(all_pos_neg, axis=1, norm='l2')

        return all_anchors, all_labels_anchors, all_pos_neg, all_labels_pos_neg



def train_both_models(train_loader, val_loader, model_1, model_2, n_epochs, lr, device, pos_weight_1, pos_weight_2, experiment_name):
    # Update classifiers as needed
    model_1.classifier = classes.BaseClassifier512(in_features=2048)
    model_2.classifier = classes.MultiClassBaseClassifier(in_features=2048, num_classes=4)

    # Move models to device
    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    # Define loss functions and optimizers
    criterion_1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight_1.to(device) if pos_weight_1 is not None else None)
    criterion_2 = nn.CrossEntropyLoss(weight=pos_weight_2.to(device) if pos_weight_2 is not None else None)
    optim_1 = optim.Adam(model_1.parameters(), lr=lr)
    optim_2 = optim.Adam(model_2.parameters(), lr=lr)

    for epoch in range(n_epochs):
        print(f"EPOCH: {epoch + 1}/{n_epochs}")
        
        model_1.train()
        model_2.train()

        running_loss_1 = 0.0
        running_loss_2 = 0.0
        corr_1 = 0
        corr_2 = 0
        total_1 = 0
        total_2 = 0

        all_labels_1 = []
        all_preds_1 = []
        all_labels_2 = []
        all_preds_2 = []

        # Add epoch start debugging
        print("\nStarting epoch debugging:")
        print(f"Number of batches in train_loader: {len(train_loader)}")
        print(f"Initial metric counters - Loss1: {running_loss_1}, Loss2: {running_loss_2}")

        for idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            binary_labels = (labels > 0).float().unsqueeze(1)  # Add dimension to match output

            # Print shapes for debugging
            if idx == 0:
                print("\nTensor shapes:")
                print(f"Images: {imgs.shape}")
                print(f"Labels: {labels.shape}")
                print(f"Binary labels: {binary_labels.shape}")

            # Zero gradients
            optim_1.zero_grad()
            optim_2.zero_grad()

            # Forward pass for model 1 (binary classification)
            feats_1 = model_1.features(imgs)
            output_1 = model_1.classifier(feats_1)
            loss_1 = criterion_1(output_1, binary_labels)

            # Forward pass for model 2 (multi-class classification)
            feats_2 = model_2.features(imgs)
            output_2 = model_2.classifier(feats_2)
            loss_2 = criterion_2(output_2, labels.long())

            if idx == 0:
                print(f"Output1 shape: {output_1.shape}")
                print(f"Output2 shape: {output_2.shape}")

            # Backward pass and optimization
            loss_1.backward()
            loss_2.backward()
            optim_1.step()
            optim_2.step()

            # Calculate predictions
            preds_1 = (torch.sigmoid(output_1) > 0.5).squeeze()
            _, preds_2 = torch.max(output_2, 1)

            # Track metrics for model 1 (binary classification)
            all_labels_1.extend(binary_labels.squeeze().cpu().numpy())
            all_preds_1.extend(preds_1.cpu().numpy())

            # Track metrics for model 2 (multi-class classification)
            all_labels_2.extend(labels.cpu().numpy())
            all_preds_2.extend(preds_2.cpu().numpy())

            # Update running losses (only once)
            running_loss_1 += loss_1.item()
            running_loss_2 += loss_2.item()

            # Update accuracy counters
            total_1 += labels.size(0)
            corr_1 += (preds_1 == binary_labels.squeeze()).sum().item()
            total_2 += labels.size(0)
            corr_2 += (preds_2 == labels).sum().item()

            # Calculate batch accuracies
            batch_bin_acc = corr_1 / total_1
            batch_multi_acc = corr_2 / total_2

            # Log batch metrics
            wandb.log({
                "batch": idx + 1,
                "batch_loss_binary": loss_1.item(),
                "batch_loss_multi": loss_2.item(),
                "batch_accuracy_binary": batch_bin_acc,
                "batch_accuracy_multi": batch_multi_acc,
            })

            # Print progress every 10 batches
            if (idx + 1) % 20 == 0:
                print(f"\nBatch {idx + 1} Stats:")
                print(f"Binary - Loss: {loss_1.item():.4f}, Acc: {batch_bin_acc:.4f}")
                print(f"Multi - Loss: {loss_2.item():.4f}, Acc: {batch_multi_acc:.4f}")
                
                # Convert current predictions and labels to numpy for sklearn metrics
                current_preds_1 = np.array(all_preds_1)
                current_labels_1 = np.array(all_labels_1)
                current_preds_2 = np.array(all_preds_2)
                current_labels_2 = np.array(all_labels_2)
                
                print("\nBinary Confusion Matrix:")
                print(confusion_matrix(current_labels_1, current_preds_1))
                
                print("\nMulti-Class Confusion Matrix:")
                print(confusion_matrix(current_labels_2, current_preds_2))
                
                print("\nClass Distributions:")
                print(f"Binary - Unique values in predictions: {np.unique(current_preds_1, return_counts=True)}")
                print(f"Binary - Unique values in labels: {np.unique(current_labels_1, return_counts=True)}")

                print(f"Multi - Unique values in predictions: {np.unique(current_preds_2, return_counts=True)}")
                print(f"Multi - Unique values in labels: {np.unique(current_labels_2, return_counts=True)}")


            # Clean up memory
            del imgs, labels, feats_1, feats_2, output_1, output_2, loss_1, loss_2, binary_labels
            torch.cuda.empty_cache()

        # Convert lists to arrays for metric calculation
        all_labels_1 = np.array(all_labels_1)
        all_preds_1 = np.array(all_preds_1)
        all_labels_2 = np.array(all_labels_2)
        all_preds_2 = np.array(all_preds_2)

        # Calculate epoch metrics
        epoch_bin_loss = running_loss_1 / len(train_loader)
        epoch_multi_loss = running_loss_2 / len(train_loader)
        epoch_bin_acc = corr_1 / total_1
        epoch_multi_acc = corr_2 / total_2

        # Print final epoch stats
        print(f"\nEpoch {epoch + 1} Final Stats:")
        print(f"Binary - Loss: {epoch_bin_loss:.4f}, Acc: {epoch_bin_acc:.4f}")
        print(f"Multi - Loss: {epoch_multi_loss:.4f}, Acc: {epoch_multi_acc:.4f}")

        # Calculate detailed metrics
        epoch_bin_precision = precision_score(all_labels_1, all_preds_1, average='binary')
        epoch_bin_recall = recall_score(all_labels_1, all_preds_1, average='binary')
        epoch_bin_f1 = f1_score(all_labels_1, all_preds_1, average='binary')
        epoch_bin_kappa = cohen_kappa_score(all_labels_1, all_preds_1)

        epoch_multi_precision = precision_score(all_labels_2, all_preds_2, average='macro')
        epoch_multi_recall = recall_score(all_labels_2, all_preds_2, average='macro')
        epoch_multi_f1 = f1_score(all_labels_2, all_preds_2, average='macro')
        epoch_multi_kappa = cohen_kappa_score(all_labels_2, all_preds_2)

        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "epoch_loss_binary": epoch_bin_loss,
            "epoch_loss_multi": epoch_multi_loss,
            "epoch_accuracy_binary": epoch_bin_acc,
            "epoch_accuracy_multi": epoch_multi_acc,
            "epoch_precision_binary": epoch_bin_precision,
            "epoch_recall_binary": epoch_bin_recall,
            "epoch_f1_binary": epoch_bin_f1,
            "epoch_kappa_binary": epoch_bin_kappa,
            "epoch_precision_multi": epoch_multi_precision,
            "epoch_recall_multi": epoch_multi_recall,
            "epoch_f1_multi": epoch_multi_f1,
            "epoch_kappa_multi": epoch_multi_kappa
        })

        
        # Validation loop
        model_1.eval()
        model_2.eval()
        val_loss_1 = 0.0
        val_loss_2 = 0.0
        val_corr_1 = 0
        val_corr_2 = 0
        val_total = 0
        
        val_labels_1 = []
        val_preds_1 = []
        val_labels_2 = []
        val_preds_2 = []
        
        print("\nStarting validation...")
        with torch.no_grad():
            for val_imgs, val_labels in val_loader:
                val_imgs = val_imgs.to(device)
                val_labels = val_labels.to(device)
                val_binary_labels = (val_labels > 0).float().unsqueeze(1)
                
                # Model 1 validation (binary)
                val_feats_1 = model_1.features(val_imgs)
                val_output_1 = model_1.classifier(val_feats_1)
                val_loss_1 += criterion_1(val_output_1, val_binary_labels).item()
                
                # Handle potential single-sample case for binary predictions
                val_preds_1_batch = (torch.sigmoid(val_output_1) > 0.5)
                if val_preds_1_batch.dim() == 0:
                    val_preds_1_batch = val_preds_1_batch.unsqueeze(0)
                val_preds_1_batch = val_preds_1_batch.squeeze(-1)  # Remove last dimension if present
                
                # Model 2 validation (multi-class)
                val_feats_2 = model_2.features(val_imgs)
                val_output_2 = model_2.classifier(val_feats_2)
                val_loss_2 += criterion_2(val_output_2, val_labels.long()).item()
                _, val_preds_2_batch = torch.max(val_output_2, 1)
                
                # Track validation metrics
                val_total += val_labels.size(0)
                val_corr_1 += (val_preds_1_batch == val_binary_labels.squeeze()).sum().item()
                val_corr_2 += (val_preds_2_batch == val_labels).sum().item()
                
                # Ensure proper dimensionality when extending lists
                val_labels_1.extend(val_binary_labels.squeeze(-1).cpu().numpy().flatten())
                val_preds_1.extend(val_preds_1_batch.cpu().numpy().flatten())
                val_labels_2.extend(val_labels.cpu().numpy().flatten())
                val_preds_2.extend(val_preds_2_batch.cpu().numpy().flatten())
        
        # Calculate validation metrics
        val_labels_1 = np.array(val_labels_1)
        val_preds_1 = np.array(val_preds_1)
        val_labels_2 = np.array(val_labels_2)
        val_preds_2 = np.array(val_preds_2)
        
        val_loss_1 /= len(val_loader)
        val_loss_2 /= len(val_loader)
        val_acc_1 = val_corr_1 / val_total
        val_acc_2 = val_corr_2 / val_total
        
        # Calculate detailed validation metrics
        val_bin_precision = precision_score(val_labels_1, val_preds_1, average='binary')
        val_bin_recall = recall_score(val_labels_1, val_preds_1, average='binary')
        val_bin_f1 = f1_score(val_labels_1, val_preds_1, average='binary')
        val_bin_kappa = cohen_kappa_score(val_labels_1, val_preds_1)
        
        val_multi_precision = precision_score(val_labels_2, val_preds_2, average='macro')
        val_multi_recall = recall_score(val_labels_2, val_preds_2, average='macro')
        val_multi_f1 = f1_score(val_labels_2, val_preds_2, average='macro')
        val_multi_kappa = cohen_kappa_score(val_labels_2, val_preds_2)
        
        # Print validation results
        print("\nValidation Results:")
        print(f"Binary - Loss: {val_loss_1:.4f}, Acc: {val_acc_1:.4f}")
        print(f"Multi - Loss: {val_loss_2:.4f}, Acc: {val_acc_2:.4f}")
        
        print("\nBinary Validation Classification Report:")
        print(classification_report(val_labels_1, val_preds_1,
                                 target_names=['Normal', 'Abnormal'],
                                 digits=4))
        
        print("\nMulti-Class Validation Classification Report:")
        print(classification_report(val_labels_2, val_preds_2,
                                 target_names=['Normal', 'Low', 'Medium', 'High'],
                                 digits=4))
        
        # Log validation metrics to wandb
        wandb.log({
            "val_loss_binary": val_loss_1,
            "val_loss_multi": val_loss_2,
            "val_accuracy_binary": val_acc_1,
            "val_accuracy_multi": val_acc_2,
            "val_precision_binary": val_bin_precision,
            "val_recall_binary": val_bin_recall,
            "val_f1_binary": val_bin_f1,
            "val_kappa_binary": val_bin_kappa,
            "val_precision_multi": val_multi_precision,
            "val_recall_multi": val_multi_recall,
            "val_f1_multi": val_multi_f1,
            "val_kappa_multi": val_multi_kappa
        })


    return model_1, model_2


def test_both_models(test_loader, model_1, model_2, device):
    """
    Test both binary and multi-class models and return predictions and labels
    
    Args:
        test_loader: DataLoader for test data
        model_1: Binary classification model
        model_2: Multi-class classification model
        device: torch device (cuda/cpu)
    
    Returns:
        Dictionary containing test predictions and labels for both models
    """
    model_1.eval()
    model_2.eval()
    
    test_labels_1 = []
    test_preds_1 = []
    test_labels_2 = []
    test_preds_2 = []
    
    print("\nStarting model testing...")
    with torch.no_grad():
        for test_imgs, test_labels in test_loader:
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_binary_labels = (test_labels > 0).float().unsqueeze(1)
            
            # Model 1 (Binary) testing
            test_feats_1 = model_1.features(test_imgs)
            test_output_1 = model_1.classifier(test_feats_1)
            test_preds_1_batch = (torch.sigmoid(test_output_1) > 0.5)
            
            # Handle dimensionality
            if test_preds_1_batch.dim() == 0:
                test_preds_1_batch = test_preds_1_batch.unsqueeze(0)
            test_preds_1_batch = test_preds_1_batch.squeeze(-1)
            
            # Model 2 (Multi-class) testing
            test_feats_2 = model_2.features(test_imgs)
            test_output_2 = model_2.classifier(test_feats_2)
            _, test_preds_2_batch = torch.max(test_output_2, 1)
            
            # Store predictions and labels
            test_labels_1.extend(test_binary_labels.squeeze(-1).cpu().numpy().flatten())
            test_preds_1.extend(test_preds_1_batch.cpu().numpy().flatten())
            test_labels_2.extend(test_labels.cpu().numpy().flatten())
            test_preds_2.extend(test_preds_2_batch.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    test_labels_1 = np.array(test_labels_1)
    test_preds_1 = np.array(test_preds_1)
    test_labels_2 = np.array(test_labels_2)
    test_preds_2 = np.array(test_preds_2)
    
    # Calculate and print test metrics
    print("\nTest Results:")
    
    # Binary classification metrics
    print("\nBinary Classification Report:")
    print(classification_report(test_labels_1, test_preds_1,
                              target_names=['Normal', 'Abnormal'],
                              digits=4))
    
    print("\nBinary Confusion Matrix:")
    print(confusion_matrix(test_labels_1, test_preds_1))
    
    # Multi-class classification metrics
    print("\nMulti-Class Classification Report:")
    print(classification_report(test_labels_2, test_preds_2,
                              target_names=['Normal', 'Low', 'Medium', 'High'],
                              digits=4))
    
    print("\nMulti-Class Confusion Matrix:")
    print(confusion_matrix(test_labels_2, test_preds_2))
    
    return {
        'binary_preds': test_preds_1,
        'binary_labels': test_labels_1,
        'multi_preds': test_preds_2,
        'multi_labels': test_labels_2
    }
