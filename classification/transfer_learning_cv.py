import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchxrayvision as xrv
import wandb
import numpy as np
from datetime import datetime
import random
from tqdm import tqdm
import utils
import metrics
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import sklearn

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))
from datasets.dataloader import get_dataloaders

# Add codev2 and DomainAdaptation to path
sys.path.append(os.path.abspath("../codev2"))
sys.path.append(os.path.abspath("../DomainAdaptation"))
from clf_manager import BinaryClassifier, MulticlassClassifier, XRVBasedClassifier

from binary_cv_old import plot_combined_conf_mat, plot_tb_stratified_binary_cm


class TransferLearningCrossValidator:
    """
    Transfer Learning Cross-Validator that trains on TB-Net and validates/tests on both TB-Net and MBOD datasets.
    """
    
    def __init__(self, model, labels_key, num_classes, device, loss_fn, 
                 tbnet_hdf5_path, mbod_hdf5_path, batch_size, epochs, 
                 optimizer_type, learning_rate, weight_decay, use_oversampling, 
                 checkpoint_save_target, exp_name, 
                 tbnet_split_file, mbod_split_file,
                 train_set_name, test_set_name, clf_task_labels_key):
        
        self.model = model
        self.labels_key = labels_key
        self.device = device
        self.loss_fn = loss_fn
        self.tbnet_hdf5_path = tbnet_hdf5_path
        self.mbod_hdf5_path = mbod_hdf5_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_oversampling = use_oversampling
        self.checkpoint_save_target = checkpoint_save_target
        self.exp_name = exp_name
        self.num_classes = num_classes
        self.tbnet_split_file = tbnet_split_file
        self.mbod_split_file = mbod_split_file
        self.train_set_name = train_set_name
        self.test_set_name = test_set_name
        self.clf_task_labels_key = clf_task_labels_key
        
        self.preprocess = T.Compose([
            T.ToTensor(),
        ])
        
        self.checkpoint_save_target = f"{self.checkpoint_save_target}/{self.exp_name}"
        
        if not os.path.isdir(self.checkpoint_save_target):
            os.makedirs(self.checkpoint_save_target)
    
    def get_train_dataloaders(self):
        """Get training dataloaders - only from TB-Net dataset"""
        augmentations = T.Compose([
            T.RandomRotation(degrees=10, expand=False, fill=0),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
        ])
        
        train_loader, _, _ = get_dataloaders(
            hdf5_path=self.tbnet_hdf5_path,
            preprocess=self.preprocess,
            train_split=0,
            batch_size=self.batch_size,
            labels_key=self.labels_key,
            split_file=self.tbnet_split_file,
            augmentations=augmentations,
            oversample=self.use_oversampling
        )
        
        return train_loader
    
    def get_eval_dataloaders(self):
        """Get evaluation dataloaders for both datasets"""
        # TB-Net validation and test sets
        _, tbnet_val_loader, tbnet_test_loader = get_dataloaders(
            hdf5_path=self.tbnet_hdf5_path,
            preprocess=self.preprocess,
            train_split=0,
            batch_size=1,
            labels_key=self.labels_key,
            split_file=self.tbnet_split_file,
            augmentations=None,
            oversample=False
        )
        
        # MBOD validation and test sets
        _, mbod_val_loader, mbod_test_loader = get_dataloaders(
            hdf5_path=self.mbod_hdf5_path,
            preprocess=self.preprocess,
            train_split=0,
            batch_size=1,
            labels_key=self.labels_key,
            split_file=self.mbod_split_file,
            augmentations=None,
            oversample=False
        )
        
        return {
            'tbnet_val': tbnet_val_loader,
            'tbnet_test': tbnet_test_loader,
            'mbod_val': mbod_val_loader,
            'mbod_test': mbod_test_loader
        }
    
    def evaluate_dataset(self, loader, epoch, fold, dataset_name, split_name="", log=True):
        """Evaluate model on a specific dataset and split"""
        self.model.eval()
        
        all_labels = []
        all_probs = []
        all_feats = []
        all_preds = []
        
        with torch.no_grad():
            for batch_imgs, batch_labels in tqdm(loader, desc=f"Evaluating {dataset_name} {split_name}"):
                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
                
                feats = self.model.features(batch_imgs)
                logits = self.model.classifier(feats)
                
                # Preprocessing for binary classification
                if self.num_classes == 2:
                    if batch_labels.dim() == 1:
                        batch_labels = batch_labels.unsqueeze(1)
                    batch_labels = batch_labels.float()
                
                loss = self.loss_fn(logits, batch_labels)
                
                loc_probs = torch.sigmoid(logits)
                all_probs.append(loc_probs.cpu().numpy())
                
                # For binary, threshold the sigmoid output
                binary_preds = (loc_probs > 0.5).int()
                all_preds.append(binary_preds.cpu().numpy())
                
                all_labels.append(batch_labels.cpu().numpy())
                all_feats.append(feats.cpu().numpy())
        
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        
        # Calculate metrics
        if self.num_classes == 2:
            spec_at_09_sens, thresh = metrics.specificity_at_sensitivity(all_labels, all_probs, min_sens=0.9)
            predictions = (all_probs > thresh)
            predictions_05 = (all_probs > 0.5)
        else:
            raise ValueError("Only binary classification supported")
        
        # Log dataset-specific metrics
        prefix = f"{dataset_name}_{split_name}_" if split_name else f"{dataset_name}_"
        if log:
            wandb.log({
                f"{prefix}spec_at_09_sens": spec_at_09_sens,
            }, step=epoch)
        
        # Confusion matrices
        global_conf_mat = confusion_matrix(all_labels, predictions)
        global_conf_mat_05 = confusion_matrix(all_labels, predictions_05)
        
        # Calculate standard metrics
        accuracy = metrics.get_accuracy(global_conf_mat)
        sensitivity = metrics.get_sensitivity(global_conf_mat)
        specificity = metrics.get_specificity(global_conf_mat)
        f1 = metrics.get_f1_score(global_conf_mat)
        kappa = metrics.get_cohens_kappa(global_conf_mat)
        
        accuracy_05 = metrics.get_accuracy(global_conf_mat_05)
        sensitivity_05 = metrics.get_sensitivity(global_conf_mat_05)
        specificity_05 = metrics.get_specificity(global_conf_mat_05)
        f1_05 = metrics.get_f1_score(global_conf_mat_05)
        kappa_05 = metrics.get_cohens_kappa(global_conf_mat_05)
        
        # Create confusion matrix plots
        comb_cm = plot_combined_conf_mat(global_conf_mat)
        comb_cm_05 = plot_combined_conf_mat(global_conf_mat_05)
        
        results = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'kappa': kappa,
            'accuracy_05': accuracy_05,
            'sensitivity_05': sensitivity_05,
            'specificity_05': specificity_05,
            'f1_05': f1_05,
            'kappa_05': kappa_05,
            'confusion_matrix': comb_cm,
            'confusion_matrix_05': comb_cm_05,
            'spec_at_09_sens': spec_at_09_sens,
            'thresh': thresh
        }
        
        return results
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        total = 0
        
        for batch_imgs, batch_labels in tqdm(train_loader, desc="Training"):
            batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
            
            features = self.model.features(batch_imgs)
            logits = self.model.classifier(features)
            
            loss_labels = batch_labels
            if self.num_classes == 2:
                if loss_labels.dim() == 1:
                    loss_labels = loss_labels.unsqueeze(1)
                loss_labels = loss_labels.float()
            
            loss = self.loss_fn(logits, loss_labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            total += batch_labels.size(0)
        
        return epoch_loss / total
    
    def train(self, iteration=None):
        """Main training loop with dual dataset evaluation"""
        best_val_specificity = 0
        patience = 100
        min_delta = 0.001
        epochs_without_improvement = 0
        
        # Get dataloaders
        train_loader = self.get_train_dataloaders()
        eval_loaders = self.get_eval_dataloaders()
        
        self.optimizer = self.optimizer_type(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):
            # Training phase (only on TB-Net)
            train_loss = self.train_epoch(train_loader)
            
            # Evaluation phase on both datasets
            all_results = {}
            
            # Evaluate on TB-Net train set for monitoring
            train_results = self.evaluate_dataset(
                train_loader, epoch, iteration, "tbnet", "train"
            )
            
            # Evaluate on TB-Net validation set
            tbnet_val_results = self.evaluate_dataset(
                eval_loaders['tbnet_val'], epoch, iteration, "tbnet", "val"
            )
            
            # Evaluate on MBOD validation set  
            mbod_val_results = self.evaluate_dataset(
                eval_loaders['mbod_val'], epoch, iteration, "mbod", "val"
            )
            
            # Log training metrics
            wandb.log({
                "train/loss": train_loss,
                "fold": iteration,
                "tbnet_train/acc": train_results['accuracy'],
                "tbnet_train/sens": train_results['sensitivity'],
                "tbnet_train/spec": train_results['specificity'],
                "tbnet_train/f1": train_results['f1'],
                "tbnet_train/kappa": train_results['kappa'],
                "cm/tbnet_train": wandb.Image(train_results['confusion_matrix']),
                "cm/tbnet_train_05": wandb.Image(train_results['confusion_matrix_05'])
            }, step=epoch)
            plt.close(train_results['confusion_matrix'])
            plt.close(train_results['confusion_matrix_05'])
            
            # Log TB-Net validation metrics
            wandb.log({
                "tbnet_val/acc": tbnet_val_results['accuracy'],
                "tbnet_val/sens": tbnet_val_results['sensitivity'],
                "tbnet_val/spec": tbnet_val_results['specificity'],
                "tbnet_val/f1": tbnet_val_results['f1'],
                "tbnet_val/kappa": tbnet_val_results['kappa'],
                "cm/tbnet_val": wandb.Image(tbnet_val_results['confusion_matrix']),
                "cm/tbnet_val_05": wandb.Image(tbnet_val_results['confusion_matrix_05'])
            }, step=epoch)
            plt.close(tbnet_val_results['confusion_matrix'])
            plt.close(tbnet_val_results['confusion_matrix_05'])
            
            # Log MBOD validation metrics
            wandb.log({
                "mbod_val/acc": mbod_val_results['accuracy'],
                "mbod_val/sens": mbod_val_results['sensitivity'],
                "mbod_val/spec": mbod_val_results['specificity'],
                "mbod_val/f1": mbod_val_results['f1'],
                "mbod_val/kappa": mbod_val_results['kappa'],
                "cm/mbod_val": wandb.Image(mbod_val_results['confusion_matrix']),
                "cm/mbod_val_05": wandb.Image(mbod_val_results['confusion_matrix_05'])
            }, step=epoch)
            plt.close(mbod_val_results['confusion_matrix'])
            plt.close(mbod_val_results['confusion_matrix_05'])
            
            # Early stopping based on TB-Net validation specificity
            val_spec = tbnet_val_results['specificity']
            if val_spec > best_val_specificity + min_delta:
                print(f"New best val spec: {val_spec:.4f} at epoch {epoch}, previous best: {best_val_specificity:.4f}")
                best_val_specificity = val_spec
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), 
                          f"{self.checkpoint_save_target}/{epoch}-{val_spec:.4f}-{iteration}.pth")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered. No improvement for {patience} epochs.")
                    break
        
        # Final evaluation on test sets
        tbnet_test_results = self.evaluate_dataset(
            eval_loaders['tbnet_test'], epoch, iteration, "tbnet", "test", log=False
        )
        
        mbod_test_results = self.evaluate_dataset(
            eval_loaders['mbod_test'], epoch, iteration, "mbod", "test", log=False
        )
        
        # Save final model
        model_name = f"final_model_{iteration}.pth" if iteration is not None else "final_model.pth"
        torch.save(self.model.state_dict(), f"{self.checkpoint_save_target}/{model_name}")
        
        return {
            'tbnet_test': tbnet_test_results,
            'mbod_test': mbod_test_results
        }
    
    def run_k_iterations(self, k, project_name, exp_name, suffix=''):
        """Run k-fold cross validation with transfer learning"""
        # Store results for both datasets
        tbnet_results = {'accuracies': [], 'sensitivities': [], 'specificities': [], 'kappas': []}
        mbod_results = {'accuracies': [], 'sensitivities': [], 'specificities': [], 'kappas': []}
        
        wandb.login()
        group_name = f"{self.checkpoint_save_target}/{self.exp_name}-{suffix}-{wandb.util.generate_id()}"
        
        for i in range(k):
            wandb.init(
                project=project_name,
                group=group_name,
                name=f"{exp_name}-fold_{i}",
                config={
                    "loss_fn": self.loss_fn.__class__.__name__,
                    "optimizer": self.optimizer_type.__name__,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "labels_key": self.labels_key,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "save_dir": self.checkpoint_save_target,
                    "fold": i,
                    "train_dataset": "TB-Net",
                    "eval_datasets": ["TB-Net", "MBOD"]
                },
            )
            
            # Train and get results
            test_results = self.train(iteration=i)
            
            # Extract TB-Net results
            tbnet_acc = test_results['tbnet_test']['accuracy']
            tbnet_sens = test_results['tbnet_test']['sensitivity']
            tbnet_spec = test_results['tbnet_test']['specificity']
            tbnet_kappa = test_results['tbnet_test']['kappa']
            
            tbnet_results['accuracies'].append(tbnet_acc)
            tbnet_results['sensitivities'].append(tbnet_sens)
            tbnet_results['specificities'].append(tbnet_spec)
            tbnet_results['kappas'].append(tbnet_kappa)
            
            # Extract MBOD results
            mbod_acc = test_results['mbod_test']['accuracy']
            mbod_sens = test_results['mbod_test']['sensitivity']
            mbod_spec = test_results['mbod_test']['specificity']
            mbod_kappa = test_results['mbod_test']['kappa']
            
            mbod_results['accuracies'].append(mbod_acc)
            mbod_results['sensitivities'].append(mbod_sens)
            mbod_results['specificities'].append(mbod_spec)
            mbod_results['kappas'].append(mbod_kappa)
            
            # Log final test results
            wandb.log({
                "tbnet_test/accuracy": tbnet_acc,
                "tbnet_test/sensitivity": tbnet_sens,
                "tbnet_test/specificity": tbnet_spec,
                "tbnet_test/f1": test_results['tbnet_test']['f1'],
                "tbnet_test/kappa": tbnet_kappa,
                "cm/tbnet_test": wandb.Image(test_results['tbnet_test']['confusion_matrix']),
                "cm/tbnet_test_05": wandb.Image(test_results['tbnet_test']['confusion_matrix_05']),
                
                "mbod_test/accuracy": mbod_acc,
                "mbod_test/sensitivity": mbod_sens,
                "mbod_test/specificity": mbod_spec,
                "mbod_test/f1": test_results['mbod_test']['f1'],
                "mbod_test/kappa": mbod_kappa,
                "cm/mbod_test": wandb.Image(test_results['mbod_test']['confusion_matrix']),
                "cm/mbod_test_05": wandb.Image(test_results['mbod_test']['confusion_matrix_05']),
            })
            
            plt.close(test_results['tbnet_test']['confusion_matrix'])
            plt.close(test_results['tbnet_test']['confusion_matrix_05'])
            plt.close(test_results['mbod_test']['confusion_matrix'])
            plt.close(test_results['mbod_test']['confusion_matrix_05'])
            
            wandb.finish()
        
        # Calculate averages
        print(f"\n=== Transfer Learning Results over {k} iterations ===")
        print("Training Dataset: TB-Net")
        print("Validation/Test Datasets: TB-Net + MBOD")
        
        print(f"\n--- TB-Net Test Results ---")
        print(f"Average Test Accuracy: {np.mean(tbnet_results['accuracies']):.4f}")
        print(f"Average Sensitivity: {np.mean(tbnet_results['sensitivities']):.4f}")
        print(f"Average Specificity: {np.mean(tbnet_results['specificities']):.4f}")
        print(f"Average Kappa: {np.mean(tbnet_results['kappas']):.4f}")
        
        print(f"\n--- MBOD Test Results (Transfer Learning) ---")
        print(f"Average Test Accuracy: {np.mean(mbod_results['accuracies']):.4f}")
        print(f"Average Sensitivity: {np.mean(mbod_results['sensitivities']):.4f}")
        print(f"Average Specificity: {np.mean(mbod_results['specificities']):.4f}")
        print(f"Average Kappa: {np.mean(mbod_results['kappas']):.4f}")
        
        return {
            'tbnet': {
                'avg_accuracy': np.mean(tbnet_results['accuracies']),
                'avg_sensitivity': np.mean(tbnet_results['sensitivities']),
                'avg_specificity': np.mean(tbnet_results['specificities']),
                'avg_kappa': np.mean(tbnet_results['kappas'])
            },
            'mbod': {
                'avg_accuracy': np.mean(mbod_results['accuracies']),
                'avg_sensitivity': np.mean(mbod_results['sensitivities']),
                'avg_specificity': np.mean(mbod_results['specificities']),
                'avg_kappa': np.mean(mbod_results['kappas'])
            }
        }