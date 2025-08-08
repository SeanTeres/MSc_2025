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

def plot_combined_conf_mat(confusion_matrix):
    if confusion_matrix.shape == (2, 2):
        # Binary case
        fig = plt.figure(figsize=(12, 8))
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca())
        plt.title("Confusion Matrix")
    elif len(confusion_matrix.shape) == 3:
        # Multilabel case
        avg_cm = np.mean(confusion_matrix, axis=0)
        norm_cm = avg_cm / (np.sum(avg_cm, axis=1, keepdims=True))
        fig = plt.figure(figsize=(12, 8))
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(norm_cm, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca())
        plt.title("Normalised Multilabel Confusion Matrix")
    else:
        # Multiclass case
        TN, FP, FN, TP = metrics.get_cm_for_class(confusion_matrix, 0)
        bin_conf_matrix = np.array([[TP, FN],
                                    [FP, TN]])
        display_labels = [i for i in range(len(confusion_matrix[0]))]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        cm_display1 = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=display_labels)
        cm_display1.plot(ax=axes[0], colorbar=False)
        axes[0].set_title("Confusion Matrix")
        cm_display2 = sklearn.metrics.ConfusionMatrixDisplay(bin_conf_matrix, display_labels=[0, 1])
        cm_display2.plot(ax=axes[1], colorbar=False)
        axes[1].set_title("Binary Confusion Matrix")
        plt.tight_layout()
    return fig

class CrossValidator:
    def __init__(self, model, labels_key, num_classes, device, loss_fn, hdf5_path, batch_size, epochs, optimizer_type, learning_rate, weight_decay, use_oversampling, checkpoint_save_target, use_multilabel, use_multiclass, use_ordinal_labels, exp_name, split_file):
        self.model = model
        self.labels_key = labels_key
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
        self.use_multilabel = use_multilabel
        self.use_multiclass = use_multiclass
        self.use_ordinal_labels = use_ordinal_labels
        self.exp_name = exp_name
        self.num_classes = num_classes
        self.split_file = split_file
        self.preprocess = T.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        # transforms.Grayscale(),
        T.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        

        if not (os.path.isdir(checkpoint_save_target)):
            os.makedirs(checkpoint_save_target)


    def local_get_dataloaders(self, train_split=None, iteration=None):
        now = datetime.now()
        currentTime = now.strftime("%Y-%m-%d_%H-%M-%S")

        if self.split_file:
            print(f"Loading from split files: {self.split_file}")
            augmentations = T.Compose([
                T.RandomRotation(degrees=10, expand=False, fill=0),
                # T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                # T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
            ])

            train_loader, _, _ = get_dataloaders(
                hdf5_path=self.hdf5_path,
                preprocess=self.preprocess,
                train_split=0,
                batch_size=self.batch_size,
                labels_key=self.labels_key,
                split_file=self.split_file,
                augmentations=augmentations,
                oversample=self.use_oversampling
            )

            _, val_loader, test_loader = get_dataloaders(
                hdf5_path=self.hdf5_path,
                preprocess=self.preprocess,
                train_split=0,
                batch_size=1,
                labels_key=self.labels_key,
                split_file=self.split_file,
                augmentations=None,
                oversample=False
            )
        else:
            if train_split is None:
                train_split = random.randint(40, 60) / 100

            train_loader, val_loader, test_loader = get_dataloaders(
                hdf5_path=self.hdf5_path,
                preprocess=self.preprocess,
                train_split=train_split,
                batch_size=self.batch_size,
                labels_key=self.labels_key,
                split_file=f"data_splits/{currentTime}_{iteration}.json"
            )

        return train_loader, val_loader, test_loader
    

    def evaluate(self, loader, epoch, fold, name="", log=True):
        self.model.eval()

        global_confusion = None

        all_labels = []
        all_probs = []
        all_feats = []
        all_preds = []

        with torch.no_grad():

            for batch_imgs, batch_labels in tqdm(loader):

                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)

                feats = self.model.features(batch_imgs)

                logits = self.model.classifier(feats)

                loss = self.loss_fn(logits, batch_labels)

                if self.use_multiclass:
                    loc_probs = torch.softmax(logits, dim=1)
                    all_probs.append(loc_probs.cpu().numpy())
                elif self.num_classes == 2:
                    loc_probs = torch.sigmoid(logits)
                    all_probs.append(loc_probs.cpu().numpy())
                else:
                    raise ValueError("wrong activation - Only binary and multiclass classification supported.")

                assert not torch.isnan(loc_probs).any(), f"NaN in locprobabilities, logits: {logits}"
                assert not torch.isinf(loc_probs).any(), f"Inf in locprobabilities, logits: {logits}"

                all_labels.append(batch_labels.cpu().numpy())
                all_feats.append(feats.cpu().numpy())
                all_preds.append(logits.argmax(dim=1).cpu().numpy())


                    
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        if self.use_multiclass:
            spec_at_09_sens, thresh = metrics.multiclass_specificity_at_sensitivity(all_labels, all_probs, min_sens=0.9)
        elif self.num_classes==2:
            spec_at_09_sens, thresh = metrics.specificity_at_sensitivity(all_labels, all_probs, min_sens=0.9)
        else:
            raise ValueError("Spec at sens err: Only multiclass and binary currently supported.")
        
        if log:
            wandb.log({f"{name}spec_at_09_sens": spec_at_09_sens}, step=epoch)

        
        if self.use_multiclass or self.num_classes == 2:
            predictions = (all_probs > thresh)
        else:
            raise ValueError("Incorrect thresholding - only multiclass or binary supported.")
        
        if not self.use_multilabel:
            global_conf_mat = metrics.confusion_matrix(all_labels, all_preds, labels=[i for i in range(self.num_classes)])
        else:
            raise ValueError("Confusion matrix not implemented for multilabel classification.")
    

        # HERE IS WHERE WE CAN PLOT TNSE ---------- to do


        accuracy = metrics.get_accuracy(global_conf_mat)
        sensitivity = metrics.get_sensitivity(global_conf_mat)
        specificity = metrics.get_specificity(global_conf_mat)
        f1 = metrics.get_f1_score(global_conf_mat)


        if self.use_multilabel:
            # This is where we get per-class metrics from function
            # ml_metrics_dict, average_auc = multilabel_metrics.calculate_per_class_metrics(
            #     global_confusion,
            #     prefix=name,
            #     y_true=all_labels,
            #     y_prob=all_probs,
            #     class_names=self.label_descs
            # )

            # if log:
                # wandb.log(ml_metrics_dict, step=epoch)
            raise ValueError("Multilabel metrics not implemented yet.")
        else:
            ml_metrics_dict = None
            average_auc = None

        comb_cm = plot_combined_conf_mat(global_conf_mat)

        if global_conf_mat.shape != (2,2) and (not self.use_multilabel):
            tn, fp, fn, tp = metrics.get_cm_for_class(global_conf_mat, 0)

            bin_cm = np.array([[tp, fn],
                  [fp, tn]])
            
            bin_acc = metrics.get_accuracy(bin_cm)
            bin_sens = metrics.get_sensitivity(bin_cm)
            bin_spec = metrics.get_specificity(bin_cm)
            bin_f1 = metrics.get_f1_score(bin_cm)
        else:
            bin_acc, bin_sens, bin_spec, bin_f1 = None, None, None, None
        
        return accuracy, sensitivity, specificity, f1, bin_acc, bin_sens, bin_spec, bin_f1, comb_cm, average_auc, 

    def train(self, train_split=None, iteration=None):
        best_val_specificity = 0
        
        patience=100
        min_delta=0.001
        epochs_without_improvement = 0


        train_loader, val_loader, test_loader = self.local_get_dataloaders(train_split=train_split, iteration=iteration)

        self.optimizer = self.optimizer_type(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # self.scheduler ------- TO do

        train_accuracy = 0

        for epoch in tqdm(range(self.epochs)):

            self.model.train()
            epoch_loss = 0
            total = 0

            for batch_imgs, batch_labels in tqdm(train_loader):

                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)

                features = self.model.features(batch_imgs)
                logits = self.model.classifier(features)


                if self.use_ordinal_labels:
                    loss_labels = utils.to_ordinal_labels(batch_labels, self.num_classes)
                else:
                    loss_labels = batch_labels

                if self.use_multilabel:
                    loss_labels = loss_labels.float()

                loss = self.loss_fn(logits, loss_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                epoch_loss += loss.item()
                total += batch_labels.size(0)

            train_acc, train_sens, train_spec, train_f1, train_bin_acc, train_bin_sens, train_bin_spec, train_bin_f1, comb_conf, train_average_auc = self.evaluate(train_loader, epoch, iteration, name="train_")


            if train_bin_acc:
                wandb.log({
                    "train_loss": epoch_loss/total,
                    "fold": iteration,
                    "train_acc":train_acc,
                    "train_sens":train_sens,
                    "train_spec":train_spec,
                    "train_f1":train_f1,
                    "train_bin_acc":train_bin_acc,
                    "train_bin_spec":train_bin_spec,
                    "train_bin_sens":train_bin_sens,
                    "train_bin_f1":train_bin_f1,
                    "train_cm":wandb.Image(comb_conf)}, 
                step=epoch)
            
            else:
                wandb.log({
                    "train_loss": epoch_loss/total,
                    "fold": iteration,
                    "train_acc": train_acc,
                    "train_sens": train_sens,
                    "train_spec": train_spec,
                    "train_f1": train_f1,
                    "train_auc": train_average_auc,
                    "train_cm": wandb.Image(comb_conf)
                }, step=epoch)
            plt.close(comb_conf)

            val_acc, val_sens, val_spec, val_f1, val_bin_acc, val_bin_sens, val_bin_spec, val_bin_f1, val_comb_conf, val_average_auc = self.evaluate(val_loader, epoch, iteration, name="val_")


            if val_bin_acc:
                wandb.log({
                    "val_acc": val_acc,
                    "val_sens": val_sens,
                    "val_spec": val_spec,
                    "val_f1": val_f1,
                    "val_bin_acc": val_bin_acc,
                    "val_bin_sens": val_bin_sens,
                    "val_bin_spec": val_bin_spec,
                    "val_bin_f1": val_bin_f1,
                    "val_cm": wandb.Image(val_comb_conf)
                }, step=epoch)

            else:
                wandb.log({
                    "val_acc": val_acc,
                    "val_sens": val_sens,
                    "val_spec": val_spec,
                    "val_f1": val_f1,
                    "val_auc": val_average_auc,
                    "val_cm": wandb.Image(val_comb_conf)
                }, step=epoch)

            plt.close(comb_conf)
            
            if val_spec > best_val_specificity + min_delta:
                print(f"New best val spec: {val_spec} at epoch {epoch}, previous best: {best_val_specificity}")
                best_val_specificity = val_spec
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), f"{self.checkpoint_save_target}/{epoch}-{val_spec}-{iteration}.pth")
            else:
                epochs_without_improvement += 1
                print(f"No improvement in val spec: {val_spec} at epoch {epoch}, previous best: {best_val_specificity}. Epochs without improvement: {epochs_without_improvement}")

                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered. No improvement for {patience} epochs.")
                    break


        test_acc, test_sens, test_spec, test_f1, test_bin_acc, test_bin_sens, test_bin_spec, test_bin_f1, comb_conf, test_average_auc = self.evaluate(test_loader, epoch, iteration, name="test_")

        if iteration is None:
            torch.save(self.model.state_dict(), f"{self.checkpoint_save_target}/final_model.pth")
        else:
            torch.save(self.model.state_dict(), f"{self.checkpoint_save_target}/final_model_{iteration}.pth")

        return test_acc, test_sens, test_spec, test_f1, test_bin_acc, test_bin_sens, test_bin_spec, test_bin_f1, comb_conf, test_average_auc

    def run_k_iterations(self, k, project_name, exp_name, suffix=''):
        accuracies = []
        sensitivities = []
        specificities = []

        wandb.login()

        for i in range(k):
            wandb.init(
                project=project_name,
                name=exp_name,
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

            acc, sens, spec, f1, bin_acc, bin_sens, bin_spec, bin_f1, comb_conf, average_auc = self.train(iteration=i)
            accuracies.append(acc)
            sensitivities.append(sens)
            specificities.append(spec)

            if bin_acc:
                wandb.log({"test_accuracy": acc,
                        "test_sensitivity": sens,
                        "test_specificity": spec,
                        "test_f1": f1,
                        "test_auc": average_auc,
                        "test_bin_accuracy": bin_acc,
                        "test_bin_sensitivity": bin_sens,
                        "test_bin_specificity": bin_spec,
                        "test_bin_f1": bin_f1,
                        "test_confusion_matrix": wandb.Image(comb_conf)})
            else:
                wandb.log({"test_accuracy": acc,
                        "test_sensitivity": sens,
                        "test_specificity": spec,
                        "test_f1": f1,
                        "test_auc": average_auc,
                        "test_confusion_matrix": wandb.Image(comb_conf)})

            plt.close(comb_conf)
            wandb.finish()

        avg_accuracy = np.mean(accuracies)
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)

        print(f"\nResults over {k} iterations:")
        print(f"Average Test Accuracy: {avg_accuracy:.2f}")
        print(f"Average Sensitivity (True Positive Rate): {avg_sensitivity:.2f}")
        print(f"Average Specificity (True Negative Rate): {avg_specificity:.2f}")

        print("Per run stats (accuracy, sensitity, specificity):")
        print(accuracies)
        print(sensitivities)
        print(specificities)

        return avg_accuracy, avg_sensitivity, avg_specificity