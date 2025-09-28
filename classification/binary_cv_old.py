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

def plot_tb_stratified_binary_cm(all_labels, all_preds, original_labels):
    """
    Create a TB-stratified binary confusion matrix.
    
    Args:
        all_labels: Ground truth profusion labels (0-3)
        all_preds: Predicted profusion labels (0-3)
        original_labels: Original multiclass_stb labels (0-7)
    """
    # Binarize profusion scores (0 = negative, 1-3 = positive)
    binary_true = (all_labels > 0).astype(int)
    binary_pred = (all_preds > 0).astype(int)
    
    # Determine TB status
    tb_status = (original_labels >= 4).astype(int)
    
    # Create empty matrix for counts
    stratified_matrix = np.zeros((4, 2), dtype=int)
    
    # Fill the matrix with counts
    # TP (Profusion+ / Profusion+)
    tp_mask = (binary_true == 1) & (binary_pred == 1)
    stratified_matrix[0, 0] = np.sum(tp_mask & (tb_status == 1))  # TB+
    stratified_matrix[0, 1] = np.sum(tp_mask & (tb_status == 0))  # TB-
    
    # FP (Profusion+ / Profusion-)
    fp_mask = (binary_true == 0) & (binary_pred == 1)
    stratified_matrix[1, 0] = np.sum(fp_mask & (tb_status == 1))  # TB+
    stratified_matrix[1, 1] = np.sum(fp_mask & (tb_status == 0))  # TB-
    
    # FN (Profusion- / Profusion+)
    fn_mask = (binary_true == 1) & (binary_pred == 0)
    stratified_matrix[2, 0] = np.sum(fn_mask & (tb_status == 1))  # TB+
    stratified_matrix[2, 1] = np.sum(fn_mask & (tb_status == 0))  # TB-
    
    # TN (Profusion- / Profusion-)
    tn_mask = (binary_true == 0) & (binary_pred == 0)
    stratified_matrix[3, 0] = np.sum(tn_mask & (tb_status == 1))  # TB+
    stratified_matrix[3, 1] = np.sum(tn_mask & (tb_status == 0))  # TB-
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define labels
    row_labels = [
        "TP (Profusion+ / Profusion+)",
        "FP (Profusion+ / Profusion-)",
        "FN (Profusion- / Profusion+)", 
        "TN (Profusion- / Profusion-)"
    ]
    col_labels = ["TB+", "TB-"]
    
    # Create heatmap
    im = ax.imshow(stratified_matrix, cmap="YlGnBu")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Label the axes
    ax.set_xlabel("TB+ Status")
    ax.set_ylabel("Predicted vs True Profusion")
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add text annotations in each cell
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, stratified_matrix[i, j], 
                   ha="center", va="center", color="black")
    
    plt.title("TB-stratified Custom Confusion Matrix")
    plt.tight_layout()
    
    return fig


def plot_combined_conf_mat(confusion_matrix):
    if confusion_matrix.shape == (2, 2):
        # Binary case
        fig = plt.figure(figsize=(12, 8))
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca(), cmap='Blues')
        plt.title("Confusion Matrix")
    elif len(confusion_matrix.shape) == 3:
        # Multilabel case
        avg_cm = np.mean(confusion_matrix, axis=0)
        norm_cm = avg_cm / (np.sum(avg_cm, axis=1, keepdims=True))
        fig = plt.figure(figsize=(12, 8))
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(norm_cm, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca(), cmap='Blues')
        plt.title("Normalised Multilabel Confusion Matrix")
    else:
        # Multiclass case
        TN, FP, FN, TP = metrics.get_cm_for_class(confusion_matrix, 0)
        bin_conf_matrix = np.array([[TP, FN],
                                    [FP, TN]])
        display_labels = [i for i in range(len(confusion_matrix[0]))]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        cm_display1 = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=display_labels)
        cm_display1.plot(ax=axes[0], colorbar=False, cmap='Blues')
        axes[0].set_title("Confusion Matrix")
        cm_display2 = sklearn.metrics.ConfusionMatrixDisplay(bin_conf_matrix, display_labels=[0, 1])
        cm_display2.plot(ax=axes[1], colorbar=False, cmap='Blues')
        axes[1].set_title("Binary Confusion Matrix")
        plt.tight_layout()
    return fig

class BinaryCrossValidator:
    def __init__(self, model, labels_key, num_classes, device, loss_fn, hdf5_path, batch_size, epochs, 
                 optimizer_type, learning_rate, weight_decay, use_oversampling, checkpoint_save_target, 
                exp_name, split_file, train_set_name, test_set_name,
                 clf_task_labels_key):
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
        self.exp_name = exp_name
        self.num_classes = num_classes
        self.split_file = split_file
        self.preprocess = T.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        # transforms.Grayscale(),
        T.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.train_set_name = train_set_name
        self.test_set_name = test_set_name
        self.clf_task_labels_key = clf_task_labels_key

        self.checkpoint_save_target = f"{self.checkpoint_save_target}/{self.exp_name}"

        if not (os.path.isdir(self.checkpoint_save_target)):
            os.makedirs(self.checkpoint_save_target)


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

        all_original_labels = []

        with torch.no_grad():

            for batch_imgs, batch_labels in tqdm(loader):

                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)

                feats = self.model.features(batch_imgs)

                logits = self.model.classifier(feats)


                loss = self.loss_fn(logits, batch_labels.float().unsqueeze(1))

            
                loc_probs = torch.sigmoid(logits)
                all_probs.append(loc_probs.cpu().numpy())
                # For binary, threshold the sigmoid output
                binary_preds = (loc_probs > 0.5).int()
                all_preds.append(binary_preds.cpu().numpy())

                # print(f"Labels: {all_labels[:10]}\n logits: {logits[:10]}\n probs: {loc_probs[:10]}\n preds: {binary_preds[:10]}")
                
                    
                all_labels.append(batch_labels.cpu().numpy())
                all_feats.append(feats.cpu().numpy())

                    
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        assert len(all_labels) == len(all_probs), "Labels and probabilities must have the same length."

        if self.num_classes==2:
            spec_at_09_sens, thresh = metrics.specificity_at_sensitivity(all_labels, all_probs, min_sens=0.9)
        else:
            raise ValueError("Spec at sens err: Only multiclass and binary currently supported.")
        
        wandb.log({
            f"{name}spec_at_09_sens": spec_at_09_sens,
            f"{name}-threshold": thresh
        }, step=epoch)

        if self.num_classes == 2:
            predictions = (all_probs > thresh)
            predictions_05 = (all_probs > 0.5)
        else:
            raise ValueError("Incorrect thresholding - only multiclass or binary supported.")
    
        global_conf_mat = confusion_matrix(all_labels, predictions)

        global_conf_mat_05 = confusion_matrix(all_labels, predictions_05)


        # HERE IS WHERE WE CAN PLOT TSNE


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



        average_auc = None

        comb_cm = plot_combined_conf_mat(global_conf_mat)

        comb_cm_05 = plot_combined_conf_mat(global_conf_mat_05)

        
        bin_acc, bin_sens, bin_spec, bin_f1, bin_kappa = None, None, None, None, None

        return accuracy, sensitivity, specificity, kappa, f1, bin_acc, bin_sens, bin_spec, bin_kappa, bin_f1, comb_cm, comb_cm_05, average_auc

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
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
            
                loss_labels = batch_labels

    
                loss = self.loss_fn(logits, batch_labels.float().unsqueeze(1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                epoch_loss += loss.item()
                total += batch_labels.size(0)

            train_acc, train_sens, train_spec, train_kappa, train_f1, train_bin_acc, train_bin_sens, train_bin_spec, train_bin_kappa, train_bin_f1, comb_conf, comb_conf_05, train_average_auc = self.evaluate(train_loader, epoch, iteration, name="train_")

            wandb.log({
                "train/loss": epoch_loss/total,
                "fold": iteration,
                "train/acc":train_acc,
                "train/sens":train_sens,
                "train/spec":train_spec,
                "train/f1":train_f1,
                "train/kappa": train_kappa,
                "cm/train":wandb.Image(comb_conf),
                "cm/train_05":wandb.Image(comb_conf_05)}, 
            step=epoch)
            plt.close(comb_conf)

            val_acc, val_sens, val_spec, val_kappa, val_f1, val_bin_acc, val_bin_sens, val_bin_spec, val_bin_kappa, val_bin_f1, val_comb_conf, val_comb_conf_05, val_average_auc = self.evaluate(val_loader, epoch, iteration, name="val_")
            scheduler.step(val_spec)
            
            wandb.log({
                "fold": iteration,
                "val/acc":val_acc,
                "val/sens":val_sens,
                "val/spec":val_spec,
                "val/f1":val_f1,
                "val/kappa": val_kappa,
                "cm/val":wandb.Image(val_comb_conf),
                "cm/val_05":wandb.Image(val_comb_conf_05),
                }, 
            step=epoch)
            plt.close(val_comb_conf)

            if val_spec > best_val_specificity + min_delta:
                print(f"New best val spec: {val_spec} at epoch {epoch}, previous best: {best_val_specificity}")
                best_val_specificity = val_spec
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), f"{self.checkpoint_save_target}/{self.exp_name}-{epoch}-{val_spec}-{iteration:.4f}.pth")
            else:
                epochs_without_improvement += 1
                print(f"No improvement in val spec: {val_spec} at epoch {epoch}, previous best: {best_val_specificity}. Epochs without improvement: {epochs_without_improvement}")

                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered. No improvement for {patience} epochs.")
                    break

        test_acc, test_sens, test_spec, test_kappa, test_f1, test_bin_acc, test_bin_sens, test_bin_spec, test_bin_kappa, test_bin_f1, test_comb_conf, test_comb_conf_05, test_average_auc = self.evaluate(test_loader, epoch, iteration, name="test_")

        if iteration is None:
            torch.save(self.model.state_dict(), f"{self.checkpoint_save_target}/final_model.pth")
        else:
            torch.save(self.model.state_dict(), f"{self.checkpoint_save_target}/final_model_{iteration}.pth")

        return test_acc, test_sens, test_spec, test_kappa, test_f1, test_bin_acc, test_bin_sens, test_bin_spec, test_bin_kappa, test_bin_f1, test_comb_conf, test_comb_conf_05, test_average_auc

    def run_k_iterations(self, k, project_name, exp_name, suffix=''):
        accuracies = []
        sensitivities = []
        specificities = []
        kappas = []

        wandb.login()

        group_name = f"{self.exp_name}-{suffix}-{wandb.util.generate_id()}"
        
       # exp_name = f"{exp_name}-fold_{i}"
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
                    "batch_size":self.batch_size,
                    "epochs": self.epochs,
                    "save_dir": self.checkpoint_save_target,
                    "fold": i
                },
            )

            acc, sens, spec, kappa, f1, bin_acc, bin_sens, bin_spec, bin_kappa, bin_f1, comb_conf, comb_conf_05, average_auc = self.train(iteration=i)

            accuracies.append(acc)
            sensitivities.append(sens)
            specificities.append(spec)
            kappas.append(kappa)


            wandb.log({"test/accuracy": acc,
                    "test/sensitivity": sens,
                    "test/specificity": spec,
                    "test/f1": f1,
                    "test/kappa": kappa,
                    "cm/test_05": wandb.Image(comb_conf_05),
                    "cm/test": wandb.Image(comb_conf),
                    })

            plt.close(comb_conf)
            wandb.finish()

        avg_accuracy = np.mean(accuracies) 
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)
        avg_kappa = np.mean(kappas)

        print(f"\nResults over {k} iterations:")
        print(f"Average Test Accuracy: {avg_accuracy:.2f}")
        print(f"Average Sensitivity (True Positive Rate): {avg_sensitivity:.2f}")
        print(f"Average Specificity (True Negative Rate): {avg_specificity:.2f}")
        print(f"Average Kappa: {avg_kappa:.2f}")

        print("Per run stats (accuracy, sensitity, specificity):")
        print(accuracies)
        print(sensitivities)
        print(specificities)
        print(kappas)

        return avg_accuracy, avg_sensitivity, avg_specificity, avg_kappa