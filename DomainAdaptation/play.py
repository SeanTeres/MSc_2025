import sys
import os
import gc
# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datasets.dataloader import get_dataloaders, get_dataloaders_with_files
import torchxrayvision as xrv
import torch
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

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity

sys.path.append(os.path.abspath("../codev2"))
from da_utils import visualize_tsne_with_kaggle_tb, analyze_mined_triplets, reinitialize_weights, ForeverDataIterator
import cl_utils
import cl_tllib_utils, cl_pml_utils
from clf_manager import XRVBasedClassifier
from clf_metrics import compute_binary_clf_metrics

from txrv_wrapper import TxrvWrapper, AdaptivePoolingLayer, init_torchxrayvision_resnet_model

from losses import calculate_jmmd_loss, JointMultipleKernelMaximumMeanDiscrepancy, GaussianKernel, CorrelationAlignmentLoss, MultipleKernelMaximumMeanDiscrepancy
def safe_mean(losses):
    if isinstance(losses, Tensor) and losses.numel() > 0:
        return losses.mean()
    
    return torch.tensor(0.0, device=losses.device if isinstance(losses, torch.Tensor) else device)

def plot_confusion_matrices(src_cm, tgt_cm, epoch):
    """Create a side-by-side plot of confusion matrices for both domains"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Source domain confusion matrix
    sns.heatmap(src_cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('TB-NET (Source)')
    ax1.set_ylabel('True')
    ax1.set_xlabel('Predicted')
    
    # Target domain confusion matrix
    sns.heatmap(tgt_cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_title('MBOD (Target)')
    ax2.set_ylabel('True')
    ax2.set_xlabel('Predicted')
    
    plt.suptitle(f'Confusion Matrices - Epoch {epoch}')
    plt.tight_layout()
    
    # Convert plot to wandb Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = wandb.Image(buf)
    plt.close()
    
    return image

if __name__ == "__main__":
    device = torch.device("cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    defaults = load_config("defaults.yaml")
    wandb_api_key = defaults["WANDB_API_KEY"]["value"]
    random_seed = defaults["RANDOM_SEED"]["value"]

    augmentations_list = transforms.Compose([
    transforms.RandomRotation(degrees=10, expand=False, fill=0),
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
    ])


    preprocess = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    cl_utils.set_random_seeds(seed=random_seed)  # Set a fixed seed for reproducibility

    try:
        cfg = load_config(defaults["DATA_CONFIG"]["path"])

        experiments_config = load_config('clda_pml.yaml')
        experiments = experiments_config["experiments"]

        for exp_cfg in experiments:
            
            exp_name = exp_cfg["name"]
            print(f"Running experiment: {exp_name}")

            if(exp_cfg.get("resolution", 999) == 512):
                print("Using model at 512x512")

                if exp_cfg.get("pretrained", False):
                    print("Using pretrained model")
                    model = xrv.models.ResNet(weights="resnet50-res512-all")
                else:
                    print("Using randomly initialized model")
                    model, _ = init_torchxrayvision_resnet_model(num_classes=1, randomly_initialise=True)

                    model = xrv.models.ResNet(weights="resnet50-res512-all")
                    reinitialize_weights(model)

            elif exp_cfg.get("resolution", 999) == 224:
                print("Using model at 224x224")
                model = xrv.models.DenseNet(weights="densenet121-res224-all")
            else:
                raise ValueError("Only 512x512 currently supported. Please set resolution to 512 in the config file.")
            
            model.tb_clf = XRVBasedClassifier(input_dim=2048, num_classes=1, name="xrv_bin_tb").to(device)
            clf_loss_fn = nn.BCEWithLogitsLoss()

            if(exp_cfg["loss_components"]["bin_tb_clf"]):
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(model.tb_clf.parameters()),
                    lr=exp_cfg["learning_rate"],
                    weight_decay=exp_cfg["learning_rate"]/10)
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=exp_cfg["learning_rate"],
                    weight_decay=exp_cfg["learning_rate"]/10)


            model = model.to(device)

            pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all").to(device)
            rand_model = xrv.models.ResNet(weights="resnet50-res512-all").to(device)
            reinitialize_weights(rand_model)
            josh_model, _ = init_torchxrayvision_resnet_model(num_classes=1, randomly_initialise=True)



            # Create an HDF5SilicosisDataset instance
            mbod_dataset_merged = HDF5Dataset(
                hdf5_path=cfg["merged_silicosis_output"]["hdf5_file"],
                labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
                images_key="images",
                augmentations=None,
                preprocess=preprocess
            )


            ilo_dataset = HDF5Dataset(
                hdf5_path=cfg["ilo_output"]["hdf5_file"],
                labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
                images_key="images",
                augmentations=None,
                preprocess=preprocess
            )

            kaggle_tb_dataset = HDF5Dataset(
                hdf5_path=cfg["kaggle_TB"]["outputpath"],
                labels_key="tuberculosis",  # Main pathology labels, 'lab' for all labels
                images_key="images",
                augmentations=None,
                preprocess=preprocess
            )

            combined_dataset_test = HDF5Dataset2(
                hdf5_path=cfg["combined_output"]["hdf5_file"],
                labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
                images_key="images",
                augmentations=None,
                preprocess=preprocess
            )

            train_loader_mbod, _, _ = get_dataloaders(
                hdf5_path=cfg["merged_silicosis_output"]["hdf5_file"],
                preprocess=preprocess,
                batch_size=exp_cfg["tgt_batch_size"],
                labels_key="multiclass_stb",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_MBOD_mlabel_stb.json",
                augmentations=augmentations_list ,
                oversample=False
            )

            train_loader_tbnet, _, _ = get_dataloaders(
                hdf5_path=cfg["kaggle_TB"]["outputpath"],
                preprocess=preprocess,
                batch_size=exp_cfg["src_batch_size"],
                labels_key="tuberculosis",
                split_file="stratified_split_tb_net.json",
                augmentations=augmentations_list,
                oversample=False
            )

            _, val_loader_mbod, test_loader_mbod = get_dataloaders(
                hdf5_path=cfg["merged_silicosis_output"]["hdf5_file"],
                preprocess=preprocess,
                batch_size=exp_cfg["tgt_batch_size"],
                labels_key="multiclass_stb",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_MBOD_mlabel_stb.json",
                augmentations=None,
                oversample=False
            )
            _, val_loader_tbnet, test_loader_tbnet = get_dataloaders(
                hdf5_path=cfg["kaggle_TB"]["outputpath"],
                preprocess=preprocess,
                batch_size=exp_cfg["src_batch_size"],
                labels_key="tuberculosis",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_tb_net.json",
                augmentations=None,
                oversample=False
                )
            def compare_model_weights(model1, model2, name1="Pretrained", name2="Random"):
                """Compare weights between two models to verify initialization differences"""
                
                # Compare first conv layer weights as a sample
                conv1_diff = torch.sum(torch.abs(model1.model.conv1.weight - model2.model.conv1.weight)).item()
                
                # Compare a random layer from the last block
                layer4_diff = torch.sum(torch.abs(model1.model.layer4[0].conv1.weight - model2.model.layer4[0].conv1.weight)).item()
                
                print(f"\nWeight Difference Summary:")
                print(f"First Conv Layer absolute difference: {conv1_diff:.2f}")
                print(f"Layer4 Conv1 absolute difference: {layer4_diff:.2f}")
                
                # Quick sanity check - are they exactly the same?
                are_identical = all(
                    torch.all(p1 == p2).item() 
                    for p1, p2 in zip(model1.parameters(), model2.parameters())
                )
                
                print(f"\nModels are{' ' if are_identical else ' not '}identical")

            # After initializing your models:
            compare_model_weights(pretrained_model, rand_model)
            compare_model_weights(pretrained_model, josh_model, name1="Pretrained", name2="Josh's Random")
            compare_model_weights(rand_model, josh_model, name1="Random", name2="Josh's Random")


    except KeyError as e:
        print(f"Missing configuration: {e}")


# -------- NOTES ---------
# Original Paper used JMMD weight as 1.0 and Triplet loss weight as 1.0
# They used batch size of 32 for both source and target
# They used INV learning rate strategy
# SGD for optimization
# FIRST, train extractor and classifier using JMMD and Cross entropy for 6000 iterations
# Then, train entire system (including Triplet Loss) for 30000 iterations
#  