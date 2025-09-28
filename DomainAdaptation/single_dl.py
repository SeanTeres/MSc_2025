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
from pytorch_metric_learning import miners, losses, reducers
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity

sys.path.append(os.path.abspath("../codev2"))
from da_utils import visualize_tsne_with_kaggle_tb, analyze_mined_triplets, reinitialize_weights, ForeverDataIterator
import cl_utils
import cl_tllib_utils, cl_pml_utils
from clf_manager import XRVBasedClassifier
from clf_metrics import compute_binary_clf_metrics

from txrv_wrapper import TxrvWrapper, AdaptivePoolingLayer, init_torchxrayvision_resnet_model



def safe_mean(losses):
    if isinstance(losses, Tensor) and losses.numel() > 0:
        return losses.mean()
    
    return torch.tensor(0.0, device=losses.device if isinstance(losses, torch.Tensor) else device)

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

            train_loader_comb, _, _ = get_dataloaders_with_files(
                hdf5_path=cfg["combined_output"]["hdf5_file"],
                preprocess=preprocess,
                batch_size=4,
                labels_key="multiclass_stb",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_mbod_tbnet_mstb.json",
                augmentations=augmentations_list ,
                oversample=False
            )


            _, val_loader_comb, test_loader_comb = get_dataloaders_with_files(
                hdf5_path=cfg["combined_output"]["hdf5_file"],
                preprocess=preprocess,
                batch_size=4,
                labels_key="multiclass_stb",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_mbod_tbnet_mstb.json",
                augmentations=None,
                oversample=False
            )
            
            if exp_cfg["distance_metric"] == "L2 squared":
                print("Using L2 squared distance metric")
                distance = LpDistance(p=2, power=2)
            elif exp_cfg["distance_metric"] == "Cosine":
                print("Using Cosine distance metric")
                distance = CosineSimilarity()
            elif exp_cfg["distance_metric"] == "L2":
                print("Using L2 distance metric")
                distance = LpDistance(p=2, power=1)
            else:
                raise ValueError(f"Unknown distance metric: {exp_cfg['distance_metric']}")
            

            sample = next(iter(train_loader_comb))
            print(len(sample))

            triplet_miner = miners.TripletMarginMiner(
                margin=0.2,
                distance=distance,
                type_of_triplets="all"
            )
            triplet_loss = losses.TripletMarginLoss(
                margin=0.2,
                distance=distance,
                reducer=reducers.ThresholdReducer(low=0.0, high=1.0)
            )
            
            num_epochs = exp_cfg["num_epochs"]
            

            for idx, sample in enumerate(train_loader_comb):
                imgs = sample[0].to(device)
                labels = sample[1]
                labels = (labels >= 4).long()  # Convert to binary labels (0 or 1)
                is_mbod = sample[2]

                feats = model.features(imgs)
                
                logits = model.tb_clf(feats)
                tb_loss = clf_loss_fn(logits, labels.unsqueeze(1).float().to(device))

                feats = F.normalize(feats, dim=1)

                anc_idx, pos_idx, neg_idx = triplet_miner(feats.detach().cpu(), labels, feats.detach().cpu(), labels)

                anc_feats = feats[anc_idx].to(device)
                pos_feats = feats[pos_idx].to(device)
                neg_feats = feats[neg_idx].to(device)

                loss = triplet_loss.compute_loss(feats, labels, (anc_idx, pos_idx, neg_idx), feats, labels)

                total_loss = safe_mean(loss) + tb_loss

                print(f"CL loss: {loss}, \n\nTB loss: {tb_loss.item()}")


                break

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