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

from losses import CorrelationAlignmentLoss, JointMultipleKernelMaximumMeanDiscrepancy, GaussianKernel, MultipleKernelMaximumMeanDiscrepancy

def calculate_jmmd_loss_fixed(encoder, src_imgs, tgt_imgs, jmmd_loss_fn, layers=["layer1"], rescale_loss=True, return_final_features=False):
    """
    Calculate JMMD loss using features from multiple layers.
    Args:
        layers: list of layer names to use for JMMD computation (e.g., ["layer1", "layer2"])
    """
    if not layers:
        raise ValueError("No layers specified for JMMD loss calculation.")

    src_features = []
    tgt_features = []
    
    # Forward pass through base layers (shared computation graph)
    src_x = encoder.model.conv1(src_imgs)
    src_x = encoder.model.bn1(src_x)
    src_x = encoder.model.relu(src_x)
    src_x = encoder.model.maxpool(src_x)
    
    tgt_x = encoder.model.conv1(tgt_imgs)
    tgt_x = encoder.model.bn1(tgt_x)
    tgt_x = encoder.model.relu(tgt_x)
    tgt_x = encoder.model.maxpool(tgt_x)
    
    # Layer 1
    src_x = encoder.model.layer1(src_x)
    tgt_x = encoder.model.layer1(tgt_x)
    
    if "layer1" in layers:
        src_feat1 = torch.flatten(encoder.model.avgpool(src_x), 1)
        tgt_feat1 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
        src_features.append(src_feat1)
        tgt_features.append(tgt_feat1)
    
    # Layer 2 (if requested)
    if "layer2" in layers or "layer3" in layers or return_final_features:
        src_x = encoder.model.layer2(src_x)
        tgt_x = encoder.model.layer2(tgt_x)
        
        if "layer2" in layers:
            src_feat2 = torch.flatten(encoder.model.avgpool(src_x), 1)
            tgt_feat2 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
            src_features.append(src_feat2)
            tgt_features.append(tgt_feat2)
    
    # Layer 3 (if requested)
    if "layer3" in layers or return_final_features:
        src_x = encoder.model.layer3(src_x)
        tgt_x = encoder.model.layer3(tgt_x)
        
        if "layer3" in layers:
            src_feat3 = torch.flatten(encoder.model.avgpool(src_x), 1)
            tgt_feat3 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
            src_features.append(src_feat3)
            tgt_features.append(tgt_feat3)
    
    # Final layer 4 for complete features (if requested)
    if return_final_features:
        src_x = encoder.model.layer4(src_x)
        tgt_x = encoder.model.layer4(tgt_x)
        
        # Get final features (equivalent to encoder.features() but reusing computation)
        src_final = torch.flatten(encoder.model.avgpool(src_x), 1)
        tgt_final = torch.flatten(encoder.model.avgpool(tgt_x), 1)

    # Compute JMMD loss with proper gradient flow
    jmmd_loss = jmmd_loss_fn(tuple(src_features), tuple(tgt_features))

    if rescale_loss:
        # Count total number of kernels across all layers
        total_kernels = sum(len(kernel_set) for kernel_set in jmmd_loss_fn.kernels)
        num_layers = len(src_features)
        jmmd_loss = jmmd_loss / (total_kernels * num_layers)

    if return_final_features:
        return jmmd_loss, src_final, tgt_final
    else:
        return jmmd_loss

def reset_experiment_state():
    """Helper function to reset all experiment state"""
    gc.collect()
    torch.cuda.empty_cache()


def safe_mean(losses):
    if isinstance(losses, Tensor) and losses.numel() > 0:
        return losses.mean()
    
    return torch.tensor(0.0, device=losses.device if isinstance(losses, torch.Tensor) else device)

def plot_confusion_matrices(src_cm, tgt_cm, epoch):
    """Create a side-by-side plot of confusion matrices for both domains"""
    from PIL import Image  # Add this import at the top of the file
    
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
    
    # Convert plot to PIL Image then to wandb Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_image = Image.open(buf)  # Convert to PIL Image
    image = wandb.Image(pil_image)  # Pass PIL Image to wandb.Image
    plt.close()
    buf.close()
    
    return image

def compute_feature_distance_stats(src_feats: torch.Tensor, tgt_feats: torch.Tensor):
    """Compute statistics about feature distances between domains"""
    with torch.no_grad():
        # L2 distances
        src_tgt_l2 = torch.cdist(src_feats, tgt_feats, p=2.0)  # Cross-domain distances
        src_src_l2 = torch.cdist(src_feats, src_feats, p=2.0)  # Within-source distances
        tgt_tgt_l2 = torch.cdist(tgt_feats, tgt_feats, p=2.0)  # Within-target distances
        
        # Remove self-distances (diagonal) for within-domain
        src_src_l2 = src_src_l2[~torch.eye(src_src_l2.shape[0], dtype=bool)]
        tgt_tgt_l2 = tgt_tgt_l2[~torch.eye(tgt_tgt_l2.shape[0], dtype=bool)]
        
        stats = {
            "cross_domain_dist/mean": src_tgt_l2.mean().item(),
            "cross_domain_dist/std": src_tgt_l2.std().item(),
            "cross_domain_dist/median": src_tgt_l2.median().item(),
            "cross_domain_dist/max": src_tgt_l2.max().item(),
            "cross_domain_dist/min": src_tgt_l2.min().item(),
            
            "source_within_dist/mean": src_src_l2.mean().item(),
            "source_within_dist/std": src_src_l2.std().item(),
            "source_within_dist/median": src_src_l2.median().item(),
            
            "target_within_dist/mean": tgt_tgt_l2.mean().item(),
            "target_within_dist/std": tgt_tgt_l2.std().item(),
            "target_within_dist/median": tgt_tgt_l2.median().item(),
        }
        
        # Compute suggested kernel bandwidths based on distance statistics
        median_dist = src_tgt_l2.median().item()
        stats.update({
            "kernel_bandwidths/cons": median_dist / 2,  # Conservative
            "kernel_bandwidths/med": median_dist,      # Medium
            "kernel_bandwidths/agg": median_dist * 2   # Aggressive
        })
        
        return stats

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

            model = None
            optimizer = None

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
            

            tgt_triplet_miner = miners.TripletMarginMiner(
                margin = exp_cfg["mining"]["tgt_margin"],
                type_of_triplets=exp_cfg["mining"]["type_of_triplets"],
                distance=distance
            )
            src_triplet_miner = miners.TripletMarginMiner(
                margin = exp_cfg["mining"]["src_margin"],
                type_of_triplets=exp_cfg["mining"]["type_of_triplets"],
                distance=distance
            )

            st_triplet_miner = miners.TripletMarginMiner(
                margin = exp_cfg["mining"]["tgt_margin"],
                type_of_triplets=exp_cfg["mining"]["type_of_triplets"],
                distance=distance
            )

            triplet_loss_tgt = losses.TripletMarginLoss(
                margin=exp_cfg["mining"]["tgt_margin"],
                distance=distance
            )
            triplet_loss_src = losses.TripletMarginLoss(         # TO DO: Consider the reducers that you can use and see if manual is the same
                margin=exp_cfg["mining"]["src_margin"],
                distance=distance
            ) 

            coral_fn = CorrelationAlignmentLoss()


            # mmd_kernels = (GaussianKernel(alpha=0.5).to(device), GaussianKernel(alpha=1.0).to(device), GaussianKernel(alpha=2.0).to(device))
            # mmd_loss_fn = MultipleKernelMaximumMeanDiscrepancy(mmd_kernels).to(device)

            kernels_per_layer = []
            for _ in exp_cfg.get("jmmd_config", {}).get("layers", ["layer1"]):
                layer_kernels = tuple(GaussianKernel(alpha=alpha).to(device) 
                                    for alpha in [0.25, 0.5, 1.0, 2.0, 3.0])
                kernels_per_layer.append(layer_kernels)

            jmmd_loss_fn = JointMultipleKernelMaximumMeanDiscrepancy(
                kernels=tuple(kernels_per_layer)
            ).to(device)

            # layer1_kernels = tuple(GaussianKernel(alpha=alpha).to(device) for alpha in [0.25, 0.5, 1.0, 2.0, 3.0])
            # jmmd_loss_fn = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, )).to(device)

            mmd_loss = MultipleKernelMaximumMeanDiscrepancy

            source_iter = ForeverDataIterator(train_loader_tbnet)
            target_iter = ForeverDataIterator(train_loader_mbod)

            steps_per_epoch = max(len(train_loader_tbnet), len(train_loader_mbod))  # e.g., 840 with b4
            num_epochs = exp_cfg["num_epochs"]
            total_steps = steps_per_epoch * num_epochs

            print(f"Total steps: {total_steps}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Number of epochs: {num_epochs}")

            print(f"Distance metric: {exp_cfg['distance_metric']}")

            wandb.init(
                project="cl-mstb-DA2",
                name=exp_name,
                config=exp_cfg,
            )


            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}")

                if(epoch == 1):
                    visualize_tsne_with_kaggle_tb(model, device, exp_cfg["name"], ilo_dataset, mbod_loader=train_loader_mbod, tb_loader=train_loader_tbnet,
                            trained=False, log_to_wandb=True, set_name="Pre-training",n_epochs=epoch)
                    plt.close("all")

                model.train()
                all_src_labels = []
                all_tgt_labels = []
                all_src_preds = []
                all_tgt_preds = []

                total_loss = 0.0
                total_src_triplet_loss = 0.0
                total_tgt_triplet_loss = 0.0
                total_st_triplet_loss = 0.0
                total_ts_triplet_loss = 0.0
                total_coral_loss = 0.0
                total_jmmd_loss = 0.0

                train_src_clf_loss = 0.0
                train_tgt_clf_loss = 0.0

                valid_steps = 0
                invalid_steps = 0

                num_src_triplets, num_tgt_triplets, num_st_triplets, num_ts_triplets = 0, 0, 0, 0

                for step in range(steps_per_epoch):
                    src_batch = next(source_iter)
                    tgt_batch = next(target_iter)

                    src_imgs, src_labels = src_batch[0].to(device), src_batch[1].to(device)
                    tgt_imgs, tgt_labels = tgt_batch[0].to(device), tgt_batch[1].to(device)

                    # ==== FIXED JMMD COMPUTATION WITH PROPER GRADIENT FLOW ====
                    # This replaces the problematic double computation
                    if src_imgs.size(0) != tgt_imgs.size(0):
                        min_batch = min(src_imgs.size(0), tgt_imgs.size(0))
                        # Compute JMMD AND get final features in one pass to avoid double computation
                        jmmd_loss, src_feats, tgt_feats = calculate_jmmd_loss_fixed(
                            model, src_imgs[:min_batch], tgt_imgs[:min_batch], 
                            jmmd_loss_fn, layers=exp_cfg.get("jmmd_config", {}).get("layers", ["layer1"]),  # Use config layers
                            rescale_loss=True, return_final_features=True
                        )
                        # Use the remaining images for feature extraction
                        if src_imgs.size(0) > min_batch:
                            remaining_src_feats = model.features(src_imgs[min_batch:])
                            src_feats = torch.cat([src_feats, remaining_src_feats], dim=0)
                        if tgt_imgs.size(0) > min_batch:
                            remaining_tgt_feats = model.features(tgt_imgs[min_batch:])
                            tgt_feats = torch.cat([tgt_feats, remaining_tgt_feats], dim=0)
                    else:
                        # Equal batch sizes - compute everything in one pass
                        jmmd_loss, src_feats, tgt_feats = calculate_jmmd_loss_fixed(
                            model, src_imgs, tgt_imgs, jmmd_loss_fn, 
                            layers=exp_cfg.get("jmmd_config", {}).get("layers", ["layer1"]),  # Use config layers
                            rescale_loss=True, return_final_features=True
                        )

                    coral_loss = coral_fn(src_feats, tgt_feats)

                    if step % 50 == 0:  # Compute every 50 steps to reduce overhead -> Do before normalization
                        dist_stats = compute_feature_distance_stats(src_feats, tgt_feats)
                        wandb.log(dist_stats)                
                    
                    if (exp_cfg["mining"]["tgt_labels"] == "TB-based"):
                        tgt_labels = (tgt_labels >= 4).long()

                        if exp_cfg["loss_components"].get("bin_tb_clf", False):
                            src_logits = model.tb_clf(src_feats)
                            tgt_logits = model.tb_clf(tgt_feats)

                            src_labels_flt = src_labels.unsqueeze(1).float()  # Ensure src_labels is a float tensor
                            tgt_labels_flt = tgt_labels.unsqueeze(1).float()  # Ensure tgt_labels is a float tensor

                            src_pred = (torch.sigmoid(src_logits) > 0.5).long().cpu().numpy()
                            tgt_pred = (torch.sigmoid(tgt_logits) > 0.5).long().cpu().numpy()
                            all_src_labels.append(src_labels_flt.cpu().numpy())
                            all_src_preds.append(src_logits.detach().cpu().numpy())
                            all_tgt_labels.append(tgt_labels_flt.cpu().numpy())
                            all_tgt_preds.append(tgt_logits.detach().cpu().numpy())

                            src_clf_loss = clf_loss_fn(src_logits, src_labels.unsqueeze(1).float()) * exp_cfg["component_weights"]["src_clf_weight"]
                            tgt_clf_loss = clf_loss_fn(tgt_logits, tgt_labels.unsqueeze(1).float()) * exp_cfg["component_weights"]["tgt_clf_weight"]
                            train_src_clf_loss += src_clf_loss.item()
                            train_tgt_clf_loss += tgt_clf_loss.item()


                            # src_clf_cm, src_clf_metrics = compute_binary_clf_metrics(src_labels_flt, src_logits.detach(), domain_name="batch_clf/TB-NET", log_to_wandb=True)
                            # tgt_clf_cm, tgt_clf_metrics = compute_binary_clf_metrics(tgt_labels_flt, tgt_logits.detach(), domain_name="batch_clf/MBOD", log_to_wandb=True)

                    else:
                        raise ValueError(f"Unknown target label structure: {exp_cfg['mining']['tgt_labels']}")
                    


                    if (exp_cfg["normalize_embeddings"] or (exp_cfg["distance_metric"] == "Cosine")):
                        src_feats = F.normalize(src_feats, p=2, dim=1)
                        tgt_feats = F.normalize(tgt_feats, p=2, dim=1)

                    
                    all_embeddings = torch.cat([src_feats, tgt_feats], dim=0)
                    all_labels = torch.cat([src_labels, tgt_labels], dim=0)

                    src_anc_idx, src_pos_idx, src_neg_idx = src_triplet_miner(src_feats, src_labels, ref_emb=src_feats, ref_labels=src_labels)
                    tgt_anc_idx, tgt_pos_idx, tgt_neg_idx = tgt_triplet_miner(tgt_feats, tgt_labels, ref_emb=tgt_feats, ref_labels=tgt_labels)

                    ts_anc_idx, ts_pos_idx, ts_neg_idx = st_triplet_miner(src_feats, src_labels, ref_emb=tgt_feats, ref_labels=tgt_labels)
                    st_anc_idx, st_pos_idx, st_neg_idx = st_triplet_miner(tgt_feats, tgt_labels, ref_emb=src_feats, ref_labels=src_labels)

                    src_loss_dict = triplet_loss_src.compute_loss(
                        src_feats, src_labels, (src_anc_idx, src_pos_idx, src_neg_idx), ref_emb=src_feats, ref_labels=src_labels
                    )
                    tgt_loss_dict = triplet_loss_tgt.compute_loss(
                        tgt_feats, tgt_labels, (tgt_anc_idx, tgt_pos_idx, tgt_neg_idx), ref_emb=tgt_feats, ref_labels=tgt_labels
                    )

                    ts_loss_dict = triplet_loss_tgt.compute_loss(
                        src_feats, src_labels, (ts_anc_idx, ts_pos_idx, ts_neg_idx), ref_emb=tgt_feats, ref_labels=tgt_labels
                    )
                    st_loss_dict = triplet_loss_src.compute_loss(
                        tgt_feats, tgt_labels, (st_anc_idx, st_pos_idx, st_neg_idx), ref_emb=src_feats, ref_labels=src_labels
                    )


                    src_triplet_loss = safe_mean(src_loss_dict["loss"]["losses"])
                    tgt_triplet_loss = safe_mean(tgt_loss_dict["loss"]["losses"])

                    ts_triplet_loss = safe_mean(ts_loss_dict["loss"]["losses"])
                    st_triplet_loss = safe_mean(st_loss_dict["loss"]["losses"])

                    src_triplet_loss_val = safe_mean(src_loss_dict["loss"]["losses"]) * exp_cfg["component_weights"]["src_triplet_weight"]
                    tgt_triplet_loss_val = safe_mean(tgt_loss_dict["loss"]["losses"]) * exp_cfg["component_weights"]["tgt_triplet_weight"]
                    
                    # FIXED: Use proper cross_domain_weight instead of inconsistent weights
                    cross_domain_weight = exp_cfg["component_weights"].get("cross_domain_weight", exp_cfg["component_weights"]["tgt_triplet_weight"])
                    ts_triplet_loss_val = safe_mean(ts_loss_dict["loss"]["losses"]) * cross_domain_weight
                    st_triplet_loss_val = safe_mean(st_loss_dict["loss"]["losses"]) * cross_domain_weight

                    coral_loss_val = coral_loss * exp_cfg["component_weights"]["coral_weight"]
                    jmmd_loss_val = jmmd_loss * exp_cfg["component_weights"]["jmmd_weight"]

                    # accumulating UNWEIGHTED losses
                    total_src_triplet_loss += src_triplet_loss.item()
                    total_tgt_triplet_loss += tgt_triplet_loss.item()
                    total_st_triplet_loss += st_triplet_loss.item()
                    total_ts_triplet_loss += ts_triplet_loss.item()
                    total_coral_loss += coral_loss.item()
                    total_jmmd_loss += jmmd_loss.item()

                    num_src_triplets += len(src_anc_idx)
                    num_tgt_triplets += len(tgt_anc_idx)
                    num_st_triplets += len(st_anc_idx)
                    num_ts_triplets += len(ts_anc_idx)

                    # print(f"Source triplet loss: {src_triplet_loss.item()}, Target triplet loss: {tgt_triplet_loss.item()}")
                    # print(f"Source triplet loss (weighted): {src_triplet_loss_val.item()}, Target triplet loss (weighted): {tgt_triplet_loss_val.item()}")


                    # Add to loss list - ABLATION FRAMEWORK
                    loss_list = []

                    # Source triplet loss
                    if exp_cfg["loss_components"].get("source_triplet", False):
                        loss_list.append(src_triplet_loss_val)

                    # Target triplet loss
                    if exp_cfg["loss_components"].get("target_triplet", False):
                        loss_list.append(tgt_triplet_loss_val)

                    # Cross-domain triplet losses (st and ts)
                    if exp_cfg["loss_components"].get("cross_domain_triplet", False):
                        loss_list.append(st_triplet_loss_val)
                        loss_list.append(ts_triplet_loss_val)
                    
                    # Coral loss
                    if exp_cfg["loss_components"].get("coral", False):
                        loss_list.append(coral_loss_val)

                    # JMMD loss
                    if exp_cfg["loss_components"].get("jmmd", False):
                        loss_list.append(jmmd_loss_val)

                    # Binary classifier losses
                    if exp_cfg["loss_components"].get("bin_tb_clf", False):
                        if exp_cfg["component_weights"]["src_clf_weight"] > 0:
                            loss_list.append(src_clf_loss)
                        if exp_cfg["component_weights"]["tgt_clf_weight"] > 0:
                            loss_list.append(tgt_clf_loss)

                    # Filter valid losses and optimize
                    valid_losses = [l for l in loss_list if l.requires_grad and not torch.isnan(l).any()]
                    if valid_losses:
                        total_loss = sum(valid_losses)
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        valid_steps += 1
                    else:
                        invalid_steps += 1
                        print(f"Step {step}: No valid losses to optimize! Loss components active: {[k for k, v in exp_cfg['loss_components'].items() if v]}")

                    wandb.log({
                        "batch/total_loss": total_loss.item() if valid_losses else 0.0,
                        "batch/coral_loss": coral_loss.item(),
                        "batch/src_triplet_loss": src_triplet_loss.item(),
                        "batch/tgt_triplet_loss": tgt_triplet_loss.item(),
                        "batch/st_triplet_loss": st_triplet_loss.item(),
                        "batch/ts_triplet_loss": ts_triplet_loss.item(),
                        "batch/src_clf_loss": src_clf_loss.item() if exp_cfg["loss_components"]["bin_tb_clf"] else 0.0,
                        "batch/tgt_clf_loss": tgt_clf_loss.item() if exp_cfg["loss_components"]["bin_tb_clf"] else 0.0,
                        "batch/valid_steps": valid_steps,
                        "batch/invalid_steps": invalid_steps,
                        "batch/jmmd_loss": jmmd_loss.item() if exp_cfg["loss_components"]["jmmd"] else 0.0,
                    })
            
                cl_metrics = cl_pml_utils.compute_comprehensive_metrics(model, device, train_loader_tbnet, train_loader_mbod, epoch=epoch+1)     # TO DO: double-check this calculation

                all_src_labels = np.concatenate(all_src_labels, axis=0)
                all_tgt_labels = np.concatenate(all_tgt_labels, axis=0)
                all_src_preds = np.concatenate(all_src_preds, axis=0)
                all_tgt_preds = np.concatenate(all_tgt_preds, axis=0)

                train_cm_src, train_src_metrics = compute_binary_clf_metrics(torch.tensor(all_src_labels), torch.tensor(all_src_preds), domain_name="train_clf/TB-NET", log_to_wandb=False)
                train_cm_tgt, train_tgt_metrics = compute_binary_clf_metrics(torch.tensor(all_tgt_labels), torch.tensor(all_tgt_preds), domain_name="train_clf/MBOD", log_to_wandb=False)


                # Log confusion matrices
                if((epoch + 1) % exp_cfg["tsne_interval"] == 0):
                    train_cms_image = plot_confusion_matrices(train_cm_src, train_cm_tgt, epoch + 1)
                    wandb.log({"cm/train_confusion_matrices": train_cms_image})

                wandb.log({
                    "epoch": epoch + 1,
                    "train/st_triplet_loss": total_st_triplet_loss,
                    "train/ts_triplet_loss": total_ts_triplet_loss,
                    "train/src_triplet_loss": total_src_triplet_loss,
                    "train/tgt_triplet_loss": total_tgt_triplet_loss,
                    "train/coral_loss": total_coral_loss,
                    "train/jmmd_loss": total_jmmd_loss,
                    "train/num_src_triplets": num_src_triplets,
                    "train/num_tgt_triplets": num_tgt_triplets,
                    "train/num_st_triplets": num_st_triplets,
                    "train/num_ts_triplets": num_ts_triplets,
                    "valid_steps": valid_steps,
                    "invalid_steps": invalid_steps,
                    "train/src_intra_similarity": cl_metrics["src_intra_similarity"],
                    "train/tgt_intra_similarity": cl_metrics["tgt_intra_similarity"],
                    "train/inter_domain_similarity": cl_metrics["inter_domain_similarity"],
                    "train/src_map": cl_metrics["src_map"],
                    "train/tgt_map": cl_metrics["tgt_map"],


                    "train_clf/src_clf_loss": train_src_clf_loss,
                    "train_clf/tgt_clf_loss": train_tgt_clf_loss,
                    "train_clf/src_specificity": train_src_metrics["specificity"],
                    "train_clf/src_sensitivity": train_src_metrics["sensitivity"],
                    "train_clf/src_f1_score": train_src_metrics["f1_score"],
                    "train_clf/src_accuracy": train_src_metrics["accuracy"],
                    "train_clf/src_specificity_at_90%_sensitivity": train_src_metrics["specificity_at_90%_sensitivity"],
                    "train_clf/src_cohen_kappa": train_src_metrics["cohen_kappa"],
                    "train_clf/tgt_specificity": train_tgt_metrics["specificity"],
                    "train_clf/tgt_sensitivity": train_tgt_metrics["sensitivity"],
                    "train_clf/tgt_f1_score": train_tgt_metrics["f1_score"],
                    "train_clf/tgt_accuracy": train_tgt_metrics["accuracy"],
                    "train_clf/tgt_specificity_at_90%_sensitivity": train_tgt_metrics["specificity_at_90%_sensitivity"],
                    "train_clf/tgt_cohen_kappa": train_tgt_metrics["cohen_kappa"],
                    "train_clf/tgt_tpr": train_tgt_metrics["tpr"],
                    "train_clf/tgt_fpr": train_tgt_metrics["fpr"],
                    "train_clf/src_tpr": train_src_metrics["tpr"],
                    "train_clf/src_fpr": train_src_metrics["fpr"],
                })

                if (epoch + 1) % exp_cfg["tsne_interval"] == 0:
                    visualize_tsne_with_kaggle_tb(model, device, exp_cfg["name"], ilo_dataset, mbod_loader=train_loader_mbod, tb_loader=train_loader_tbnet,
                                                trained=True, log_to_wandb=True, set_name="Training",n_epochs=epoch + 1)
                    visualize_tsne_with_kaggle_tb(model, device, exp_cfg["name"], ilo_dataset, mbod_loader=val_loader_mbod, tb_loader=val_loader_tbnet,
                                                trained=True, log_to_wandb=True, set_name="Validation",n_epochs=epoch + 1)
                    
                    plt.close('all')

                
    
                model.eval()
                with torch.no_grad():
                    val_cl_metrics = cl_pml_utils.compute_comprehensive_metrics(model, device, val_loader_tbnet, val_loader_mbod, epoch=epoch+1)
                    val_steps_per_epoch = max(len(val_loader_tbnet), len(val_loader_mbod))

                    val_src_iter = ForeverDataIterator(val_loader_tbnet)
                    val_tgt_iter = ForeverDataIterator(val_loader_mbod)

                    val_all_src_labels = []
                    val_all_src_preds = []
                    val_all_tgt_labels = []
                    val_all_tgt_preds = []

                    for step in range(val_steps_per_epoch):
                        val_src_batch = next(val_src_iter)
                        val_tgt_batch = next(val_tgt_iter)

                        val_src_imgs, val_src_labels = val_src_batch[0].to(device), val_src_batch[1].to(device)
                        val_tgt_imgs, val_tgt_labels = val_tgt_batch[0].to(device), val_tgt_batch[1].to(device)

                        val_src_feats = model.features(val_src_imgs)
                        val_tgt_feats = model.features(val_tgt_imgs)

                        val_src_logits = model.tb_clf(val_src_feats)
                        val_tgt_logits = model.tb_clf(val_tgt_feats)

                        val_tgt_labels = (val_tgt_labels >= 4).long()

                        # Accumulate for epoch-level metrics
                        val_src_labels_flt = val_src_labels.unsqueeze(1).float()
                        val_tgt_labels_flt = val_tgt_labels.unsqueeze(1).float()
                        val_src_pred = (torch.sigmoid(val_src_logits) > 0.5).long().cpu().numpy()
                        val_tgt_pred = (torch.sigmoid(val_tgt_logits) > 0.5).long().cpu().numpy()
                        val_all_src_labels.append(val_src_labels_flt.cpu().numpy())
                        val_all_src_preds.append(val_src_logits.detach().cpu().numpy())
                        val_all_tgt_labels.append(val_tgt_labels_flt.cpu().numpy())
                        val_all_tgt_preds.append(val_tgt_logits.detach().cpu().numpy())

                    # After all batches, concatenate and compute metrics
                    val_all_src_labels = np.concatenate(val_all_src_labels, axis=0)
                    val_all_src_preds = np.concatenate(val_all_src_preds, axis=0)
                    val_all_tgt_labels = np.concatenate(val_all_tgt_labels, axis=0)
                    val_all_tgt_preds = np.concatenate(val_all_tgt_preds, axis=0)

                    val_cm_src, val_src_metrics = compute_binary_clf_metrics(torch.tensor(val_all_src_labels), torch.tensor(val_all_src_preds), domain_name="val_clf/TB-NET", log_to_wandb=False)
                    val_cm_tgt, val_tgt_metrics = compute_binary_clf_metrics(torch.tensor(val_all_tgt_labels), torch.tensor(val_all_tgt_preds), domain_name="val_clf/MBOD", log_to_wandb=False)

                    # Log confusion matrices
                    if((epoch + 1) % exp_cfg["tsne_interval"] == 0):
                        val_cms_image = plot_confusion_matrices(val_cm_src, val_cm_tgt, epoch + 1)
                        wandb.log({"cm/val_confusion_matrices": val_cms_image})

                    wandb.log({
                        "epoch": epoch + 1,
                        "val_clf/src_specificity": val_src_metrics["specificity"],
                        "val_clf/src_sensitivity": val_src_metrics["sensitivity"],
                        "val_clf/src_f1_score": val_src_metrics["f1_score"],
                        "val_clf/src_accuracy": val_src_metrics["accuracy"],
                        "val_clf/src_specificity_at_90%_sensitivity": val_src_metrics["specificity_at_90%_sensitivity"],
                        "val_clf/src_cohen_kappa": val_src_metrics["cohen_kappa"],
                        "val_clf/tgt_specificity": val_tgt_metrics["specificity"],
                        "val_clf/tgt_sensitivity": val_tgt_metrics["sensitivity"],
                        "val_clf/tgt_f1_score": val_tgt_metrics["f1_score"],
                        "val_clf/tgt_accuracy": val_tgt_metrics["accuracy"],
                        "val_clf/tgt_specificity_at_90%_sensitivity": val_tgt_metrics["specificity_at_90%_sensitivity"],
                        "val_clf/tgt_cohen_kappa": val_tgt_metrics["cohen_kappa"],
                        "val/src_intra_similarity": val_cl_metrics["src_intra_similarity"],
                        "val/tgt_intra_similarity": val_cl_metrics["tgt_intra_similarity"],
                        "val/inter_domain_similarity": val_cl_metrics["inter_domain_similarity"],
                        "val/src_map": val_cl_metrics["src_map"],
                        "val/tgt_map": val_cl_metrics["tgt_map"],
                        "val_clf/src_tpr": val_src_metrics["tpr"],
                        "val_clf/src_fpr": val_src_metrics["fpr"],
                        "val_clf/tgt_tpr": val_tgt_metrics["tpr"],
                        "val_clf/tgt_fpr": val_tgt_metrics["fpr"],
                    })


            print(f"Experiment {exp_name} completed successfully. Starting next experiment...")
            wandb.finish()
            del model, optimizer, jmmd_loss_fn
            reset_experiment_state()
    except KeyError as e:
        print(f"Missing configuration: {e}")


# -------- NOTES ---------
# FIXED ISSUES:
# 1. JMMD gradient flow now properly maintained through single computational graph
# 2. Cross-domain triplet loss weights now use proper cross_domain_weight
# 3. Added better error logging for invalid losses
# 4. Eliminated double computation by computing JMMD and final features together
# 5. Project name includes "FIXED" to differentiate from original runs