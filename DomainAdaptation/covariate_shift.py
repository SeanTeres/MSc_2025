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
from collections import OrderedDict

class SingleLayerFeatureExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.features = None
        self.layer_name = layer_name
        self.hook = self._register_hook(layer_name)
    
    def _get_activation(self):
        def hook(model, input, output):
            self.features = output.detach()
        return hook
    
    def _register_hook(self, layer_name):
        if layer_name == 'conv1':
            return self.model.model.conv1.register_forward_hook(self._get_activation())
        elif layer_name == 'layer1':
            return self.model.model.layer1.register_forward_hook(self._get_activation())
        elif layer_name == 'layer2':
            return self.model.model.layer2.register_forward_hook(self._get_activation())
        elif layer_name == 'layer3':
            return self.model.model.layer3.register_forward_hook(self._get_activation())
        elif layer_name == 'layer4':
            return self.model.model.layer4.register_forward_hook(self._get_activation())
    
    def remove_hook(self):
        self.hook.remove()

    def get_features(self, x):
        _ = self.model(x)  # Forward pass to trigger hook
        return self.features
    
def compute_layer_mmd(source_features, target_features, kernel_type='gaussian'):
    """Compute Maximum Mean Discrepancy between source and target features."""
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Flatten spatial dimensions if present
    if len(source_features.shape) > 2:
        source_features = source_features.view(source_features.size(0), -1)
        target_features = target_features.view(target_features.size(0), -1)
    
    source_features = source_features.cpu().numpy()
    target_features = target_features.cpu().numpy()
    
    # Compute kernel matrices
    K_ss = rbf_kernel(source_features)
    K_tt = rbf_kernel(target_features)
    K_st = rbf_kernel(source_features, target_features)
    
    # Compute MMD
    mmd = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    return mmd

def analyze_layer_shifts(source_loader, target_loader, model, device):
    """Analyze domain shift at each layer, one layer at a time."""
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    layer_shifts = {}
    
    model.eval()
    
    for layer_idx, layer_name in enumerate(layer_names):
        print(f"Processing layer: {layer_name}")
        
        # Extract features for this layer only
        feature_extractor = SingleLayerFeatureExtractor(model, layer_name)
        
        # Process source domain for current layer
        source_features = []
        with torch.no_grad():
            for batch in source_loader:
                images = batch[0].to(device)
                features = feature_extractor.get_features(images)
                source_features.append(features)
        
        # Concatenate source features
        source_layer = torch.cat(source_features, dim=0)
        
        # Process target domain for current layer
        target_features = []
        with torch.no_grad():
            for batch in target_loader:
                images = batch[0].to(device)
                features = feature_extractor.get_features(images)
                target_features.append(features)
        
        # Concatenate target features
        target_layer = torch.cat(target_features, dim=0)
        
        # Compute MMD
        mmd = compute_layer_mmd(source_layer, target_layer)
        
        # Store results
        layer_shifts[layer_name] = {
            'mmd': mmd,
            'layer_depth': layer_idx
        }
        
        # Clean up to free memory
        feature_extractor.remove_hook()
        del source_features
        del target_features
        del source_layer
        del target_layer
        torch.cuda.empty_cache()
        gc.collect()
        
    return layer_shifts

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

            train_loader_mbod, _, _ = get_dataloaders(
                hdf5_path=cfg["merged_silicosis_output"]["hdf5_file"],
                preprocess=preprocess,
                batch_size=4,
                labels_key="multiclass_stb",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_MBOD_mlabel_stb.json",
                augmentations=augmentations_list ,
                oversample=False
            )

            train_loader_tbnet, _, _ = get_dataloaders(
                hdf5_path=cfg["kaggle_TB"]["outputpath"],
                preprocess=preprocess,
                batch_size=4,
                labels_key="tuberculosis",
                split_file="stratified_split_tb_net.json",
                augmentations=augmentations_list,
                oversample=False
            )

            _, val_loader_mbod, test_loader_mbod = get_dataloaders(
                hdf5_path=cfg["merged_silicosis_output"]["hdf5_file"],
                preprocess=preprocess,
                batch_size=4,
                labels_key="multiclass_stb",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_MBOD_mlabel_stb.json",
                augmentations=None,
                oversample=False
            )
            _, val_loader_tbnet, test_loader_tbnet = get_dataloaders(
                hdf5_path=cfg["kaggle_TB"]["outputpath"],
                preprocess=preprocess,
                batch_size=4,
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
            

            layer_kernels = (cl_tllib_utils.GaussianKernel(alpha=0.5), cl_tllib_utils.GaussianKernel(1.), cl_tllib_utils.GaussianKernel(2.))
            jmmd_loss_fn = cl_tllib_utils.JointMultipleKernelMaximumMeanDiscrepancy((layer_kernels,))

            layer_shifts = analyze_layer_shifts(train_loader_tbnet, train_loader_mbod, model, device)
            print("Layer shifts:", layer_shifts)


    except KeyError as e:
        print(f"Missing configuration: {e}")

