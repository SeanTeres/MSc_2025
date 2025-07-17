import sys
import os
import gc
# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

from datasets.hdf_dataset import HDF5Dataset
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datasets.dataloader import get_dataloaders
import torchxrayvision as xrv
import torch
from train_utils import classes, helpers
import torch.nn.functional as F
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score
import seaborn as sns
from sklearn.calibration import calibration_curve
import io
import torchvision.transforms as transforms
import os
from tsne import visualize_tsne
import math
import random
import pandas as pd

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import cl_utils

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries"""
    import random
    import numpy as np
    import torch
    import os
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For sklearn (if used)
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass
    
    print(f"🌱 Random seeds set to {seed} for reproducibility")
    return seed


if __name__ == "__main__":
    set_random_seeds()  # Set a fixed seed for reproducibility

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    config = load_config("/home/sean/MSc_2025/codev2/config.yaml")

    preprocess = transforms.Compose([
    # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        # Get the path to the generated HDF5 file
        hdf5_file_path = config["merged_silicosis_output"]["hdf5_file"]
        ilo_hdf5_file_path = config["ilo_output"]["hdf5_file"]
     

        # Create an HDF5SilicosisDataset instance
        mbod_dataset_merged = HDF5Dataset(
            hdf5_path=hdf5_file_path,
            labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )


        ilo_dataset = HDF5Dataset(
            hdf5_path=ilo_hdf5_file_path,
            labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
            images_key="images",
            augmentations=None,
            preprocess=preprocess
        )

        
        multiclass_stb_mapping = {
            0: "Profusion 0, No TB",
            1: "Profusion 1, No TB",
            2: "Profusion 2, No TB",
            3: "Profusion 3, No TB",
            4: "Profusion 0, With TB",
            5: "Profusion 1, With TB",
            6: "Profusion 2, With TB",
            7: "Profusion 3, With TB"
        }

        excel_path = "/home/sean/MSc_2025/codev2/end-end_clf_logs/quad_clf.xlsx"
        if os.path.exists(excel_path):
            results_df = pd.read_excel(excel_path)
        else:
            results_df = pd.DataFrame()

        experiment_names = ["Quad_OG-n1_prof-n2_v1-XRV_clf_025-m_03_fast-b16", "Quad_Original-XRV_clf_025-m_03_fast-b16", "Quadruplet_Orig_Paper-n1_mstb-025_mcprof"]

        for experiment_name in experiment_names:


            if "BIN" in experiment_name:
                active_classifier = "binary_profusion"
            else:
                active_classifier = "multiclass_profusion"

            model = xrv.models.ResNet(weights="resnet50-res512-all")

            if(active_classifier == "multiclass_profusion"):
                # model.mc_prof_clf = cl_utils.MulticlassClassifier(input_dim=2048, num_classes=4, name="MC-Prof", dropout_rate=0.5)

                model.mc_prof_clf = cl_utils.XRVBasedClassifier(input_dim=2048, num_classes=4, name="XRV-Base")  # Add XRV classifier for 4 classes

                # model.mc_prof_clf = cl_utils.ShallowMulticlassClassifier(input_dim=2048, num_classes=4, name="MC-Prof", dropout_rate=0.5) # TO DO: Check shallower classifier
                classifier_optimizer = torch.optim.Adam(model.mc_prof_clf.parameters(), 
                                                        lr=1e-3,  # Changed from 1e-3
                                                        weight_decay=1e-4)  # Changed from 1e-3
                clf_loss_fn = nn.CrossEntropyLoss()

            elif(active_classifier == "binary_profusion"):
                model.mc_prof_clf = cl_utils.XRVBasedClassifier(input_dim=2048, num_classes=1, name="XRV-Base")  # Add XRV classifier for 2 classes
                classifier_optimizer = torch.optim.Adam(model.mc_prof_clf.parameters(), lr=1e-3, weight_decay=1e-4)
                clf_loss_fn = nn.BCEWithLogitsLoss()

            model = model.to(device)

            encoder_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

            triplet_loss_fn = nn.TripletMarginLoss(margin=0.05, p=2)

            checkpoint_dir = os.path.join("checkpoints", experiment_name)
            checkpoint_path = os.path.join(checkpoint_dir, "best_bin_spec_model.pth")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            n_epochs = checkpoint['epoch']

            model.mc_prof

            

            margin_1 = 0.05
            margin_2 = 0.05 * 0.25

            if "RNR" in experiment_name:
                loss_type = "RelativeNegativeRanking"
                
            elif "TNR" in experiment_name:
                loss_type = "TieredNegativeRanking"
                
            elif "SNR" in experiment_name:
                loss_type = "SequentialNegativeRanking"
                
            elif "DoubleTriplet" in experiment_name:
                loss_type = "DoubleTriplet"

            elif ("Original" in experiment_name) or ("OG" in experiment_name):
                loss_type = "Original"
                
            else:
                assert ValueError(f"Unknown experiment type: {experiment_name}")
                loss_type="NONE"

            quadruplet_loss_fn = cl_utils.QuadrupletMarginLoss(margin1=margin_1, margin2=margin_2, p=2, type=loss_type, mask_ilo_tb=False)

            batch_size=16
            mbod_merged_loader = torch.utils.data.DataLoader(mbod_dataset_merged, batch_size=batch_size, shuffle=True)



            train_loader, val_loader, test_loader = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=None,
                oversample=False,
                scaling_factor = 0
            )

        # TO DO : FINISH THIS SCRIPT 
        # We want to load the pretrained encoder, freeze it, and then finetune classifier
        


    except KeyError as e:
        print(f"Missing configuration: {e}")


