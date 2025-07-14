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

        wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')
        wandb.init(project='MBOD-cl-3', name='Quadruplet_Orig_Paper_n1_mstb-025_mprof',
            config={
                "seed": 42,  # ADD THIS
                "experiment_type": "Quadruplet (Original)",
                "n2_selection": "Original Paper", # Decide how we select N2 samples (label-based only) ---> Remember, no SH constraints here (for now)
                "n1_selection": "MSTB-based",
                "prioritize_prof_n1": False,  # Prioritize highest profusion difference for N1 samples
                "beta_factor": 0.25, # The ratio between the two margins in quadruplet loss
                "dataset": "MBOD ONLY",
                "labeling_scheme": "MSTB",
                "batch_size": 32,
                "n_epochs": 1000,
                "learning_rate": 1e-4,
                "oversample": True,
                "OS_factor": 0.5,  # Oversampling factor
                "initial_margin": 0.05,      
                "final_margin": 0.5,        
                "margin_scheduling": True,   # Enable margin scheduling
                "scheduling_fraction": 0.75,  # Complete scheduling in first x% of training
                "mining": "BSHN-v2",
                "augmentations": True,
                "filtered_dataset": True,
                "loss_function": "Quadruplet",
                "p_ilo_anchor": 0.0,
                "p_ilo_final": 0.1,
                "num_classes": 8,  # Explicitly specify 8 classes
                "p_ilo_scheduling": False,  # Enable p_ilo scheduling
                "mask_ilo_tb": False,  # Mask ILO TB-based loss
                "use_classification": True,
                "active_classifier": "multiclass_profusion",
                "lambda_clf": 0.25,  # Weight for classifier loss
            })
        
        experiment_name = wandb.run.name

        model = xrv.models.ResNet(weights="resnet50-res512-all")

        if(wandb.config.use_classification and wandb.config.active_classifier == "multiclass_profusion"):
            model.mc_prof_clf = cl_utils.MulticlassClassifier(input_dim=2048, num_classes=4, name="MC-Prof", dropout_rate=0.3)
            classifier_optimizer = torch.optim.Adam(model.mc_prof_clf.parameters(), lr=1e-3, weight_decay=1e-4)
            clf_loss_fn = nn.CrossEntropyLoss()
            
        elif(wandb.config.use_classification and wandb.config.active_classifier == "binary_profusion"):
            model.bin_prof_clf= cl_utils.BinaryClassifier(input_dim=2048, name="Bin-Prof", dropout_rate=0.3)
            classifier_optimizer = torch.optim.Adam(model.bin_prof_clf.parameters(), lr=1e-3, weight_decay=1e-4)
            clf_loss_fn = nn.BCEWithLogitsLoss()

        model = model.to(device)

        encoder_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        triplet_loss_fn = nn.TripletMarginLoss(margin=wandb.config.initial_margin, p=2)

        margin_1 = wandb.config.initial_margin
        margin_2 = wandb.config.initial_margin * wandb.config.beta_factor

        if "RNR" in wandb.config.experiment_type:
            loss_type = "RelativeNegativeRanking"
            
        elif "TNR" in wandb.config.experiment_type:
            loss_type = "TieredNegativeRanking"
            
        elif "SNR" in wandb.config.experiment_type:
            loss_type = "SequentialNegativeRanking"
            
        elif "DoubleTriplet" in wandb.config.experiment_type:
            loss_type = "DoubleTriplet"

        elif "Original" in wandb.config.experiment_type:
            loss_type = "Original"
            
        else:
            assert ValueError(f"Unknown experiment type: {wandb.config.experiment_type}")
            loss_type="NONE"

        quadruplet_loss_fn = cl_utils.QuadrupletMarginLoss(margin1=margin_1, margin2=margin_2, p=2, type=loss_type, mask_ilo_tb=wandb.config.mask_ilo_tb)

        n_epochs = wandb.config.n_epochs
        margin = wandb.config.initial_margin
        batch_size = wandb.config.batch_size
        
        preprocess = transforms.Compose([
           # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
           # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        mbod_merged_loader = torch.utils.data.DataLoader(mbod_dataset_merged, batch_size=wandb.config.batch_size, shuffle=True)
        

        if(wandb.config.augmentations):

            augmentations_list = transforms.Compose([
                transforms.RandomRotation(degrees=10, expand=False, fill=0),
                # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
            ])
            # Get the dataloaders
            train_loader, _, _ = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=augmentations_list,
                oversample=wandb.config.oversample,
                scaling_factor = wandb.config.OS_factor 
            )

            _, val_loader, test_loader = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=None,
                oversample=None,
                scaling_factor = wandb.config.OS_factor 
            )

        else:
            train_loader, _, _ = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=None,
                oversample=wandb.config.oversample,
                scaling_factor = wandb.config.OS_factor
            )

            _, val_loader, test_loader = get_dataloaders(
                hdf5_path=hdf5_file_path,
                preprocess=preprocess,
                batch_size=wandb.config.batch_size,
                labels_key="multiclass_stb",
                split_file="stratified_split_mstb_new.json",
                augmentations=None,
                oversample=None,
                scaling_factor = wandb.config.OS_factor
            )

        ilo_images = []
        ilo_labels = []

        for idx in range(len(ilo_dataset)):
            image, label = ilo_dataset[idx]
            
            # Convert image to a PyTorch tensor and move to GPU
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).to(device)
            label_tensor = torch.tensor(label, dtype=torch.long).to(device)

            ilo_images.append(image_tensor)
            ilo_labels.append(label_tensor)

        # Stack all tensors into a single tensor for efficient access
        ilo_images = torch.cat(ilo_images, dim=0)  # Shape: (N, 1, H, W)
        ilo_labels = torch.stack(ilo_labels)       # Shape: (N,)

        # visualize_tsne(model, device, ilo_dataset, mbod_merged_loader, trained=False, log_to_wandb=True, set_name="pre-training", entire_dataset=True, is_mstb=False)
        # visualize_tsne(model, device, ilo_dataset, train_loader, trained=False, log_to_wandb=True, set_name="pre-training", entire_dataset=False, is_mstb=False)
        
        
        results = cl_utils.train_model_quadruplet(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            triplet_loss_fn=triplet_loss_fn,
            quadruplet_loss_fn=quadruplet_loss_fn,
            clf_loss_fn=clf_loss_fn if wandb.config.use_classification else None,
            encoder_optimizer=encoder_optimizer,
            classifier_optimizer=classifier_optimizer,
            device=device,
            n_epochs=n_epochs,
            ilo_dataset=ilo_dataset,
            mbod_merged_loader=mbod_merged_loader,
            experiment_name=experiment_name,
            margin_scheduling=wandb.config.margin_scheduling,
            initial_margin=wandb.config.initial_margin,
            final_margin=wandb.config.final_margin,
            scheduling_fraction=wandb.config.scheduling_fraction,
            mining_strat=wandb.config.mining,
            p_ilo_anchor=wandb.config.p_ilo_anchor,
            lambda_clf=wandb.config.lambda_clf,
            use_classification= wandb.config.use_classification
        )

        test_results = cl_utils.test_quadruplet_model(
            model=model,
            test_loader=test_loader,
            device=device,
            triplet_loss_fn=triplet_loss_fn,
            quadruplet_loss_fn=quadruplet_loss_fn,
            ilo_dataset=ilo_dataset,
            experiment_name=experiment_name,
            log_to_wandb=True,
            clf_loss_fn=clf_loss_fn if wandb.config.use_classification else None,
)

       

    except KeyError as e:
        print(f"Missing configuration: {e}")


