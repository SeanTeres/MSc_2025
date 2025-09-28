import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchxrayvision as xrv
import wandb
import numpy as np
import yaml

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))
from datasets.dataloader import get_dataloaders
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2

# Add codev2 and DomainAdaptation to path
sys.path.append(os.path.abspath("../DomainAdaptation"))
from da_utils import reinitialize_weights, visualize_tsne_with_kaggle_tb
from clf_manager import BinaryClassifier, MulticlassClassifier, XRVBasedClassifier

from transfer_learning_cv import TransferLearningCrossValidator

def load_config(config_path="defaults.yaml"):
    """Load the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error reading the configuration file: {e}")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    
    # Load transfer learning configuration
    with open("transfer_learning_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Extract configuration parameters
    PROJECT_NAME = config["PROJECT_NAME"]
    RUN_NAME = config["RUN_NAME"]
    USE_ORDINAL_LABELS = config["USE_ORDINAL_LABELS"]
    OVERSAMPLE = config["OVERSAMPLE"]
    USE_MULTILABEL = config["USE_MULTILABEL"]
    USE_MULTICLASS = config["USE_MULTICLASS"]
    CHECKPOINT_SAVE_DIR = config["CHECKPOINT_SAVE_DIR"]
    LABELS_KEY = config["LABELS_KEY"]
    NUM_CLASSES = config["NUM_CLASSES"]
    RESOLUTION = config["RESOLUTION"]
    PRETRAINED = config["PRETRAINED"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    WEIGHT_DECAY = config["WEIGHT_DECAY"]
    SPLIT_FILE_MBOD = config["SPLIT_FILE_MBOD"]
    DATA_PATH_MBOD = config["DATA_PATH_MBOD"]
    SPLIT_FILE_TBNET = config["SPLIT_FILE_TBNET"]
    DATA_PATH_TBNET = config["DATA_PATH_TBNET"]
    EPOCHS = config["EPOCHS"]
    WEIGHTED_LOSS = config["WEIGHTED_LOSS"]
    CLF_TASK_LABELS_KEY = config["CLF_TASK_LABELS_KEY"]
    TRAIN_SET_NAME = config["TRAIN_SET_NAME"]
    TEST_SET_NAME = config["TEST_SET_NAME"]
    CLF_TYPE = config["CLF_TYPE"]
    
    # Load defaults (for API key and random seed)
    defaults = load_config("defaults.yaml")
    wandb_api_key = defaults["WANDB_API_KEY"]["value"]
    random_seed = defaults["RANDOM_SEED"]["value"]
    
    try:
        print(f"Setting up transfer learning: Train on {TRAIN_SET_NAME}, Evaluate on both datasets")
        print(f"TB-Net data: {DATA_PATH_TBNET}")
        print(f"MBOD data: {DATA_PATH_MBOD}")
        
        # Initialize model
        if RESOLUTION == 512:
            model = xrv.models.ResNet(weights="resnet50-res512-all")
            if not PRETRAINED:
                reinitialize_weights(model)
        else:
            raise ValueError(f"Unsupported resolution: {RESOLUTION}")
        
        # Set up classifier and loss function
        if NUM_CLASSES == 2:
            loss_fn = nn.BCEWithLogitsLoss()
            
            if CLF_TYPE == "Linear":
                model.classifier = XRVBasedClassifier(input_dim=2048, num_classes=1, name="bin_XRV-Base")
            elif CLF_TYPE == "MLP":
                model.classifier = BinaryClassifier(input_dim=2048, name="bin_mlp-Base")
            else:
                raise ValueError(f"Unsupported classifier type: {CLF_TYPE}")
        else:
            raise ValueError(f"Only binary classification supported: {NUM_CLASSES}")
        
        model = model.to(device)
        
        # Initialize Transfer Learning Cross Validator
        transfer_cv = TransferLearningCrossValidator(
            model=model,
            labels_key=LABELS_KEY,
            device=device,
            loss_fn=loss_fn,
            tbnet_hdf5_path=DATA_PATH_TBNET,
            mbod_hdf5_path=DATA_PATH_MBOD,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            optimizer_type=torch.optim.Adam,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            use_oversampling=OVERSAMPLE,
            checkpoint_save_target=CHECKPOINT_SAVE_DIR,
            exp_name=RUN_NAME,
            num_classes=NUM_CLASSES,
            tbnet_split_file=SPLIT_FILE_TBNET,
            mbod_split_file=SPLIT_FILE_MBOD,
            train_set_name=TRAIN_SET_NAME,
            test_set_name=TEST_SET_NAME,
            clf_task_labels_key=CLF_TASK_LABELS_KEY,
        )
        
        print("Starting transfer learning cross-validation...")
        print("Training Dataset: TB-Net")
        print("Validation/Test Datasets: TB-Net + MBOD")
        
        # Run k-fold cross validation
        results = transfer_cv.run_k_iterations(
            k=5, 
            project_name=PROJECT_NAME, 
            exp_name=RUN_NAME, 
            suffix="tl"
        )
        
        print("\n=== FINAL TRANSFER LEARNING RESULTS ===")
        print(f"TB-Net (Source Domain) - Avg Accuracy: {results['tbnet']['avg_accuracy']:.4f}")
        print(f"TB-Net (Source Domain) - Avg Specificity: {results['tbnet']['avg_specificity']:.4f}")
        print(f"MBOD (Target Domain) - Avg Accuracy: {results['mbod']['avg_accuracy']:.4f}")
        print(f"MBOD (Target Domain) - Avg Specificity: {results['mbod']['avg_specificity']:.4f}")
        
        print("\nTransfer learning completed successfully!")
        
    except KeyError as e:
        print(f"Missing configuration: {e}")
    except Exception as e:
        print(f"Error during transfer learning: {e}")
        raise