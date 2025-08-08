import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchxrayvision as xrv
import wandb
import numpy as np
import yaml

from train_utils import CrossValidator

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))
from datasets.dataloader import get_dataloaders
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2

# Add codev2 and DomainAdaptation to path
sys.path.append(os.path.abspath("../codev2"))
sys.path.append(os.path.abspath("../DomainAdaptation"))
from da_utils import reinitialize_weights, visualize_tsne_with_kaggle_tb
from clf_manager import BinaryClassifier, MulticlassClassifier, XRVBasedClassifier

with open("clf_config.yaml", "r") as f:
    config = yaml.safe_load(f)

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
SPLIT_FILE = config["SPLIT_FILE"]
DATA_PATH = config["DATA_PATH"]
EPOCHS = config["EPOCHS"]
WEIGHTED_LOSS = config["WEIGHTED_LOSS"]

def load_config(config_path="config.yaml"):
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
    Returns:
        A dictionary containing the configuration settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error reading the configuration file: {e}")


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)


    defaults = load_config("defaults.yaml")
    wandb_api_key = defaults["WANDB_API_KEY"]["value"]
    random_seed = defaults["RANDOM_SEED"]["value"]

    try:

        if RESOLUTION == 512:
            model = xrv.models.ResNet(weights="resnet50-res512-all")

            if not PRETRAINED:
                reinitialize_weights(model)
        else:
            raise ValueError(f"Unsupported resolution: {RESOLUTION}")
        
        if NUM_CLASSES == 4:

            if WEIGHTED_LOSS:
                raise ValueError("Weighted loss function is not yet supported.")  
            else:
                loss_fn = nn.CrossEntropyLoss()
                model.classifier = XRVBasedClassifier(input_dim=2048, num_classes=NUM_CLASSES, name="XRV-Base")
        else:
            raise ValueError(f"Unsupported number of classes: {NUM_CLASSES}")
        
        model = model.to(device)
        cv = CrossValidator(
            model=model,
            labels_key=LABELS_KEY,
            device=device,
            loss_fn=loss_fn,
            hdf5_path=DATA_PATH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            optimizer_type=torch.optim.Adam,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            use_oversampling=OVERSAMPLE,
            checkpoint_save_target=CHECKPOINT_SAVE_DIR,
            use_multilabel=USE_MULTILABEL,
            use_multiclass=USE_MULTICLASS,
            use_ordinal_labels=USE_ORDINAL_LABELS,
            exp_name=RUN_NAME,
            num_classes=NUM_CLASSES,
            split_file=SPLIT_FILE
            )
        
        ave_acc, ave_sens, ave_spec = cv.run_k_iterations(1,f"{PROJECT_NAME}", exp_name=RUN_NAME, suffix=f"")
        

        
# FULL TO DO:
# 1) Add support for binary profusion, binary tb and maybe multiclass_stb?
# 2) COHEN's KAPPA must be logged
# 3) Add support for weighted loss and/or focal loss
# 4) Make sure we are able to also run this on TB-based datasets (TB-Net, MC, SZ, etc.)
# 5) t-SNE plots
# 6) LR scheduler?        
        


        

    except KeyError as e:
        print(f"Missing configuration: {e}")
