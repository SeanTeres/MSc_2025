import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchxrayvision as xrv
import wandb
import numpy as np
import yaml
import torch.nn.functional as F


# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))
from datasets.dataloader import get_dataloaders
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2

# Add codev2 and DomainAdaptation to path
sys.path.append(os.path.abspath("../DomainAdaptation"))
from da_utils import reinitialize_weights, visualize_tsne_with_kaggle_tb
from clf_manager import BinaryClassifier, MulticlassClassifier, XRVBasedClassifier

# Add to your imports
from torch.nn import functional as F

# Define focal loss with gamma=2
def focal_loss(logits, targets, gamma=2):
    BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    focal_loss = (1-pt)**gamma * BCE_loss
    return focal_loss.mean()

class BinaryClassifier2(nn.Module):
    def __init__(self, in_features):
        super(BinaryClassifier2, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, 1024)  # Input layer
        self.dropout = nn.Dropout(0.1)  # Dropout for regularization
        self.fc1 = nn.Linear(1024, 512)  # First hidden layer
        self.dropout1 = nn.Dropout(0.1)  # Add dropout for regularization
        self.fc2 = nn.Linear(512, 256)   # Second hidden layer
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 1)     # Output layer (single node for binary classification)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)  # Apply dropout after the first layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # Raw logit output (no sigmoid here)
        return x


from binary_cv_old import BinaryCrossValidator

with open("bin_config.yaml", "r") as f:
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
SPLIT_FILE_MBOD = config["SPLIT_FILE_MBOD"]
DATA_PATH_MBOD = config["DATA_PATH_MBOD"]
SPLIT_FILE_TBNET = config["SPLIT_FILE_TBNET"]
DATA_PATH_TBNET = config["DATA_PATH_TBNET"]
EPOCHS = config["EPOCHS"]
WEIGHTED_LOSS = config["WEIGHTED_LOSS"]
CLF_TASK_LABELS_KEY = config["CLF_TASK_LABELS_KEY"]
TRAIN_SET_NAME = config["TRAIN_SET_NAME"]
TEST_SET_NAME = config["TEST_SET_NAME"]
CLF_TYPE = config["CLF_TYPE"]  # "Linear" or "MLP"
LOSS_FUNC = config["LOSS_FUNC"]  # "CrossEntropy" or "BCEWithLogits" or "FocalLoss"

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        elif NUM_CLASSES == 2:

            if(LOSS_FUNC == "BCEWithLogitsLoss"):
                loss_fn = nn.BCEWithLogitsLoss()

                if WEIGHTED_LOSS:
                    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3]).to(device))
            elif(LOSS_FUNC == "FocalLoss"):

                loss_fn = focal_loss
            else:
                raise ValueError(f"Unsupported loss function for 2 classes: {LOSS_FUNC}")

            if(CLF_TYPE == "Linear"):
                model.classifier = XRVBasedClassifier(input_dim=2048, num_classes=1, name="bin_XRV-Base")
            elif (CLF_TYPE == "MLP"):
                model.classifier = BinaryClassifier(input_dim=2048, name="bin_mlp-Base")
            elif(CLF_TYPE == "MLP2"):
                model.classifier = BinaryClassifier2(in_features=2048)
        else:
            raise ValueError(f"Unsupported number of classes: {NUM_CLASSES}")
        
        model = model.to(device)
        cv = BinaryCrossValidator(
            model=model,
            labels_key=LABELS_KEY,
            device=device,
            loss_fn=loss_fn,
            hdf5_path=DATA_PATH_TBNET,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            optimizer_type=torch.optim.Adam,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            use_oversampling=OVERSAMPLE,
            checkpoint_save_target=CHECKPOINT_SAVE_DIR,
            exp_name=RUN_NAME,
            num_classes=NUM_CLASSES,
            split_file=SPLIT_FILE_TBNET,
            train_set_name=TRAIN_SET_NAME,
            test_set_name=TEST_SET_NAME,
            clf_task_labels_key=CLF_TASK_LABELS_KEY,
            )
        
        ave_acc, ave_sens, ave_spec, avg_kappa = cv.run_k_iterations(5,f"{PROJECT_NAME}", exp_name=RUN_NAME, suffix=f"")
        

        
# FULL TO DO:
# 1) Add support for binary profusion, binary tb and maybe multiclass_stb?
# 3) Add support for weighted loss and/or focal loss
# 4) Make sure we are able to also run this on TB-based datasets (TB-Net, MC, SZ, etc.)
# 5) t-SNE plots
# 6) LR scheduler?        
        


        

    except KeyError as e:
        print(f"Missing configuration: {e}")
