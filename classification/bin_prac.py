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
import sklearn.metrics as skmetrics
import metrics
import random
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import SubsetRandomSampler, Subset
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
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_combined_conf_mat(confusion_matrix):
    if confusion_matrix.shape == (2, 2):
        # Binary case
        fig = plt.figure(figsize=(12, 8))
        cm_display = skmetrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca(), cmap='Blues')
        plt.title("Confusion Matrix")
    elif len(confusion_matrix.shape) == 3:
        # Multilabel case
        avg_cm = np.mean(confusion_matrix, axis=0)
        norm_cm = avg_cm / (np.sum(avg_cm, axis=1, keepdims=True))
        fig = plt.figure(figsize=(12, 8))
        cm_display = skmetrics.ConfusionMatrixDisplay(norm_cm, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca(), cmap='Blues')
        plt.title("Normalised Multilabel Confusion Matrix")
    else:
        # Multiclass case
        TN, FP, FN, TP = metrics.get_cm_for_class(confusion_matrix, 0)
        bin_conf_matrix = np.array([[TP, FN],
                                    [FP, TN]])
        display_labels = [i for i in range(len(confusion_matrix[0]))]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        cm_display1 = skmetrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=display_labels)
        cm_display1.plot(ax=axes[0], colorbar=False, cmap='Blues')
        axes[0].set_title("Confusion Matrix")
        cm_display2 = skmetrics.ConfusionMatrixDisplay(bin_conf_matrix, display_labels=[0, 1])
        cm_display2.plot(ax=axes[1], colorbar=False, cmap='Blues')
        axes[1].set_title("Binary Confusion Matrix")
        plt.tight_layout()
    return fig

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
FREEZE_ENC = config["FREEZE_ENC"]

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

def extract_image_features(model, device, batch_imgs):
    x = model.model.conv1(batch_imgs)
    x = model.model.bn1(x)
    x = model.model.relu(x)
    x = model.model.maxpool(x)

    l1_out = model.model.layer1(x)
    l2_out = model.model.layer2(l1_out)
    l3_out = model.model.layer3(l2_out)
    l4_out = model.model.layer4(l3_out)

    feats = model.model.avgpool(l4_out)
    feats = torch.flatten(feats, 1)

    return feats

def evaluate_model(dataloader, epoch, loss_fn, name="", fold=None):
    print(f"\nEVALUATING on {name}\n")
    metrics_dict = {}
    model.eval()

    all_labels = []
    all_probs = []
    all_feats = []
    all_logits = []

    total_loss = 0.0

    with torch.no_grad():

        for batch_imgs, batch_labels in tqdm(dataloader, desc=f"Evaluating {name} {fold}"):
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

            feats = model.features(batch_imgs)
            logits = model.classifier(feats)
            probs = torch.sigmoid(logits)


            loss_val = loss_fn(logits, batch_labels.float().unsqueeze(1))
            total_loss += loss_val.item()

            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())


        avg_loss = total_loss / len(dataloader)
        print(f"Average Loss - {name} {fold}: {avg_loss}")

        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()
        all_logits = np.array(all_logits).flatten()

        # print(all_labels)
        # print(all_probs)

        preds_05 = all_probs > 0.5
        
        # Compute metrics
        cm_05 = skmetrics.confusion_matrix(all_labels, preds_05)

        spec_at_09_sens, threshold = metrics.specificity_at_sensitivity(all_labels, all_probs, 0.9)
        preds_opt = all_probs > threshold
        cm_opt = skmetrics.confusion_matrix(all_labels, preds_opt)

        acc_05 = metrics.get_accuracy(cm_05)
        f1_05 = metrics.get_f1_score(cm_05)
        sens_05 = metrics.get_sensitivity(cm_05)
        spec_05 = metrics.get_specificity(cm_05)
        kappa_05 = metrics.get_cohens_kappa(cm_05)


        acc_opt = metrics.get_accuracy(cm_opt)
        sens_opt = metrics.get_sensitivity(cm_opt)
        spec_opt = metrics.get_specificity(cm_opt)
        f1_opt = metrics.get_f1_score(cm_opt)
        kappa_opt = metrics.get_cohens_kappa(cm_opt)

        try:
            auc_score = skmetrics.roc_auc_score(all_labels, all_probs)
            fpr, tpr, _ = skmetrics.roc_curve(all_labels, all_probs)

        except ValueError as e:
            print(f"Warning: Could not calculate auc")
            auc_score = None
            fpr, tpr = None, None

        

        metrics_dict["auc"] = auc_score
        metrics_dict["fpr"] = fpr
        metrics_dict["tpr"] = tpr
        metrics_dict["loss"] = total_loss

        metrics_dict["cm_05"] = cm_05
        metrics_dict["acc_05"] = acc_05
        metrics_dict["f1_05"] = f1_05
        metrics_dict["sens_05"] = sens_05
        metrics_dict["spec_05"] = spec_05
        metrics_dict["kappa_05"] = kappa_05

        metrics_dict["spec_at_09_sens"] = spec_at_09_sens
        metrics_dict["threshold"] = threshold
        metrics_dict["cm_opt"] = cm_opt
        metrics_dict["acc_opt"] = acc_opt
        metrics_dict["f1_opt"] = f1_opt
        metrics_dict["sens_opt"] = sens_opt
        metrics_dict["spec_opt"] = spec_opt
        metrics_dict["kappa_opt"] = kappa_opt

        cm_opt_img = plot_combined_conf_mat(cm_opt)
        cm_05_img = plot_combined_conf_mat(cm_05)


    # print(f"labels: {all_labels} \n logits: {all_logits} \n probs: {all_probs}")
    

    return metrics_dict


def get_all_dataloaders(data_path_mbod, split_file_mbod, data_path_tbnet, split_file_tbnet, labels_key, batch_size, use_oversampling):

    preprocess = transforms.Compose([
    # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    augmentations = transforms.Compose([
    transforms.RandomRotation(degrees=10, expand=False, fill=0),
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
    ])

    train_loader_mbod, _, _ = get_dataloaders(
        hdf5_path=data_path_mbod,
        preprocess=preprocess,
        train_split=0,
        batch_size=batch_size,
        labels_key=labels_key,
        split_file=split_file_mbod,
        augmentations=augmentations,
        oversample=use_oversampling
    )

    _, val_loader_mbod, test_loader_mbod = get_dataloaders(
        hdf5_path=data_path_mbod,
        preprocess=preprocess,
        train_split=0,
        batch_size=1,
        labels_key=labels_key,
        split_file=split_file_mbod,
        augmentations=None,
        oversample=False
    )

    train_loader_tbnet, _, _ = get_dataloaders(
        hdf5_path=data_path_tbnet,
        preprocess=preprocess,
        train_split=0,
        batch_size=batch_size,
        labels_key=labels_key,
        split_file=split_file_tbnet,
        augmentations=augmentations,
        oversample=use_oversampling
    )

    _, val_loader_tbnet, test_loader_tbnet = get_dataloaders(
        hdf5_path=data_path_tbnet,
        preprocess=preprocess,
        train_split=0,
        batch_size=1,
        labels_key=labels_key,
        split_file=split_file_tbnet,
        augmentations=None,
        oversample=False
    )

    return train_loader_mbod, val_loader_mbod, test_loader_mbod, train_loader_tbnet, val_loader_tbnet, test_loader_tbnet


def train(config, model, device, optimizer, fold):

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    
    train_loader_mbod, val_loader_mbod, test_loader_mbod, train_loader_tbnet, val_loader_tbnet, test_loader_tbnet = get_all_dataloaders(
        config["DATA_PATH_MBOD"], config["SPLIT_FILE_MBOD"], config["DATA_PATH_TBNET"], config["SPLIT_FILE_TBNET"], config["LABELS_KEY"], config["BATCH_SIZE"], config["OVERSAMPLE"]
    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    train_loader = train_loader_tbnet if config["TRAIN_SET_NAME"] == "TB-NET" else train_loader_mbod

    for epoch in tqdm(range(EPOCHS), desc=f"Training on {config['TRAIN_SET_NAME']}"):
        model.train()
        total_loss = 0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()

            feats = model.features(batch_imgs)
            outputs = model.classifier(feats)
            loss = loss_fn(outputs, batch_labels.float().unsqueeze(1))
            # print(f"loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
        train_dict = evaluate_model(train_loader, epoch, loss_fn, name="train", fold=fold)

        cm_05_img = plot_combined_conf_mat(train_dict["cm_05"])
        cm_opt_img = plot_combined_conf_mat(train_dict["cm_opt"])

        
        wandb.log({
            "train/loss": train_dict["loss"],
            "train/auc": train_dict["auc"],
            "train/fpr": train_dict["fpr"],
            "train/tpr": train_dict["tpr"],
            "train/acc_05": train_dict["acc_05"],
            "train/f1_05": train_dict["f1_05"],
            "train/sens_05": train_dict["sens_05"],
            "train/spec_05": train_dict["spec_05"],
            "train/kappa_05": train_dict["kappa_05"],
            "train/spec_at_09_sens": train_dict["spec_at_09_sens"],
            "train/threshold": train_dict["threshold"],
            "cm/train_cm_opt": wandb.Image(cm_opt_img),
            "cm/train_cm_05": wandb.Image(cm_05_img),
            "train/acc_opt": train_dict["acc_opt"],
            "train/f1_opt": train_dict["f1_opt"],
            "train/sens_opt": train_dict["sens_opt"],
            "train/spec_opt": train_dict["spec_opt"],
            "train/kappa_opt": train_dict["kappa_opt"],
        }, step=epoch)

        val_dict = evaluate_model(val_loader_tbnet if config["TRAIN_SET_NAME"] == "TB-NET" else val_loader_mbod, epoch, loss_fn, name="val", fold=fold)

        val_cm_05_img = plot_combined_conf_mat(val_dict["cm_05"])
        val_cm_opt_img = plot_combined_conf_mat(val_dict["cm_opt"])

        scheduler.step(val_dict["spec_opt"])

        wandb.log({
            "val/auc": val_dict["auc"],
            "val/fpr": val_dict["fpr"],
            "val/tpr": val_dict["tpr"],
            "val/loss": val_dict["loss"],
            "val/acc_05": val_dict["acc_05"],
            "val/f1_05": val_dict["f1_05"],
            "val/sens_05": val_dict["sens_05"],
            "val/spec_05": val_dict["spec_05"],
            "val/kappa_05": val_dict["kappa_05"],
            "val/spec_at_09_sens": val_dict["spec_at_09_sens"],
            "val/threshold": val_dict["threshold"],
            "cm/val_cm_opt": wandb.Image(val_cm_opt_img),
            "cm/val_cm_05": wandb.Image(val_cm_05_img),
            "val/acc_opt": val_dict["acc_opt"],
            "val/f1_opt": val_dict["f1_opt"],
            "val/sens_opt": val_dict["sens_opt"],
            "val/spec_opt": val_dict["spec_opt"],
            "val/kappa_opt": val_dict["kappa_opt"],
        }, step=epoch)
    

    test_dict = evaluate_model(test_loader_tbnet if config["TEST_SET_NAME"] == "TB-NET" else test_loader_mbod, epoch, loss_fn, name="test", fold=fold)

    return test_dict
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)


    defaults = load_config("defaults.yaml")
    wandb_api_key = defaults["WANDB_API_KEY"]["value"]
    random_seed = defaults["RANDOM_SEED"]["value"]
    
    set_random_seeds(random_seed)

    accuracies = []
    sensitivities = []
    specificities = []
    kappas = []

    group_name = f"{RUN_NAME}-{wandb.util.generate_id()}"

    for k in range(config["NUM_FOLDS"]):

        if(k>0):
            random_seed = random.randint(0, 100)
            print(f"RANDOM SEED: {random_seed}")
            set_random_seeds(random_seed)
        else:
            print(f"INITIAL RANDOM SEED (Should be 42): {random_seed}")

        if RESOLUTION == 512:
            model = xrv.models.ResNet(weights="resnet50-res512-all")

            if not PRETRAINED:
                # raise ValueError("need to still fix random init.")
                reinitialize_weights(model)
        else:
            raise ValueError(f"Unsupported resolution: {RESOLUTION}")
        
        if NUM_CLASSES == 4:

            if WEIGHTED_LOSS:
                raise ValueError("Weighted loss function is not yet supported.")  
            else:
                loss_fn = nn.CrossEntropyLoss()
        elif NUM_CLASSES == 2:

            if(LOSS_FUNC == "BCEWithLogitsLoss"):
                pos_weight = torch.tensor([0.3]).to(device) if WEIGHTED_LOSS else None
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        

        if FREEZE_ENC:
            for name, param in model.named_parameters():
                # Only freeze parameters that are NOT in the classifier
                if not name.startswith("classifier"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # Only optimize classifier parameters
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        wandb.init(
            project=PROJECT_NAME,
            name=f"{RUN_NAME}-fold_{k}",
            group=group_name,
            config={
                "loss_fn": loss_fn.__class__.__name__,
                "optimizer": optimizer,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "labels_key": LABELS_KEY, 
                "batch_size":BATCH_SIZE,
                "epochs": EPOCHS,
                "clf_head": CLF_TYPE,
                "clf_task": CLF_TASK_LABELS_KEY,
                "resolution": RESOLUTION,
                "train_set": TRAIN_SET_NAME,
                "test_set": TEST_SET_NAME,
                "weighted_loss": WEIGHTED_LOSS,
                "loss_fn": LOSS_FUNC,
                "random_seed":random_seed,
                "fold": k
            },
        )
        model = model.to(device)

        test_dict = train(config=config, model=model, device=device, optimizer=optimizer, fold=k)

        test_cm_05_img = plot_combined_conf_mat(test_dict["cm_05"])
        test_cm_opt_img = plot_combined_conf_mat(test_dict["cm_opt"])

        wandb.log({
        "test/auc": test_dict["auc"],
        "test/fpr": test_dict["fpr"],
        "test/tpr": test_dict["tpr"],
        "test/loss": test_dict["loss"],
        "test/acc_05": test_dict["acc_05"],
        "test/f1_05": test_dict["f1_05"],
        "test/sens_05": test_dict["sens_05"],
        "test/spec_05": test_dict["spec_05"],
        "test/kappa_05": test_dict["kappa_05"],
        "test/spec_at_09_sens": test_dict["spec_at_09_sens"],
        "test/threshold": test_dict["threshold"],
        "cm/test_cm_opt": wandb.Image(test_cm_opt_img),
        "cm/test_cm_05": wandb.Image(test_cm_05_img),
        "test/acc_opt": test_dict["acc_opt"],
        "test/f1_opt": test_dict["f1_opt"],
        "test/sens_opt": test_dict["sens_opt"],
        "test/spec_opt": test_dict["spec_opt"],
        "test/kappa_opt": test_dict["kappa_opt"],
    }, step=k)
    

        kappas.append(test_dict["kappa_opt"])
        accuracies.append(test_dict["acc_opt"])
        sensitivities.append(test_dict["sens_opt"])
        specificities.append(test_dict["spec_opt"])

        print(f"Fold {k} - Test: {test_dict}")

        wandb.finish()

    mean_acc = np.mean(accuracies)
    mean_sens = np.mean(sensitivities)
    mean_spec = np.mean(specificities)
    mean_kappa = np.mean(kappas)

    print(f"Mean - Acc: {mean_acc}, Sens: {mean_sens}, Spec: {mean_spec}, Kappa: {mean_kappa}")




# FULL TO DO:
# 1) Add support for binary profusion, binary tb and maybe multiclass_stb?
# 3) Add support for weighted loss and/or focal loss
# 4) Make sure we are able to also run this on TB-based datasets (TB-Net, MC, SZ, etc.)
# 5) t-SNE plots
# 6) LR scheduler?        

