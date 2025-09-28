# TO DO:
# Binary spec_at_sens per-class?

# 3) Log all (CL and clf) metrics and tSNEs correctly

import sys
import os
import gc
import yaml
import numpy as np
import matplotlib.pyplot as plt
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
from tqdm import tqdm

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))
from datasets.hdf_dataset import HDF5Dataset
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import h5py
from datasets.dataloader import get_dataloaders

# Add codev2 to the Python path
sys.path.append(os.path.abspath("../codev2"))
import cl_utils
from train_utils import classes, helpers
from tsne import visualize_tsne
from clf_manager import MulticlassClassifier, BinaryClassifier, XRVBasedClassifier

# Add classification to the python path
sys.path.append(os.path.abspath("../classification"))
import metrics
from cross_validation import plot_combined_conf_mat, plot_tb_stratified_binary_cm



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
    defaults = load_config("defaults.yaml")
    wandb_api_key = defaults["WANDB_API_KEY"]["value"]
    random_seed = defaults["RANDOM_SEED"]["value"]
    set_random_seeds(seed=random_seed)  # Set a fixed seed for reproducibility


def mine_quadruplets(model, device, batch_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels, return_single_match=True):
    all_embeddings = []
    all_labels = []
    anchors = []
    positives = []
    negatives = []
    negatives_2 = []
    
    batch_shn_failures, batch_ap_failures, batch_n2_failures, batch_quadruplet_count = 0, 0, 0, 0

    for i, positive_label in enumerate(batch_labels):

        positive_embedding = embeddings[i].unsqueeze(0)  # shape [1, C]
    
        if np.random.rand() < cfg["P_ILO_ANCHOR"]:
            ilo_indices = torch.where(ilo_labels == (positive_label % 4))[0]
            
            # Use ILO as anchor
            ilo_idx = np.random.choice(ilo_indices.cpu().numpy())
            anchor_embedding = ilo_images[ilo_idx].unsqueeze(0)  # shape [1, C]
            anchor_embedding = model.features(anchor_embedding)  # Get features of the anchor
            anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
            anchor_label = ilo_labels[ilo_idx].item()  # Get the label of the anchor
            n_ilo_anchors += 1
        
        else:
            batch_matching_indices = [j for j in range(len(batch_labels)) 
                                            if batch_labels[j] == positive_label and j != i]
            
            if batch_matching_indices:
                batch_anchor_idx = np.random.choice(batch_matching_indices)
                anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                anchor_label = batch_labels[batch_anchor_idx]
            else:
                batch_ap_failures += 1
                continue
        

        if cfg["MINING_STRATEGY"] == "BSHN":

            prof_pos_score = positive_label % 4
            pos_tb_status = 1 if positive_label >= 4 else 0


            if cfg["MINING_STRATEGY"] == "BSHN":

                if cfg["N1_SELECTION"] == "Profusion-based": # In line with our original approach. Strongly enforce profusion separation. 
                    negative_indices = [j for j, label in enumerate(batch_labels) 
                                    if label != positive_label and (label % 4) != prof_pos_score]     
                                    
                elif cfg["N1_SELECTION"] == "MSTB-based":
                    negative_indices = [j for j, label in enumerate(batch_labels) 
                                    if label != positive_label]
                    
        else:
            raise ValueError(f"Unsupported N1 selection strategy: {cfg['N1_SELECTION']}")
        
        if negative_indices:
            negative_embeddings = embeddings[negative_indices]
            anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)
            
            # Compute distances
            dists = F.pairwise_distance(anchor_repeated, negative_embeddings)
            positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
            
            # Find semi-hard negatives
            semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance +current_margin))
            semi_hard_dists = dists[semi_hard_mask]
            
            if semi_hard_dists.numel() > 0:
                # Use semi-hard negative
                hard_idx_in_masked = torch.argmin(semi_hard_dists).item()
                semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)
                selected_neg_idx = semi_hard_indices[hard_idx_in_masked].item()


                negative_embedding = negative_embeddings[selected_neg_idx].unsqueeze(0)
                negative_label = batch_labels[negative_indices[selected_neg_idx]]

                if cfg["N2_SELECTION"] == "Proposed":
                    # Implement the proposed N2 selection strategy
                    negative_indices_2 = [j for j, label in enumerate(batch_labels) 
                        if (label % 4) == (negative_label % 4) and 
                        (int(label >= 4) != int(negative_label >= 4))]
                    
                elif cfg["N2_SELECTION"] == "Original":
                    negative_indices_2 = [j for j, label in enumerate(batch_labels)
                            if label != positive_label and label != negative_label]
                
                if negative_indices_2:
                    neg_2_idx = np.random.choice(negative_indices_2)
                    negative_embedding_2 = embeddings[neg_2_idx].unsqueeze(0)
                    negative_label_2 = batch_labels[neg_2_idx]

                    anchors.append(anchor_embedding)
                    positives.append(positive_embedding)
                    negatives.append(negative_embedding)
                    negatives_2.append(negative_embedding_2)

                    batch_quadruplet_count += 1
                else:
                    batch_n2_failures += 1
            
            else:
                batch_shn_failures += 1
                continue


    
    mining_stats = {
        "ap_failures": batch_ap_failures,
        "shn_failures": batch_shn_failures,
        "n2_failures": batch_n2_failures,
        "quadruplet_count": batch_quadruplet_count
    }
    
    # print(f"QUADRUPLET COUNT: {batch_quadruplet_count}")
    if batch_quadruplet_count > 0:

        batch_anchors = torch.cat(anchors, dim=0)
        batch_positives = torch.cat(positives, dim=0)
        batch_negatives = torch.cat(negatives, dim=0)
        batch_negatives_2 = torch.cat(negatives_2, dim=0)


        return batch_anchors, batch_positives, batch_negatives, batch_negatives_2, mining_stats
    else:
        # Return empty tensors instead of lists
        empty_tensor = torch.tensor([], device=device)
        return empty_tensor, empty_tensor, empty_tensor, empty_tensor, mining_stats

def mine_quadruplets_all(model, device, cfg, current_margin, loader, ilo_images, ilo_labels):
    all_embeddings = []
    all_labels = []

    anchors = []
    positives = []
    negatives = []
    negatives_2 = []

    for batch_idx, sample in enumerate(loader):

        imgs = sample[0].to(device)
        labels = sample[1].to(device)

        feats = model.features(imgs)
        embeddings = F.normalize(feats, p=2, dim=1)

        all_embeddings.append(embeddings.detach().cpu())
        all_labels.append(labels.detach().cpu())

        batch_labels = labels.cpu().numpy()

        for i, positive_label in enumerate(batch_labels):
            positive_embedding = embeddings[i].unsqueeze(0)  # [1, C]

            # Decide anchor
            if np.random.rand() < cfg["P_ILO_ANCHOR"]:
                ilo_indices = torch.where(ilo_labels == (positive_label % 4))[0]
                ilo_idx = np.random.choice(ilo_indices.cpu().numpy())
                anchor_embedding = ilo_images[ilo_idx].unsqueeze(0)
                anchor_embedding = model.features(anchor_embedding)
                anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                anchor_label = ilo_labels[ilo_idx].item()
            else:
                batch_matching_indices = [j for j in range(len(batch_labels)) if batch_labels[j] == positive_label and j != i]
                if batch_matching_indices:
                    batch_anchor_idx = np.random.choice(batch_matching_indices)
                    anchor_embedding = embeddings[batch_anchor_idx].unsqueeze(0)
                    anchor_label = batch_labels[batch_anchor_idx]
                else:
                    continue

            # Select N1 negatives
            prof_pos_score = positive_label % 4
            if cfg["N1_SELECTION"] == "Profusion-based":
                negative_indices = [j for j, label in enumerate(batch_labels)
                                    if label != positive_label and (label % 4) != prof_pos_score]
            elif cfg["N1_SELECTION"] == "MSTB-based":
                negative_indices = [j for j, label in enumerate(batch_labels)
                                    if label != positive_label]
            else:
                raise ValueError(f"Unsupported N1 selection strategy: {cfg['N1_SELECTION']}")

            if not negative_indices:
                continue

            negative_embeddings = embeddings[negative_indices]
            anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)
            dists = F.pairwise_distance(anchor_repeated, negative_embeddings)
            positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)

            # Semi-hard mask
            semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + current_margin))
            semi_hard_indices = torch.nonzero(semi_hard_mask).squeeze(1)

            if semi_hard_indices.numel() == 0:
                continue

            for idx in semi_hard_indices:
                negative_embedding = negative_embeddings[idx].unsqueeze(0)
                negative_label = batch_labels[negative_indices[idx]]

                # Select N2 negatives
                if cfg["N2_SELECTION"] == "Proposed":
                    negative_indices_2 = [j for j, label in enumerate(batch_labels)
                                          if (label % 4) == (negative_label % 4) and
                                          (int(label >= 4) != int(negative_label >= 4))]
                elif cfg["N2_SELECTION"] == "Original":
                    negative_indices_2 = [j for j, label in enumerate(batch_labels)
                                          if label != positive_label and label != negative_label]
                else:
                    raise ValueError(f"Unsupported N2 selection strategy: {cfg['N2_SELECTION']}")

                if not negative_indices_2:
                    continue

                for neg2_idx in negative_indices_2:
                    negative_embedding_2 = embeddings[neg2_idx].unsqueeze(0)
                    negative_label_2 = batch_labels[neg2_idx]

                    # Append all quadruplets
                    anchors.append(anchor_embedding)
                    positives.append(positive_embedding)
                    negatives.append(negative_embedding)
                    negatives_2.append(negative_embedding_2)

                print(f"ANCHORS: {len(anchors)}, POSITIVES: {len(positives)}, NEGATIVES: {len(negatives)}, NEGATIVES_2: {len(negatives_2)}")
    batch_anchors = torch.cat(anchors, dim=0)
    batch_positives = torch.cat(positives, dim=0)
    batch_negatives = torch.cat(negatives, dim=0)
    batch_negatives_2 = torch.cat(negatives_2, dim=0)
    return batch_anchors, batch_positives, batch_negatives, batch_negatives_2


def evaluate_model(model, device, dataloader, quadruplet_loss_fn, clf_loss_fn, ilo_images, ilo_labels, current_margin, epoch, name, cfg, fold=None):
    print(f"EVALUATING on :{name} - fold {fold}\n")

    clf_metrics_dict = {}
    cl_metrics_dict = {}

    all_labels = []
    all_preds = []
    all_probs = []
    all_original_labels = []

    total_quad_loss, total_clf_loss = 0.0, 0.0

    
    model.eval()

    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(dataloader, desc=f"Evaluating {name} - fold {fold}"):
            batch_imgs = batch_imgs.to(device)
            batch_cpu_labels = batch_labels

            batch_labels = batch_labels.to(device)

            feats = model.features(batch_imgs)
            embeddings = F.normalize(feats, p=2, dim=1)

            anchors, positives, negatives, negatives_2, mining_stats = mine_quadruplets(model, device, batch_cpu_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels)

            quad_loss = quadruplet_loss_fn(anchors, positives, negatives, negatives_2) if len(anchors) != 0 else torch.tensor(0.0, device=device)
            
            clf_results = cl_utils.compute_classification_loss(model, embeddings, batch_labels, clf_loss_fn, active_classifier="multiclass_profusion")

            pred_labels = torch.argmax(clf_results['predictions'].detach().cpu(), dim=1)
            pred_probs = torch.softmax(clf_results['predictions'].detach().cpu(), dim=1)
            gt_labels = clf_results['prof_labels'].detach().cpu()
            
            all_labels.append(gt_labels.numpy())
            all_preds.append(pred_labels.numpy())
            all_probs.append(pred_probs.numpy())
            all_original_labels.append(batch_cpu_labels)

            
            total_clf_loss += clf_results['loss'].item()
            total_quad_loss += quad_loss.item()

        all_original_labels = np.concatenate(all_original_labels)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

        avg_quad_loss = total_quad_loss / len(dataloader)
        avg_clf_loss = total_clf_loss / len(dataloader)

        global_conf_mat = metrics.confusion_matrix(all_labels, all_preds, labels=[i for i in range(4)])

       # print(f"\n\nCM: {global_conf_mat}")

        #print(f"ALL PREDS: {np.unique(all_preds)} \n ALL PROBS: {np.unique(all_probs)}\n ALL LABELS: {np.unique(all_labels)}")

        if cfg["USE_MULTICLASS"] or cfg["NUM_CLASSES"] == 2:
            spec_at_90_sens, thresh = metrics.multiclass_specificity_at_sensitivity(all_labels, all_probs, min_sens=0.9)

           # print(f"SPECIFICITY at 90% SENSITIVITY: {spec_at_90_sens}\n\n THRESHOLD: {thresh}")

            predictions = (all_probs > thresh)

            # print(f"OPTIMAL PREDS: {predictions}")


        accuracy = metrics.get_accuracy(global_conf_mat)
        sensitivity = metrics.get_sensitivity(global_conf_mat)
        specificity = metrics.get_specificity(global_conf_mat)
        f1 = metrics.get_f1_score(global_conf_mat)
        kappa = metrics.get_cohens_kappa(global_conf_mat)

        comb_cm = plot_combined_conf_mat(global_conf_mat)

        if(cfg["LABELS_KEY_MBOD"] == "multiclass_stb"):
            tb_stratified_cm = plot_tb_stratified_binary_cm(all_labels, all_preds, all_original_labels)

            #print(f"TB STRATIFIED CM: {tb_stratified_cm}")

        # Convert multiclass data to binary format
        if all_probs.ndim > 1:  # If we have multiclass probabilities
            # Convert labels to binary (0 vs any profusion)
            binary_labels = (all_labels > 0).astype(np.int32)
            
            # Sum probabilities for non-zero classes (classes 1, 2, 3)
            binary_probs = all_probs[:, 1:].sum(axis=1)
        else:
            binary_labels = all_labels
            binary_probs = all_probs

        # Now call with binary format data
        bin_spec_at_90_sens, bin_thresh = metrics.specificity_at_sensitivity(binary_labels, binary_probs, 0.9)

        if global_conf_mat.shape != (2,2):
            tn, fp, fn, tp = metrics.get_cm_for_class(global_conf_mat, 0)
            bin_cm = np.array([[tp, fn],
                  [fp, tn]])
           # print(f"Binary CM: {bin_cm}")
            bin_acc = metrics.get_accuracy(bin_cm)
            bin_sens = metrics.get_sensitivity(bin_cm)
            bin_spec = metrics.get_specificity(bin_cm)
            bin_f1 = metrics.get_f1_score(bin_cm)
            bin_kappa = metrics.get_cohens_kappa(bin_cm)
        
    
    clf_metrics_dict = {
        "avg_quad_loss": avg_quad_loss,
        "avg_clf_loss": avg_clf_loss,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "kappa": kappa,
        "bin_accuracy": bin_acc,
        "bin_sensitivity": bin_sens,
        "bin_specificity": bin_spec,
        "bin_f1": bin_f1,
        "bin_kappa": bin_kappa,
        "spec_at_90_sens": spec_at_90_sens,
        "bin_spec_at_90_sens": bin_spec_at_90_sens,
        "bin_thresh": bin_thresh,
        # "threshold": thresh
    }

    return clf_metrics_dict, comb_cm, tb_stratified_cm

def train(model, device, cfg, fold, clf_loss_fn, quadruplet_loss_fn):
    preprocess = transforms.Compose([
    # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    augmentations_list = transforms.Compose([
    transforms.RandomRotation(degrees=10, expand=False, fill=0),
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
    ])


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

    mbod_dataset = HDF5Dataset(
        hdf5_path = cfg["DATA_PATH_MBOD"],
        labels_key=cfg["LABELS_KEY_MBOD"],
        images_key="images",
        augmentations=None,
        preprocess=preprocess,
    )

    ilo_dataset = HDF5Dataset(
        hdf5_path = cfg["DATA_PATH_ILO"],
        labels_key=cfg["LABELS_KEY_ILO"],
        images_key="images",
        augmentations=None,
        preprocess=preprocess,
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

    if cfg["LABELS_KEY_MBOD"] == "multiclass_stb":
        num_classes = 8
    elif cfg["LABELS_KEY_MBOD"] == "profusion_score":
        num_classes = 4
    else:
        raise ValueError(f"Unsupported labels key: {cfg['LABELS_KEY_MBOD']}")
    train_loader_mbod, _, _ = get_dataloaders(
        hdf5_path=cfg["DATA_PATH_MBOD"],
        preprocess=preprocess,
        batch_size=cfg["BATCH_SIZE"],
        labels_key=cfg["LABELS_KEY_MBOD"],
        split_file=cfg["SPLIT_FILE_MBOD"],
        augmentations=augmentations_list,
        oversample=cfg["OVERSAMPLE"],
    )

    train_loader_mbod_viz, _, _ = get_dataloaders(
        hdf5_path=cfg["DATA_PATH_MBOD"],
        preprocess=preprocess,
        batch_size=cfg["BATCH_SIZE"],
        labels_key=cfg["LABELS_KEY_MBOD"],
        split_file=cfg["SPLIT_FILE_MBOD"],
        augmentations=augmentations_list,
        oversample=False,
    )

    _, val_loader_mbod, test_loader_mbod = get_dataloaders(
        hdf5_path=cfg["DATA_PATH_MBOD"],
        preprocess=preprocess,
        batch_size=cfg["BATCH_SIZE"],
        labels_key=cfg["LABELS_KEY_MBOD"],
        split_file=cfg["SPLIT_FILE_MBOD"],
        augmentations=None,
        oversample=False,
    )

    classifier_optimizer = torch.optim.Adam(model.mc_prof_clf.parameters(), 
                                            lr=1e-3,  
                                            weight_decay=1e-4)

    encoder_optimizer = torch.optim.Adam(model.model.parameters(), 
                                          lr=cfg["LEARNING_RATE"],  
                                          weight_decay=cfg["LEARNING_RATE"])
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    best_val_specificity = 0

    patience = 50
    min_delta = 0.001
    epochs_without_improvement = 0

    for epoch in tqdm(range(cfg["EPOCHS"]), desc=f"Training - fold {fold}"):
        model.train()
        total_quad_loss = 0.0
        total_clf_loss = 0.0
        total_loss_val = 0.0
        num_quadruplets, ap_failures, shn_failures, n2_failures = 0, 0, 0, 0
        all_labels, all_preds, all_probs, all_original_labels = [], [], [], []

        if cfg["MARGIN_SCHEDULING"]:
            current_margin = cl_utils.get_sin_scheduled_margin(epoch, True, cfg["INITIAL_MARGIN"], cfg["FINAL_MARGIN"], cfg["EPOCHS"], cfg["SCHEDULING_FRACTION"])
            quadruplet_loss_fn.margin1 = current_margin
            quadruplet_loss_fn.margin2 = current_margin * cfg["MARGIN_BETA_FACTOR"]

        for batch_idx, sample in enumerate(train_loader_mbod):

            batch_imgs = sample[0].to(device)
            batch_labels = sample[1].to(device)
            batch_cpu_labels = batch_labels.cpu().detach().numpy()


            feats = model.features(batch_imgs)
            embeddings = F.normalize(feats, p=2, dim=1)

            anchors, positives, negatives, negatives_2, mining_stats = mine_quadruplets(model, device, batch_cpu_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels)

            num_quadruplets += mining_stats["quadruplet_count"]

            if(len(anchors) > 0):
                quad_loss = quadruplet_loss_fn(anchors, positives, negatives, negatives_2)
                total_quad_loss += quad_loss.item()
            else:
                quad_loss = torch.tensor(0.0, device=device)
                total_quad_loss += quad_loss.item()

            clf_results = cl_utils.compute_classification_loss(model, embeddings, batch_labels, clf_loss_fn, active_classifier="multiclass_profusion")

            pred_labels = torch.argmax(clf_results['predictions'].detach().cpu(), dim=1)
            pred_probs = torch.softmax(clf_results['predictions'].detach().cpu(), dim=1)
            gt_labels = clf_results['prof_labels'].detach().cpu()
            
            all_labels.append(gt_labels.numpy())
            all_preds.append(pred_labels.numpy())
            all_probs.append(pred_probs.numpy())
            all_original_labels.append(batch_cpu_labels)

            loss_val = quad_loss + cfg["LAMBDA_CLF"] * clf_results['loss']

            total_loss_val += loss_val.item()
            total_clf_loss += clf_results['loss'].item()
            total_quad_loss += quad_loss.item()
            
            encoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            loss_val.backward()
            encoder_optimizer.step()
            classifier_optimizer.step()

        print(f"EPOCH: {epoch + 1}, Total Loss: {total_loss_val / len(train_loader_mbod)}, "
              f"CLF Loss: {total_clf_loss / len(train_loader_mbod)}, "
              f"Quad Loss: {total_quad_loss / len(train_loader_mbod)}")
        
        train_dict, comb_cm, tb_stratified_cm = evaluate_model(model, device, train_loader_mbod, quadruplet_loss_fn, clf_loss_fn, ilo_images, ilo_labels, margin_1, epoch, "Train", cfg, fold=fold)


        wandb.log({
            "train/avg_quad_loss": train_dict["avg_quad_loss"],
            "train/avg_clf_loss": train_dict["avg_clf_loss"],
            "train/accuracy": train_dict["accuracy"],
            "train/specificity": train_dict["specificity"],
            "train/sensitivity": train_dict["sensitivity"],
            "train/f1": train_dict["f1"],
            "train/kappa": train_dict["kappa"],
            "train/spec_at_90_sens": train_dict["spec_at_90_sens"],
            # "train/threshold": train_dict["threshold"],
            
            "train/bin_spec_at_90_sens": train_dict["bin_spec_at_90_sens"],
            "train/bin_threshold": train_dict["bin_thresh"],
            "train/bin_accuracy": train_dict["bin_accuracy"],
            "train/bin_specificity": train_dict["bin_specificity"],
            "train/bin_sensitivity": train_dict["bin_sensitivity"],
            "train/bin_f1": train_dict["bin_f1"],
            "train/bin_kappa": train_dict["bin_kappa"],
            "cm/train": wandb.Image(comb_cm),
            "cm/train_tb_stratified": wandb.Image(tb_stratified_cm),
            "current_margin": quadruplet_loss_fn.margin1,
            "current_margin_2": quadruplet_loss_fn.margin2
        }, step=epoch)

        val_dict, val_comb_cm, val_tb_stratified_cm = evaluate_model(model, device, val_loader_mbod, quadruplet_loss_fn, clf_loss_fn, ilo_images, ilo_labels, current_margin, epoch, "Validation", cfg, fold=fold)
        wandb.log({
            "val/avg_quad_loss": val_dict["avg_quad_loss"],
            "val/avg_clf_loss": val_dict["avg_clf_loss"],
            "val/accuracy": val_dict["accuracy"],
            "val/specificity": val_dict["specificity"],
            "val/sensitivity": val_dict["sensitivity"],
            "val/f1": val_dict["f1"],
            "val/kappa": val_dict["kappa"],
            "val/spec_at_90_sens": val_dict["spec_at_90_sens"],
            # "val/threshold": val_dict["threshold"],

            "val/bin_spec_at_90_sens": val_dict["bin_spec_at_90_sens"],
            "val/bin_threshold": val_dict["bin_thresh"],
            "val/bin_accuracy": val_dict["bin_accuracy"],
            "val/bin_specificity": val_dict["bin_specificity"],
            "val/bin_sensitivity": val_dict["bin_sensitivity"],
            "val/bin_f1": val_dict["bin_f1"],
            "val/bin_kappa": val_dict["bin_kappa"],
            "cm/val": wandb.Image(val_comb_cm),
            "cm/val_tb_stratified": wandb.Image(val_tb_stratified_cm)
        }, step=epoch)

        if (epoch +  1) % cfg["TSNE_INTERVAL"] == 0:

            visualize_tsne(model, device, ilo_dataset, train_loader_mbod_viz, 
                            trained=True, log_to_wandb=True, 
                            n_epochs=epoch+1, set_name="training", entire_dataset=False)
            visualize_tsne(model, device, ilo_dataset, val_loader_mbod, 
                            trained=True, log_to_wandb=True,
                            n_epochs=epoch+1, set_name="validation", entire_dataset=False)
        # scheduler to do
    
    test_dict, test_comb_cm, test_tb_stratified_cm = evaluate_model(model, device, test_loader_mbod, quadruplet_loss_fn, clf_loss_fn, ilo_images, ilo_labels, current_margin, epoch, "Test", cfg, fold=fold)
    return test_dict, test_comb_cm, test_tb_stratified_cm
try:
    cfg = load_config("cl_config.yaml")

    defaults = load_config("defaults.yaml")
    wandb_api_key = defaults["WANDB_API_KEY"]["value"]
    random_seed = defaults["RANDOM_SEED"]["value"]
    set_random_seeds(seed=random_seed)  # Set a fixed seed for reproducibility


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")



    if cfg["LABELS_KEY_MBOD"] == "multiclass_stb":
        num_classes = 8
    elif cfg["LABELS_KEY_MBOD"] == "profusion_score":
        num_classes = 4
    else:
        raise ValueError(f"Unsupported labels key: {cfg['LABELS_KEY_MBOD']}")

    experiment_name = cfg["RUN_NAME"]

    group_name = f"{experiment_name}-{wandb.util.generate_id()}"

    accuracies = []
    sensitivities = []
    specificities = []
    kappas = []

    for k in range(cfg["NUM_FOLDS"]):

        if (k > 0):
            random_seed = random.randint(0, 100)
            print(f"RANDOM SEED: {random_seed} for FOLD: {k}")
            set_random_seeds(random_seed)
        
        

        model = xrv.models.ResNet(weights="resnet50-res512-all")

        model.mc_prof_clf = XRVBasedClassifier(input_dim=2048, num_classes=4, name="XRV-Base")

        model = model.to(device) 

        # model.mc_prof_clf = cl_utils.ShallowMulticlassClassifier(input_dim=2048, num_classes=4, name="MC-Prof", dropout_rate=0.5) # TO DO: Check shallower classifier

        margin_1 = cfg["INITIAL_MARGIN"]
        margin_2 = cfg["INITIAL_MARGIN"] * cfg["MARGIN_BETA_FACTOR"]

        quadruplet_loss_fn = cl_utils.QuadrupletMarginLoss(margin1=margin_1, margin2=margin_2, p=2, type="Original")
        clf_loss_fn = nn.CrossEntropyLoss()

        wandb.init(
            project = cfg["PROJECT_NAME"],
            name = f"{experiment_name}-fold_{k}",
            group=group_name,
            config={
                "learning_rate": cfg["LEARNING_RATE"],
                "batch_size": cfg["BATCH_SIZE"],
                "num_classes": num_classes,
                "experiment_name": experiment_name,
                "fold": k,
                "random_seed": random_seed
            }
        )

        test_dict, test_comb_cm, test_tb_stratified_cm = train(model, device, cfg, k, clf_loss_fn, quadruplet_loss_fn)

        accuracies.append(test_dict["accuracy"])
        sensitivities.append(test_dict["sensitivity"])
        specificities.append(test_dict["specificity"])
        kappas.append(test_dict["kappa"])

        wandb.log({
            "test/avg_quad_loss": test_dict["avg_quad_loss"],
            "test/avg_clf_loss": test_dict["avg_clf_loss"],
            "test/accuracy": test_dict["accuracy"],
            "test/specificity": test_dict["specificity"],
            "test/sensitivity": test_dict["sensitivity"],
            "test/f1": test_dict["f1"],
            "test/kappa": test_dict["kappa"],
            "test/spec_at_90_sens": test_dict["spec_at_90_sens"],
            # "test/threshold": test_dict["threshold"],

            "test/bin_spec_at_90_sens": test_dict["bin_spec_at_90_sens"],
            "test/bin_threshold": test_dict["bin_thresh"],
            "test/bin_accuracy": test_dict["bin_accuracy"],
            "test/bin_specificity": test_dict["bin_specificity"],
            "test/bin_sensitivity": test_dict["bin_sensitivity"],
            "test/bin_f1": test_dict["bin_f1"],
            "test/bin_kappa": test_dict["bin_kappa"],
            "cm/test": wandb.Image(test_comb_cm),
            "cm/test_tb_stratified": wandb.Image(test_tb_stratified_cm)
        }, step=k)

        wandb.finish()


    mean_acc = np.mean(accuracies)
    mean_sens = np.mean(sensitivities)
    mean_spec = np.mean(specificities)
    mean_kappa = np.mean(kappas)

    print(f"Mean - Acc: {mean_acc}, Sens: {mean_sens}, Spec: {mean_spec}, Kappa: {mean_kappa}")



except KeyError as e:
    print(f"Missing configuration: {e}")
