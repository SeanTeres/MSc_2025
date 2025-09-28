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

def plot_per_class_map(class_map, title="Per-Class Mean Average Precision"):
    """
    Plot a bar chart of per-class mAP values.
    
    Args:
        class_map: Dictionary mapping class IDs to mAP values
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = sorted(class_map.keys())
    map_values = [class_map[c] for c in classes]
    
    # Create bar plot
    bars = ax.bar(classes, map_values, color='steelblue', alpha=0.8)
    
    # Add horizontal line for overall mAP
    overall_map = np.mean(list(map_values))
    ax.axhline(y=overall_map, color='r', linestyle='-', label=f'Overall mAP: {overall_map:.3f}')
    
    # Add values above bars
    for bar, val in zip(bars, map_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('mAP')
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(classes)
    
    # Add class names if appropriate
    if len(classes) <= 8:  # For small number of classes
        class_names = {
            0: "Prof 0", 1: "Prof 1", 2: "Prof 2", 3: "Prof 3"
        }
        ax.set_xticklabels([class_names.get(c, str(c)) for c in classes])
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig

def compute_map_per_class(embeddings, labels):
    from sklearn.metrics import average_precision_score
    import numpy as np
    from collections import defaultdict

    # Normalize embeddings (redundant if already normalized)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T)  # [N, N]
    labels = labels.numpy()
    sim_matrix = sim_matrix.numpy()

    class_to_aps = defaultdict(list)

    for i in range(len(labels)):
        current_label = labels[i]
        true = (labels == current_label).astype(np.int32)
        pred = sim_matrix[i]

        # Remove self-comparison
        true = np.delete(true, i)
        pred = np.delete(pred, i)

        if true.sum() == 0:
            continue

        ap = average_precision_score(true, pred)
        class_to_aps[current_label].append(ap)

    # Average APs per class
    class_map = {}
    for class_id, aps in class_to_aps.items():
        if len(aps) > 0:
            class_map[int(class_id)] = np.mean(aps)

    overall_map = np.mean(list(class_map.values()))
    
    return overall_map, class_map

def compute_retrieval_metrics_minimal(embeddings, labels):
    """
    Calculate minimal set of retrieval metrics: mAP, precision@1, recall@5
    
    Args:
        embeddings: Embedding vectors (N, dim)
        labels: Ground truth profusion labels (0-3)
    
    Returns:
        Dictionary with retrieval metrics
    """
    metrics_dict = {}
    
    # Convert to numpy for calculation
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    # Calculate pairwise cosine similarities (faster than Euclidean for normalized vectors)
    similarities = np.dot(embeddings, embeddings.T)  # Assumes normalized embeddings
    
    # Replace diagonal with -inf to exclude self-matches
    np.fill_diagonal(similarities, -np.inf)
    
    # Calculate metrics
    ap_scores = []
    precision_at_1 = []
    recall_at_5 = []
    
    for i in range(len(embeddings)):
        query_label = labels[i]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(-similarities[i])
        
        # Get relevant items (same class as query)
        relevant_indices = np.where(labels == query_label)[0]
        relevant_indices = relevant_indices[relevant_indices != i]
        total_relevant = len(relevant_indices)
        
        if total_relevant == 0:
            continue  # Skip queries with no relevant items
        
        # Calculate metrics
        retrieved_relevant = 0
        ap_score = 0.0
        
        for k, idx in enumerate(sorted_indices):
            rank = k + 1  # 1-based ranking
            is_relevant = (labels[idx] == query_label)
            
            if is_relevant:
                retrieved_relevant += 1
                ap_score += retrieved_relevant / rank
                
            # Precision@1
            if rank == 1:
                precision_at_1.append(1.0 if is_relevant else 0.0)
                
            # Recall@5
            if rank == 5:
                recall_at_5.append(retrieved_relevant / total_relevant)
                
        # Normalize AP by number of relevant items
        if retrieved_relevant > 0:
            ap_score /= total_relevant
            ap_scores.append(ap_score)
    
    # Calculate final metrics
    metrics_dict["mAP"] = np.mean(ap_scores) if ap_scores else 0.0
    metrics_dict["precision@1"] = np.mean(precision_at_1) if precision_at_1 else 0.0
    metrics_dict["recall@5"] = np.mean(recall_at_5) if recall_at_5 else 0.0
    
    return metrics_dict

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


def mine_triplets(model, device, batch_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels, return_single_match=True):
    all_embeddings = []
    all_labels = []
    anchors = []
    positives = []
    negatives = []
    
    batch_shn_failures, batch_ap_failures, batch_triplet_count = 0, 0, 0

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

                anchors.append(anchor_embedding)
                positives.append(positive_embedding)
                negatives.append(negative_embedding)

                batch_triplet_count += 1

            else:
                batch_shn_failures += 1
                continue


    
    mining_stats = {
        "ap_failures": batch_ap_failures,
        "shn_failures": batch_shn_failures,
        "triplet_count": batch_triplet_count
    }
    
    # print(f"triplet COUNT: {batch_triplet_count}")
    if batch_triplet_count > 0:

        batch_anchors = torch.cat(anchors, dim=0)
        batch_positives = torch.cat(positives, dim=0)
        batch_negatives = torch.cat(negatives, dim=0)


        return batch_anchors, batch_positives, batch_negatives, mining_stats
    else:
        # Return empty tensors instead of lists
        empty_tensor = torch.tensor([], device=device)
        return empty_tensor, empty_tensor, empty_tensor, mining_stats


def evaluate_model(model, device, dataloader, triplet_loss_fn, clf_loss_fn, ilo_images, ilo_labels, current_margin, epoch, name, cfg, fold=None):
    print(f"EVALUATING on :{name} - fold {fold}\n")

    clf_metrics_dict = {}
    cl_metrics_dict = {}

    all_labels = []
    all_preds = []
    all_probs = []
    all_original_labels = []
    all_embeddings = []
    total_trip_loss, total_clf_loss = 0.0, 0.0

    
    model.eval()

    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(dataloader, desc=f"Evaluating {name} - fold {fold}"):
            batch_imgs = batch_imgs.to(device)
            batch_cpu_labels = batch_labels

            batch_labels = batch_labels.to(device)

            feats = model.features(batch_imgs)
            embeddings = F.normalize(feats, p=2, dim=1)

            all_embeddings.append(embeddings.detach().cpu())

            anchors, positives, negatives, mining_stats = mine_triplets(model, device, batch_cpu_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels)

            trip_loss = triplet_loss_fn(anchors, positives, negatives) if len(anchors) != 0 else torch.tensor(0.0, device=device)
            
            clf_results = cl_utils.compute_classification_loss(model, embeddings, batch_labels, clf_loss_fn, active_classifier="multiclass_profusion")

            pred_labels = torch.argmax(clf_results['predictions'].detach().cpu(), dim=1)
            pred_probs = torch.softmax(clf_results['predictions'].detach().cpu(), dim=1)
            gt_labels = clf_results['prof_labels'].detach().cpu()
            
            all_labels.append(gt_labels.numpy())
            all_preds.append(pred_labels.numpy())
            all_probs.append(pred_probs.numpy())
            all_original_labels.append(batch_cpu_labels)

            
            total_clf_loss += clf_results['loss'].item()
            total_trip_loss += trip_loss.item()

        all_original_labels = np.concatenate(all_original_labels)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        all_embeddings = torch.cat(all_embeddings, dim=0)

        profusion_labels = all_original_labels % 4
        retrieval_metrics = compute_retrieval_metrics_minimal(all_embeddings, profusion_labels)
        cl_metrics_dict.update(retrieval_metrics)

        overall_map, class_map = compute_map_per_class(all_embeddings, profusion_labels)
        per_class_map_fig = plot_per_class_map(class_map, f"{name} Per-Class mAP (Epoch {epoch})") # Create figure for per-class mAP
        cl_metrics_dict["overall_map"] = overall_map
        cl_metrics_dict["class_map"] = class_map
        cl_metrics_dict["per_class_map_fig"] = per_class_map_fig
    



        avg_trip_loss = total_trip_loss / len(dataloader)
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
        "avg_trip_loss": avg_trip_loss,
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

    return clf_metrics_dict, cl_metrics_dict, comb_cm, tb_stratified_cm

def train(model, device, cfg, fold, clf_loss_fn, triplet_loss_fn):
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
        total_trip_loss = 0.0
        total_clf_loss = 0.0
        total_loss_val = 0.0
        num_triplets, ap_failures, shn_failures, n2_failures = 0, 0, 0, 0
        all_labels, all_preds, all_probs, all_original_labels = [], [], [], []

        if cfg["MARGIN_SCHEDULING"]:
            current_margin = cl_utils.get_sin_scheduled_margin(epoch, True, cfg["INITIAL_MARGIN"], cfg["FINAL_MARGIN"], cfg["EPOCHS"], cfg["SCHEDULING_FRACTION"])
            triplet_loss_fn.margin = current_margin
            triplet_loss_fn.margin2 = current_margin * cfg["MARGIN_BETA_FACTOR"]

        for batch_idx, sample in enumerate(train_loader_mbod):

            batch_imgs = sample[0].to(device)
            batch_labels = sample[1].to(device)
            batch_cpu_labels = batch_labels.cpu().detach().numpy()


            feats = model.features(batch_imgs)
            embeddings = F.normalize(feats, p=2, dim=1)

            anchors, positives, negatives, mining_stats = mine_triplets(model, device, batch_cpu_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels)

            num_triplets += mining_stats["triplet_count"]

            if(len(anchors) > 0):
                trip_loss = triplet_loss_fn(anchors, positives, negatives)
                total_trip_loss += trip_loss.item()
            else:
                trip_loss = torch.tensor(0.0, device=device)
                total_trip_loss += trip_loss.item()

            clf_results = cl_utils.compute_classification_loss(model, embeddings, batch_labels, clf_loss_fn, active_classifier="multiclass_profusion")

            pred_labels = torch.argmax(clf_results['predictions'].detach().cpu(), dim=1)
            pred_probs = torch.softmax(clf_results['predictions'].detach().cpu(), dim=1)
            gt_labels = clf_results['prof_labels'].detach().cpu()
            
            all_labels.append(gt_labels.numpy())
            all_preds.append(pred_labels.numpy())
            all_probs.append(pred_probs.numpy())
            all_original_labels.append(batch_cpu_labels)

            loss_val = trip_loss + cfg["LAMBDA_CLF"] * clf_results['loss']

            total_loss_val += loss_val.item()
            total_clf_loss += clf_results['loss'].item()
            total_trip_loss += trip_loss.item()
            
            encoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            loss_val.backward()
            encoder_optimizer.step()
            classifier_optimizer.step()

        print(f"EPOCH: {epoch + 1}, Total Loss: {total_loss_val / len(train_loader_mbod)}, "
              f"CLF Loss: {total_clf_loss / len(train_loader_mbod)}, "
              f"trip Loss: {total_trip_loss / len(train_loader_mbod)}")
        
        train_dict, train_cl_dict, comb_cm, tb_stratified_cm = evaluate_model(model, device, train_loader_mbod, triplet_loss_fn, clf_loss_fn, ilo_images, ilo_labels, current_margin, epoch, "Train", cfg, fold=fold)


        wandb.log({
            "train/avg_trip_loss": train_dict["avg_trip_loss"],
            "train/avg_clf_loss": train_dict["avg_clf_loss"],
            "train/accuracy": train_dict["accuracy"],
            "train/specificity": train_dict["specificity"],
            "train/sensitivity": train_dict["sensitivity"],
            "train/f1": train_dict["f1"],
            "train/kappa": train_dict["kappa"],
            "train/spec_at_90_sens": train_dict["spec_at_90_sens"],
            # "train/threshold": train_dict["threshold"],
            "train/mAP": train_cl_dict["mAP"],
            "train/precision@1": train_cl_dict["precision@1"],
            "train/recall@5": train_cl_dict["recall@5"],
            "train/bin_spec_at_90_sens": train_dict["bin_spec_at_90_sens"],
            "train/bin_threshold": train_dict["bin_thresh"],
            "train/bin_accuracy": train_dict["bin_accuracy"],
            "train/bin_specificity": train_dict["bin_specificity"],
            "train/bin_sensitivity": train_dict["bin_sensitivity"],
            "train/bin_f1": train_dict["bin_f1"],
            "train/bin_kappa": train_dict["bin_kappa"],
            "cm/train": wandb.Image(comb_cm),
            "cm/train_tb_stratified": wandb.Image(tb_stratified_cm),
            "current_margin": triplet_loss_fn.margin,
        }, step=epoch)

        val_dict, val_cl_dict, val_comb_cm, val_tb_stratified_cm = evaluate_model(model, device, val_loader_mbod, triplet_loss_fn, clf_loss_fn, ilo_images, ilo_labels, current_margin, epoch, "Validation", cfg, fold=fold)
        wandb.log({
            "val/avg_trip_loss": val_dict["avg_trip_loss"],
            "val/avg_clf_loss": val_dict["avg_clf_loss"],
            "val/accuracy": val_dict["accuracy"],
            "val/specificity": val_dict["specificity"],
            "val/sensitivity": val_dict["sensitivity"],
            "val/f1": val_dict["f1"],
            "val/kappa": val_dict["kappa"],
            "val/spec_at_90_sens": val_dict["spec_at_90_sens"],
            # "val/threshold": val_dict["threshold"],

            "val/mAP": val_cl_dict["mAP"],
            "val/precision@1": val_cl_dict["precision@1"],
            "val/recall@5": val_cl_dict["recall@5"],
            "val/overall_map": val_cl_dict["overall_map"],
            "val/per_class_map_fig": wandb.Image(val_cl_dict["per_class_map_fig"]),

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
    
    test_dict, test_comb_cm, test_tb_stratified_cm = evaluate_model(model, device, test_loader_mbod, triplet_loss_fn, clf_loss_fn, ilo_images, ilo_labels, current_margin, epoch, "Test", cfg, fold=fold)
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

        triplet_loss_fn = nn.TripletMarginLoss(margin=margin_1, p=2)
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

        test_dict, test_cl_dict, test_comb_cm, test_tb_stratified_cm = train(model, device, cfg, k, clf_loss_fn, triplet_loss_fn)

        accuracies.append(test_dict["accuracy"])
        sensitivities.append(test_dict["sensitivity"])
        specificities.append(test_dict["specificity"])
        kappas.append(test_dict["kappa"])

        wandb.log({
            "test/avg_trip_loss": test_dict["avg_trip_loss"],
            "test/avg_clf_loss": test_dict["avg_clf_loss"],
            "test/accuracy": test_dict["accuracy"],
            "test/specificity": test_dict["specificity"],
            "test/sensitivity": test_dict["sensitivity"],
            "test/f1": test_dict["f1"],
            "test/kappa": test_dict["kappa"],
            "test/spec_at_90_sens": test_dict["spec_at_90_sens"],
            # "test/threshold": test_dict["threshold"],

            "test/mAP": test_cl_dict["mAP"],
            "test/precision@1": test_cl_dict["precision@1"],
            "test/recall@5": test_cl_dict["recall@5"],
            "test/overall_map": test_cl_dict["overall_map"],
            "test/per_class_map_fig": wandb.Image(test_cl_dict["per_class_map_fig"]),


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
