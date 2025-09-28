# Training script for triplet loss only (no classifier)
# Uses KNN for classification evaluation when best validation mAP is achieved

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
from sklearn.neighbors import KNeighborsClassifier
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

# Add classification to the python path
sys.path.append(os.path.abspath("../classification"))
import metrics
from cross_validation import plot_combined_conf_mat, plot_tb_stratified_binary_cm

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

def evaluate_knn_classifier(train_embeddings, train_labels, val_embeddings, val_labels, k=5):
    """
    Train KNN classifier and evaluate on validation set
    
    Args:
        train_embeddings: Training embeddings (N, dim)
        train_labels: Training labels (N,)
        val_embeddings: Validation embeddings (M, dim)
        val_labels: Validation labels (M,)
        k: Number of neighbors
    
    Returns:
        Dictionary with classification metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(train_embeddings):
        train_embeddings = train_embeddings.detach().cpu().numpy()
    if torch.is_tensor(train_labels):
        train_labels = train_labels.detach().cpu().numpy()
    if torch.is_tensor(val_embeddings):
        val_embeddings = val_embeddings.detach().cpu().numpy()
    if torch.is_tensor(val_labels):
        val_labels = val_labels.detach().cpu().numpy()
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_embeddings, train_labels)
    
    # Make predictions
    val_preds = knn.predict(val_embeddings)
    val_probs = knn.predict_proba(val_embeddings)
    
    # Calculate confusion matrix
    global_conf_mat = confusion_matrix(val_labels, val_preds, labels=[i for i in range(4)])
    
    # Calculate metrics
    accuracy = metrics.get_accuracy(global_conf_mat)
    sensitivity = metrics.get_sensitivity(global_conf_mat)
    specificity = metrics.get_specificity(global_conf_mat)
    f1 = metrics.get_f1_score(global_conf_mat)
    kappa = metrics.get_cohens_kappa(global_conf_mat)
    
    # Binary metrics (0 vs any profusion)
    binary_labels = (val_labels > 0).astype(np.int32)
    binary_probs = val_probs[:, 1:].sum(axis=1)
    
    bin_spec_at_90_sens, bin_thresh = metrics.specificity_at_sensitivity(binary_labels, binary_probs, 0.9)
    
    if global_conf_mat.shape != (2,2):
        tn, fp, fn, tp = metrics.get_cm_for_class(global_conf_mat, 0)
        bin_cm = np.array([[tp, fn], [fp, tn]])
        bin_acc = metrics.get_accuracy(bin_cm)
        bin_sens = metrics.get_sensitivity(bin_cm)
        bin_spec = metrics.get_specificity(bin_cm)
        bin_f1 = metrics.get_f1_score(bin_cm)
        bin_kappa = metrics.get_cohens_kappa(bin_cm)
    
    # Generate confusion matrix plots
    comb_cm = plot_combined_conf_mat(global_conf_mat)
    
    clf_metrics_dict = {
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
        "bin_spec_at_90_sens": bin_spec_at_90_sens,
        "bin_thresh": bin_thresh,
    }
    
    return clf_metrics_dict, comb_cm

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

def mine_triplets(model, device, batch_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels):
    """Mine triplets for contrastive learning"""
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
            if len(ilo_indices) > 0:
                ilo_idx = np.random.choice(ilo_indices.cpu().numpy())
                anchor_embedding = ilo_images[ilo_idx].unsqueeze(0)  # shape [1, C]
                anchor_embedding = model.features(anchor_embedding)  # Get features of the anchor
                anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                anchor_label = ilo_labels[ilo_idx].item()  # Get the label of the anchor
            else:
                batch_ap_failures += 1
                continue
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

            if cfg["N1_SELECTION"] == "Profusion-based":
                negative_indices = [j for j, label in enumerate(batch_labels) 
                                if label != positive_label and (label % 4) != prof_pos_score]     
                                
            elif cfg["N1_SELECTION"] == "MSTB-based":
                negative_indices = [j for j, label in enumerate(batch_labels) 
                                if label != positive_label]
        else:
            raise ValueError(f"Unsupported mining strategy: {cfg['MINING_STRATEGY']}")
        
        if negative_indices:
            negative_embeddings = embeddings[negative_indices]
            anchor_repeated = anchor_embedding.repeat(negative_embeddings.size(0), 1)
            
            # Compute distances
            dists = F.pairwise_distance(anchor_repeated, negative_embeddings)
            positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
            
            # Find semi-hard negatives
            semi_hard_mask = (dists > positive_distance) & (dists < (positive_distance + current_margin))
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
    
    if batch_triplet_count > 0:
        batch_anchors = torch.cat(anchors, dim=0)
        batch_positives = torch.cat(positives, dim=0)
        batch_negatives = torch.cat(negatives, dim=0)
        return batch_anchors, batch_positives, batch_negatives, mining_stats
    else:
        # Return empty tensors instead of lists
        empty_tensor = torch.tensor([], device=device)
        return empty_tensor, empty_tensor, empty_tensor, mining_stats

def evaluate_model(model, device, dataloader, triplet_loss_fn, ilo_images, ilo_labels, current_margin, epoch, name, cfg, fold=None):
    """Evaluate model using only retrieval metrics (no classifier)"""
    print(f"EVALUATING on: {name} - fold {fold}\n")

    all_labels = []
    all_original_labels = []
    all_embeddings = []
    total_triplet_loss = 0.0
    
    model.eval()

    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(dataloader, desc=f"Evaluating {name} - fold {fold}"):
            batch_imgs = batch_imgs.to(device)
            batch_cpu_labels = batch_labels
            batch_labels = batch_labels.to(device)

            feats = model.features(batch_imgs)
            embeddings = F.normalize(feats, p=2, dim=1)

            all_embeddings.append(embeddings.detach().cpu())
            all_original_labels.append(batch_cpu_labels)

            # Mine triplets for loss calculation
            anchors, positives, negatives, mining_stats = mine_triplets(model, device, batch_cpu_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels)
            triplet_loss = triplet_loss_fn(anchors, positives, negatives) if len(anchors) != 0 else torch.tensor(0.0, device=device)
            total_triplet_loss += triplet_loss.item()

        all_original_labels = np.concatenate(all_original_labels)
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Calculate retrieval metrics using profusion labels
        profusion_labels = all_original_labels % 4
        retrieval_metrics = compute_retrieval_metrics_minimal(all_embeddings, profusion_labels)

        avg_triplet_loss = total_triplet_loss / len(dataloader)

    eval_dict = {
        "avg_triplet_loss": avg_triplet_loss,
        "mAP": retrieval_metrics["mAP"],
        "precision@1": retrieval_metrics["precision@1"],
        "recall@5": retrieval_metrics["recall@5"]
    }
    
    return eval_dict, all_embeddings, profusion_labels

def train(model, device, cfg, fold, triplet_loss_fn):
    """Train model using only triplet loss"""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    augmentations_list = transforms.Compose([
        transforms.RandomRotation(degrees=10, expand=False, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
    ])

    # Load datasets
    ilo_dataset = HDF5Dataset(
        hdf5_path=cfg["DATA_PATH_ILO"],
        labels_key=cfg["LABELS_KEY_ILO"],
        images_key="images",
        augmentations=None,
        preprocess=preprocess,
    )

    # Prepare ILO images
    ilo_images = []
    ilo_labels = []

    for idx in range(len(ilo_dataset)):
        image, label = ilo_dataset[idx]
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).to(device)
        label_tensor = torch.tensor(label, dtype=torch.long).to(device)
        ilo_images.append(image_tensor)
        ilo_labels.append(label_tensor)

    ilo_images = torch.cat(ilo_images, dim=0)
    ilo_labels = torch.stack(ilo_labels)

    # Get data loaders
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
        augmentations=None,
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

    # Only encoder optimizer (no classifier)
    encoder_optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=cfg["LEARNING_RATE"],  
                                        weight_decay=cfg["WEIGHT_DECAY"])
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    best_val_mAP = 0
    patience = 50
    min_delta = 0.001
    epochs_without_improvement = 0

    for epoch in tqdm(range(cfg["EPOCHS"]), desc=f"Training - fold {fold}"):
        model.train()
        total_triplet_loss = 0.0
        num_triplets = 0

        if cfg["MARGIN_SCHEDULING"]:
            current_margin = cl_utils.get_sin_scheduled_margin(epoch, True, cfg["INITIAL_MARGIN"], cfg["FINAL_MARGIN"], cfg["EPOCHS"], cfg["SCHEDULING_FRACTION"])
            triplet_loss_fn.margin = current_margin

        for batch_idx, sample in enumerate(train_loader_mbod):
            batch_imgs = sample[0].to(device)
            batch_labels = sample[1].to(device)
            batch_cpu_labels = batch_labels.cpu().detach().numpy()

            feats = model.features(batch_imgs)
            embeddings = F.normalize(feats, p=2, dim=1)

            anchors, positives, negatives, mining_stats = mine_triplets(model, device, batch_cpu_labels, embeddings, cfg, current_margin, ilo_images, ilo_labels)

            num_triplets += mining_stats["triplet_count"]

            if len(anchors) > 0:
                triplet_loss = triplet_loss_fn(anchors, positives, negatives)
                total_triplet_loss += triplet_loss.item()
            else:
                triplet_loss = torch.tensor(0.0, device=device)

            encoder_optimizer.zero_grad()
            triplet_loss.backward()
            encoder_optimizer.step()

        print(f"EPOCH: {epoch + 1}, Triplet Loss: {total_triplet_loss / len(train_loader_mbod)}")
        
        # Evaluate on training set
        train_dict, train_embeddings, train_prof_labels = evaluate_model(model, device, train_loader_mbod_viz, triplet_loss_fn, ilo_images, ilo_labels, current_margin, epoch, "Train", cfg, fold=fold)

        wandb.log({
            "train/avg_triplet_loss": train_dict["avg_triplet_loss"],
            "train/mAP": train_dict["mAP"],
            "train/precision@1": train_dict["precision@1"],
            "train/recall@5": train_dict["recall@5"],
            "current_margin": triplet_loss_fn.margin,
        }, step=epoch)

        # Evaluate on validation set
        val_dict, val_embeddings, val_prof_labels = evaluate_model(model, device, val_loader_mbod, triplet_loss_fn, ilo_images, ilo_labels, current_margin, epoch, "Validation", cfg, fold=fold)
        
        wandb.log({
            "val/avg_triplet_loss": val_dict["avg_triplet_loss"],
            "val/mAP": val_dict["mAP"],
            "val/precision@1": val_dict["precision@1"],
            "val/recall@5": val_dict["recall@5"],
        }, step=epoch)

        # Check if this is the best validation mAP
        if val_dict["mAP"] > best_val_mAP + min_delta:
            best_val_mAP = val_dict["mAP"]
            epochs_without_improvement = 0
            
            print(f"New best validation mAP: {best_val_mAP:.4f} - Running KNN classifier evaluation")
            
            # Run KNN classifier evaluation
            knn_metrics, knn_cm = evaluate_knn_classifier(train_embeddings, train_prof_labels, val_embeddings, val_prof_labels, k=5)
            
            # Log KNN classification metrics
            wandb.log({
                "val_knn/accuracy": knn_metrics["accuracy"],
                "val_knn/sensitivity": knn_metrics["sensitivity"],
                "val_knn/specificity": knn_metrics["specificity"],
                "val_knn/f1": knn_metrics["f1"],
                "val_knn/kappa": knn_metrics["kappa"],
                "val_knn/bin_accuracy": knn_metrics["bin_accuracy"],
                "val_knn/bin_sensitivity": knn_metrics["bin_sensitivity"],
                "val_knn/bin_specificity": knn_metrics["bin_specificity"],
                "val_knn/bin_f1": knn_metrics["bin_f1"],
                "val_knn/bin_kappa": knn_metrics["bin_kappa"],
                "val_knn/bin_spec_at_90_sens": knn_metrics["bin_spec_at_90_sens"],
                "val_knn/bin_thresh": knn_metrics["bin_thresh"],
                "cm/val_knn": wandb.Image(knn_cm),
                "best_val_mAP": best_val_mAP
            }, step=epoch)
            
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Generate t-SNE visualizations
        if (epoch + 1) % cfg["TSNE_INTERVAL"] == 0:
            visualize_tsne(model, device, ilo_dataset, train_loader_mbod_viz, 
                          trained=True, log_to_wandb=True, 
                          n_epochs=epoch+1, set_name="training", entire_dataset=False)
            visualize_tsne(model, device, ilo_dataset, val_loader_mbod, 
                          trained=True, log_to_wandb=True,
                          n_epochs=epoch+1, set_name="validation", entire_dataset=False)
    
    # Final test evaluation
    test_dict, test_embeddings, test_prof_labels = evaluate_model(model, device, test_loader_mbod, triplet_loss_fn, ilo_images, ilo_labels, current_margin, epoch, "Test", cfg, fold=fold)
    
    # Final KNN evaluation on test set
    final_knn_metrics, final_knn_cm = evaluate_knn_classifier(train_embeddings, train_prof_labels, test_embeddings, test_prof_labels, k=5)
    
    return test_dict, final_knn_metrics, final_knn_cm

try:
    cfg = load_config("cl_config.yaml")

    defaults = load_config("defaults.yaml")
    wandb_api_key = defaults["WANDB_API_KEY"]["value"]
    random_seed = defaults["RANDOM_SEED"]["value"]
    set_random_seeds(seed=random_seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    experiment_name = cfg["RUN_NAME"] + "_triplet_only"
    group_name = f"{experiment_name}-{wandb.util.generate_id()}"

    test_maps = []
    test_knn_accuracies = []
    test_knn_sensitivities = []
    test_knn_specificities = []
    test_knn_kappas = []

    for k in range(cfg["NUM_FOLDS"]):
        if k > 0:
            random_seed = random.randint(0, 100)
            print(f"RANDOM SEED: {random_seed} for FOLD: {k}")
            set_random_seeds(random_seed)

        # Load model (only feature extractor, no classifier)
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = model.to(device)

        margin_1 = cfg["INITIAL_MARGIN"]
        triplet_loss_fn = nn.TripletMarginLoss(margin=margin_1, p=2)

        wandb.init(
            project=cfg["PROJECT_NAME"],
            name=f"{experiment_name}-fold_{k}",
            group=group_name,
            config={
                "learning_rate": cfg["LEARNING_RATE"],
                "batch_size": cfg["BATCH_SIZE"],
                "experiment_name": experiment_name,
                "fold": k,
                "random_seed": random_seed,
                "training_mode": "triplet_only"
            }
        )

        test_dict, test_knn_metrics, test_knn_cm = train(model, device, cfg, k, triplet_loss_fn)

        test_maps.append(test_dict["mAP"])
        test_knn_accuracies.append(test_knn_metrics["accuracy"])
        test_knn_sensitivities.append(test_knn_metrics["sensitivity"])
        test_knn_specificities.append(test_knn_metrics["specificity"])
        test_knn_kappas.append(test_knn_metrics["kappa"])

        # Log final test metrics
        wandb.log({
            "test/mAP": test_dict["mAP"],
            "test/precision@1": test_dict["precision@1"],
            "test/recall@5": test_dict["recall@5"],
            "test/avg_triplet_loss": test_dict["avg_triplet_loss"],
            "test_knn/accuracy": test_knn_metrics["accuracy"],
            "test_knn/sensitivity": test_knn_metrics["sensitivity"],
            "test_knn/specificity": test_knn_metrics["specificity"],
            "test_knn/f1": test_knn_metrics["f1"],
            "test_knn/kappa": test_knn_metrics["kappa"],
            "test_knn/bin_accuracy": test_knn_metrics["bin_accuracy"],
            "test_knn/bin_sensitivity": test_knn_metrics["bin_sensitivity"],
            "test_knn/bin_specificity": test_knn_metrics["bin_specificity"],
            "test_knn/bin_f1": test_knn_metrics["bin_f1"],
            "test_knn/bin_kappa": test_knn_metrics["bin_kappa"],
            "test_knn/bin_spec_at_90_sens": test_knn_metrics["bin_spec_at_90_sens"],
            "cm/test_knn": wandb.Image(test_knn_cm)
        }, step=k)

        wandb.finish()

    # Print final results
    mean_mAP = np.mean(test_maps)
    mean_acc = np.mean(test_knn_accuracies)
    mean_sens = np.mean(test_knn_sensitivities)
    mean_spec = np.mean(test_knn_specificities)
    mean_kappa = np.mean(test_knn_kappas)

    print(f"Mean Test Results:")
    print(f"mAP: {mean_mAP:.4f}")
    print(f"KNN - Acc: {mean_acc:.4f}, Sens: {mean_sens:.4f}, Spec: {mean_spec:.4f}, Kappa: {mean_kappa:.4f}")

except KeyError as e:
    print(f"Missing configuration: {e}")