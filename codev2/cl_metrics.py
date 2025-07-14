import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score


def get_cm_for_class(cm, k):
    TP = TN = FP = FN = 0

    TP = cm[k, k]

    FP = cm[:, k].sum() - TP

    FN = cm[k, :].sum() - TP

    TN = cm.sum() - TP - FP - FN

    return TN, FP, FN, TP


def get_specificity(cm):
    if cm.shape == (2, 2):
        # Binary case
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    elif len(cm.shape) == 3:
        # Multilabel case
        specs = []
        for i in range(len(cm)):
            TN, FP, FN, TP = cm[i].ravel()
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0

            specs.append(spec)

        specificity = np.average(specs)
    else:
        # Multilabel case
        specs = []
        for i in range(len(cm)):
            TN, FP, FN, TP = get_cm_for_class(cm, i)
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0

            specs.append(spec)

        specificity = np.average(specs)

    return specificity


def get_sensitivity(cm):
    if cm.shape == (2, 2):
        # Binary case
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    elif len(cm.shape) == 3:
        # Multilabel case
        senses = []
        for i in range(len(cm)):
            TN, FP, FN, TP = cm[i].ravel()
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0

            senses.append(sens)

        sensitivity = np.average(senses)
    else:
        # Multiclass case
        senses = []
        for i in range(len(cm)):
            TN, FP, FN, TP = get_cm_for_class(cm, i)
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0

            senses.append(sens)

        sensitivity = np.average(senses)

    return sensitivity


def get_f1_score(cm):
    if cm.shape == (2, 2):
        # Binary case
        TN, FP, FN, TP = cm.ravel()
        f1 = TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0
    elif len(cm.shape) == 3:
        # Multilabel case
        f1s = []
        for i in range(len(cm)):
            TN, FP, FN, TP = cm[i].ravel()
            f = TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0

            f1s.append(f)

        f1 = np.average(f1s)
    else:
        # Multiclass case
        f1s = []
        for i in range(len(cm)):
            TN, FP, FN, TP = get_cm_for_class(cm, i)
            f = TP / (TP + 0.5 * (FP + FN)) if (TP + 0.5 * (FP + FN)) > 0 else 0

            f1s.append(f)

        f1 = np.average(f1s)

    return f1


def get_accuracy(cm):
    if len(cm.shape) == 3:
        # Multilabel case
        accs = []
        for i in range(len(cm)):
            TN, FP, FN, TP = cm[i].ravel()
            total = TN + FP + FN + TP
            acc = (TP + TN) / total if total > 0 else 0
            accs.append(acc)

        accuracy = np.average(accs)
    else:
        # Multiclass and binary case
        correct = np.trace(cm)
        total = np.sum(cm)
        accuracy = correct/total

    return accuracy


def specificity_at_sensitivity(y_true, y_prob, min_sens=0.9):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    mask = tpr >= min_sens
    if not np.any(mask):
        return 0.0, 1.0
    idx = np.where(mask)[0][0]
    specificity = 1 - fpr[idx]
    threshold = thresholds[idx]
    return specificity, threshold


def multilabel_specificity_at_sensitivity(y_true, y_prob, min_sens=0.9):
    specs = []
    thresholds = []
    for i in range(y_true.shape[1]):
        spec, thresh = specificity_at_sensitivity(y_true[:, i], y_prob[:, i], min_sens)
        specs.append(spec)
        thresholds.append(thresh)
    return np.mean(specs), np.array(thresholds)


def multiclass_specificity_at_sensitivity(y_true, y_prob, min_sens=0.9):
    specs = []
    thresholds = []
    for i in range(y_prob.shape[1]):
        y_true_bin = (y_true == i).astype(int)
        y_prob_bin = y_prob[:, i]
        spec, thresh = specificity_at_sensitivity(y_true_bin, y_prob_bin, min_sens)
        specs.append(spec)
        thresholds.append(thresh)
    return np.mean(specs), np.array(thresholds)





def calculate_classification_metrics(predictions, labels, task_type="multiclass", binary_target_class=None):
    """
    Calculate comprehensive classification metrics
    
    Args:
        predictions: Model predictions (logits)
        labels: True labels
        task_type: "binary" or "multiclass"
        binary_target_class: If provided, convert multiclass to binary (target_class vs rest)
    
    Returns:
        Dictionary of metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    
    metrics = {}
    
    # Convert multiclass to binary if requested
    if task_type == "multiclass" and binary_target_class is not None:
        # Convert to binary: target_class vs rest
        labels_np = labels.numpy()

        if binary_target_class == "profusion_present":
            # Prof 1-3 = 1 (positive), Prof 0 = 0 (negative)
            y_true_binary = (labels_np > 0).astype(int)
            
            if predictions.dim() > 1:
                probs = torch.softmax(predictions, dim=1).numpy()
                pred_classes = torch.argmax(predictions, dim=1).numpy()
                # Probability of having ANY profusion (Prof 1-3)
                y_prob_binary = 1 - probs[:, 0]  # 1 - P(Prof 0)
            else:
                pred_classes = predictions.numpy()
                y_prob_binary = None
            
            # Binary prediction: Prof 1-3 vs Prof 0
            y_pred_binary = (pred_classes > 0).astype(int)
        else:
            # Original behavior: specific class vs rest
            y_true_binary = (labels_np == binary_target_class).astype(int)
            
            if predictions.dim() > 1:
                probs = torch.softmax(predictions, dim=1).numpy()
                pred_classes = torch.argmax(predictions, dim=1).numpy()
                # Probability of being the target class
                y_prob_binary = probs[:, binary_target_class]
            else:
                pred_classes = predictions.numpy()
                y_prob_binary = None
            
            y_pred_binary = (pred_classes == binary_target_class).astype(int)
        
        
        # Now treat as binary classification
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['kappa'] = cohen_kappa_score(y_true_binary, y_pred_binary)
        
        # AUC (using probabilities)
        try:
            if y_prob_binary is not None:
                metrics['auc'] = roc_auc_score(y_true_binary, y_prob_binary)
            else:
                metrics['auc'] = 0.0
        except:
            metrics['auc'] = 0.0
        
        # Specificity
        try:
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                metrics['specificity'] = 0.0
        except:
            metrics['specificity'] = 0.0
        
        # Specificity @ Sensitivity
        try:
            if y_prob_binary is not None:
                spec_at_sens, threshold = specificity_at_sensitivity(y_true_binary, y_prob_binary, min_sens=0.9)
                metrics['spec_at_sens'] = spec_at_sens
                metrics['threshold_at_sens'] = threshold
            else:
                metrics['spec_at_sens'] = 0.0
                metrics['threshold_at_sens'] = 1.0
        except:
            metrics['spec_at_sens'] = 0.0
            metrics['threshold_at_sens'] = 1.0
        
        # Store the binary confusion matrix for plotting
        metrics['confusion_matrix'] = confusion_matrix(y_true_binary, y_pred_binary)
        metrics['y_true_binary'] = y_true_binary
        metrics['y_pred_binary'] = y_pred_binary
        metrics['y_prob_binary'] = y_prob_binary
        
        return metrics
    
    elif task_type == "binary":
        # Your existing binary code...
        probs = torch.sigmoid(predictions.squeeze()).numpy()
        pred_classes = (probs > 0.5).astype(int)
        labels_np = labels.numpy()
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels_np, pred_classes)
        metrics['f1'] = f1_score(labels_np, pred_classes, zero_division=0)
        metrics['precision'] = precision_score(labels_np, pred_classes, zero_division=0)
        metrics['recall'] = recall_score(labels_np, pred_classes, zero_division=0)
        metrics['kappa'] = cohen_kappa_score(labels_np, pred_classes)
        
        # AUC (using probabilities)
        try:
            metrics['auc'] = roc_auc_score(labels_np, probs)
        except:
            metrics['auc'] = 0.0
        
        # Specificity
        try:
            cm = confusion_matrix(labels_np, pred_classes)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                metrics['specificity'] = 0.0
        except:
            metrics['specificity'] = 0.0
        
        return metrics
    
    else:
        # Your existing multiclass code...
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=1).numpy()
            probs = torch.softmax(predictions, dim=1).numpy()
        else:
            pred_classes = predictions.numpy()
            probs = None
        
        labels_np = labels.numpy()
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels_np, pred_classes)
        metrics['f1'] = f1_score(labels_np, pred_classes, average='weighted', zero_division=0)
        metrics['precision'] = precision_score(labels_np, pred_classes, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(labels_np, pred_classes, average='weighted', zero_division=0)
        metrics['kappa'] = cohen_kappa_score(labels_np, pred_classes)
        
        # AUC (multiclass)
        try:
            if probs is not None:
                metrics['auc'] = roc_auc_score(labels_np, probs, multi_class='ovr', average='weighted')
            else:
                metrics['auc'] = 0.0
        except:
            metrics['auc'] = 0.0
        
        # Specificity (macro average for multiclass)
        try:
            cm = confusion_matrix(labels_np, pred_classes)
            n_classes = cm.shape[0]
            
            specificities = []
            for i in range(n_classes):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp
                
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(specificity)
            
            metrics['specificity'] = np.mean(specificities)
        except:
            metrics['specificity'] = 0.0
    
    return metrics


def calculate_embedding_alignment_metrics(embeddings, labels):
    """
    Calculate embedding alignment metrics including intra-class and inter-class distances
    
    Args:
        embeddings: Normalized embeddings from the model
        labels: Class labels for each embedding
        
    Returns:
        dict: Dictionary containing alignment metrics
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Get unique classes
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    # Initialize metrics
    intra_class_dists = []
    inter_class_dists = []
    
    # Calculate intra-class distances (within same class)
    for class_idx in unique_classes:
        class_mask = labels == class_idx
        class_embeddings = embeddings[class_mask]
        
        if len(class_embeddings) <= 1:
            continue  # Skip classes with only one sample
            
        # Calculate pairwise distances within class
        from sklearn.metrics import pairwise_distances
        class_dists = pairwise_distances(class_embeddings)
        # Get upper triangle (excluding diagonal)
        intra_dists = class_dists[np.triu_indices_from(class_dists, k=1)]
        intra_class_dists.extend(intra_dists)
    
    # Calculate inter-class distances (between different classes)
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            class_i_mask = labels == unique_classes[i]
            class_j_mask = labels == unique_classes[j]
            
            class_i_embeddings = embeddings[class_i_mask]
            class_j_embeddings = embeddings[class_j_mask]
            
            if len(class_i_embeddings) == 0 or len(class_j_embeddings) == 0:
                continue
                
            # Calculate pairwise distances between classes
            from sklearn.metrics import pairwise_distances
            inter_dists = pairwise_distances(class_i_embeddings, class_j_embeddings)
            inter_class_dists.extend(inter_dists.flatten())
    
    # Calculate metrics
    intra_class_mean = np.mean(intra_class_dists) if intra_class_dists else np.nan
    inter_class_mean = np.mean(inter_class_dists) if inter_class_dists else np.nan
    
    # Embedding alignment ratio (higher is better)
    embedding_ratio = inter_class_mean / intra_class_mean if intra_class_mean > 0 else np.nan
    
    # Calculate silhouette score if there are enough samples
    silhouette = np.nan
    davies_bouldin = np.nan
    
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        if len(unique_classes) > 1 and all(np.sum(labels == c) > 1 for c in unique_classes):
            silhouette = silhouette_score(embeddings, labels)
            davies_bouldin = davies_bouldin_score(embeddings, labels)
    except:
        pass
    
    return {
        'intra_class_distance': intra_class_mean,
        'inter_class_distance': inter_class_mean,
        'embedding_ratio': embedding_ratio,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin
    }