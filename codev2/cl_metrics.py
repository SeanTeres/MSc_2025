from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


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


def calculate_per_class_metrics(confusion_matrix, prefix, y_true, y_prob, class_names=None):
    # Handle both 2D (multiclass) and 3D (multilabel) confusion matrices
    if len(confusion_matrix.shape) == 2:
        # Multiclass case - convert 2D CM to per-class metrics
        n_classes = confusion_matrix.shape[0]
        
        if class_names is None:
            class_names = [f"Class_{i:02d}" for i in range(n_classes)]
        
        metrics_dict = {}
        
        for i, class_name in enumerate(class_names):
            # Extract TP, FP, FN, TN for class i from multiclass confusion matrix
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp  # Sum of column i minus TP
            fn = confusion_matrix[i, :].sum() - tp  # Sum of row i minus TP
            tn = confusion_matrix.sum() - tp - fp - fn
            
            # Calculate all metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as sensitivity
            sensitivity = recall  # Sensitivity = Recall = TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            support = tp + fn  # actual positives
            
            # For multiclass, convert to one-vs-rest for AUC calculation
            try:
                y_true_binary = (y_true == i).astype(int)
                y_prob_class = y_prob[:, i] if y_prob.ndim > 1 else y_prob
                auc = roc_auc_score(y_true_binary, y_prob_class)
            except:
                auc = 0.0  # Handle cases where AUC can't be calculated
            
            # Log individual class metrics
            metrics_dict.update({
                f"{prefix}precision/{class_name}": precision,
                f"{prefix}sensitivity/{class_name}": sensitivity,
                f"{prefix}specificity/{class_name}": specificity,
                f"{prefix}f1/{class_name}": f1,
                f"{prefix}accuracy/{class_name}": accuracy,
                f"{prefix}auc/{class_name}": auc,
                f"{prefix}support/{class_name}": int(support),
                f"{prefix}tp/{class_name}": int(tp),
                f"{prefix}fp/{class_name}": int(fp),
                f"{prefix}fn/{class_name}": int(fn),
                f"{prefix}tn/{class_name}": int(tn)
            })
    
    elif len(confusion_matrix.shape) == 3:
        # Original multilabel case - keep existing code
        n_classes = confusion_matrix.shape[0]

        if class_names is None:
            class_names = [f"Class_{i:02d}" for i in range(n_classes)]

        metrics_dict = {}

        for i, class_name in enumerate(class_names):
            cm = confusion_matrix[i]
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

            # Calculate all metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as sensitivity
            sensitivity = recall  # Sensitivity = Recall = TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            support = tp + fn  # actual positives
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])

            # Log individual class metrics
            metrics_dict.update({
                f"{prefix}precision/{class_name}": precision,
                f"{prefix}sensitivity/{class_name}": sensitivity,
                f"{prefix}specificity/{class_name}": specificity,
                f"{prefix}f1/{class_name}": f1,
                f"{prefix}accuracy/{class_name}": accuracy,
                f"{prefix}auc/{class_name}": auc,
                f"{prefix}support/{class_name}": int(support),
                f"{prefix}tp/{class_name}": int(tp),
                f"{prefix}fp/{class_name}": int(fp),
                f"{prefix}fn/{class_name}": int(fn),
                f"{prefix}tn/{class_name}": int(tn)
            })
    else:
        raise ValueError("Expected 2D (multiclass) or 3D (multilabel) confusion matrix")

    return metrics_dict