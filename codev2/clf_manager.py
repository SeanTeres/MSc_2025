import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import wandb
import seaborn as sns


class ClassifierBase(nn.Module, ABC):
    """Base class for all classifiers"""
    
    def __init__(self, input_dim: int, num_classes: int, name: str):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.name = name
        
    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        pass

class BinaryClassifier(ClassifierBase):
    """Binary classifier for TB detection or profusion presence"""
    
    def __init__(self, input_dim: int, name: str, dropout_rate: float = 0.3):
        super().__init__(input_dim, 2, name)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # No sigmoid here
        )
        self.loss_fn = nn.BCEWithLogitsLoss()  # More stable
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions.squeeze(), targets.float())
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        probs = torch.sigmoid(predictions.squeeze())
        pred_binary = (probs > 0.5).float()
        
        accuracy = (pred_binary == targets.float()).float().mean()
        
        tp = ((pred_binary == 1) & (targets == 1)).float().sum()
        fp = ((pred_binary == 1) & (targets == 0)).float().sum()
        fn = ((pred_binary == 0) & (targets == 1)).float().sum()
        tn = ((pred_binary == 0) & (targets == 0)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        return {
            f'{self.name}_accuracy': accuracy.item(),
            f'{self.name}_precision': precision.item(),
            f'{self.name}_recall': recall.item(),
            f'{self.name}_f1': f1.item(),
            f'{self.name}_specificity': specificity.item()
        }
    
class XRVBasedClassifier(ClassifierBase):
    "Linear FC as used in their original model"
    "We just change the number of classes and ignore their pretrained weights"
    def __init__(self, input_dim: int, num_classes: int, name: str):
        super().__init__(input_dim, num_classes, name)
        self.fc = nn.Linear(input_dim, num_classes)


    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.CrossEntropyLoss()(predictions, targets.long())
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        pred_classes = torch.argmax(predictions, dim=1)
        accuracy = (pred_classes == targets).float().mean()
        
        metrics = {f'{self.name}_accuracy': accuracy.item()}
        
        # Per-class metrics
        for class_id in range(self.num_classes):
            class_mask = (targets == class_id)
            if class_mask.sum() > 0:
                class_acc = (pred_classes[class_mask] == targets[class_mask]).float().mean()
                metrics[f'{self.name}_class_{class_id}_accuracy'] = class_acc.item()
                tp = ((pred_classes == class_id) & (targets == class_id)).float().sum()
                fp = ((pred_classes == class_id) & (targets != class_id)).float().sum()
                fn = ((pred_classes != class_id) & (targets == class_id)).float().sum()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                metrics[f'{self.name}_class_{class_id}_f1'] = f1.item()
        return metrics
    
class ShallowMulticlassClassifier(ClassifierBase):
    """Shallow multiclass classifier for profusion scores or full MSTB"""
    def __init__(self, input_dim: int, num_classes: int, name: str, dropout_rate: float = 0.3):
        super().__init__(input_dim, num_classes, name)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)  # Output layer for multiclass
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets.long())
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        pred_classes = torch.argmax(predictions, dim=1)
        accuracy = (pred_classes == targets).float().mean()
        metrics = {f'{self.name}_accuracy': accuracy.item()}
        # Per-class metrics (same as original)
        for class_id in range(self.num_classes):
            class_mask = (targets == class_id)
            if class_mask.sum() > 0:
                class_acc = (pred_classes[class_mask] == targets[class_mask]).float().mean()
                metrics[f'{self.name}_class_{class_id}_accuracy'] = class_acc.item()
                tp = ((pred_classes == class_id) & (targets == class_id)).float().sum()
                fp = ((pred_classes == class_id) & (targets != class_id)).float().sum()
                fn = ((pred_classes != class_id) & (targets == class_id)).float().sum()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                metrics[f'{self.name}_class_{class_id}_f1'] = f1.item()
        return metrics

class MulticlassClassifier(ClassifierBase):
    """Multiclass classifier for profusion scores or full MSTB"""
    
    def __init__(self, input_dim: int, num_classes: int, name: str, dropout_rate: float = 0.3):
        super().__init__(input_dim, num_classes, name)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)  # Output layer for multiclass
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets.long())
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        pred_classes = torch.argmax(predictions, dim=1)
        accuracy = (pred_classes == targets).float().mean()
        
        # Per-class accuracy and F1
        metrics = {f'{self.name}_accuracy': accuracy.item()}
        
        # Calculate per-class metrics
        for class_id in range(self.num_classes):
            class_mask = (targets == class_id)
            if class_mask.sum() > 0:
                # Class accuracy
                class_acc = (pred_classes[class_mask] == targets[class_mask]).float().mean()
                metrics[f'{self.name}_class_{class_id}_accuracy'] = class_acc.item()
                
                # Class F1 score
                tp = ((pred_classes == class_id) & (targets == class_id)).float().sum()
                fp = ((pred_classes == class_id) & (targets != class_id)).float().sum()
                fn = ((pred_classes != class_id) & (targets == class_id)).float().sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                metrics[f'{self.name}_class_{class_id}_f1'] = f1.item()
        
        return metrics
    
class ClassifierManager:
    """Manages multiple classifiers for different label schemes"""
    
    def __init__(self, embedding_dim: int, device: torch.device, shared_optimizer: Optional[torch.optim.Optimizer] = None):
        self.embedding_dim = embedding_dim
        self.device = device
        self.classifiers: Dict[str, ClassifierBase] = {}
        self.optimizer: torch.optim.Optimizer = shared_optimizer
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.loss_weights: Dict[str, float] = {}
        
        # Label scheme mappings for MSTB
        self.label_mappings = {
            "binary_tb": {
                "description": "TB Detection (0=No TB, 1=TB)",
                "classes": ["No TB", "TB"]
            },
            "binary_profusion": {
                "description": "Profusion Presence (0=No Profusion, 1=Profusion Present)",
                "classes": ["No Profusion", "Profusion Present"]
            },
            "multiclass_profusion": {
                "description": "Profusion Severity (0-3)",
                "classes": ["Prof 0", "Prof 1", "Prof 2", "Prof 3"]
            },
            "multiclass_mstb": {
                "description": "Full MSTB Classification (0-7)",
                "classes": ["Prof 0, No TB", "Prof 1, No TB", "Prof 2, No TB", "Prof 3, No TB",
                           "Prof 0, TB", "Prof 1, TB", "Prof 2, TB", "Prof 3, TB"]
            }
        }
        
    def add_classifier(self, classifier_type: str, lr: float = 1e-3, weight_decay: float = 1e-4, loss_weight: float = 1.0):
        """
        Add a classifier to the manager
        
        Args:
            classifier_type: One of ['binary_tb', 'binary_profusion', 'multiclass_profusion', 'multiclass_mstb']
            lr: Learning rate for the classifier
            weight_decay: Weight decay for regularization
        """
        if classifier_type not in self.label_mappings:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Create the appropriate classifier
        if classifier_type == "binary_tb":
            classifier = BinaryClassifier(self.embedding_dim, "binary_tb")
        elif classifier_type == "binary_profusion":
            classifier = BinaryClassifier(self.embedding_dim, "binary_profusion")
        elif classifier_type == "multiclass_profusion":
            classifier = MulticlassClassifier(self.embedding_dim, 4, "multiclass_profusion")
        elif classifier_type == "multiclass_mstb":
            classifier = MulticlassClassifier(self.embedding_dim, 8, "multiclass_mstb")
        
        # Move to device and store
        classifier = classifier.to(self.device)
        self.classifiers[classifier_type] = classifier
        self.loss_weights[classifier_type] = loss_weight # Default weight
        
        print(f"Added classifier: {classifier_type}")
        print(f"  - {self.label_mappings[classifier_type]['description']}")
        print(f"  - Classes: {self.label_mappings[classifier_type]['classes']}")
        print(f"  - Loss weight: {loss_weight}")

    def prepare_targets(self, labels: torch.Tensor, classifier_name: str) -> torch.Tensor:
        """
        Convert MSTB labels (0-7) to appropriate targets for each classifier
        
        Args:
            labels: Original MSTB labels (0-7)
            classifier_name: Name of the classifier
            
        Returns:
            Transformed targets for the specific classifier
        """
        if classifier_name == "binary_tb":
            # TB detection: 0-3 -> 0 (No TB), 4-7 -> 1 (TB)
            return (labels >= 4).long()
        
        elif classifier_name == "binary_profusion":
            # Profusion presence: 0,4 -> 0 (No Profusion), 1-3,5-7 -> 1 (Profusion Present)
            return ((labels % 4) > 0).long()
        
        elif classifier_name == "multiclass_profusion":
            # Profusion scores: 0-7 -> 0-3 (extract profusion level)
            return (labels % 4).long()
        
        elif classifier_name == "multiclass_mstb":
            # Full MSTB: 0-7 -> 0-7 (no transformation)
            return labels.long()
        
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")
    
    def train_step(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                   active_classifiers: List[str] = None) -> Dict[str, float]:
        """
        Train all active classifiers on a batch of embeddings
        
        Args:
            embeddings: Batch of embeddings from the encoder
            labels: Batch of MSTB labels
            active_classifiers: List of classifiers to train (None = all)
            
        Returns:
            Dictionary of training metrics
        """
        if active_classifiers is None:
            active_classifiers = list(self.classifiers.keys())
        
        loss_metrics = {}
        total_clf_loss = 0.0
        total_weighted_loss = 0.0
        
        for name in active_classifiers:
            if name not in self.classifiers:
                continue
                
            classifier = self.classifiers[name]
            loss_weight = self.loss_weights.get(name, 1.0)
            
            # Prepare targets
            targets = self.prepare_targets(labels, name)
            
            # Forward pass
            # self.optimizer.zero_grad()
            predictions = classifier(embeddings)
            raw_loss = classifier.compute_loss(predictions, targets)
            
            # Apply individual weighting
            weighted_loss = loss_weight * raw_loss

            total_clf_loss += raw_loss
            total_weighted_loss += weighted_loss
            
            # Compute metrics
            with torch.no_grad():
                loss_metrics[f'{name}_loss'] = raw_loss.item()
                loss_metrics[f'{name}_weighted_loss'] = weighted_loss.item()


        
        # ✅ FIXED: Return tensors for backpropagation
        return {
            'total_clf_loss': total_clf_loss,
            'total_weighted_clf_loss': total_weighted_loss,
            'metrics': loss_metrics
        }    
    def evaluate(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                 active_classifiers: List[str] = None) -> Dict[str, float]:
        """
        Evaluate all active classifiers
        
        Args:
            embeddings: Batch of embeddings
            labels: Batch of MSTB labels
            active_classifiers: List of classifiers to evaluate (None = all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if active_classifiers is None:
            active_classifiers = list(self.classifiers.keys())
        
        loss_metrics = {}
        
        with torch.no_grad():
            for name in active_classifiers:
                if name not in self.classifiers:
                    continue
                    
                classifier = self.classifiers[name]
                targets = self.prepare_targets(labels, name)
                
                predictions = classifier(embeddings)
                loss = classifier.compute_loss(predictions, targets)
                weighted_loss = self.loss_weights.get(name, 1.0) * loss
                
                loss_metrics[f'{name}_loss'] = loss.item()
                loss_metrics[f'{name}_weighted_loss'] = weighted_loss.item()

        return loss_metrics

    def get_predictions(self, embeddings: torch.Tensor, 
                       active_classifiers: List[str] = None) -> Dict[str, torch.Tensor]:
        """Get predictions from all active classifiers"""
        if active_classifiers is None:
            active_classifiers = list(self.classifiers.keys())
        
        predictions = {}
        
        with torch.no_grad():
            for name in active_classifiers:
                if name not in self.classifiers:
                    continue
                predictions[name] = self.classifiers[name](embeddings)
        
        return predictions


    def get_confusion_matrix(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                            classifier_name: str) -> np.ndarray:
        """
        Get confusion matrix for a specific classifier
        
        Args:
            embeddings: All embeddings
            labels: All MSTB labels
            classifier_name: Name of the classifier
            
        Returns:
            Confusion matrix as numpy array
        """
        if classifier_name not in self.classifiers:
            return None
            
        classifier = self.classifiers[classifier_name]
        targets = self.prepare_targets(labels, classifier_name)
        
        with torch.no_grad():
            predictions = classifier(embeddings)
            
            if classifier_name in ["binary_tb", "binary_profusion"]:
                # Binary classification
                probs = torch.sigmoid(predictions.squeeze()).cpu().numpy()
                pred_classes = (probs > 0.5).astype(int)
            else:
                # Multiclass classification
                pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            
            targets_np = targets.cpu().numpy()
            
            # Create confusion matrix
            cm = confusion_matrix(targets_np, pred_classes)
            return cm

    def create_confusion_matrix_plot(self, cm: np.ndarray, classifier_name: str, 
                                    set_name: str = "train", epoch: int = 0):
        """
        Create a confusion matrix plot for wandb logging
        
        Args:
            cm: Confusion matrix
            classifier_name: Name of the classifier
            set_name: "train" or "val" 
            epoch: Current epoch number
            
        Returns:
            matplotlib figure object
        """
        if cm is None:
            return None
            
        # Get class names
        if classifier_name == "binary_profusion":
            class_names = ["No Profusion", "Profusion Present"]
            title = "Binary Profusion Classification"
        elif classifier_name == "binary_tb":
            class_names = ["No TB", "TB"]
            title = "Binary TB Classification"
        elif classifier_name == "multiclass_profusion":
            class_names = ["Prof 0", "Prof 1", "Prof 2", "Prof 3"]
            title = "Multiclass Profusion Classification"
        elif classifier_name == "multiclass_mstb":
            class_names = ["Prof 0, TB-", "Prof 1, TB-", "Prof 2, TB-", "Prof 3, TB-", "Prof 0, TB+", "Prof 1, TB+", "Prof 2, TB+", "Prof 3, TB+"]
            title = "Multiclass MSTB Classification"
        else:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
            title = f"{classifier_name} Classification"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20)
        
        # Set title
        ax.set_title(f'{title}\n{set_name.title()} Set - Epoch {epoch}', fontsize=14, pad=20)
        
        # Set tick labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color=text_color, fontsize=12)
        
        # Set labels
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # Calculate and add accuracy text
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            metrics_text = f'Accuracy: {accuracy*100:.3f}%\nSensitivity: {sensitivity*100:.3f}%\nSpecificity: {specificity*100:.3f}%'
        else:
            accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
            metrics_text = f'Accuracy: {accuracy*100:.3f}%'

        # Add metrics text box
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def log_confusion_matrix_to_wandb(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                                    classifier_name: str, set_name: str = "train", epoch: int = 0):
        """
        Create and log confusion matrix visualization to wandb
        
        Args:
            embeddings: All embeddings
            labels: All MSTB labels
            classifier_name: Name of the classifier
            set_name: "train" or "val" 
            epoch: Current epoch number
        """

        # Get confusion matrix
        cm = self.get_confusion_matrix(embeddings, labels, classifier_name)
        
        if cm is None:
            return
        
        # Create plot
        fig = self.create_confusion_matrix_plot(cm, classifier_name, set_name, epoch)
        
        if fig is not None:
            # Log to wandb
            wandb.log({
                f"{set_name}_{classifier_name}_confusion_matrix": wandb.Image(fig)
            })
            
            # Close the figure to free memory
            plt.close(fig)
            
            print(f"Logged {classifier_name} confusion matrix visualization to wandb ({set_name} set)")

    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates for all classifiers"""
        return {name: optimizer.param_groups[0]['lr'] 
                for name, optimizer in self.optimizers.items()}
    
    def save_state(self, filepath: str):
        """Save all classifier states"""
        state = {
            'classifiers': {name: clf.state_dict() for name, clf in self.classifiers.items()},
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'schedulers': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'embedding_dim': self.embedding_dim,
            'active_classifiers': list(self.classifiers.keys())
        }
        torch.save(state, filepath)
        print(f"Saved classifier manager state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load all classifier states"""
        state = torch.load(filepath, map_location=self.device)
        
        for name, clf_state in state['classifiers'].items():
            if name in self.classifiers:
                self.classifiers[name].load_state_dict(clf_state)
        
        for name, opt_state in state['optimizers'].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(opt_state)
        
        for name, sch_state in state['schedulers'].items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(sch_state)
        
        print(f"Loaded classifier manager state from {filepath}")
        print(f"Active classifiers: {state.get('active_classifiers', [])}")
    
    def _compute_comprehensive_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, name: str) -> Dict[str, float]:
        """
        Compute comprehensive metrics for a classifier
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            name: Classifier name
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy for sklearn
        if name in ["binary_tb", "binary_profusion"]:
            # Binary classification
            probs = torch.sigmoid(predictions.squeeze()).cpu().numpy()
            pred_binary = (probs > 0.5).astype(int)
            targets_np = targets.cpu().numpy()
            
            # Multiclass weighted metrics (treating as 2-class)
            metrics[f'{name}_accuracy'] = accuracy_score(targets_np, pred_binary)
            metrics[f'{name}_f1_weighted'] = f1_score(targets_np, pred_binary, average='weighted', zero_division=0)
            metrics[f'{name}_precision_weighted'] = precision_score(targets_np, pred_binary, average='weighted', zero_division=0)
            metrics[f'{name}_recall_weighted'] = recall_score(targets_np, pred_binary, average='weighted', zero_division=0)
            
            # Binarized metrics (same as weighted for binary)
            metrics[f'{name}_f1_binary'] = f1_score(targets_np, pred_binary, zero_division=0)
            metrics[f'{name}_precision_binary'] = precision_score(targets_np, pred_binary, zero_division=0)
            metrics[f'{name}_recall_binary'] = recall_score(targets_np, pred_binary, zero_division=0)
            
            # Confusion matrix for sensitivity/specificity
            try:
                cm = confusion_matrix(targets_np, pred_binary)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics[f'{name}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    metrics[f'{name}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    metrics[f'{name}_sensitivity'] = 0.0
                    metrics[f'{name}_specificity'] = 0.0
            except:
                metrics[f'{name}_sensitivity'] = 0.0
                metrics[f'{name}_specificity'] = 0.0
            
            # Kappa
            metrics[f'{name}_kappa'] = cohen_kappa_score(targets_np, pred_binary)
            
        else:
            # Multiclass classification
            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Multiclass weighted metrics
            metrics[f'{name}_accuracy'] = accuracy_score(targets_np, pred_classes)
            metrics[f'{name}_f1_weighted'] = f1_score(targets_np, pred_classes, average='weighted', zero_division=0)
            metrics[f'{name}_precision_weighted'] = precision_score(targets_np, pred_classes, average='weighted', zero_division=0)
            metrics[f'{name}_recall_weighted'] = recall_score(targets_np, pred_classes, average='weighted', zero_division=0)
            
            # Binarized metrics (macro average)
            metrics[f'{name}_f1_binary'] = f1_score(targets_np, pred_classes, average='macro', zero_division=0)
            metrics[f'{name}_precision_binary'] = precision_score(targets_np, pred_classes, average='macro', zero_division=0)
            metrics[f'{name}_recall_binary'] = recall_score(targets_np, pred_classes, average='macro', zero_division=0)
            
            # Sensitivity and specificity for multiclass (macro average)
            try:
                cm = confusion_matrix(targets_np, pred_classes)
                n_classes = cm.shape[0]
                
                sensitivities = []
                specificities = []
                
                for i in range(n_classes):
                    tp = cm[i, i]
                    fn = cm[i, :].sum() - tp
                    fp = cm[:, i].sum() - tp
                    tn = cm.sum() - tp - fn - fp
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    sensitivities.append(sensitivity)
                    specificities.append(specificity)
                
                metrics[f'{name}_sensitivity'] = np.mean(sensitivities)
                metrics[f'{name}_specificity'] = np.mean(specificities)
            except:
                metrics[f'{name}_sensitivity'] = 0.0
                metrics[f'{name}_specificity'] = 0.0
            
            # Kappa
            metrics[f'{name}_kappa'] = cohen_kappa_score(targets_np, pred_classes)
        
        return metrics

    def evaluate_epoch(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                    active_classifiers: List[str] = None) -> Dict[str, float]:
        """
        Evaluate all active classifiers with comprehensive metrics
        
        Args:
            embeddings: All embeddings from the epoch
            labels: All MSTB labels from the epoch
            active_classifiers: List of classifiers to evaluate (None = all)
            
        Returns:
            Dictionary of comprehensive evaluation metrics
        """
        if active_classifiers is None:
            active_classifiers = list(self.classifiers.keys())
        
        all_metrics = {}
        
        with torch.no_grad():
            for name in active_classifiers:
                if name not in self.classifiers:
                    continue
                    
                classifier = self.classifiers[name]
                targets = self.prepare_targets(labels, name)
                
                predictions = classifier(embeddings)
                loss = classifier.compute_loss(predictions, targets)
                
                # Store loss
                all_metrics[f'{name}_loss'] = loss.item()
                
                # Compute comprehensive metrics
                metrics = self._compute_comprehensive_metrics(predictions, targets, name)
                all_metrics.update(metrics)
        
        return all_metrics

    def print_summary(self):
        """Print a summary of all active classifiers"""
        print("\n" + "="*80)
        print("CLASSIFIER MANAGER SUMMARY")
        print("="*80)
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Device: {self.device}")
        print(f"Active classifiers: {len(self.classifiers)}")
        
        # Print shared optimizer learning rate
        if self.optimizer:
            print(f"Shared optimizer learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        for name, classifier in self.classifiers.items():
            print(f"\n{name.upper()}:")
            print(f"  - {self.label_mappings[name]['description']}")
            print(f"  - Classes: {self.label_mappings[name]['classes']}")
            print(f"  - Parameters: {sum(p.numel() for p in classifier.parameters()):,}")