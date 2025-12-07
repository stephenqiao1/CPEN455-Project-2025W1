import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

class avg_acc_logger:
    def __init__(self):
        self.total_samples = 0
        self.correct_samples = 0

    def update(self, is_correct):
        self.total_samples += len(is_correct)
        self.correct_samples += is_correct.sum().item()

    def compute_accuracy(self):
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples

class avg_logger:
    def __init__(self):
        self.total_count = 0
        self.total_value = 0.0

    def update(self, value, count=1):
        self.total_value += value * count
        self.total_count += count

    def compute_average(self):
        if self.total_count == 0:
            return 0.0
        return self.total_value / self.total_count

class comprehensive_metrics_logger:
    """
    Comprehensive metrics logger that computes:
    - Accuracy
    - Precision, Recall, F1-score (macro and per-class)
    - Confusion matrix
    - ROC-AUC
    - Per-class accuracy
    """
    def __init__(self):
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []  # Store probabilities for ROC-AUC
        
    def update(self, labels, predictions, probs=None):
        """
        Update with batch results.
        
        Args:
            labels: True labels (tensor or list)
            predictions: Predicted labels (tensor or list)
            probs: Predicted probabilities [batch_size, num_classes] (optional, for ROC-AUC)
        """
        if labels is None or predictions is None:
            return
            
        # Convert to numpy if needed
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        elif isinstance(predictions, list):
            predictions = np.array(predictions)
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        elif probs is not None and isinstance(probs, list):
            probs = np.array(probs)
        
        # Ensure 1D arrays
        if labels.ndim > 1:
            labels = labels.flatten()
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        self.all_labels.extend(labels.tolist() if hasattr(labels, 'tolist') else labels)
        self.all_predictions.extend(predictions.tolist() if hasattr(predictions, 'tolist') else predictions)
        
        if probs is not None:
            # Ensure probs is 2D [batch_size, num_classes]
            if probs.ndim == 1:
                # If 1D, assume it's predictions and create one-hot
                probs = None  # Skip ROC-AUC if we don't have proper probabilities
            elif probs.ndim == 2:
                if len(self.all_probs) == 0:
                    self.all_probs = probs
                else:
                    self.all_probs = np.vstack([self.all_probs, probs])
    
    def compute_metrics(self):
        """
        Compute all metrics.
        
        Returns:
            dict with all computed metrics
        """
        if len(self.all_labels) == 0 or len(self.all_predictions) == 0:
            return {
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "precision_ham": 0.0,
                "recall_ham": 0.0,
                "f1_ham": 0.0,
                "precision_spam": 0.0,
                "recall_spam": 0.0,
                "f1_spam": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "roc_auc": 0.0,
                "accuracy_ham": 0.0,
                "accuracy_spam": 0.0,
            }
        
        labels = np.array(self.all_labels)
        predictions = np.array(self.all_predictions)
        
        # Basic accuracy
        accuracy = (labels == predictions).mean()
        
        # Precision, Recall, F1 (macro average)
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        # Ensure 2x2 shape even if one class is missing
        if cm.shape == (1, 1):
            if labels[0] == 0:
                cm = np.array([[cm[0, 0], 0], [0, 0]])
            else:
                cm = np.array([[0, 0], [0, cm[0, 0]]])
        elif cm.shape == (1, 2) or cm.shape == (2, 1):
            # Handle edge cases
            cm = np.array([[cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0, 
                           cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0],
                          [cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0,
                           cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0]])
        
        # ROC-AUC (if probabilities available)
        roc_auc = 0.0
        if len(self.all_probs) > 0 and self.all_probs.shape[1] >= 2:
            try:
                # Use spam probability (class 1) for ROC-AUC
                if len(self.all_probs.shape) == 2:
                    spam_probs = self.all_probs[:, 1]
                    roc_auc = roc_auc_score(labels, spam_probs)
            except Exception:
                roc_auc = 0.0
        
        # Per-class accuracy
        accuracy_ham = 0.0
        accuracy_spam = 0.0
        ham_mask = labels == 0
        spam_mask = labels == 1
        if ham_mask.sum() > 0:
            accuracy_ham = (predictions[ham_mask] == labels[ham_mask]).mean()
        if spam_mask.sum() > 0:
            accuracy_spam = (predictions[spam_mask] == labels[spam_mask]).mean()
        
        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_ham": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
            "recall_ham": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
            "f1_ham": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            "precision_spam": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
            "recall_spam": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
            "f1_spam": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            "confusion_matrix": cm.tolist(),
            "roc_auc": float(roc_auc),
            "accuracy_ham": float(accuracy_ham),
            "accuracy_spam": float(accuracy_spam),
        }