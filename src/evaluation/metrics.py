"""Evaluation metrics for model performance."""
import torch
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_probs),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }
    
    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    metrics["thresholds"] = thresholds
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing metrics and predictions
    """
    model.to(device)
    model.eval()
    
    y_true = []
    y_probs = []

    with torch.no_grad():
        for inputs, labels, _, _ in test_loader:
            inputs = inputs.to(device).float()
            outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
            y_probs.extend(outputs)
            y_true.extend(labels.numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Binary predictions (threshold at 0.5)
    y_pred = (y_probs >= 0.5).astype(int)
    
    # Compute all metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    # Add raw predictions
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_probs"] = y_probs
    
    # Print metrics
    print("\n📊 Test Set Evaluation:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"AUC      : {metrics['auc']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_acc']:.4f}")
    
    return metrics

