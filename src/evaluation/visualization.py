"""Visualization utilities for evaluation results."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (AUROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    lr_list: List[float],
    losses: List[float],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training history (learning rate vs validation loss).
    
    Args:
        lr_list: List of learning rates
        losses: List of validation losses
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(6, 6))
    plt.plot(lr_list, losses, color="green", marker='o')
    plt.xlabel('Learning Rate')
    plt.xscale('log')
    plt.ylabel('Validation Loss')
    plt.title('Learning Rate vs Validation Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or ['0', '1'],
                yticklabels=class_names or ['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

