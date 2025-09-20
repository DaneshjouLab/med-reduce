"""Training utilities, callbacks, and metrics."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from transformers import TrainerCallback
from typing import Dict, Any, Optional, Tuple
import wandb

from src.utils import env_path, get_gpu_memory

class LossLoggerCallback(TrainerCallback):
    """Log training metrics to structured JSON Lines file."""
    
    def __init__(self, log_dir: str, phase: str, model_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{model_name}_{phase}_log.jsonl")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        with open(self.log_file, "a") as f:
            json.dump({"step": state.global_step, **logs}, f)
            f.write("\n")

class WandbCallback(TrainerCallback):
    """Weights & Biases logging callback."""
    
    def __init__(self, model_name: str, phase: str):
        self.model_name = model_name
        self.phase = phase
        self.best_accuracy = 0.0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Add model name and phase
            logs["model"] = self.model_name
            logs["phase"] = self.phase
            
            # Track GPU memory if available
            gpu_memory = get_gpu_memory()
            if gpu_memory > 0:
                logs["gpu_memory_mb"] = gpu_memory
            
            # Log to wandb
            wandb.log(logs)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # Track best accuracy
            if "eval_accuracy" in metrics:
                self.best_accuracy = max(self.best_accuracy, metrics["eval_accuracy"])
                metrics["best_accuracy"] = self.best_accuracy
            
            # Log evaluation metrics
            wandb.log(metrics)

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray], model_name: Optional[str] = None) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Tuple of (logits, labels)
        model_name: Optional model name for saving visualizations
        
    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute metrics
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    # For binary classification, use probability of positive class
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    auc = roc_auc_score(labels, probs[:, 1])
    
    # Save visualizations if model_name provided
    if model_name:
        save_confusion_matrix(labels, predictions, model_name)
        save_class_distribution(predictions, model_name)
    
    return {"accuracy": acc, "f1": f1, "auc": auc}

def save_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, model_name: str):
    """Save confusion matrix visualization."""
    plot_dir = os.path.join(env_path("PLOT_DIR", "."), model_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    conf_mat = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, "conf_mat.png"), dpi=300, bbox_inches="tight")
    plt.close()

def save_class_distribution(predictions: np.ndarray, model_name: str):
    """Save class distribution to JSON."""
    plot_dir = os.path.join(env_path("PLOT_DIR", "."), model_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    unique, counts = np.unique(predictions, return_counts=True)
    class_breakdown = {str(k): int(v) for k, v in zip(unique, counts)}
    
    with open(os.path.join(plot_dir, "class_breakdown.json"), "w") as f:
        json.dump(class_breakdown, f, indent=4)

def profile_model(model: torch.nn.Module, resolution: int) -> float:
    """
    Profile model to get FLOPs count.
    
    Args:
        model: Model to profile
        resolution: Input resolution
        
    Returns:
        GFLOPs count, or -1 if profiling fails
    """
    try:
        from thop import profile
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = torch.randn(1, 3, resolution, resolution).to(device)
        model.to(device)
        flops, _ = profile(model, inputs=(dummy_input,))
        return flops / 1e9  # Convert to GFLOPs
    except Exception as e:
        print(f"FLOP profiling failed: {e}")
        return -1

def create_compute_metrics_fn(model_name: str):
    """
    Create a compute_metrics function with model_name bound.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Function that computes metrics
    """
    def _compute_metrics(eval_pred):
        return compute_metrics(eval_pred, model_name)
    return _compute_metrics