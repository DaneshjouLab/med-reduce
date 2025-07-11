import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pynvml
from .constants import HF_MODELS

# Constants
GPU_AVAILABLE = torch.cuda.is_available()


def env_path(key, default):
    """Get environment variable or default value."""
    return os.environ.get(key, default)


def compute_metrics(eval_pred, model_name):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    # For binary classification, use the probability of the positive class
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    # Use the probability of class 1 (positive class) for ROC AUC
    auc = roc_auc_score(labels, probs[:, 1])

    plot_dir = os.path.join(
        env_path("PLOT_DIR", "."), model_name
    )
    os.makedirs(plot_dir, exist_ok=True)

    conf_mat = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"{model_name}_conf_mat")
    plt.savefig(os.path.join(plot_dir, "conf_mat.png"), dpi=300, bbox_inches="tight")
    plt.close()

    unique, counts = np.unique(predictions, return_counts=True)
    class_breakdown = {str(k): int(v) for k, v in zip(unique, counts)}
    with open(os.path.join(plot_dir, "class_breakdown.json"), "w") as f:
        json.dump(class_breakdown, f)

    return {"accuracy": acc, "f1": f1, "auc": auc}


def get_gpu_memory(device_id=0):
    if not GPU_AVAILABLE:
        return -1
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2
    except:
        return -1
    

def freeze_backbone(model, model_type):
    if model_type in HF_MODELS:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    elif model_type == "simclr":
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")