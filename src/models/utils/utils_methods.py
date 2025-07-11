# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""Utility methods for model training and evaluation.
This module provides functions for computing evaluation metrics, managing GPU memory,
freezing model backbones, and handling environment paths."""

import os
import shutil
import json
import numpy as np
from thop import profile
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
    """
    Compute evaluation metrics from model predictions.
    """
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
    with open(os.path.join(plot_dir, "class_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(class_breakdown, f)

    return {"accuracy": acc, "f1": f1, "auc": auc}


def get_gpu_memory(device_id=0):
    """
    Get the used GPU memory in MB for a specific device.
    """
    if not GPU_AVAILABLE:
        return -1
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2
    except pynvml.NVMLError:
        return -1
    except Exception:   # pylint: disable=broad-exception-caught
        return -1


def freeze_backbone(model, model_type):
    """
    Freeze the backbone of the model based on its type.
    """
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

def get_flops(model, resolution):
    """
    Profile FLOPs for the given model and resolution.

    Args:
        model: The model to profile.
        resolution (int): Input image resolution.

    Returns:
        float: FLOPs in giga units, or -1 if profiling fails.
    """
    try:
        dummy_input = torch.randn(1, 3, resolution, resolution).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy_input,))
        return flops / 1e9
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"FLOP profiling failed: {e}")
        return -1

def check_disk_space(min_gb=1):
    """
    Checks if there is at least min_gb GB of free disk space.

    Args:
        min_gb (int): Minimum required free disk space in GB.

    Raises:
        RuntimeError: If not enough disk space is available.
    """
    total, used, free = shutil.disk_usage("/")
    print(
        f"Disk space: Total={total // (2**30)} GB, "
        f"Used={used // (2**30)} GB, "
        f"Free={free // (2**30)} GB"
    )
    if free < min_gb * (2**30):
        raise RuntimeError(f"Not enough disk space. Please free up at least {min_gb}GB.")

def save_model_and_preprocessor(model, preprocessors, typ, name, learning_rate):
    """
    Saves the trained model and preprocessor to disk.

    Args:
        model: The trained model.
        preprocessors (dict): Preprocessors for each model type.
        typ (str): Model type.
        name (str): Model name.
        learning_rate (float): Learning rate used for training.

    Returns:
        str: Path to the saved model directory.
    """
    model_dir = os.path.join(env_path("MODEL_DIR", "."), f"{name}_lr_{learning_rate}")
    os.makedirs(model_dir, exist_ok=True)
    if typ in HF_MODELS:
        model.save_pretrained(model_dir)
        preprocessors[typ].save_pretrained(model_dir)
    elif typ == "simclr":
        torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_type": "simclr",
                "backbone": "resnet50",
                "num_classes": model.classifier.out_features,
            }, f)
    return model_dir
