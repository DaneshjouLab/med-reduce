# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/metrics/visualization.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.training_utils import env_path

def _plot_dir(model_name: str) -> str:
    out = os.path.join(env_path("PLOT_DIR", "./plots"), model_name)
    os.makedirs(out, exist_ok=True)
    return out

def save_confusion_matrix(
    labels: Iterable[int],
    predictions: Iterable[int],
    model_name: str,
    normalize: bool = False,
    filename: str = "conf_mat.png",
):
    """Save a confusion matrix image (optionally normalized)."""
    labels = np.asarray(labels)
    preds = np.asarray(predictions)
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} — Confusion Matrix" + (" (norm.)" if normalize else ""))
    out = os.path.join(_plot_dir(model_name), filename)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

def save_class_distribution(
    predictions: Iterable[int],
    model_name: str,
    filename: str = "class_breakdown.json",
):
    """Save class histogram of predictions as JSON."""
    preds = np.asarray(predictions)
    unique, counts = np.unique(preds, return_counts=True)
    data = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    out = os.path.join(_plot_dir(model_name), filename)
    with open(out, "w") as f:
        import json
        json.dump(data, f, indent=2)
