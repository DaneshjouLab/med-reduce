# src/metrics/metrics.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.evaluation.visualization import save_confusion_matrix, save_class_distribution

def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits_t = torch.tensor(logits)
    probs = torch.softmax(logits_t, dim=1).cpu().numpy()
    return probs

def compute_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray],
    model_name: Optional[str] = None,
    average: str = "weighted",
    save_viz: bool = True,
) -> Dict[str, float]:
    """
    Compute accuracy, F1, and AUC (binary or multiclass-ovr).
    - eval_pred: (logits [N,C], labels [N])
    """
    logits, labels = eval_pred
    labels = np.asarray(labels)
    preds = np.argmax(logits, axis=-1)
    probs = _softmax_np(logits)

    # core metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average)

    # AUC: binary if C==2 else OvR; guard for degenerate cases
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average=average)
    except Exception:
        auc = float("nan")

    # optional viz
    if save_viz and model_name:
        try:
            save_confusion_matrix(labels, preds, model_name=model_name, normalize=False)
            save_class_distribution(preds, model_name=model_name)
        except Exception as e:
            print(f"[compute_metrics] Visualization failed: {e}")

    return {"accuracy": float(acc), "f1": float(f1), "auc": float(auc)}

def create_compute_metrics_fn(
    model_name: Optional[str] = None,
    average: str = "weighted",
    save_viz: bool = True,
):
    """Closure for 🤗 Trainer: returns a callable(eval_pred)->metrics dict."""
    def _fn(eval_pred):
        return compute_metrics(eval_pred, model_name=model_name, average=average, save_viz=save_viz)
    return _fn
