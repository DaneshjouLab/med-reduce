# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""Shared classification-metric computation (AUROC + macro F1).

Factored out of ``linear_probe_embedding_engine`` so metric computation is
consistent — single-label (binary / multi-class one-vs-rest) and multi-label
(per-label, uncertain-label masking) settings.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from src.utils.logging_core import get_logger

log = get_logger(__name__)


def compute_auroc_and_f1(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    multi_label: bool = False,
    label_names: Optional[list[str]] = None,
) -> Tuple[float, float, Optional[dict]]:
    """Compute macro AUROC, macro F1, and a per-class AUROC dict.

    Args:
        all_labels: ``[N]`` int labels (single-label) or ``[N, C]`` with values in
            {0, 1, -1} (multi-label; -1 = uncertain/masked).
        all_probs: ``[N, C]`` class probabilities (softmax) or per-label sigmoid
            probabilities for multi-label.
        multi_label: Whether this is a multi-label task.
        label_names: Optional names used as keys in the per-class AUROC dict.

    Returns:
        (val_auroc, val_f1, per_class_auroc_dict). ``per_class_auroc_dict`` is
        ``None`` for the binary single-label case (no per-class breakdown).
    """
    per_class_auroc_dict: Optional[dict] = None

    if multi_label:
        n_labels = all_probs.shape[1]
        per_class_auroc = []
        per_class_auroc_dict = {}
        for i in range(n_labels):
            y_true_i = all_labels[:, i]
            y_score_i = all_probs[:, i]
            # Mask out uncertain labels (-1)
            valid_mask = y_true_i >= 0
            y_true_i = y_true_i[valid_mask]
            y_score_i = y_score_i[valid_mask]
            # Need both 0s and 1s for AUROC
            if len(np.unique(y_true_i)) < 2:
                continue
            try:
                auc_i = roc_auc_score(y_true_i, y_score_i)
                per_class_auroc.append(auc_i)
                name = label_names[i] if label_names and i < len(label_names) else str(i)
                per_class_auroc_dict[name] = float(auc_i)
            except ValueError:
                pass

        val_auroc = float(np.mean(per_class_auroc)) if per_class_auroc else float('nan')

        # Multi-label F1: per-label F1 on valid entries, then macro-average
        per_class_f1 = []
        for i in range(n_labels):
            y_true_i = all_labels[:, i]
            y_pred_i = (all_probs[:, i] > 0.5).astype(int)
            valid_mask = y_true_i >= 0
            y_true_i = y_true_i[valid_mask]
            y_pred_i = y_pred_i[valid_mask]
            if len(y_true_i) == 0 or len(np.unique(y_true_i)) < 1:
                continue
            try:
                f1_i = f1_score(y_true_i, y_pred_i, average='binary', zero_division=0)
                per_class_f1.append(f1_i)
            except ValueError:
                pass
        val_f1 = float(np.mean(per_class_f1)) if per_class_f1 else float('nan')

        return val_auroc, val_f1, per_class_auroc_dict

    # --- Single-label (binary / multi-class one-vs-rest) ---
    unique_labels_in_val = np.unique(all_labels)
    num_classes = all_probs.shape[1]

    if len(unique_labels_in_val) < 2:
        log.warning(
            f"Cannot compute AUROC - only {len(unique_labels_in_val)} class(es) present in validation set "
            f"(classes: {unique_labels_in_val.tolist()}). Returning NaN to exclude from averaging."
        )
        val_auroc = float('nan')
    else:
        try:
            if num_classes == 2:
                val_auroc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                per_class_auroc = []
                per_class_auroc_dict = {}
                for cls in unique_labels_in_val:
                    y_true_binary = (all_labels == cls).astype(int)
                    y_score_cls = all_probs[:, cls]
                    try:
                        cls_auroc = roc_auc_score(y_true_binary, y_score_cls)
                        per_class_auroc.append(cls_auroc)
                        cls_name = label_names[int(cls)] if label_names and int(cls) < len(label_names) else str(int(cls))
                        per_class_auroc_dict[cls_name] = float(cls_auroc)
                    except ValueError:
                        pass

                if per_class_auroc:
                    val_auroc = float(np.mean(per_class_auroc))
                else:
                    val_auroc = float('nan')
        except ValueError as e:
            log.warning(f"Could not compute AUROC: {e}")
            val_auroc = float('nan')

    try:
        all_preds = np.argmax(all_probs, axis=1)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
    except ValueError as e:
        log.warning(f"Could not compute F1: {e}")
        val_f1 = float('nan')

    return val_auroc, val_f1, per_class_auroc_dict
