# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""
Training engine for linear probing on pre-computed embeddings.

This is Stage 2 of the two-stage approach:
Stage 1: Extract embeddings at each resolution
Stage 2: Train linear classifier on embeddings 
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import math
import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from src.utils.logging_core import get_logger
from src.engines.training_core import (
    _maybe_scheduler_step,
    _create_grad_scaler,
    _update_history_and_log,
)

log = get_logger(__name__)


def train_probe_on_embeddings(
    *,
    classifier: nn.Module,
    loaders: Dict[str, Any],  # {"train": DataLoader, "val": DataLoader}
    loss_fn,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Tuple[Any, Dict[str, Any]]] = None,
    device: torch.device,
    epochs: int,
    grad_clip: Optional[float] = None,
    mixed_precision: bool = True,
    log_interval: int = 50,
    wandb_logger=None,
    metric_key: str = "val_acc",
    label_names: list[str] | None = None,
    multi_label: bool = False,
) -> Dict[str, Any]:
    """
    Train a linear classifier on pre-computed embeddings.

    Args:
        classifier: Linear classifier
        loaders: Dict with 'train' and 'val' DataLoaders for embeddings
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Optional (scheduler, metadata) tuple
        device: Device to train on
        epochs: Number of epochs
        grad_clip: Optional gradient clipping value
        mixed_precision: Whether to use mixed precision
        log_interval: Logging interval
        wandb_logger: Optional WandB logger
        metric_key: Metric to track for best model

    Returns:
        Dict with best_metric, history, and final_lr
    """
    classifier.train()

    scaler = _create_grad_scaler(mixed_precision)

    sched, sched_meta = scheduler or (None, {})
    best_metric = -math.inf if not metric_key.endswith("loss") else math.inf
    best_state_dict = None

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auroc": [], "val_f1": [], "lr": []}

    for epoch in range(1, epochs + 1):
        classifier.train()
        running_loss, n_seen = 0.0, 0

        for step, batch in enumerate(loaders["train"], start=1):
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=mixed_precision):
                logits = classifier(embeddings)
                loss = loss_fn(logits, labels)

            if mixed_precision:
                scaler.scale(loss).backward()

                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
                optimizer.step()

            if sched is not None:
                _maybe_scheduler_step(sched_meta, sched, on="batch")

            running_loss += float(loss.item()) * labels.size(0)
            n_seen += labels.size(0)

            if step % log_interval == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                if wandb_logger:
                    wandb_logger.log({"train/loss": float(loss.item()), "lr": cur_lr})

        train_loss = running_loss / max(n_seen, 1)

        val_loss, val_acc, val_auroc, val_f1, per_class_auroc_dict = _run_validation_on_embeddings(
            classifier=classifier,
            loaders=loaders,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=mixed_precision,
            label_names=label_names,
            multi_label=multi_label,
        )

        if sched is not None:
            if metric_key.endswith("loss"):
                metric_for_scheduler = val_loss
            elif metric_key == "val_auroc":
                metric_for_scheduler = val_auroc
            else:
                metric_for_scheduler = val_acc
            _maybe_scheduler_step(sched_meta, sched, on="epoch", metric=metric_for_scheduler)

        cur_lr = optimizer.param_groups[0]["lr"]
        _update_history_and_log(
            history=history,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics={"val_acc": val_acc, "val_auroc": val_auroc, "val_f1": val_f1},
            cur_lr=cur_lr,
            wandb_logger=wandb_logger,
            log=log,
        )

        if metric_key.endswith("loss"):
            current_metric = val_loss
        elif metric_key == "val_auroc":
            current_metric = val_auroc
        else:
            current_metric = val_acc

        # Skip NaN metrics when determining best model
        if not np.isnan(current_metric):
            is_better = (
                (current_metric < best_metric) if metric_key.endswith("loss")
                else (current_metric > best_metric)
            )

            if is_better:
                best_metric = current_metric
                del best_state_dict  # free previous snapshot before allocating new one
                best_state_dict = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    if best_state_dict is not None:
        classifier.load_state_dict(best_state_dict)

    if best_metric == -math.inf or best_metric == math.inf:
        best_metric = float('nan')

    return {
        "best_metric": best_metric,
        "history": history,
        "final_lr": optimizer.param_groups[0]["lr"],
        "per_class_auroc": per_class_auroc_dict,
    }


def _run_validation_on_embeddings(
    classifier: nn.Module,
    loaders: Dict[str, Any],
    loss_fn,
    device: torch.device,
    mixed_precision: bool = True,
    label_names: list[str] | None = None,
    multi_label: bool = False,
) -> Tuple[float, float, float, float, dict | None]:
    classifier.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loaders["val"]:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            with autocast(device_type=device.type, enabled=mixed_precision):
                logits = classifier(embeddings)
                loss = loss_fn(logits, labels)

            val_loss += float(loss.item()) * labels.size(0)

            if multi_label:
                # Multi-label: sigmoid probabilities, exact-match accuracy
                probs = torch.sigmoid(logits.float())
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).all(dim=1).sum().item()
            else:
                # Single-label: softmax probabilities, argmax accuracy
                probs = torch.softmax(logits.float(), dim=1)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

            val_total += labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    val_loss = val_loss / max(val_total, 1)
    val_acc = val_correct / max(val_total, 1)

    all_labels_list, all_probs_list = all_labels, all_probs
    all_labels = np.concatenate(all_labels_list)
    all_probs = np.concatenate(all_probs_list)
    del all_labels_list, all_probs_list  # free the chunk lists

    per_class_auroc_dict = None

    if multi_label:
        # Multi-label AUROC: per-label, then macro-average
        n_labels = all_probs.shape[1]
        per_class_auroc = []
        per_class_auroc_dict = {}
        for i in range(n_labels):
            y_true_i = all_labels[:, i]
            y_score_i = all_probs[:, i]
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

        # Multi-label F1
        try:
            all_preds = (all_probs > 0.5).astype(int)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
        except ValueError as e:
            log.warning(f"Could not compute F1: {e}")
            val_f1 = float('nan')

    else:
        # Single-label AUROC and F1 (unchanged)
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

        # Compute Macro F1
        try:
            all_preds = np.argmax(all_probs, axis=1)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
        except ValueError as e:
            log.warning(f"Could not compute F1: {e}")
            val_f1 = float('nan')

    return val_loss, val_acc, val_auroc, val_f1, per_class_auroc_dict
