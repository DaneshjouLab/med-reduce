# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
"""Training engine utilities for fine-tuning and linear probing.

This module provides helper functions used by other training engines:
- _accuracy: Computes top-1 accuracy
- _maybe_scheduler_step: Step schedulers based on configuration/meta
- _evaluate: Evaluate model on a loader with loss + accuracy
- _create_grad_scaler: Create a GradScaler with common configuration
- _update_history_and_log: Update training history and log results
- _is_metric_better: Check if current metric is better than best metric

Imported by:
- src.engines.linear_probe_engine
- src.engines.finetune_engine
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

try:
    # PyTorch 2.0+ unified AMP API
    from torch.amp import autocast, GradScaler
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.numel()
    return correct / max(total, 1)


def _maybe_scheduler_step(scheduler_meta: Dict[str, Any], scheduler, *, on: str, metric=None):
    """Step scheduler if meta says so. `on` ∈ {'batch','epoch','val'}."""
    step_when = str(scheduler_meta.get("step_per", "epoch"))
    if step_when == on:
        if "monitor" in scheduler_meta or metric is not None:
            metric_value = metric if metric is not None else scheduler_meta.get("monitor")
            scheduler.step(metric_value)
        else:
            scheduler.step()


def _evaluate(
    model: nn.Module,
    loader,
    loss_fn,
    device: torch.device,
    mixed_precision: bool,
) -> Tuple[float, float]:
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = _preprocess_batch(batch, device)

            # Device-aware autocast (CPU/GPU) and version-safe
            with autocast(
                device_type=getattr(device, "type", "cuda"), enabled=mixed_precision
            ):
                output = model(x)
                if hasattr(output, 'logits'):
                    logits = output.logits
                elif isinstance(output, dict) and 'logits' in output:
                    logits = output['logits']
                else:
                    logits = output
                loss = loss_fn(logits, y)

            bsz = y.size(0)
            running_loss += float(loss.item()) * bsz
            running_acc += _accuracy(logits, y) * bsz
            n += bsz

    return running_loss / max(n, 1), running_acc / max(n, 1)


def _create_grad_scaler(mixed_precision: bool = True) -> GradScaler:
    """Create a GradScaler with common configuration for mixed precision training.

    Args:
        mixed_precision: Whether to enable mixed precision training.

    Returns:
        A configured GradScaler instance.
    """
    try:
        return GradScaler(
            enabled=mixed_precision,
            init_scale=2.0**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
    except TypeError:
        # Fallback for older PyTorch versions that don't support all parameters
        return GradScaler(enabled=mixed_precision)


def _update_history_and_log(  # pylint: disable=too-many-arguments
    *,
    history: Dict[str, List[float]],
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    cur_lr: float,
    val_auroc: Optional[float] = None,
    wandb_logger: Optional[Any] = None,
    log: Optional[Any] = None,
) -> None:
    """Update training history and log results.

    Args:
        history: Dictionary to store training history.
        epoch: Current epoch number.
        train_loss: Training loss for current epoch.
        val_loss: Validation loss for current epoch.
        val_acc: Validation accuracy for current epoch.
        cur_lr: Current learning rate.
        val_auroc: Optional validation AUROC for current epoch.
        wandb_logger: Optional wandb logger instance.
        log: Optional logger instance.
    """
    # Update history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    if val_auroc is not None and "val_auroc" in history:
        history["val_auroc"].append(val_auroc)
    history["lr"].append(cur_lr)

    # Log to wandb if available
    if wandb_logger:
        log_dict = {
            "epoch": epoch,
            "train/loss_epoch": train_loss,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "lr": cur_lr,
        }
        if val_auroc is not None:
            log_dict["val/auroc"] = val_auroc
        wandb_logger.log(log_dict)

    # Log to console if logger is available
    if log:
        if val_auroc is not None:
            log.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_auroc=%.4f | lr=%.2e",
                epoch,
                train_loss,
                val_loss,
                val_acc,
                val_auroc,
                cur_lr,
            )
        else:
            log.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.2e",
                epoch,
                train_loss,
                val_loss,
                val_acc,
                cur_lr,
        )


def _is_metric_better(
    metric_key: str, current_metric: float, best_metric: float
) -> Tuple[bool, float]:
    """Check if current metric is better than best metric.

    Args:
        metric_key: Metric name, if ends with "loss" will minimize, otherwise maximize.
        current_metric: Current metric value.
        best_metric: Best metric value so far.

    Returns:
        Tuple of (is_better, best_metric_value)
    """
    minimize = metric_key.endswith("loss")
    is_better = (
        (current_metric < best_metric) if minimize else (current_metric > best_metric)
    )

    if is_better:
        return True, current_metric
    return False, best_metric


def _preprocess_batch(batch, device):
    """Preprocess a batch by moving data to the appropriate device.

    Args:
        batch: Either a dictionary with 'image'/'label' keys or a tuple (x, y).
        device: The target device for tensors.

    Returns:
        Tuple of (input_tensor, target_tensor) on the specified device.
    """
    if isinstance(batch, dict):
        x, y = batch.get("image", batch.get("pixel_values")), batch.get(
            "label", batch.get("label")
        )
    else:
        x, y = batch

    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _run_validation_and_scheduler(  # pylint: disable=too-many-arguments
    *,
    model: nn.Module,
    loaders: Dict[str, Any],
    loss_fn: Any,
    device: torch.device,
    mixed_precision: bool,
    sched: Any,
    sched_meta: Dict[str, Any],
    metric_key: str,
) -> Tuple[float, float]:
    """Run validation and handle scheduler steps.

    Args:
        model: The model to evaluate.
        loaders: Dictionary of data loaders.
        loss_fn: Loss function.
        device: Target device.
        mixed_precision: Whether to use mixed precision.
        sched: Optional scheduler.
        sched_meta: Scheduler metadata.
        metric_key: Metric key to monitor.

    Returns:
        Tuple of (validation_loss, validation_accuracy).
    """
    # Run validation
    val_loss, val_acc = _evaluate(
        model=model,
        loader=loaders["val"],
        loss_fn=loss_fn,
        device=device,
        mixed_precision=mixed_precision,
    )

    # Handle scheduler steps
    if sched is not None:
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched_meta["monitor"] = val_loss if metric_key.endswith("loss") else val_acc
            _maybe_scheduler_step(sched_meta, sched, on="val")
        else:
            _maybe_scheduler_step(sched_meta, sched, on="epoch")

    return val_loss, val_acc


def _update_best_model_state(
    *,
    model: nn.Module,
    metric_key: str,
    val_loss: float,
    val_acc: float,
    best_metric: float,
) -> Tuple[Dict[str, torch.Tensor], float, bool]:
    """Update best model state if current metric is better.

    Args:
        model: The model to get state from.
        metric_key: Metric key to monitor.
        val_loss: Validation loss.
        val_acc: Validation accuracy.
        best_metric: Best metric value so far.

    Returns:
        Tuple of (best_state_dict, best_metric, is_better).
    """
    monitor = val_loss if metric_key.endswith("loss") else val_acc
    is_better, updated_best_metric = _is_metric_better(metric_key, monitor, best_metric)

    best_state_dict = None
    if is_better:
        best_state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    return best_state_dict, updated_best_metric, is_better

def _get_embeddings(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        mixed_precision: bool = True, # Ignored here but kept for compatibility
        max_samples: Optional[int] = None, # Include this if you want a limit
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts embeddings (pre-logits) and targets by calling the external 
        extract_embeddings utility. Returns Tensors for saving.
        """
        # NOTE: Assuming extract_embeddings is imported or defined locally
        model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
                # Handle different batch formats
                if isinstance(batch, dict):
                    pixel_values = batch['pixel_values'].to(device)
                    batch_labels = batch['label'].to(device)
                else:
                    pixel_values, batch_labels = batch[0].to(device), batch[1].to(device)
                
                # Get embeddings before classifier
                if hasattr(model, 'backbone'):
                    # DINOv3 wrapper
                    outputs = model.backbone(pixel_values=pixel_values)
                    emb = outputs.pooler_output
                elif hasattr(model, 'vit'):
                    # ViT models
                    outputs = model.vit(pixel_values=pixel_values)
                    emb = outputs.last_hidden_state[:, 0]  # CLS token
                elif hasattr(model, 'dinov2'):
                    # DINOv2 models
                    outputs = model.dinov2(pixel_values=pixel_values)
                    emb = outputs.last_hidden_state[:, 0]
                else:
                    # Fallback: use model's forward but extract features
                    outputs = model(pixel_values=pixel_values, output_hidden_states=True)
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        emb = outputs.hidden_states[-1][:, 0]
                    else:
                        raise ValueError("Cannot extract embeddings from this model")
                
                embeddings.append(emb.cpu().numpy())
                labels.append(batch_labels.cpu().numpy())
                
                # Early stop if max_samples reached
                if max_samples and len(embeddings) * emb.shape[0] >= max_samples:
                    break
        
        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)
        
        if max_samples:
            embeddings = embeddings[:max_samples]
            labels = labels[:max_samples]
        
        # Convert NumPy arrays back to Tensors for torch.save in the training loop
        return torch.from_numpy(embeddings), torch.from_numpy(labels)
