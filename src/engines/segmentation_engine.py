# src/engines/segmentation_engine.py
# -*- coding: utf-8 -*-
"""
Training engine for segmentation models.

Provides end-to-end training loop for semantic segmentation tasks.
Unlike classification (which uses two-stage training with embeddings),
segmentation trains the full model including spatial decoder.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import math
import torch
from torch import nn

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
from src.evaluation.segmentation_metrics import RunningSegmentationMetrics

log = get_logger(__name__)


def _run_segmentation_validation(
    model: nn.Module,
    val_loader,
    device: torch.device,
    mixed_precision: bool = True,
    compute_auroc: bool = False,
) -> Dict[str, float]:
    """
    Run validation epoch and compute segmentation metrics.

    Optimized implementation that:
    - Computes metrics incrementally (no memory accumulation)
    - Keeps tensors on GPU during metric computation
    - Avoids expensive CPU transfers per batch

    Args:
        model: Segmentation model
        val_loader: Validation data loader
        device: Device to run on
        mixed_precision: Whether to use mixed precision
        compute_auroc: Whether to compute AUROC (slower, uses more memory)

    Returns:
        Dictionary with:
            - val_loss: Average validation loss
            - val_dice: Dice coefficient
            - val_iou: Intersection over Union
            - val_pixel_acc: Pixel accuracy
            - val_auroc: AUROC (only if compute_auroc=True)
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0

    # Use running metrics accumulator - computes incrementally on GPU
    # This avoids storing all logits/masks in memory
    metrics_accumulator = RunningSegmentationMetrics(
        threshold=0.5, device=device, compute_auroc=compute_auroc
    )

    with torch.no_grad():
        for batch in val_loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            masks = batch["mask_target"].to(device, non_blocking=True)

            batch_size = images.size(0)
            n_samples += batch_size

            with autocast(device_type=device.type, enabled=mixed_precision):
                outputs = model(pixel_values=images, labels=masks)
                loss = outputs.loss
                logits = outputs.logits

            running_loss += loss.item() * batch_size

            # Update running metrics on GPU (avoid CPU transfer)
            metrics_accumulator.update(logits, masks)

    # Compute average loss
    val_loss = running_loss / n_samples

    # Compute final metrics from accumulated statistics
    metrics = metrics_accumulator.compute()

    result = {
        "val_loss": val_loss,
        "val_dice": metrics["dice"],
        "val_iou": metrics["iou"],
        "val_pixel_acc": metrics["pixel_acc"],
    }

    if compute_auroc:
        result["val_auroc"] = metrics["auroc"]

    return result


def train_segmentation(
    *,
    model: nn.Module,
    loaders: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Tuple[Any, Dict[str, Any]]] = None,
    device: torch.device,
    epochs: int,
    grad_clip: Optional[float] = None,
    mixed_precision: bool = True,
    log_interval: int = 50,
    wandb_logger = None,
    metric_key: str = "val_dice",
    save_checkpoints: bool = True,
    checkpoint_dir: Optional[Path] = None,
    compute_auroc: bool = False,
) -> Dict[str, Any]:
    """
    Train a segmentation model end-to-end.

    Args:
        model: Segmentation model (e.g., DINOv3ForSegmentation)
        loaders: Dict with 'train' and 'val' DataLoaders
        optimizer: Optimizer
        scheduler: Optional (scheduler, metadata) tuple
        device: Device to train on
        epochs: Number of epochs
        grad_clip: Optional gradient clipping value
        mixed_precision: Whether to use mixed precision training
        log_interval: Logging interval (in epochs)
        wandb_logger: Optional WandB logger
        metric_key: Metric to track for best model (default: "val_dice")
        save_checkpoints: Whether to save checkpoints
        checkpoint_dir: Directory to save checkpoints
        compute_auroc: Whether to compute AUROC during validation

    Returns:
        Dictionary with:
            - history: Dict of training metrics per epoch
            - best_metric: Best validation metric achieved
            - best_metric_name: Name of the tracked metric
            - final_val_dice: Final Dice score
            - final_val_iou: Final IoU score
            - final_val_pixel_acc: Final pixel accuracy
            - final_val_auroc: Final AUROC (only if compute_auroc=True)
    """
    model.train()

    scaler = _create_grad_scaler(mixed_precision)
    sched, sched_meta = scheduler or (None, {})

    # Initialize best model tracking
    # Dice/IoU are maximized (higher is better)
    best_metric = -math.inf
    best_checkpoint_path = None
    best_epoch = 0

    # Initialize history with proper segmentation metric names
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_pixel_acc": [],
        "lr": []
    }
    if compute_auroc:
        history["val_auroc"] = []

    log.info(f"\n{'='*60}")
    log.info("Starting Segmentation Training")
    log.info(f"{'='*60}")
    log.info(f"Epochs: {epochs}")
    log.info(f"Device: {device}")
    log.info(f"Mixed precision: {mixed_precision}")
    log.info(f"Gradient clipping: {grad_clip}")
    log.info(f"Metric key: {metric_key}")
    log.info(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # ========== Training ==========
        model.train()
        running_loss = 0.0
        n_train_samples = 0

        for step, batch in enumerate(loaders["train"], start=1):
            images = batch["pixel_values"].to(device)
            masks = batch["mask_target"].to(device)

            batch_size = images.size(0)
            n_train_samples += batch_size

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with autocast(device_type=device.type, enabled=mixed_precision):
                outputs = model(pixel_values=images, labels=masks)
                loss = outputs.loss

            # Backward pass
            if mixed_precision:
                scaler.scale(loss).backward()

                # Gradient clipping (after unscaling)
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            running_loss += loss.item() * batch_size

        # Compute average training loss
        avg_train_loss = running_loss / n_train_samples

        # ========== Validation ==========
        val_metrics = _run_segmentation_validation(
            model, loaders["val"], device, mixed_precision, compute_auroc=compute_auroc
        )

        # ========== Scheduler Step ==========
        if sched:
            _maybe_scheduler_step(
                sched_meta, sched, on="epoch", metric=val_metrics["val_loss"]
            )

        # ========== Logging ==========
        current_lr = optimizer.param_groups[0]["lr"]

        # Build metrics dict for logging
        log_metrics = {
            "val_dice": val_metrics["val_dice"],
            "val_iou": val_metrics["val_iou"],
            "val_pixel_acc": val_metrics["val_pixel_acc"],
        }
        if compute_auroc:
            log_metrics["val_auroc"] = val_metrics["val_auroc"]

        # Log to console and wandb using proper metric names
        _update_history_and_log(
            history=history,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_metrics["val_loss"],
            cur_lr=current_lr,
            metrics=log_metrics,
            wandb_logger=wandb_logger,
            log=log,
        )

        # ========== Track Best Model ==========
        current_metric = val_metrics[metric_key]

        if current_metric > best_metric:  # Dice/IoU are maximized
            best_metric = current_metric
            best_epoch = epoch

            log.info(
                f"  New best {metric_key}: {best_metric:.4f} at epoch {epoch}"
            )

            # Save checkpoint directly to disk (avoids expensive deepcopy)
            if save_checkpoints and checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Remove old best checkpoint to save disk space
                if best_checkpoint_path is not None and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()

                best_checkpoint_path = checkpoint_dir / f"best_model_dice{best_metric:.4f}.pt"

                # Save directly to disk - much faster than deepcopy + later save
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "metric": best_metric,
                    "metric_name": metric_key,
                    "epoch": best_epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_metrics["val_dice"],
                    "val_iou": val_metrics["val_iou"],
                    "val_pixel_acc": val_metrics["val_pixel_acc"],
                }
                if compute_auroc:
                    checkpoint_data["val_auroc"] = val_metrics["val_auroc"]
                torch.save(checkpoint_data, best_checkpoint_path)

                log.info(f"  Saved checkpoint to {best_checkpoint_path}")

    # ========== Restore Best Model ==========
    if best_checkpoint_path is not None and best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info(
            f"\nRestored best model from epoch {best_epoch} "
            f"with {metric_key}={best_metric:.4f}\n"
        )

    # ========== Return Results ==========
    result = {
        "history": history,
        "best_metric": best_metric,
        "best_metric_name": metric_key,
        "best_epoch": best_epoch,
        "final_val_dice": history["val_dice"][-1] if history["val_dice"] else 0.0,
        "final_val_iou": history["val_iou"][-1] if history["val_iou"] else 0.0,
        "final_val_pixel_acc": history["val_pixel_acc"][-1] if history["val_pixel_acc"] else 0.0,
    }
    if compute_auroc:
        result["final_val_auroc"] = history["val_auroc"][-1] if history["val_auroc"] else 0.0
    return result
