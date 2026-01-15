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
import copy
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
from src.evaluation.segmentation_metrics import compute_segmentation_metrics

log = get_logger(__name__)


def _run_segmentation_validation(
    model: nn.Module,
    val_loader,
    device: torch.device,
    mixed_precision: bool = True,
) -> Dict[str, float]:
    """
    Run validation epoch and compute segmentation metrics.

    Args:
        model: Segmentation model
        val_loader: Validation data loader
        device: Device to run on
        mixed_precision: Whether to use mixed precision

    Returns:
        Dictionary with:
            - val_loss: Average validation loss
            - val_dice: Dice coefficient
            - val_iou: Intersection over Union
            - val_pixel_acc: Pixel accuracy
    """
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_masks = []
    n_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["pixel_values"].to(device)
            masks = batch["mask_target"].to(device)

            batch_size = images.size(0)
            n_samples += batch_size

            with autocast(device_type=device.type, enabled=mixed_precision):
                outputs = model(pixel_values=images, labels=masks)
                loss = outputs.loss
                logits = outputs.logits

            running_loss += loss.item() * batch_size

            # Collect predictions and targets for metrics
            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())

    # Compute average loss
    val_loss = running_loss / n_samples

    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Compute segmentation metrics
    metrics = compute_segmentation_metrics(all_logits, all_masks)

    return {
        "val_loss": val_loss,
        "val_dice": metrics["dice"],
        "val_iou": metrics["iou"],
        "val_pixel_acc": metrics["pixel_acc"],
    }


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

    Returns:
        Dictionary with:
            - history: Dict of training metrics per epoch
            - best_metric: Best validation metric achieved
            - best_metric_name: Name of the tracked metric
            - final_val_dice: Final Dice score
            - final_val_iou: Final IoU score
            - final_val_pixel_acc: Final pixel accuracy
    """
    model.train()

    scaler = _create_grad_scaler(mixed_precision)
    sched, sched_meta = scheduler or (None, {})

    # Initialize best model tracking
    # Dice/IoU are maximized (higher is better)
    best_metric = -math.inf
    best_state_dict = None
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
            model, loaders["val"], device, mixed_precision
        )

        # ========== Scheduler Step ==========
        if sched:
            _maybe_scheduler_step(
                sched_meta, sched, on="epoch", metric=val_metrics["val_loss"]
            )

        # ========== Logging ==========
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to console and wandb using proper metric names
        _update_history_and_log(
            history=history,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_metrics["val_loss"],
            cur_lr=current_lr,
            metrics={
                "val_dice": val_metrics["val_dice"],
                "val_iou": val_metrics["val_iou"],
                "val_pixel_acc": val_metrics["val_pixel_acc"],
            },
            wandb_logger=wandb_logger,
            log=log,
        )

        # ========== Track Best Model ==========
        current_metric = val_metrics[metric_key]

        if current_metric > best_metric:  # Dice/IoU are maximized
            best_metric = current_metric
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

            log.info(
                f"  🏆 New best {metric_key}: {best_metric:.4f} at epoch {epoch}"
            )

            # Save checkpoint
            if save_checkpoints and checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = checkpoint_dir / f"best_model_dice{best_metric:.4f}.pt"

                torch.save({
                    "model_state_dict": best_state_dict,
                    "metric": best_metric,
                    "metric_name": metric_key,
                    "epoch": best_epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_metrics["val_dice"],
                    "val_iou": val_metrics["val_iou"],
                    "val_pixel_acc": val_metrics["val_pixel_acc"],
                }, checkpoint_path)

                log.info(f"  💾 Saved checkpoint to {checkpoint_path}")

    # ========== Restore Best Model ==========
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        log.info(
            f"\n✓ Restored best model from epoch {best_epoch} "
            f"with {metric_key}={best_metric:.4f}\n"
        )

    # ========== Return Results ==========
    return {
        "history": history,
        "best_metric": best_metric,
        "best_metric_name": metric_key,
        "best_epoch": best_epoch,
        "final_val_dice": history["val_dice"][-1] if history["val_dice"] else 0.0,
        "final_val_iou": history["val_iou"][-1] if history["val_iou"] else 0.0,
        "final_val_pixel_acc": history["val_pixel_acc"][-1] if history["val_pixel_acc"] else 0.0,
    }
