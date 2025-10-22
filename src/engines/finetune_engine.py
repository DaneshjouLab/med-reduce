# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""Training engine for full fine-tuning (end-to-end)."""
# pylint: disable=duplicate-code
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import math
import torch
from torch import nn

try:
    # PyTorch 2.0+ unified AMP API
    from torch.amp import autocast
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast

# pylint: disable=import-error
from src.utils.logging import get_logger
from src.engines.utils.training_core import (
    _maybe_scheduler_step,
    _create_grad_scaler,
    _update_history_and_log,
    _preprocess_batch,
    _run_validation_and_scheduler,
    _update_best_model_state,
)

log = get_logger(__name__)


def train_finetune(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    *,
    model: nn.Module,
    loaders: Dict[str, Any],  # {"train": DataLoader, "val": DataLoader}
    loss_fn,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Tuple[Any, Dict[str, Any]]] = None,  # (scheduler, meta)
    device: torch.device,
    epochs: int,
    grad_clip: Optional[float] = None,
    mixed_precision: bool = True,
    log_interval: int = 50,
    wandb_logger=None,
    metric_key: str = "val_acc",
    accumulation_steps: int = 1,
    zero_grad_set_to_none: bool = True,
) -> Dict[str, Any]:
    """
    Generic engine for full fine-tuning. Agnostic to datasets & transforms.

    Args:
        model: torch.nn.Module to train end-to-end.
        loaders: dict with "train" and "val" DataLoaders.
        loss_fn: callable (logits, targets) -> loss tensor.
        optimizer: torch optimizer over model parameters.
        scheduler: optional (scheduler, meta) where meta may contain:
            - step_per: {"batch","epoch","val"} (default "epoch")
            - monitor: metric value for schedulers like ReduceLROnPlateau (filled automatically)
        device: torch.device.
        epochs: number of epochs.
        grad_clip: optional float, max norm for gradient clipping.
        mixed_precision: bool, enable autocast + GradScaler.
        log_interval: int steps for logging.
        wandb_logger: object with .log(dict) and optional .watch(...) methods.
        metric_key: "val_acc" (maximize) or "...loss" (minimize) to pick best checkpoint.
        accumulation_steps: gradient accumulation steps (>1 to simulate larger batch).
        zero_grad_set_to_none: pass to optimizer.zero_grad for perf.

    Returns:
        dict with "best_metric", "history", and "final_lr".
    """
    assert accumulation_steps >= 1, "accumulation_steps must be >= 1"

    # Initialize GradScaler with backward compatibility
    scaler = _create_grad_scaler(mixed_precision)

    sched, sched_meta = scheduler or (None, {})
    best_metric = -math.inf if not metric_key.endswith("loss") else math.inf
    best_state = None

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

        for step, batch in enumerate(loaders["train"], start=1):
            x, y = _preprocess_batch(batch, device)

            with autocast(device_type=device.type, enabled=mixed_precision):
                logits = model(x)
                loss = loss_fn(logits, y)
                loss_to_backprop = loss / accumulation_steps

            if mixed_precision:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            # Step on accumulation boundary
            if step % accumulation_steps == 0:
                if grad_clip is not None:
                    if mixed_precision:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

                # Per-batch scheduler step (if configured)
                if sched is not None:
                    _maybe_scheduler_step(sched_meta, sched, on="batch")

            bsz = y.size(0)
            running_loss += float(loss.item()) * bsz
            n_seen += bsz

            if step % log_interval == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                if wandb_logger:
                    wandb_logger.log({"train/loss": float(loss.item()), "lr": cur_lr})

        # ---- validation and scheduler step
        val_loss, val_acc = _run_validation_and_scheduler(
            model=model,
            loaders=loaders,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=mixed_precision,
            sched=sched,
            sched_meta=sched_meta,
            metric_key=metric_key,
        )

        # Aggregate + log
        train_loss = running_loss / max(n_seen, 1)
        cur_lr = optimizer.param_groups[0]["lr"]

        _update_history_and_log(
            history=history,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            cur_lr=cur_lr,
            wandb_logger=wandb_logger,
            log=log,
        )

        updated_state, best_metric, is_better = _update_best_model_state(
            model=model,
            metric_key=metric_key,
            val_loss=val_loss,
            val_acc=val_acc,
            best_metric=best_metric,
        )

        if is_better:
            best_state = updated_state

    # Restore best weights so caller can save/export
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_metric": best_metric,
        "history": history,
        "final_lr": optimizer.param_groups[0]["lr"],
    }
