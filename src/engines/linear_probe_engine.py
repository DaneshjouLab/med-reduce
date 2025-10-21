# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""Training engine for linear probing."""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import math
import torch
from torch import nn

# --- AMP import (robust across versions) ---
try:
    # PyTorch 2.0+ unified AMP API
    from torch.amp import autocast, GradScaler
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler

# pylint: disable=import-error
from src.utils.logging import get_logger
from src.engines.utils.training_loops import (
    _maybe_scheduler_step,
    _evaluate,
)

log = get_logger(__name__)


def train_probe(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
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
) -> Dict[str, Any]:
    """
    Generic engine for linear probing. Agnostic to dataset & transforms.
    Returns a dict with best metric, histories, and final lr.
    """
    model.train()

    # Initialize GradScaler safely across torch versions
    try:
        scaler = GradScaler(
            enabled=mixed_precision,
            init_scale=2.0**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
    except TypeError:
        scaler = GradScaler(enabled=mixed_precision)

    sched, sched_meta = scheduler or (None, {})
    best_metric = -math.inf
    best_state_dict = None

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0

        for step, batch in enumerate(loaders["train"], start=1):
            if isinstance(batch, dict):
                x, y = batch["image"], batch["label"]
            else:
                x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=mixed_precision):
                logits = model(x)
                loss = loss_fn(logits, y)

            if mixed_precision:
                scaler.scale(loss).backward()

                if grad_clip is not None:
                    # Unscale before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # Step and update scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if sched is not None:
                _maybe_scheduler_step(sched_meta, sched, on="batch")

            running_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)

            if step % log_interval == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                if wandb_logger:
                    wandb_logger.log({"train/loss": float(loss.item()), "lr": cur_lr})

        train_loss = running_loss / max(n_seen, 1)

        # ---- validation
        val_loss, val_acc = _evaluate(
            model=model,
            loader=loaders["val"],
            loss_fn=loss_fn,
            device=device,
            mixed_precision=mixed_precision,
        )

        # scheduler on epoch or val metric
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched_meta["monitor"] = (
                    val_loss if metric_key.endswith("loss") else val_acc
                )
                _maybe_scheduler_step(sched_meta, sched, on="val")
            else:
                _maybe_scheduler_step(sched_meta, sched, on="epoch")

        # logging
        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        if wandb_logger:
            wandb_logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": train_loss,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "lr": cur_lr,
                }
            )

        log.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.2e",
            epoch,
            train_loss,
            val_loss,
            val_acc,
            cur_lr,
        )

        monitor = val_loss if metric_key.endswith("loss") else val_acc
        is_better = (
            (monitor < best_metric)
            if metric_key.endswith("loss")
            else (monitor > best_metric)
        )
        if is_better:
            best_metric = monitor
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # restore best (optional: caller can save now)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "best_metric": best_metric,
        "history": history,
        "final_lr": optimizer.param_groups[0]["lr"],
    }
