# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""Training engine for linear probing."""
# pylint: disable=duplicate-code
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import math
import torch
from torch import nn

# --- AMP import (robust across versions) ---
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
    scaler = _create_grad_scaler(mixed_precision)

    sched, sched_meta = scheduler or (None, {})
    best_metric = -math.inf if not metric_key.endswith("loss") else math.inf
    best_state_dict = None

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0

        for step, batch in enumerate(loaders["train"], start=1):
            x, y = _preprocess_batch(batch, device)

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

        # logging
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
            best_state_dict = updated_state

    # restore best (optional: caller can save now)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "best_metric": best_metric,
        "history": history,
        "final_lr": optimizer.param_groups[0]["lr"],
    }
