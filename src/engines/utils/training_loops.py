# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""Training engine utilities for fine-tuning and linear probing.

This module provides helper functions used by other training engines:
- _accuracy: Computes top-1 accuracy
- _maybe_scheduler_step: Step schedulers based on configuration/meta
- _evaluate: Evaluate model on a loader with loss + accuracy

Imported by:
- src.engines.linear_probe_engine
- src.engines.finetune_engine
"""
from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
from torch import nn

try:
    # PyTorch 2.0+ unified AMP API
    from torch.amp import autocast
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.numel()
    return correct / max(total, 1)


def _maybe_scheduler_step(scheduler_meta: Dict[str, Any], scheduler, *, on: str):
    """Step scheduler if meta says so. `on` ∈ {'batch','epoch','val'}."""
    step_when = str(scheduler_meta.get("step_per", "epoch"))
    if step_when == on:
        if "monitor" in scheduler_meta:
            # e.g., ReduceLROnPlateau
            scheduler.step(scheduler_meta["monitor"])
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
            if isinstance(batch, dict):
                x, y = batch.get("image"), batch.get("label")
            else:
                x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Device-aware autocast (CPU/GPU) and version-safe
            with autocast(device_type=getattr(device, "type", "cuda"), enabled=mixed_precision):
                logits = model(x)
                loss = loss_fn(logits, y)

            bsz = y.size(0)
            running_loss += float(loss.item()) * bsz
            running_acc += _accuracy(logits, y) * bsz
            n += bsz

    return running_loss / max(n, 1), running_acc / max(n, 1)
