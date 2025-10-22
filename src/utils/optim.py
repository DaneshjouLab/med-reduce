# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
Optimization utilities for PyTorch models.

This module provides functionality for:
- Building optimizers with weight decay parameter groups
- Creating learning rate schedulers with various policies
- Handling scheduler updates during training
"""

# src/utils/optim.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterable, Tuple, Dict, Any
import math
import torch  # pylint: disable=import-error
from torch.optim import Optimizer  # pylint: disable=import-error
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, StepLR, ReduceLROnPlateau  # pylint: disable=import-error


def _param_groups_decay(model_or_params: Iterable, weight_decay: float) -> list:
    """
    Split parameters into (decay / no_decay) groups.
    - no_decay: bias, LayerNorm/BatchNorm weights.
    """
    if isinstance(model_or_params, torch.nn.Module):
        params = list(model_or_params.named_parameters())
    else:
        # assume an iterable of parameters
        params = [(f"p{i}", p) for i, p in enumerate(model_or_params)]

    decay, no_decay = [], []
    for name, p in params:
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_optimizer(cfg, params) -> Optimizer:
    opt_cfg = (
        getattr(cfg, "train").get("optimizer", {}) if hasattr(cfg, "train") else {}
    )
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 0.05))

    if isinstance(params, torch.nn.Module) or (
        hasattr(params, "__iter__") and hasattr(next(iter(params)), "ndim")
    ):
        param_groups = _param_groups_decay(params, wd)
    else:
        # already groups
        param_groups = params

    if name == "adamw":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        nesterov = bool(opt_cfg.get("nesterov", True))
        return torch.optim.SGD(
            param_groups, lr=lr, momentum=momentum, nesterov=nesterov
        )
    # If we reach this point, the optimizer is not supported
    raise ValueError(f"Unsupported optimizer: {name}")


def _warmup_cosine_lambda_fn(epochs: int, warmup_epochs: int, min_lr_ratio: float):
    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # cosine from 1.0 -> min_lr_ratio
        progress = (current_epoch - warmup_epochs) / float(
            max(1, epochs - warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def _build_scheduler(
    cfg, optimizer: Optimizer
) -> Tuple[_LRScheduler | None, Dict[str, Any]]:
    sch_cfg = (
        getattr(cfg, "train").get("scheduler", {}) if hasattr(cfg, "train") else {}
    )
    name = str(sch_cfg.get("name", "cosine")).lower()
    epochs = int(sch_cfg.get("epochs", getattr(cfg.train, "epochs", 50)))
    warmup_epochs = int(sch_cfg.get("warmup_epochs", 0))

    if name in ("cosine", "cosineanneal", "cosine_anneal"):
        min_lr = float(sch_cfg.get("min_lr", 1e-6))
        base_lr = float(
            getattr(cfg.train.get("optimizer", {}), "lr", 1e-4)
            if hasattr(cfg, "train")
            else 1e-4
        )
        min_lr_ratio = max(min_lr / max(base_lr, 1e-12), 0.0)
        lr_lambda = _warmup_cosine_lambda_fn(epochs, warmup_epochs, min_lr_ratio)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, {"by": "epoch"}
    if name == "step":
        step_size = int(sch_cfg.get("step_size", 30))
        gamma = float(sch_cfg.get("gamma", 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler, {"by": "epoch"}
    if name in ("plateau", "reduceonplateau"):
        patience = int(sch_cfg.get("patience", 5))
        factor = float(sch_cfg.get("factor", 0.5))
        scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
        return scheduler, {"by": "val_metric"}
    if name in ("none", "off"):
        return None, {"by": "none"}
    # If we reach this point, the scheduler is not supported
    raise ValueError(f"Unsupported scheduler: {name}")


def step_scheduler(
    scheduler,
    meta: Dict[str, Any],
    epoch: int = None, # pylint: disable=unused-argument
    val_metric: float | None = None
):
    """
    Step the scheduler depending on configuration (by epoch or by val metric).
    meta["by"] is returned from _build_scheduler.

    Args:
        scheduler: The scheduler to step
        meta: Metadata dictionary with scheduling policy
        epoch: Current epoch (not used but kept for API compatibility)
        val_metric: Validation metric for ReduceLROnPlateau schedulers
    """
    if scheduler is None:
        return
    if meta.get("by") == "epoch":
        scheduler.step()
    elif meta.get("by") == "val_metric":
        # lower is better by default; pass -val_metric if you want higher-better
        scheduler.step(val_metric)


def make_optimizer_and_scheduler(
    cfg, params
) -> Tuple[Optimizer, Tuple[_LRScheduler | None, Dict[str, Any]]]:
    """
    Factory: returns (optimizer, (scheduler, meta)).
    Use step_scheduler(scheduler, meta, epoch, val_metric) to step it.
    """
    optimizer = _build_optimizer(cfg, params)
    scheduler, meta = _build_scheduler(cfg, optimizer)
    return optimizer, (scheduler, meta)
