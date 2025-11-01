# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""K-Fold Cross Validation wrapper for fine-tuning."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import os
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold

# pylint: disable=import-error
from src.engines.finetune_engine import train_finetune
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.utils.optim import make_optimizer_and_scheduler
from src.losses.classification import cross_entropy_loss
from src.models.factory import create_model, create_preprocessor
from src.data.datamodule import BaseDataModule
from src.utils.training_utils import profile_model
from src.config import LoggingConfig, RuntimeConfig, DataConfig, ModelConfig, LossConfig, TrainingConfig

log = get_logger(__name__)


class FinetuneCVWrapper:
    """Performs K-fold cross-validation for model fine-tuning."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup base data module
        self.dm = BaseDataModule(
            cfg=cfg,
            dataset_name=cfg.data.dataset_name,
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )
        self.dm.setup("fit")

        self.dataset = self.dm.train_dataset  # assume dm exposes this
        self.k_folds = int(getattr(cfg.train, "k_folds", 5))
        self.subset_frac = float(getattr(cfg.train, "subset_frac", 1.0))

        self.model_info = cfg.model
        self.loss_fn = cross_entropy_loss(
            label_smoothing=float(getattr(cfg.loss, "label_smoothing", 0.0)),
            reduction=str(getattr(cfg.loss, "reduction", "mean")),
        )

        self.wandb = WandbLogger(
            project=getattr(cfg.logging, "project", "crossval-finetune"),
            run_name=getattr(cfg.logging, "run_name", "cv_run"),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            tags=["crossval"],
        )

    def _make_model_and_optim(self):
        model = create_model(self.model_info, resolution=self.cfg.data.image_size).to(self.device)
        try:
            preprocessor = create_preprocessor(self.model_info, resolution=self.cfg.data.image_size)
        except Exception:
            preprocessor = None

        optimizer, (scheduler, sched_meta) = make_optimizer_and_scheduler(self.cfg, model.parameters())
        return model, optimizer, (scheduler, sched_meta), preprocessor

    def _get_fold_loaders(self, train_idx, val_idx):
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)

        return {
            "train": DataLoader(train_subset, batch_size=self.cfg.data.batch_size, shuffle=True),
            "val": DataLoader(val_subset, batch_size=self.cfg.data.batch_size, shuffle=False),
        }

    def train(self) -> Dict[str, Any]:
        """Run K-fold cross-validation."""
        np.random.seed(int(getattr(self.cfg.train, "seed", 42)))
        if self.subset_frac < 1.0:
            subset_size = int(len(self.dataset) * self.subset_frac)
            indices = np.random.choice(len(self.dataset), subset_size, replace=False)
            self.dataset = Subset(self.dataset, indices)
            log.info(f"Using subset of {subset_size} samples ({self.subset_frac*100:.0f}%) for CV.")

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=int(getattr(self.cfg.train, "seed", 42)))
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            log.info(f"🔹 Fold {fold+1}/{self.k_folds}")

            model, optimizer, (scheduler, sched_meta), preprocessor = self._make_model_and_optim()
            loaders = self._get_fold_loaders(train_idx, val_idx)

            result = train_finetune(
                model=model,
                loaders=loaders,
                loss_fn=self.loss_fn,
                optimizer=optimizer,
                scheduler=(scheduler, sched_meta),
                device=self.device,
                epochs=int(self.cfg.train.epochs),
                grad_clip=getattr(self.cfg.train, "grad_clip", None),
                mixed_precision=bool(getattr(self.cfg.train, "mixed_precision", True)),
                log_interval=int(getattr(self.cfg.train, "log_interval", 50)),
                wandb_logger=self.wandb,
                metric_key=str(getattr(self.cfg.train, "metric_key", "val_acc")),
            )
            fold_metrics.append(result["best_metric"])

        mean_metric = float(np.mean(fold_metrics))
        log.info(f"✅ Mean CV metric ({self.cfg.train.metric_key}): {mean_metric:.4f}")

        return {"fold_metrics": fold_metrics, "mean_metric": mean_metric}


def run(cfg: Any) -> Dict[str, Any]:
    """Entry point (mirrors finetune wrapper structure)."""
    wrapper = FinetuneCVWrapper(cfg)
    return wrapper.train()
