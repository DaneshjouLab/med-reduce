# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/wrappers/probe.py
# -*- coding: utf-8 -*-
"""Linear probing wrapper for training classification heads on frozen backbones."""
from __future__ import annotations
from typing import Any, Dict

import os
import torch
from torch.utils.data import DataLoader

# pylint: disable=import-error
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.utils.optim import make_optimizer_and_scheduler
from src.losses.classification import cross_entropy_loss
from src.models.factory import (
    create_model, create_preprocessor, freeze_backbone, save_model
)
from src.data.datamodule import BaseDataModule
from src.engines.linear_probe_engine import train_probe
from src.utils.training_utils import profile_model

log = get_logger(__name__)


class ProbeWrapper:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Orchestrates linear probing:
      - builds model (+preprocessor if you need it for your datasets),
      - freezes backbone,
      - prepares dataloaders via DataModule,
      - creates optimizer/scheduler/loss,
      - calls the probe engine,
      - saves best model.
    """

    def __init__(self, cfg: Any):
        """
        Expected cfg fields (suggested):
          cfg.model: {type, model_id, config{num_labels,...}}
          cfg.data: {dataset_name, data_dir, image_size, batch_size, num_workers}
          cfg.train: {epochs, optimizer{...}, scheduler{...}, grad_clip, mixed_precision}
          cfg.loss: {label_smoothing, ignore_index, reduction}
          cfg.logging: {project, entity, run_name, wandb_enabled}
          cfg.runtime: {run_dir}
        """
        self.cfg = cfg
        setup_logging()

        # Build model
        self.model_info = cfg.model
        self.model = create_model(self.model_info, resolution=cfg.data.image_size)

        # Optionally create preprocessor (useful if datamodule needs it)
        try:
            self.preprocessor = create_preprocessor(
                self.model_info, resolution=cfg.data.image_size
            )
        except (ImportError, AttributeError, KeyError) as e:
            log.debug(f"Preprocessor creation failed: {e}")
            self.preprocessor = None

        # Freeze backbone for linear probing
        freeze_backbone(self.model, self.model_info["type"])

        # Data
        self.dm = BaseDataModule(
            cfg=cfg,
            dataset_name=cfg.data.dataset_name,
            data_dir=cfg.data.data_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )
        self.dm.setup("fit")

        # Optimizer & scheduler
        self.optimizer, (self.scheduler, self.sched_meta) = (
            make_optimizer_and_scheduler(cfg, self.model.parameters())
        )

        # Loss
        self.loss_fn = cross_entropy_loss(
            label_smoothing=float(getattr(cfg.loss, "label_smoothing", 0.0)),
            class_weight=None,
            ignore_index=int(getattr(cfg.loss, "ignore_index", -100)),
            reduction=str(getattr(cfg.loss, "reduction", "mean")),
        )

        # W&B
        self.wandb = WandbLogger(
            project=getattr(cfg.logging, "project", "resolution-aware-probe"),
            run_name=getattr(cfg.logging, "run_name", None),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            entity=getattr(cfg.logging, "entity", None),
            tags=getattr(cfg.logging, "tags", ["probe"]),
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _make_loaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for training and validation."""
        return {
            "train": self.dm.train_dataloader(),
            "val": self.dm.val_dataloader(),
            # "test": self.dm.test_dataloader(),  # optional
        }

    def train(self) -> Dict[str, Any]:
        """Run linear probing training."""
        log.info("Starting linear probe...")

        # Optional: profile FLOPs once
        gflops = profile_model(self.model, self.cfg.data.image_size)
        if self.wandb:
            self.wandb.log({"model/gflops": gflops})

        results = train_probe(
            model=self.model,
            loaders=self._make_loaders(),
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=(self.scheduler, self.sched_meta),
            device=self.device,
            epochs=int(self.cfg.train.epochs),
            grad_clip=getattr(self.cfg.train, "grad_clip", None),
            mixed_precision=bool(
                getattr(self.cfg.train, "mixed_precision", True)
            ),
            log_interval=int(getattr(self.cfg.train, "log_interval", 50)),
            wandb_logger=self.wandb,
            metric_key=str(getattr(self.cfg.train, "metric_key", "val_acc")),
        )

        # Save best model (HF format if applicable)
        run_dir = getattr(self.cfg.runtime, "run_dir", "./runs/probe")
        os.makedirs(run_dir, exist_ok=True)
        try:
            save_model(
                self.model,
                self.model_info,
                save_dir=run_dir,
                preprocessor=self.preprocessor
            )
        except (OSError, ValueError, AttributeError) as e:
            log.warning(f"Failed to save model in HF format; error: {e}")

        # Finish W&B
        if self.wandb:
            self.wandb.log({"best/metric": results.get("best_metric", None)})
            self.wandb.finish()

        log.info("Linear probe finished.")
        return results


def run(cfg: Any) -> Dict[str, Any]:
    """Convenience function to run from a CLI entrypoint."""
    wrapper = ProbeWrapper(cfg)
    return wrapper.train()
