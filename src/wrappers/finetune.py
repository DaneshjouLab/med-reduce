# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""Fine-tuning wrapper for end-to-end optimization."""
from __future__ import annotations
from typing import Any, Dict

import os
import torch
from torch.utils.data import DataLoader

# pylint: disable=import-error
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.utils.optim import make_optimizer_and_scheduler
from src.losses.classification import cross_entropy_loss
from src.models.factory import create_model, create_preprocessor, save_model
from src.data.datamodule import BaseDataModule
from src.engines.finetune_engine import train_finetune
from src.utils.training_utils import profile_model
from src.engines.training_core import _get_embeddings
from src.config import LoggingConfig, RuntimeConfig, DataConfig, ModelConfig, LossConfig, TrainingConfig

# Create config instances
logging_cfg = LoggingConfig(
    save_umap_embeddings=True,
    umap_max_samples=1000,
    run_name="vit_flowers_umap"
)

runtime_cfg = RuntimeConfig(run_dir="./runs/experiment_1")

log = get_logger(__name__)


class FinetuneWrapper:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Orchestrates full fine-tuning:
      - builds model (+preprocessor if needed),
      - prepares dataloaders via DataModule,
      - creates optimizer/scheduler/loss,
      - calls the finetune engine,
      - saves best model.
    """

    def __init__(self, cfg: Any):
        """
        Expected cfg fields (suggested):
          cfg.model, cfg.data, cfg.train, cfg.loss, cfg.logging, cfg.runtime
        """
        self.cfg = cfg
        setup_logging()

        # Model
        self.model_info = cfg.model
        self.model = create_model(self.model_info, resolution=cfg.data.image_size)

        # Optional preprocessor
        try:
            self.preprocessor = create_preprocessor(
                self.model_info, resolution=cfg.data.image_size
            )
        except (ImportError, AttributeError, KeyError) as e:
            log.debug(f"Preprocessor creation failed: {e}")
            self.preprocessor = None

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
            project=getattr(cfg.logging, "project", "resolution-aware-finetune"),
            run_name=getattr(cfg.logging, "run_name", None),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            entity=getattr(cfg.logging, "entity", None),
            tags=getattr(cfg.logging, "tags", ["finetune"]),
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _make_loaders(self) -> Dict[str, DataLoader]:
        return {
            "train": self.dm.train_dataloader(),
            "val": self.dm.val_dataloader(),
        }

    def train(self) -> Dict[str, Any]:
        """Run fine-tuning training."""
        log.info("Starting fine-tune...")

        gflops = profile_model(self.model, self.cfg.data.image_size)
        if self.wandb:
            self.wandb.log({"model/gflops": gflops})

        # Create loaders once
        loaders = self._make_loaders()
        
        # Get UMAP config settings (with defaults)
        umap_enabled = getattr(self.cfg.logging, "save_umap_embeddings", False)
        max_samples = getattr(self.cfg.logging, "umap_max_samples", None)
        run_name = getattr(self.cfg.logging, "run_name", "run")
        run_dir = getattr(self.cfg.runtime, "run_dir", "./runs/finetune")
        umap_dir = os.path.join(run_dir, "umap_embeddings")
        
        if umap_enabled:
            os.makedirs(umap_dir, exist_ok=True)
            
            log.info("Extracting embeddings before training (epoch 0)...")
            embeddings, labels = _get_embeddings(
                model=self.model,
                dataloader=loaders["val"],
                device=self.device,
                mixed_precision=bool(getattr(self.cfg.train, "mixed_precision", True)),
                max_samples=max_samples,
            )
            save_path = os.path.join(umap_dir, f"{run_name}_e000.pt")
            torch.save({"embeddings": embeddings, "labels": labels.long()}, save_path)
            log.info(f"Saved pre-training embeddings to {save_path}")

        # Run training
        results = train_finetune(
            model=self.model,
            loaders=loaders,  # Pass the already-created loaders
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=(self.scheduler, self.sched_meta),
            device=self.device,
            epochs=int(self.cfg.train.epochs),
            grad_clip=getattr(self.cfg.train, "grad_clip", None),
            mixed_precision=bool(getattr(self.cfg.train, "mixed_precision", True)),
            log_interval=int(getattr(self.cfg.train, "log_interval", 50)),
            wandb_logger=self.wandb,
            metric_key=str(getattr(self.cfg.train, "metric_key", "val_acc")),
        )

        if umap_enabled:
            epochs = int(self.cfg.train.epochs)
            log.info(f"Extracting embeddings after training (epoch {epochs})...")
            embeddings, labels = _get_embeddings(
                model=self.model,
                dataloader=loaders["val"],
                device=self.device,
                mixed_precision=bool(getattr(self.cfg.train, "mixed_precision", True)),
                max_samples=max_samples,
            )
            save_path = os.path.join(umap_dir, f"{run_name}_e{epochs:03d}.pt")
            torch.save({"embeddings": embeddings, "labels": labels.long()}, save_path)
            log.info(f"Saved post-training embeddings to {save_path}")

        # Save model
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

        if self.wandb:
            self.wandb.log({"best/metric": results.get("best_metric", None)})
            self.wandb.finish()

        log.info("Fine-tune finished.")
        return results


def run(cfg: Any) -> Dict[str, Any]:
    wrapper = FinetuneWrapper(cfg)
    return wrapper.train()
