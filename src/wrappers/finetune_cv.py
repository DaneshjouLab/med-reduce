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
import hydra
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
from src.engines.training_core import _get_embeddings

log = get_logger(__name__)


class FinetuneCVWrapper:
    """Performs K-fold cross-validation for model fine-tuning."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        setup_logging()

        # Setup base data module
        self.dm = hydra.utils.instantiate(cfg.datamodule, full_cfg=cfg)
        self.dm.setup("fit")

        raw_dataset = self.dm.train_set
        if isinstance(raw_dataset, Subset):
            log.info("Detected Subset from random_split, unwrapping to base dataset for CV")
            # Extract the underlying dataset and indices
            self.base_dataset = raw_dataset.dataset
            self.base_indices = np.array(raw_dataset.indices)
            log.info(f"Unwrapped Subset: base dataset type={type(self.base_dataset).__name__}, "
                    f"using {len(self.base_indices)} samples from random_split")
        else:
            self.base_dataset = raw_dataset
            self.base_indices = None
            log.info(f"Using dataset directly: type={type(self.base_dataset).__name__}")
            
        self.k_folds = int(getattr(cfg.train, "k_folds", 5))
        self.subset_frac = float(getattr(cfg.train, "subset_frac", 1.0))

        self.model_info = cfg.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.loss_fn = cross_entropy_loss(
            label_smoothing=float(getattr(cfg.loss, "label_smoothing", 0.0)),
            class_weight=None,
            ignore_index=int(getattr(cfg.loss, "ignore_index", -100)),
            reduction=str(getattr(cfg.loss, "reduction", "mean")),
        )

        self.wandb = WandbLogger(
            project=getattr(cfg.logging, "project", "crossval-finetune"),
            run_name=getattr(cfg.logging, "run_name", "cv_run"),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            tags=["crossval"],
        )

        # UMAP configuration
        self.umap_enabled = getattr(cfg.logging, "save_umap_embeddings", False)
        self.umap_max_samples = getattr(cfg.logging, "umap_max_samples", None)
        self.run_name = getattr(cfg.logging, "run_name", "cv_run")
        self.run_dir = getattr(cfg.runtime, "run_dir", "./runs/finetune")
        self.umap_base_dir = os.path.join(self.run_dir, "umap_embeddings")

        if self.umap_enabled:
            os.makedirs(self.umap_base_dir, exist_ok=True)
            log.info(f"UMAP embeddings will be saved to {self.umap_base_dir}")

        # Checkpoint configuration
        self.save_checkpoints = getattr(cfg.logging, "save_checkpoints", True)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")

        if self.save_checkpoints:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            log.info(f"Model checkpoints will be saved to {self.checkpoint_dir}")

    def _make_model_and_optim(self):
        model = create_model(self.model_info, resolution=self.cfg.data.image_size).to(self.device)
        try:
            preprocessor = create_preprocessor(self.model_info, resolution=self.cfg.data.image_size)
        except Exception:
            preprocessor = None

        optimizer, (scheduler, sched_meta) = make_optimizer_and_scheduler(self.cfg, model.parameters())
        return model, optimizer, (scheduler, sched_meta), preprocessor

    def _get_fold_loaders(self, train_idx, val_idx):
        """
        Create dataloaders for a fold.
        
        Uses torch.utils.data.Subset to avoid nesting issues.
        Subset is lightweight and works with all PyTorch datasets including ISICHFRawSplit.
        """
        # If we have base_indices (from unwrapped Subset), map through them
        if self.base_indices is not None:
            actual_train_idx = self.base_indices[train_idx]
            actual_val_idx = self.base_indices[val_idx]
        else:
            actual_train_idx = train_idx
            actual_val_idx = val_idx
        
        train_subset = Subset(self.base_dataset, actual_train_idx)
        val_subset = Subset(self.base_dataset, actual_val_idx)

        num_workers = getattr(self.cfg.datamodule, 'num_workers', 8)
        pin_memory = getattr(self.cfg.datamodule, 'pin_memory', True)
        persistent_workers = getattr(self.cfg.datamodule, 'persistent_workers', False)
        prefetch_factor = getattr(self.cfg.datamodule, 'prefetch_factor', 2)

        return {
            "train": DataLoader(
                train_subset, 
                batch_size=self.cfg.data.batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and num_workers > 0,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            ),
            "val": DataLoader(
                val_subset, 
                batch_size=self.cfg.data.batch_size, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and num_workers > 0,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            ),
        }

    def _save_fold_embeddings(self, model, dataloader, fold, epoch):
        """Extract and save embeddings for a specific fold and epoch."""
        embeddings, labels = _get_embeddings(
            model=model,
            dataloader=dataloader,
            device=self.device,
            mixed_precision=bool(getattr(self.cfg.train, "mixed_precision", True)),
            max_samples=self.umap_max_samples,
        )

        # Create fold-specific directory
        fold_dir = os.path.join(self.umap_base_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save with epoch number
        save_path = os.path.join(fold_dir, f"{self.run_name}_e{epoch:03d}.pt")
        torch.save({"embeddings": embeddings, "labels": labels.long()}, save_path)
        log.info(f"Saved embeddings to {save_path}")

    def _save_checkpoint(self, model, fold, metric, optimizer=None):
        """Save model checkpoint for a specific fold."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "fold": fold,
            "metric": metric,
            "model_config": self.model_info,
            "cfg": self.cfg,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{self.run_name}_fold{fold+1}_metric{metric:.4f}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        log.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def train(self) -> Dict[str, Any]:
        """Run K-fold cross-validation."""
        np.random.seed(int(getattr(self.cfg.train, "seed", 42)))
        
        # Determine the working set of indices
        if self.base_indices is not None:
            working_indices = self.base_indices.copy()
        else:
            working_indices = np.arange(len(self.base_dataset))
        
        # Apply additional subset if needed
        if self.subset_frac < 1.0:
            subset_size = int(len(working_indices) * self.subset_frac)
            selected = np.random.choice(len(working_indices), subset_size, replace=False)
            working_indices = working_indices[selected]
            log.info(f"Using subset of {subset_size} samples ({self.subset_frac*100:.0f}%) for CV.")

        # Update base_indices to reflect our working set
        self.base_indices = working_indices
        
        # Create k-fold splits on the working indices
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=int(getattr(self.cfg.train, "seed", 42)))
        fold_metrics = []

        # Split on index range, not the dataset itself
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(working_indices)))):
            log.info(f"🔹 Fold {fold+1}/{self.k_folds}")

            model, optimizer, (scheduler, sched_meta), preprocessor = self._make_model_and_optim()
            model.to(self.device)
            loaders = self._get_fold_loaders(train_idx, val_idx)

            # Extract pre-training embeddings (epoch 0)
            if self.umap_enabled:
                log.info(f"Extracting embeddings before training (fold {fold+1}, epoch 0)...")
                self._save_fold_embeddings(model, loaders["val"], fold, epoch=0)

            # Run training
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

            # Save best model checkpoint for this fold
            if self.save_checkpoints:
                self._save_checkpoint(
                    model=model,
                    fold=fold,
                    metric=result["best_metric"],
                    optimizer=optimizer
                )

            # Extract post-training embeddings
            if self.umap_enabled:
                epochs = int(self.cfg.train.epochs)
                log.info(f"Extracting embeddings after training (fold {fold+1}, epoch {epochs})...")
                self._save_fold_embeddings(model, loaders["val"], fold, epoch=epochs)

            fold_metrics.append(result["best_metric"])

        mean_metric = float(np.mean(fold_metrics))
        std_metric = float(np.std(fold_metrics))
        log.info(f"Mean CV metric ({self.cfg.train.metric_key}): {mean_metric:.4f} ± {std_metric:.4f}")

        return {
            "fold_metrics": fold_metrics, 
            "mean_metric": mean_metric,
            "std_metric": std_metric
        }


def run(cfg: Any) -> Dict[str, Any]:
    """Entry point."""
    wrapper = FinetuneCVWrapper(cfg)
    return wrapper.train()