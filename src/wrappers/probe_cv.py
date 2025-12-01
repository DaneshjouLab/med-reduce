# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""K-Fold Cross Validation wrapper for hyperparameter tuning with linear probing."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import os
import torch
import numpy as np
import hydra
import json
import copy
import random
import itertools
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold

# pylint: disable=import-error
from src.engines.linear_probe_engine import train_probe
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.utils.optim import make_optimizer_and_scheduler
from src.losses.classification import cross_entropy_loss
from src.models.factory import create_model, create_preprocessor
from src.data.datamodule import BaseDataModule
from src.utils.training_utils import profile_model
from src.engines.training_core import _get_embeddings
from src.utils.teacher_cache import TeacherEmbeddingCache, create_clean_image_dataloader

log = get_logger(__name__)


class ProbeCVWrapper:
    """
    Performs K-fold cross-validation for hyperparameter tuning with linear probing.

    Hyperparameter tuning is performed using 5-fold cross-validation on the training split.
    For each domain-teacher pair at the highest resolution:
      - 512px for dermatology and radiology
      - 1024px for pathology

    Grid search over:
      - Learning rate η ∈ {1×10⁻⁴, 3×10⁻⁴, 1×10⁻³}
      - Weight decay λ ∈ {0, 1×10⁻⁴, 1×10⁻³}
      - Batch size b ∈ {32, 64}
      - Decoder dropout (segmentation only) ∈ {0.0, 0.1, 0.3}

    The selected hyperparameters can then be used for training at other resolutions.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        setup_logging()

        current_resolution = getattr(cfg.data, "image_size", None) or getattr(cfg.dataset, "image_size", None)

        expected_high_res_map = {
            "dermatology": 512,
            "radiology": 512,
            "pathology": 1024,
        }

        domain = getattr(cfg, "domain", None)
        expected_high_res = expected_high_res_map.get(domain, 512)  # Default to 512

        hyperparam_search = getattr(cfg.train, "hyperparam_search", None)
        if hyperparam_search and getattr(hyperparam_search, "enabled", False):
            if current_resolution and current_resolution < expected_high_res:
                log.warning(
                    f"⚠️  Hyperparameter search should be performed at highest resolution! "
                    f"Current: {current_resolution}px, Expected: {expected_high_res}px for {domain or 'this domain'}"
                )
            else:
                log.info(f"✓ Hyperparameter search at highest resolution: {current_resolution}px")

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
            project=getattr(cfg.logging, "project", "crossval-probe"),
            run_name=getattr(cfg.logging, "run_name", "cv_run"),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            tags=["crossval"],
        )

        # UMAP configuration
        self.umap_enabled = getattr(cfg.logging, "save_umap_embeddings", False)
        self.umap_max_samples = getattr(cfg.logging, "umap_max_samples", None)
        self.run_name = getattr(cfg.logging, "run_name", "cv_run")
        self.run_dir = getattr(cfg.runtime, "run_dir", "./runs/probe")
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

        # Teacher embedding cache configuration (for distillation)
        self.distillation_enabled = False
        self.teacher_cache = None
        if hasattr(cfg, 'distillation') and getattr(cfg.distillation, 'enabled', False):
            self.distillation_enabled = True
            distill_cfg = cfg.distillation

            # Initialize teacher cache
            teacher_model_info = distill_cfg.get('teacher_model', cfg.model)
            cache_dir = getattr(distill_cfg, 'teacher_cache_dir', './cache/teacher_embeddings')
            full_resolution = getattr(distill_cfg, 'full_resolution', 224)

            self.teacher_cache = TeacherEmbeddingCache(
                cache_dir=cache_dir,
                teacher_model_info=teacher_model_info,
                full_resolution=full_resolution,
                device=self.device,
            )

            log.info(f"Distillation enabled with teacher cache at {cache_dir}")

        # Hyperparameter search configuration
        self.hyperparam_search_enabled = False
        self.pretuned_hyperparams = None
        if hyperparam_search:
            self.hyperparam_search_enabled = getattr(hyperparam_search, "enabled", False)
            self.hyperparam_use_cv = getattr(hyperparam_search, "use_cv", True)
            self.param_grid = getattr(hyperparam_search, "param_grid", {})

            # Check if we should load pre-tuned hyperparameters
            load_from_file = getattr(hyperparam_search, "load_from_file", None)
            if load_from_file and os.path.exists(load_from_file):
                self.pretuned_hyperparams = self._load_pretuned_hyperparams(load_from_file)

            if self.hyperparam_search_enabled:
                self.search_dir = os.path.join(self.run_dir, "hyperparam_search")
                os.makedirs(self.search_dir, exist_ok=True)
                log.info(f"Hyperparameter search results will be saved to {self.search_dir}")

    def _load_pretuned_hyperparams(self, filepath: str) -> Dict[str, Any]:
        """Load pre-tuned hyperparameters from a JSON file."""
        log.info(f"📥 Loading pre-tuned hyperparameters from {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)

        hyperparams = data.get("best_hyperparameters", {})
        metadata = data.get("metadata", {})

        log.info(f"✓ Loaded hyperparameters: {hyperparams}")
        log.info(f"  Original resolution: {metadata.get('resolution')}px")
        log.info(f"  Domain: {metadata.get('domain')}")
        log.info(f"  Teacher: {metadata.get('teacher_model')}")
        log.info(f"  Validation metric: {data.get('validation_metric', {}).get('mean'):.4f} ± {data.get('validation_metric', {}).get('std'):.4f}")

        return hyperparams

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

    def _get_all_hyperparam_configs(self) -> List[Dict[str, Any]]:
        """Get all hyperparameter configurations from the parameter grid (full grid search)."""
        if not self.param_grid:
            return []

        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*self.param_grid.values()))

        return [dict(zip(keys, combo)) for combo in combos]

    def _apply_hyperparams_to_cfg(self, params: Dict[str, Any]) -> Any:
        """Apply hyperparameters to a config copy."""
        cfg = copy.deepcopy(self.cfg)

        # Learning rate and optimizer params
        if "lr" in params:
            cfg.train.optimizer.lr = params["lr"]
        if "weight_decay" in params:
            cfg.train.optimizer.weight_decay = params["weight_decay"]

        # Data params
        if "batch_size" in params:
            cfg.data.batch_size = params["batch_size"]
            if hasattr(cfg, "datamodule"):
                cfg.datamodule.batch_size = params["batch_size"]

        # Loss params
        if "label_smoothing" in params:
            cfg.loss.label_smoothing = params["label_smoothing"]

        # Model params (for segmentation)
        if "decoder_dropout" in params:
            if not hasattr(cfg.model, "config"):
                cfg.model.config = {}
            cfg.model.config.decoder_dropout = params["decoder_dropout"]

        return cfg

    def _run_cv_for_hyperparams(self, params: Dict[str, Any]) -> Tuple[float, float]:
        """
        Run full 5-fold cross-validation with specific hyperparameters.

        Returns the mean and std of validation metrics across all folds.
        """
        # Apply hyperparameters to config
        trial_cfg = self._apply_hyperparams_to_cfg(params)

        # Determine the working set of indices
        if self.base_indices is not None:
            working_indices = self.base_indices.copy()
        else:
            working_indices = np.arange(len(self.base_dataset))

        # Create k-fold splits
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=int(getattr(self.cfg.train, "seed", 42)))
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(working_indices)))):
            log.info(f"  Fold {fold+1}/{self.k_folds}")

            # Get actual indices
            if self.base_indices is not None:
                actual_train_idx = working_indices[train_idx]
                actual_val_idx = working_indices[val_idx]
            else:
                actual_train_idx = working_indices[train_idx]
                actual_val_idx = working_indices[val_idx]

            # Create dataloaders
            train_subset = Subset(self.base_dataset, actual_train_idx)
            val_subset = Subset(self.base_dataset, actual_val_idx)

            num_workers = getattr(trial_cfg.datamodule, 'num_workers', 8)
            pin_memory = getattr(trial_cfg.datamodule, 'pin_memory', True)
            persistent_workers = getattr(trial_cfg.datamodule, 'persistent_workers', False)
            prefetch_factor = getattr(trial_cfg.datamodule, 'prefetch_factor', 2)

            loaders = {
                "train": DataLoader(
                    train_subset,
                    batch_size=trial_cfg.data.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers and num_workers > 0,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                ),
                "val": DataLoader(
                    val_subset,
                    batch_size=trial_cfg.data.batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers and num_workers > 0,
                    prefetch_factor=prefetch_factor if num_workers > 0 else None,
                ),
            }

            # Create model and optimizer
            model = create_model(self.model_info, resolution=trial_cfg.data.image_size).to(self.device)
            optimizer, (scheduler, sched_meta) = make_optimizer_and_scheduler(trial_cfg, model.parameters())

            # Update loss function with potential label smoothing change
            loss_fn = cross_entropy_loss(
                label_smoothing=float(getattr(trial_cfg.loss, "label_smoothing", 0.0)),
                class_weight=None,
                ignore_index=int(getattr(trial_cfg.loss, "ignore_index", -100)),
                reduction=str(getattr(trial_cfg.loss, "reduction", "mean")),
            )

            # Run training for this fold
            result = train_probe(
                model=model,
                loaders=loaders,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=(scheduler, sched_meta),
                device=self.device,
                epochs=int(trial_cfg.train.epochs),
                grad_clip=getattr(trial_cfg.train, "grad_clip", None),
                mixed_precision=bool(getattr(trial_cfg.train, "mixed_precision", True)),
                log_interval=int(getattr(trial_cfg.train, "log_interval", 50)),
                wandb_logger=None,  # Disable wandb for hyperparameter search trials
                metric_key=str(getattr(trial_cfg.train, "metric_key", "val_acc")),
            )

            fold_metrics.append(result["best_metric"])

        mean_metric = float(np.mean(fold_metrics))
        std_metric = float(np.std(fold_metrics))

        return mean_metric, std_metric

    def _run_hyperparam_search(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run hyperparameter search using 5-fold cross-validation.

        Each hyperparameter configuration is evaluated with full 5-fold CV.
        This should only be run at the highest resolution:
          - 512px for dermatology and radiology
          - 1024px for pathology
        Returns the best hyperparameters and all trial results.
        """
        log.info(f"\n{'='*60}")
        log.info(f"Starting Hyperparameter Search (5-Fold Cross-Validation)")
        log.info(f"  Grid search strategy: Full grid search")
        log.info(f"  Evaluation: {self.k_folds}-fold CV per configuration")
        log.info(f"{'='*60}\n")

        results = []
        all_configs = self._get_all_hyperparam_configs()

        if not all_configs:
            log.warning("No hyperparameter grid specified, skipping search")
            return {}, []

        log.info(f"Total configurations to evaluate: {len(all_configs)}")

        for i, params in enumerate(all_configs, start=1):
            log.info(f"\n{'─'*60}")
            log.info(f"Trial {i}/{len(all_configs)}: {params}")
            log.info(f"{'─'*60}")

            try:
                # Run 5-fold CV evaluation
                mean_metric, std_metric = self._run_cv_for_hyperparams(params)

                result_entry = {
                    "trial": i,
                    "params": params,
                    "mean_metric": mean_metric,
                    "std_metric": std_metric,
                }
                results.append(result_entry)

                log.info(f"✓ Trial {i} completed: {mean_metric:.4f} ± {std_metric:.4f}")

            except Exception as e:
                log.error(f"✗ Trial {i} failed with error: {e}")
                result_entry = {
                    "trial": i,
                    "params": params,
                    "mean_metric": float('-inf'),
                    "std_metric": 0.0,
                    "error": str(e),
                }
                results.append(result_entry)

        # Find best configuration
        metric_key = str(getattr(self.cfg.train, "metric_key", "val_acc"))
        reverse = not metric_key.endswith("loss")  # Higher is better unless it's a loss
        valid_results = [r for r in results if r["mean_metric"] != float('-inf')]

        if not valid_results:
            raise RuntimeError("All hyperparameter trials failed!")

        best_result = sorted(valid_results, key=lambda x: x["mean_metric"], reverse=reverse)[0]
        best_params = best_result["params"]
        best_metric = best_result["mean_metric"]
        best_std = best_result["std_metric"]

        log.info(f"\n{'='*60}")
        log.info(f"🏆 Best Hyperparameters Found:")
        log.info(f"  Params: {best_params}")
        log.info(f"  Validation Metric: {best_metric:.4f} ± {best_std:.4f}")
        log.info(f"  Trial: {best_result['trial']}")
        log.info(f"{'='*60}\n")

        # Save results to JSON
        self._save_hyperparam_results(results, best_result)

        return best_params, results

    def _save_hyperparam_results(self, results: List[Dict[str, Any]], best_result: Dict[str, Any]):
        """Save hyperparameter search results to JSON file."""
        output = {
            "search_config": {
                "param_grid": self.param_grid,
                "k_folds": self.k_folds,
                "resolution": getattr(self.cfg.data, "image_size", None),
                "domain": getattr(self.cfg, "domain", None),
                "teacher_model": getattr(self.cfg.model, "name", None),
                "strategy": "5-fold_cv_per_config",
                "note": "Each config evaluated with full 5-fold cross-validation.",
            },
            "best_result": best_result,
            "all_results": results,
        }

        # Save full results
        output_path = os.path.join(self.search_dir, "hyperparam_search_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        log.info(f"📊 Hyperparameter search results saved to {output_path}")

        # Save best hyperparameters in a separate file for easy reuse
        best_params_output = {
            "best_hyperparameters": best_result["params"],
            "validation_metric": {
                "mean": best_result["mean_metric"],
                "std": best_result["std_metric"],
            },
            "metadata": {
                "resolution": getattr(self.cfg.data, "image_size", None),
                "domain": getattr(self.cfg, "domain", None),
                "teacher_model": getattr(self.cfg.model, "name", None),
                "metric_key": str(getattr(self.cfg.train, "metric_key", "val_acc")),
                "k_folds": self.k_folds,
            },
            "usage": "These hyperparameters should be reused for all lower-resolution experiments for this domain-teacher pair.",
        }

        best_params_path = os.path.join(self.search_dir, "best_hyperparameters.json")
        with open(best_params_path, "w") as f:
            json.dump(best_params_output, f, indent=2)

        log.info(f"💾 Best hyperparameters saved to {best_params_path}")
        log.info(f"   Use these hyperparameters for lower-resolution experiments with same domain-teacher pair")

    def _ensure_teacher_embeddings_cached(self):
        """Ensure teacher embeddings are cached before training starts."""
        if not self.distillation_enabled or self.teacher_cache is None:
            return

        dataset_name = self.cfg.datamodule.dataset_name
        data_dir = getattr(self.cfg.datamodule, 'data_dir', None)

        # Check if embeddings are already cached
        if self.teacher_cache.exists(dataset_name, split="train"):
            log.info("✓ Teacher embeddings already cached")
            return

        log.info(f"\n{'='*60}")
        log.info("Caching teacher embeddings for distillation")
        log.info(f"{'='*60}\n")

        # Create dataloader for clean full-resolution images
        batch_size = getattr(self.cfg.datamodule, 'batch_size', 256)
        num_workers = getattr(self.cfg.datamodule, 'num_workers', 8)

        dataloader = create_clean_image_dataloader(
            dataset_name=dataset_name,
            data_dir=data_dir,
            split="train",
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=self.teacher_cache.full_resolution,
        )

        # Cache embeddings
        self.teacher_cache.cache_embeddings(
            dataloader=dataloader,
            dataset_name=dataset_name,
            split="train",
            force_recompute=False,
        )

        log.info(f"{'='*60}\n")

    def train(self) -> Dict[str, Any]:
        """
        Run K-fold cross-validation.

        Workflow:
        1. If hyperparameter search is enabled, perform grid search with 5-fold CV
           at the highest resolution and return the best hyperparameters.
        2. If pre-tuned hyperparameters are loaded (for lower-resolution experiments),
           apply them and run standard CV evaluation.
        3. Otherwise, run standard CV evaluation with current config.
        """
        # Ensure teacher embeddings are cached if distillation is enabled
        self._ensure_teacher_embeddings_cached()

        # If hyperparameter search is enabled, run it and return results
        if self.hyperparam_search_enabled:
            log.info("🔍 Hyperparameter search enabled - running 5-fold CV-based hyperparameter tuning")
            best_params, search_results = self._run_hyperparam_search()

            # Return the best results from the hyperparameter search
            # No need to re-run CV since we already did it for the best config
            best_result = [r for r in search_results if r["params"] == best_params][0]
            return {
                "best_hyperparameters": best_params,
                "fold_metrics": [],  # Not tracked individually in hyperparam search
                "mean_metric": best_result["mean_metric"],
                "std_metric": best_result["std_metric"],
            }

        # If pre-tuned hyperparameters are loaded, apply them
        if self.pretuned_hyperparams:
            log.info(f"\n{'='*60}")
            log.info("📌 Applying pre-tuned hyperparameters from high-resolution search")
            log.info(f"{'='*60}\n")

            self.cfg = self._apply_hyperparams_to_cfg(self.pretuned_hyperparams)

            # Update loss function if label_smoothing changed
            self.loss_fn = cross_entropy_loss(
                label_smoothing=float(getattr(self.cfg.loss, "label_smoothing", 0.0)),
                class_weight=None,
                ignore_index=int(getattr(self.cfg.loss, "ignore_index", -100)),
                reduction=str(getattr(self.cfg.loss, "reduction", "mean")),
            )

        # Run standard K-fold cross-validation
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
            result = train_probe(
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
    wrapper = ProbeCVWrapper(cfg)
    return wrapper.train()