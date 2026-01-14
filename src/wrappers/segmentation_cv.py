# src/wrappers/segmentation_cv.py
# -*- coding: utf-8 -*-
"""
K-Fold Cross-Validation wrapper for segmentation tasks.

This wrapper orchestrates the full segmentation training pipeline:
- Data loading and split management
- Model creation (DINOv3ForSegmentation)
- K-fold cross-validation training
- Results aggregation and reporting

Unlike classification (two-stage with embeddings), segmentation
trains end-to-end without embedding caching.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pathlib import Path

import os
import copy
import json
import itertools
import torch
import numpy as np
import hydra
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.utils.optim import make_optimizer_and_scheduler
from src.utils.split_manager import SplitManager
from src.engines.segmentation_engine import train_segmentation

log = get_logger(__name__)


class SegmentationCVWrapper:
    """
    K-Fold Cross-Validation wrapper for segmentation training.

    Provides end-to-end training pipeline with:
    - Consistent train/val/test splits
    - K-fold cross-validation
    - Frozen backbone (linear probing style)
    - Metrics aggregation across folds
    - Optional hyperparameter search
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        setup_logging()

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

        # Initialize datamodule
        log.info("Initializing datamodule...")
        self.dm = hydra.utils.instantiate(cfg.datamodule, full_cfg=cfg)
        self.dm.setup("fit")
        log.info(f"✓ Datamodule initialized: {self.dm.__class__.__name__}")

        # Split management
        self.dataset_name = self.dm.dataset_identifier
        split_dir = getattr(cfg, "split_dir", "./splits")
        self.split_manager = SplitManager(
            split_dir=split_dir,
            dataset_name=self.dataset_name,
            seed=int(getattr(cfg.train, "seed", 42)),
        )
        log.info(f"✓ Split manager initialized (seed={cfg.train.seed})")

        # Training config
        self.k_folds = int(getattr(cfg.train, "k_folds", 5))
        self.run_dir = Path(getattr(cfg.runtime, "run_dir", "./runs/segmentation"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"✓ Run directory: {self.run_dir}")

        # Hyperparameter search setup
        self.hyperparam_search_enabled = False
        self.pretuned_hyperparams = None

        hyperparam_search = getattr(cfg.train, "hyperparam_search", None)
        if hyperparam_search:
            self.hyperparam_search_enabled = getattr(hyperparam_search, "enabled", False)
            self.param_grid = getattr(hyperparam_search, "param_grid", {})

            if self.hyperparam_search_enabled:
                self.search_dir = self.run_dir / "hyperparam_search"
                self.search_dir.mkdir(exist_ok=True, parents=True)
                log.info(f"✓ Hyperparameter search enabled")

            # Load pre-tuned hyperparameters if available
            load_from_file = getattr(hyperparam_search, "load_from_file", None)
            if load_from_file and os.path.exists(load_from_file):
                self.pretuned_hyperparams = self._load_pretuned_hyperparams(load_from_file)

        # Wandb logging
        self.wandb = WandbLogger(
            project=getattr(cfg.logging, "project", "segmentation"),
            run_name=getattr(cfg.logging, "run_name", "seg_run"),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            tags=["segmentation", "k-fold-cv"],
        )

    def _create_model(self) -> torch.nn.Module:
        """
        Create segmentation model from config.

        Creates a DINOv3ForSegmentation model with config-specified parameters.
        Optionally freezes the backbone for faster training (default: freeze).

        Returns:
            Initialized model on appropriate device
        """
        from src.models.dinov3_segmentation import (
            DINOv3ForSegmentation,
            DINOv3SegmentationConfig
        )

        model_cfg = self.cfg.model

        # Extract configuration
        backbone_model_id = model_cfg.get("model_id", "facebook/dinov3-vits16-pretrain-lvd1689m")
        num_classes = model_cfg.config.get("num_classes", 1)
        hidden_size = model_cfg.config.get("hidden_size", 384)
        dropout_rate = model_cfg.config.get("dropout_rate", 0.1)
        loss_type = model_cfg.config.get("loss_type", "dice_bce")
        dice_weight = model_cfg.config.get("dice_weight", 0.5)

        # Get image size from data config
        image_size = int(getattr(self.cfg.data, "image_size", 256))

        log.info(f"\nCreating segmentation model:")
        log.info(f"  Backbone: {backbone_model_id}")
        log.info(f"  Num classes: {num_classes}")
        log.info(f"  Hidden size: {hidden_size}")
        log.info(f"  Image size: {image_size}x{image_size}")
        log.info(f"  Dropout: {dropout_rate}")
        log.info(f"  Loss type: {loss_type}")
        log.info(f"  Dice weight: {dice_weight}\n")

        # Create config
        config = DINOv3SegmentationConfig(
            backbone_model_id=backbone_model_id,
            num_classes=num_classes,
            hidden_size=hidden_size,
            patch_size=16,
            image_size=image_size,
            dropout_rate=dropout_rate,
            loss_type=loss_type,
            dice_weight=dice_weight,
        )

        # Instantiate model
        model = DINOv3ForSegmentation(config)

        # Optionally freeze backbone
        freeze_backbone = getattr(model_cfg, "freeze_backbone", True)
        if freeze_backbone:
            from src.models.factory import freeze_backbone as freeze_fn
            freeze_fn(model.backbone, "dinov3")
            log.info("✓ Frozen DINOv3 backbone (training only segmentation head)")

            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            log.info(f"  Trainable params: {trainable_params:,} / {total_params:,} "
                    f"({100*trainable_params/total_params:.1f}%)\n")
        else:
            log.info("✓ Training full model (backbone + segmentation head)\n")

        return model.to(self.device)

    def _train_single_fold(
        self,
        fold: int,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train model on a single fold.

        Args:
            fold: Fold number (0-indexed)
            train_indices: Training set indices for this fold
            val_indices: Validation set indices for this fold

        Returns:
            Dictionary with training results:
                - history: Training curves
                - best_metric: Best validation metric
                - final_val_dice, final_val_iou, final_val_pixel_acc
        """
        log.info(f"\n{'='*60}")
        log.info(f"Training Fold {fold+1}/{self.k_folds}")
        log.info(f"{'='*60}")
        log.info(f"Train samples: {len(train_indices)}")
        log.info(f"Val samples: {len(val_indices)}\n")

        # ========== Create Data Loaders ==========
        full_dataset = self.dm.full_dataset
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=getattr(self.cfg.datamodule, "pin_memory", True),
            persistent_workers=getattr(self.cfg.datamodule, "persistent_workers", False),
            prefetch_factor=getattr(self.cfg.datamodule, "prefetch_factor", 2),
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=getattr(self.cfg.datamodule, "pin_memory", True),
            persistent_workers=getattr(self.cfg.datamodule, "persistent_workers", False),
            prefetch_factor=getattr(self.cfg.datamodule, "prefetch_factor", 2),
        )

        loaders = {"train": train_loader, "val": val_loader}

        # ========== Create Model ==========
        model = self._create_model()

        # ========== Create Optimizer and Scheduler ==========
        optimizer, (scheduler, sched_meta) = make_optimizer_and_scheduler(
            self.cfg, model.parameters()
        )

        # ========== Training ==========
        checkpoint_dir = self.run_dir / f"fold_{fold+1}"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        result = train_segmentation(
            model=model,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=(scheduler, sched_meta),
            device=self.device,
            epochs=int(self.cfg.train.epochs),
            grad_clip=getattr(self.cfg.train, "grad_clip", None),
            mixed_precision=bool(getattr(self.cfg.train, "mixed_precision", True)),
            log_interval=int(getattr(self.cfg.train, "log_interval", 50)),
            wandb_logger=self.wandb,
            metric_key=str(getattr(self.cfg.train, "metric_key", "val_dice")),
            save_checkpoints=bool(getattr(self.cfg.logging, "save_checkpoints", True)),
            checkpoint_dir=checkpoint_dir,
        )

        log.info(f"\n✓ Fold {fold+1} complete:")
        log.info(f"  Best {result['best_metric_name']}: {result['best_metric']:.4f}")
        log.info(f"  Final Dice: {result['final_val_dice']:.4f}")
        log.info(f"  Final IoU: {result['final_val_iou']:.4f}")
        log.info(f"  Final Pixel Acc: {result['final_val_pixel_acc']:.4f}\n")

        return result

    def _train_final_model(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train final model on full 80% train set and evaluate on 20% test set.

        This implements the paper's Train@R, Test@R protocol:
        - Train on full training set (80%)
        - Evaluate on held-out test set (20%)
        - Report test set performance (not CV mean)

        Args:
            train_indices: Full training set indices (80% of data)
            test_indices: Test set indices (20% of data)

        Returns:
            Dictionary with final model results:
                - history: Training curves
                - best_metric: Best test metric
                - final_val_dice, final_val_iou, final_val_pixel_acc
        """
        log.info(f"\n{'='*60}")
        log.info(f"Final Evaluation: Train on 80% → Test on 20%")
        log.info(f"{'='*60}")
        log.info(f"Train samples: {len(train_indices)}")
        log.info(f"Test samples: {len(test_indices)}\n")

        # ========== Create Data Loaders ==========
        full_dataset = self.dm.full_dataset
        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=getattr(self.cfg.datamodule, "pin_memory", True),
            persistent_workers=getattr(self.cfg.datamodule, "persistent_workers", False),
            prefetch_factor=getattr(self.cfg.datamodule, "prefetch_factor", 2),
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=getattr(self.cfg.datamodule, "pin_memory", True),
            persistent_workers=getattr(self.cfg.datamodule, "persistent_workers", False),
            prefetch_factor=getattr(self.cfg.datamodule, "prefetch_factor", 2),
        )

        loaders = {"train": train_loader, "val": test_loader}  # Use test as "val"

        # ========== Create Model ==========
        model = self._create_model()

        # ========== Create Optimizer and Scheduler ==========
        optimizer, (scheduler, sched_meta) = make_optimizer_and_scheduler(
            self.cfg, model.parameters()
        )

        # ========== Training ==========
        checkpoint_dir = self.run_dir / "final_model"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        result = train_segmentation(
            model=model,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=(scheduler, sched_meta),
            device=self.device,
            epochs=int(self.cfg.train.epochs),
            grad_clip=getattr(self.cfg.train, "grad_clip", None),
            mixed_precision=bool(getattr(self.cfg.train, "mixed_precision", True)),
            log_interval=int(getattr(self.cfg.train, "log_interval", 50)),
            wandb_logger=self.wandb,
            metric_key=str(getattr(self.cfg.train, "metric_key", "val_dice")),
            save_checkpoints=bool(getattr(self.cfg.logging, "save_checkpoints", True)),
            checkpoint_dir=checkpoint_dir,
        )

        log.info(f"\n✓ Final model training complete:")
        log.info(f"  Test Dice: {result['final_val_dice']:.4f}")
        log.info(f"  Test IoU: {result['final_val_iou']:.4f}")
        log.info(f"  Test Pixel Acc: {result['final_val_pixel_acc']:.4f}\n")

        return result

    def _load_pretuned_hyperparams(self, filepath: str) -> Dict[str, Any]:
        """Load pre-tuned hyperparameters from JSON."""
        log.info(f"📥 Loading pre-tuned hyperparameters from {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)

        hyperparams = data.get("best_hyperparameters", {})
        metadata = data.get("metadata", {})

        log.info(f"✓ Loaded hyperparameters: {hyperparams}")
        log.info(f"  Original resolution: {metadata.get('resolution')}px")

        return hyperparams

    def _apply_hyperparams_to_cfg(self, params: Dict[str, Any]) -> Any:
        """Apply hyperparameters to config."""
        cfg = copy.deepcopy(self.cfg)

        if "lr" in params:
            cfg.train.optimizer.lr = params["lr"]
        if "weight_decay" in params:
            cfg.train.optimizer.weight_decay = params["weight_decay"]
        if "batch_size" in params:
            cfg.data.batch_size = params["batch_size"]
        if "decoder_dropout" in params:
            cfg.model.config.dropout_rate = params["decoder_dropout"]

        return cfg

    def _get_all_hyperparam_configs(self) -> List[Dict[str, Any]]:
        """Get all hyperparameter configurations from grid."""
        if not self.param_grid:
            return []

        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*self.param_grid.values()))

        return [dict(zip(keys, combo)) for combo in combos]

    def _run_cv_for_hyperparams(
        self,
        params: Dict[str, Any],
        cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[float, float]:
        """
        Run K-fold CV with specific hyperparameters.

        Args:
            params: Hyperparameters to evaluate
            cv_folds: List of (train_indices, val_indices) for each fold

        Returns:
            Tuple of (mean_metric, std_metric)
        """
        trial_cfg = self._apply_hyperparams_to_cfg(params)

        fold_metrics = []

        for fold, (train_fold_indices, val_fold_indices) in enumerate(cv_folds):
            log.info(f"  Fold {fold+1}/{len(cv_folds)}")

            # Create data loaders
            full_dataset = self.dm.full_dataset
            train_subset = Subset(full_dataset, train_fold_indices)
            val_subset = Subset(full_dataset, val_fold_indices)

            train_loader = DataLoader(
                train_subset,
                batch_size=trial_cfg.data.batch_size,
                shuffle=True,
                num_workers=self.cfg.datamodule.num_workers,
                pin_memory=getattr(self.cfg.datamodule, "pin_memory", True),
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=trial_cfg.data.batch_size,
                shuffle=False,
                num_workers=self.cfg.datamodule.num_workers,
                pin_memory=getattr(self.cfg.datamodule, "pin_memory", True),
            )

            loaders = {"train": train_loader, "val": val_loader}

            # Create model with trial config
            model = self._create_model()

            # Create optimizer and scheduler with trial config
            optimizer, (scheduler, sched_meta) = make_optimizer_and_scheduler(
                trial_cfg, model.parameters()
            )

            # Train
            result = train_segmentation(
                model=model,
                loaders=loaders,
                optimizer=optimizer,
                scheduler=(scheduler, sched_meta),
                device=self.device,
                epochs=int(trial_cfg.train.epochs),
                grad_clip=getattr(trial_cfg.train, "grad_clip", None),
                mixed_precision=bool(getattr(trial_cfg.train, "mixed_precision", True)),
                log_interval=int(getattr(trial_cfg.train, "log_interval", 50)),
                wandb_logger=None,  # Disable wandb during hyperparameter search
                metric_key=str(getattr(trial_cfg.train, "metric_key", "val_dice")),
                save_checkpoints=False,  # Don't save checkpoints during search
                checkpoint_dir=None,
            )

            fold_metrics.append(result["best_metric"])

        mean_metric = float(np.mean(fold_metrics))
        std_metric = float(np.std(fold_metrics))

        return mean_metric, std_metric

    def _run_hyperparam_search(
        self,
        cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run hyperparameter search using K-fold CV.

        Args:
            cv_folds: K-fold cross-validation splits

        Returns:
            Tuple of (best_params, all_results)
        """
        log.info(f"\n{'='*60}")
        log.info(f"Hyperparameter Tuning")
        log.info(f"{'='*60}\n")

        all_configs = self._get_all_hyperparam_configs()
        if not all_configs:
            log.warning("No hyperparameter grid specified")
            return {}, []

        log.info(f"Total configurations to evaluate: {len(all_configs)}")

        results = []
        for i, params in enumerate(all_configs, start=1):
            log.info(f"\nTrial {i}/{len(all_configs)}: {params}")

            try:
                mean_metric, std_metric = self._run_cv_for_hyperparams(
                    params=params,
                    cv_folds=cv_folds,
                )

                result_entry = {
                    "trial": i,
                    "params": params,
                    "mean_metric": mean_metric,
                    "std_metric": std_metric,
                }
                results.append(result_entry)

                log.info(f"✓ Trial {i}: {mean_metric:.4f} ± {std_metric:.4f}")

            except Exception as e:
                log.error(f"✗ Trial {i} failed: {e}")
                result_entry = {
                    "trial": i,
                    "params": params,
                    "mean_metric": float('-inf'),
                    "std_metric": 0.0,
                    "error": str(e),
                }
                results.append(result_entry)

        metric_key = str(getattr(self.cfg.train, "metric_key", "val_dice"))
        reverse = not metric_key.endswith("loss")  # Maximize Dice/IoU, minimize loss
        valid_results = [r for r in results if r["mean_metric"] != float('-inf')]

        if not valid_results:
            raise RuntimeError(f"All hyperparameter trials failed!")

        best_result = sorted(valid_results, key=lambda x: x["mean_metric"], reverse=reverse)[0]
        best_params = best_result["params"]

        log.info(f"\n{'='*60}")
        log.info(f"🏆 Best Hyperparameters:")
        log.info(f"  {best_params}")
        log.info(f"  Metric: {best_result['mean_metric']:.4f} ± {best_result['std_metric']:.4f}")
        log.info(f"{'='*60}\n")

        self._save_hyperparam_results(results, best_result)

        return best_params, results

    def _to_serializable(self, obj: Any) -> Any:
        """Convert OmegaConf objects to regular Python objects for JSON serialization."""
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_serializable(item) for item in obj]
        else:
            return obj

    def _save_hyperparam_results(self, results: List[Dict[str, Any]], best_result: Dict[str, Any]):
        """Save hyperparameter search results."""
        output = {
            "search_config": {
                "param_grid": self._to_serializable(self.param_grid),
                "k_folds": self.k_folds,
                "image_size": int(getattr(self.cfg.data, "image_size", 256)),
                "model": self.cfg.model.name,
            },
            "best_result": self._to_serializable(best_result),
            "all_results": self._to_serializable(results),
        }

        output_path = self.search_dir / "hyperparam_search_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        best_params_output = {
            "best_hyperparameters": self._to_serializable(best_result["params"]),
            "validation_metric": {
                "mean": best_result["mean_metric"],
                "std": best_result["std_metric"],
            },
            "metadata": {
                "image_size": int(getattr(self.cfg.data, "image_size", 256)),
                "model": self.cfg.model.name,
            },
        }

        best_params_path = self.search_dir / "best_hyperparameters.json"
        with open(best_params_path, "w") as f:
            json.dump(best_params_output, f, indent=2)

        log.info(f"💾 Results saved to {output_path}")
        log.info(f"💾 Best params saved to {best_params_path}")

    def run(self) -> Dict[str, Any]:
        """
        Run K-fold cross-validation for segmentation.

        This is the main entry point for the wrapper. It:
        1. Creates or loads train/test splits
        2. Creates K CV folds from training data
        3. Trains a model on each fold
        4. Aggregates results across folds

        Returns:
            Dictionary with aggregated results:
                - mean_metric: Mean of primary metric across folds
                - std_metric: Standard deviation of primary metric
                - fold_metrics: List of metrics for each fold
                - fold_results: Full results for each fold
        """
        log.info(f"\n{'='*60}")
        log.info("Starting K-Fold Cross-Validation")
        log.info(f"{'='*60}")
        log.info(f"Dataset: {self.dataset_name}")
        log.info(f"K-folds: {self.k_folds}")
        log.info(f"{'='*60}\n")

        # ========== Setup Splits ==========
        full_dataset = self.dm.full_dataset
        dataset_size = len(full_dataset)
        log.info(f"Full dataset size: {dataset_size}")

        # Get stratification labels (if available)
        # For segmentation, stratification might not be applicable
        try:
            stratify_labels = self.dm.get_labels_for_stratification(full_dataset)
            log.info(f"Using stratification based on labels")
        except Exception:
            stratify_labels = None
            log.info(f"No stratification (pure random splits)")

        # Create or load train/test splits
        if not self.split_manager.exists():
            log.info("Creating new train/test splits...")
            splits = self.split_manager.create_splits(
                dataset_size=dataset_size,
                use_val_split=False,
                train_ratio=0.8,
                stratify_labels=stratify_labels,
            )
            log.info("✓ Splits created")
        else:
            log.info("Loading existing train/test splits...")
            splits = self.split_manager.load_splits()
            log.info("✓ Splits loaded")

        train_indices = splits["train"]
        log.info(f"Training set size: {len(train_indices)}\n")

        # ========== Create CV Folds ==========
        log.info(f"Creating {self.k_folds} CV folds...")
        cv_folds = self.split_manager.create_cv_folds(
            train_indices=train_indices,
            n_folds=self.k_folds,
            stratify_labels=stratify_labels,
        )
        log.info(f"✓ CV folds created\n")

        # ========== Hyperparameter Search (if enabled) ==========
        if self.hyperparam_search_enabled:
            best_params, _ = self._run_hyperparam_search(cv_folds)

            # Apply best hyperparameters to config
            self.cfg = self._apply_hyperparams_to_cfg(best_params)
            log.info("✓ Applied best hyperparameters to config\n")

        elif self.pretuned_hyperparams:
            log.info("📌 Applying pre-tuned hyperparameters")
            self.cfg = self._apply_hyperparams_to_cfg(self.pretuned_hyperparams)
            log.info("✓ Applied pre-tuned hyperparameters to config\n")

        # ========== Final Evaluation: Train@R, Test@R ==========
        # Get test set indices
        test_indices = splits["test"]
        log.info(f"Test set size: {len(test_indices)}\n")

        # Train final model on full 80% train, evaluate on 20% test
        final_result = self._train_final_model(
            train_indices=train_indices,
            test_indices=test_indices,
        )

        # Report test set performance (not CV mean)
        metric_key = str(getattr(self.cfg.train, "metric_key", "val_dice"))

        log.info(f"\n{'='*60}")
        log.info(f"Final Test Set Results")
        log.info(f"{'='*60}")
        log.info(f"Test Dice: {final_result['final_val_dice']:.4f}")
        log.info(f"Test IoU: {final_result['final_val_iou']:.4f}")
        log.info(f"Test Pixel Acc: {final_result['final_val_pixel_acc']:.4f}")
        log.info(f"Best {metric_key}: {final_result['best_metric']:.4f}")
        log.info(f"{'='*60}\n")

        # Log to wandb
        if self.wandb and self.wandb.enabled:
            self.wandb.log({
                "test_dice": final_result['final_val_dice'],
                "test_iou": final_result['final_val_iou'],
                "test_pixel_acc": final_result['final_val_pixel_acc'],
                f"best_{metric_key}": final_result['best_metric'],
            })

        return {
            "test_dice": final_result['final_val_dice'],
            "test_iou": final_result['final_val_iou'],
            "test_pixel_acc": final_result['final_val_pixel_acc'],
            "best_metric": final_result['best_metric'],
            "metric_key": metric_key,
            "history": final_result['history'],
        }


def run(cfg: Any) -> Dict[str, Any]:
    """
    Entry point for segmentation training.

    Called by train.py when mode == "segmentation".

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with cross-validation results
    """
    wrapper = SegmentationCVWrapper(cfg)
    return wrapper.run()
