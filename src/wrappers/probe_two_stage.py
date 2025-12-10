# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""
Two-stage linear probing wrapper with embedding caching.

Pipeline:
(A) Data loading & split definition
    - Load dataset and create consistent train/val/test splits
    - Store split indices for reuse across all experiments

(B) Image preprocessing
    - Load images at target resolution R
    - No augmentations for LP (augmentations only for distillation)

(C) Frozen DINOv3 encoder (Stage 1)
    - For each resolution R: extract and cache embeddings
    - Embeddings stored in: cache/{dataset}/{model}/{R}px/

(D) Linear Probing (Stage 2)
    - Hyperparameter tuning at full resolution R* only
      * Use 5-fold CV on cached embeddings
      * Search over LR, weight decay, batch size, etc.
    - Final LP training at each resolution R
      * Load cached embeddings at resolution R
      * Train with fixed hyperparameters from tuning
      * Evaluate on test set
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import os
import torch
import numpy as np
import hydra
import json
import copy
import itertools
from torch.utils.data import DataLoader
from pathlib import Path

# pylint: disable=import-error
from src.engines.linear_probe_embedding_engine import train_probe_on_embeddings
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.utils.optim import make_optimizer_and_scheduler
from src.losses.classification import cross_entropy_loss
from src.models.factory import create_model, freeze_backbone
from src.utils.embedding_cache import EmbeddingCache
from src.utils.split_manager import SplitManager
from src.data.embedding_dataset import EmbeddingDataset, SubsetEmbeddingDataset

log = get_logger(__name__)


class ProbeTwoStageWrapper:
    """
    Two-stage linear probing with embedding caching.

    Stage 1: Extract and cache embeddings at each resolution
    Stage 2: Train linear probe on cached embeddings
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        setup_logging()

        self.current_resolution = getattr(cfg.data, "image_size", None) or getattr(cfg.dataset, "image_size", 224)

        self.expected_high_res_map = {
            "dermatology": 512,
        }

        self.domain = getattr(cfg, "domain", None)
        self.expected_high_res = self.expected_high_res_map.get(self.domain, 512)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize data module
        self.dm = hydra.utils.instantiate(cfg.datamodule, full_cfg=cfg)
        self.dm.setup("fit")

        # Dataset info
        self.dataset_name = getattr(cfg.datamodule, "dataset_name", "unknown")
        self.model_info = cfg.model
        self.model_name = self.model_info.get("name", "dinov3")

        split_dir = getattr(cfg, "split_dir", "./splits")
        self.split_manager = SplitManager(
            split_dir=split_dir,
            dataset_name=self.dataset_name,
            seed=int(getattr(cfg.train, "seed", 42)),
        )

        cache_dir = getattr(cfg, "embedding_cache_dir", "./cache/embeddings")
        self.embedding_cache = EmbeddingCache(
            cache_dir=cache_dir,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            device=self.device,
        )

        self.k_folds = int(getattr(cfg.train, "k_folds", 5))
        self.loss_fn = cross_entropy_loss(
            label_smoothing=float(getattr(cfg.loss, "label_smoothing", 0.0)),
            class_weight=None,
            ignore_index=int(getattr(cfg.loss, "ignore_index", -100)),
            reduction=str(getattr(cfg.loss, "reduction", "mean")),
        )

        self.wandb = WandbLogger(
            project=getattr(cfg.logging, "project", "two-stage-probe"),
            run_name=getattr(cfg.logging, "run_name", "two_stage_run"),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            tags=["two-stage", "linear-probe"],
        )

        self.run_dir = getattr(cfg.runtime, "run_dir", "./runs/probe_two_stage")
        os.makedirs(self.run_dir, exist_ok=True)

        self.hyperparam_search_enabled = False
        self.pretuned_hyperparams = None

        hyperparam_search = getattr(cfg.train, "hyperparam_search", None)
        if hyperparam_search:
            self.hyperparam_search_enabled = getattr(hyperparam_search, "enabled", False)
            self.param_grid = getattr(hyperparam_search, "param_grid", {})

            if self.hyperparam_search_enabled:
                if self.current_resolution < self.expected_high_res:
                    log.warning(
                        f"⚠️  Hyperparameter search should be at highest resolution! "
                        f"Current: {self.current_resolution}px, Expected: {self.expected_high_res}px"
                    )
                else:
                    log.info(f"✓ Hyperparameter search at highest resolution: {self.current_resolution}px")

                self.search_dir = os.path.join(self.run_dir, "hyperparam_search")
                os.makedirs(self.search_dir, exist_ok=True)

            # Load pre-tuned hyperparameters if available
            load_from_file = getattr(hyperparam_search, "load_from_file", None)
            if load_from_file and os.path.exists(load_from_file):
                self.pretuned_hyperparams = self._load_pretuned_hyperparams(load_from_file)

    def _load_pretuned_hyperparams(self, filepath: str) -> Dict[str, Any]:
        """Load pre-tuned hyperparameters from JSON."""
        log.info(f"📥 Loading pre-tuned hyperparameters from {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)

        hyperparams = data.get("best_hyperparameters", {})
        metadata = data.get("metadata", {})

        log.info(f"✓ Loaded hyperparameters: {hyperparams}")
        log.info(f"  Original resolution: {metadata.get('resolution')}px")
        log.info(f"  Domain: {metadata.get('domain')}")

        return hyperparams

    def _setup_splits(self) -> Dict[str, np.ndarray]:
        """
        Setup and save data splits.

        Returns:
            Dict with 'train', 'test', and optionally 'val' indices
        """
        log.info(f"\n{'='*60}")
        log.info("(A) Data Loading & Split Definition")
        log.info(f"{'='*60}\n")

        full_dataset = self.dm.train_set
        dataset_size = len(full_dataset)

        stratify_labels = None
        if hasattr(full_dataset, 'targets'):
            stratify_labels = np.array(full_dataset.targets)
        elif hasattr(full_dataset, 'labels'):
            stratify_labels = np.array(full_dataset.labels)

        if not self.split_manager.exists():
            log.info("Creating new splits...")
            splits = self.split_manager.create_splits(
                dataset_size=dataset_size,
                use_val_split=False,
                train_ratio=0.8,
                stratify_labels=stratify_labels,
            )
        else:
            log.info("Loading existing splits...")
            splits = self.split_manager.load_splits()

        return splits

    def _extract_embeddings_for_split(
        self,
        model: torch.nn.Module,
        split_indices: np.ndarray,
        split_name: str,
        resolution: int,
        force_recompute: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and cache embeddings for a data split.

        Args:
            model: Frozen DINOv3 model
            split_indices: Indices for this split
            split_name: 'train', 'val', or 'test'
            resolution: Image resolution
            force_recompute: Force re-extraction even if cached

        Returns:
            Tuple of (embeddings, labels)
        """
        if not force_recompute and self.embedding_cache.exists(resolution, split_name):
            log.info(f"✓ {split_name} embeddings already cached at {resolution}px")
            return self.embedding_cache.load(resolution, split_name)

        log.info(f"🔄 Extracting {split_name} embeddings at {resolution}px...")

        full_dataset = self.dm.train_set if split_name in ["train", "val"] else self.dm.test_set
        from torch.utils.data import Subset
        subset = Subset(full_dataset, split_indices)

        dataloader = DataLoader(
            subset,
            batch_size=getattr(self.cfg.data, "batch_size", 256),
            shuffle=False,
            num_workers=getattr(self.cfg.datamodule, "num_workers", 8),
            pin_memory=True,
        )

        embeddings, labels = self.embedding_cache.extract_and_cache(
            model=model,
            dataloader=dataloader,
            resolution=resolution,
            split=split_name,
            model_info=self.model_info,
            mixed_precision=bool(getattr(self.cfg.train, "mixed_precision", True)),
            force_recompute=force_recompute,
        )

        return embeddings, labels

    def _extract_all_embeddings(
        self,
        resolution: int,
        splits: Dict[str, np.ndarray],
        force_recompute: bool = False,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract embeddings for all splits at a given resolution.

        Args:
            resolution: Image resolution
            splits: Dict of split indices
            force_recompute: Force re-extraction

        Returns:
            Dict mapping split_name -> (embeddings, labels)
        """
        log.info(f"\n{'='*60}")
        log.info(f"(C) Frozen DINOv3 Encoder - Extracting Embeddings at {resolution}px")
        log.info(f"{'='*60}\n")

        model = create_model(self.model_info, resolution=resolution)
        model = model.to(self.device)
        model.eval()

        freeze_backbone(model, self.model_info.get("type", "dinov3"))

        all_embeddings = {}
        for split_name, split_indices in splits.items():
            embeddings, labels = self._extract_embeddings_for_split(
                model=model,
                split_indices=split_indices,
                split_name=split_name,
                resolution=resolution,
                force_recompute=force_recompute,
            )
            all_embeddings[split_name] = (embeddings, labels)

        del model
        torch.cuda.empty_cache()

        return all_embeddings

    def _create_linear_classifier(self, embedding_dim: int, num_classes: int) -> torch.nn.Module:
        """Create a simple linear classifier for embeddings."""
        return torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Dropout(float(getattr(self.cfg.model.get("config", {}), "dropout_rate", 0.1))),
            torch.nn.Linear(embedding_dim, num_classes),
        ).to(self.device)

    def _apply_hyperparams_to_cfg(self, params: Dict[str, Any]) -> Any:
        """Apply hyperparameters to config."""
        cfg = copy.deepcopy(self.cfg)

        if "lr" in params:
            cfg.train.optimizer.lr = params["lr"]
        if "weight_decay" in params:
            cfg.train.optimizer.weight_decay = params["weight_decay"]
        if "batch_size" in params:
            cfg.data.batch_size = params["batch_size"]
        if "label_smoothing" in params:
            cfg.loss.label_smoothing = params["label_smoothing"]

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
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[float, float]:
        """
        Run 5-fold CV with specific hyperparameters on embeddings.

        Args:
            params: Hyperparameters to evaluate
            train_embeddings: Training embeddings
            train_labels: Training labels
            cv_folds: List of (train_indices, val_indices) for each fold

        Returns:
            Tuple of (mean_metric, std_metric)
        """
        trial_cfg = self._apply_hyperparams_to_cfg(params)

        fold_metrics = []

        for fold, (train_fold_indices, val_fold_indices) in enumerate(cv_folds):
            log.info(f"  Fold {fold+1}/{len(cv_folds)}")

            train_dataset = SubsetEmbeddingDataset(
                train_embeddings, train_labels, torch.from_numpy(train_fold_indices)
            )
            val_dataset = SubsetEmbeddingDataset(
                train_embeddings, train_labels, torch.from_numpy(val_fold_indices)
            )

            loaders = {
                "train": DataLoader(
                    train_dataset,
                    batch_size=trial_cfg.data.batch_size,
                    shuffle=True,
                    num_workers=0,  # Embeddings are already in memory
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=trial_cfg.data.batch_size,
                    shuffle=False,
                    num_workers=0,
                ),
            }

            embedding_dim = train_embeddings.shape[1]
            num_classes = int(train_labels.max().item()) + 1
            classifier = self._create_linear_classifier(embedding_dim, num_classes)

            optimizer, (scheduler, sched_meta) = make_optimizer_and_scheduler(
                trial_cfg, classifier.parameters()
            )

            loss_fn = cross_entropy_loss(
                label_smoothing=float(getattr(trial_cfg.loss, "label_smoothing", 0.0)),
                class_weight=None,
                ignore_index=-100,
                reduction="mean",
            )

            result = train_probe_on_embeddings(
                classifier=classifier,
                loaders=loaders,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=(scheduler, sched_meta),
                device=self.device,
                epochs=int(trial_cfg.train.epochs),
                grad_clip=getattr(trial_cfg.train, "grad_clip", None),
                mixed_precision=bool(getattr(trial_cfg.train, "mixed_precision", True)),
                log_interval=int(getattr(trial_cfg.train, "log_interval", 50)),
                wandb_logger=None,
                metric_key=str(getattr(trial_cfg.train, "metric_key", "val_acc")),
            )

            fold_metrics.append(result["best_metric"])

        mean_metric = float(np.mean(fold_metrics))
        std_metric = float(np.std(fold_metrics))

        return mean_metric, std_metric

    def _run_hyperparam_search(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run hyperparameter search using 5-fold CV on embeddings.

        This should only be run at the highest resolution.

        Returns:
            Tuple of (best_params, all_results)
        """
        log.info(f"\n{'='*60}")
        log.info(f"(D1) Hyperparameter Tuning at {self.current_resolution}px")
        log.info(f"{'='*60}\n")

        splits = self.split_manager.load_splits()
        train_indices = splits["train"]

        stratify_labels = None
        full_dataset = self.dm.train_set
        if hasattr(full_dataset, 'targets'):
            stratify_labels = np.array(full_dataset.targets)
        elif hasattr(full_dataset, 'labels'):
            stratify_labels = np.array(full_dataset.labels)

        cv_folds = self.split_manager.create_cv_folds(
            train_indices=train_indices,
            n_folds=self.k_folds,
            stratify_labels=stratify_labels,
        )

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
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
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

        metric_key = str(getattr(self.cfg.train, "metric_key", "val_acc"))
        reverse = not metric_key.endswith("loss")
        valid_results = [r for r in results if r["mean_metric"] != float('-inf')]

        if not valid_results:
            raise RuntimeError("All hyperparameter trials failed!")

        best_result = sorted(valid_results, key=lambda x: x["mean_metric"], reverse=reverse)[0]
        best_params = best_result["params"]

        log.info(f"\n{'='*60}")
        log.info(f"🏆 Best Hyperparameters:")
        log.info(f"  {best_params}")
        log.info(f"  Metric: {best_result['mean_metric']:.4f} ± {best_result['std_metric']:.4f}")
        log.info(f"{'='*60}\n")

        self._save_hyperparam_results(results, best_result)

        return best_params, results

    def _save_hyperparam_results(self, results: List[Dict[str, Any]], best_result: Dict[str, Any]):
        """Save hyperparameter search results."""
        output = {
            "search_config": {
                "param_grid": self.param_grid,
                "k_folds": self.k_folds,
                "resolution": self.current_resolution,
                "domain": self.domain,
                "model": self.model_name,
            },
            "best_result": best_result,
            "all_results": results,
        }

        output_path = os.path.join(self.search_dir, "hyperparam_search_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        best_params_output = {
            "best_hyperparameters": best_result["params"],
            "validation_metric": {
                "mean": best_result["mean_metric"],
                "std": best_result["std_metric"],
            },
            "metadata": {
                "resolution": self.current_resolution,
                "domain": self.domain,
                "teacher_model": self.model_name,
            },
        }

        best_params_path = os.path.join(self.search_dir, "best_hyperparameters.json")
        with open(best_params_path, "w") as f:
            json.dump(best_params_output, f, indent=2)

        log.info(f"💾 Results saved to {output_path}")
        log.info(f"💾 Best params saved to {best_params_path}")

    def _train_final_probe(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        test_embeddings: torch.Tensor,
        test_labels: torch.Tensor,
        resolution: int,
    ) -> Dict[str, Any]:
        """
        Train final linear probe at a specific resolution.

        Args:
            train_embeddings: Training embeddings
            train_labels: Training labels
            test_embeddings: Test embeddings
            test_labels: Test labels
            resolution: Image resolution

        Returns:
            Dict with training results
        """
        log.info(f"\n{'='*60}")
        log.info(f"(D2) Final Linear Probing at {resolution}px")
        log.info(f"{'='*60}\n")

        train_dataset = EmbeddingDataset(train_embeddings, train_labels)
        test_dataset = EmbeddingDataset(test_embeddings, test_labels)

        loaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=self.cfg.data.batch_size,
                shuffle=True,
                num_workers=0,
            ),
            "val": DataLoader(
                test_dataset,
                batch_size=self.cfg.data.batch_size,
                shuffle=False,
                num_workers=0,
            ),
        }

        embedding_dim = train_embeddings.shape[1]
        num_classes = int(train_labels.max().item()) + 1
        classifier = self._create_linear_classifier(embedding_dim, num_classes)

        optimizer, (scheduler, sched_meta) = make_optimizer_and_scheduler(
            self.cfg, classifier.parameters()
        )

        result = train_probe_on_embeddings(
            classifier=classifier,
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

        log.info(f"✓ Final probe at {resolution}px: {result['best_metric']:.4f}")

        return result

    def run(self) -> Dict[str, Any]:
        """
        Run the two-stage linear probing pipeline.

        Returns:
            Dict with results
        """
        splits = self._setup_splits()

        all_embeddings = self._extract_all_embeddings(
            resolution=self.current_resolution,
            splits=splits,
            force_recompute=False,
        )

        train_embeddings, train_labels = all_embeddings["train"]
        test_embeddings, test_labels = all_embeddings["test"]

        if self.hyperparam_search_enabled:
            best_params, _ = self._run_hyperparam_search(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
            )

            self.cfg = self._apply_hyperparams_to_cfg(best_params)
            self.loss_fn = cross_entropy_loss(
                label_smoothing=float(getattr(self.cfg.loss, "label_smoothing", 0.0)),
                class_weight=None,
                ignore_index=-100,
                reduction="mean",
            )

        elif self.pretuned_hyperparams:
            log.info("📌 Applying pre-tuned hyperparameters")
            self.cfg = self._apply_hyperparams_to_cfg(self.pretuned_hyperparams)
            self.loss_fn = cross_entropy_loss(
                label_smoothing=float(getattr(self.cfg.loss, "label_smoothing", 0.0)),
                class_weight=None,
                ignore_index=-100,
                reduction="mean",
            )

        result = self._train_final_probe(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            resolution=self.current_resolution,
        )

        return result


def run(cfg: Any) -> Dict[str, Any]:
    """Entry point."""
    wrapper = ProbeTwoStageWrapper(cfg)
    return wrapper.run()
