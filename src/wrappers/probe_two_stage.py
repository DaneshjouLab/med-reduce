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
import traceback
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import OmegaConf

# pylint: disable=import-error
from src.engines.linear_probe_embedding_engine import train_probe_on_embeddings
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.utils.optim import make_optimizer_and_scheduler
from src.utils.training_utils import profile_model, calculate_inference_latency, get_gpu_memory
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

        self.dataset_name = self.dm.dataset_identifier
        self.model_info = cfg.model
        self.model_name = self.model_info.get("name", "dinov3")

        self.seed = int(getattr(cfg.train, "seed", 42))

        split_dir = getattr(cfg, "split_dir", "./splits")
        self.split_manager = SplitManager(
            split_dir=split_dir,
            dataset_name=self.dataset_name,
            seed=self.seed,
        )

        cache_dir = getattr(cfg, "embedding_cache_dir", "./cache/embeddings")
        self.embedding_cache = EmbeddingCache(
            cache_dir=cache_dir,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            seed=self.seed,
            device=self.device,
        )

        self.k_folds = int(getattr(cfg.train, "k_folds", 5))

        self.force_recompute = getattr(cfg.datamodule, "force_recompute_embeddings", False)

        self.class_weights = None
        self.loss_fn = None  # Will be created after computing class weights

        self.wandb = WandbLogger(
            project=getattr(cfg.logging, "project", "two-stage-probe"),
            run_name=getattr(cfg.logging, "run_name", "two_stage_run"),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            tags=["two-stage", "linear-probe", f"{self.current_resolution}px", self.model_name],
        )

        self.wandb.log({
            "config/resolution": self.current_resolution,
            "config/model_name": self.model_name,
            "config/domain": self.domain,
            "config/seed": self.seed,
        })

        base_run_dir = getattr(cfg.runtime, "run_dir", "./runs/probe_two_stage")
        self.run_dir = os.path.join(base_run_dir, f"seed_{self.seed}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Efficiency metrics storage
        self.efficiency_metrics = {
            "resolution": self.current_resolution,
            "model_name": self.model_name,
            "domain": self.domain,
            "encoder_gflops": None,
            "encoder_latency_ms": None,
            "embedding_extraction_time_s": None,
            "peak_gpu_memory_mb": None,
        }

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

    def _compute_class_weights(self, train_labels: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
        """
        Compute class weights for balanced loss function.

        Uses inverse frequency: weight_i = n_samples / (n_classes * count_i)

        Args:
            train_labels: Training labels
            num_classes: Total number of classes. If None, inferred from max label + 1.

        Returns:
            Tensor of class weights (one per class)
        """
        train_labels_np = train_labels.cpu().numpy()
        n_samples = len(train_labels_np)

        if num_classes is None:
            num_classes = int(train_labels_np.max()) + 1

        # Count samples per class (including classes with 0 samples)
        class_counts = np.zeros(num_classes, dtype=np.float64)
        unique_classes, counts = np.unique(train_labels_np, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            class_counts[cls] = count

        # Compute weights, handling classes with 0 samples
        class_weights = np.zeros(num_classes, dtype=np.float64)
        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weights[i] = n_samples / (num_classes * class_counts[i])
            else:
                # Assign weight of 0 for missing classes (won't affect loss since no samples)
                class_weights[i] = 0.0

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        log.info(f"\n{'='*60}")
        log.info("Class Weight Computation")
        log.info(f"{'='*60}")
        log.info(f"Training set size: {n_samples}")
        log.info(f"Number of classes: {num_classes}")
        for cls in range(num_classes):
            count = int(class_counts[cls])
            weight = class_weights[cls]
            pct = count / n_samples * 100 if n_samples > 0 else 0
            log.info(f"  Class {cls}: {count} samples ({pct:.1f}%) -> weight: {weight:.4f}")
        log.info(f"{'='*60}\n")

        return class_weights_tensor

    def _setup_splits(self) -> Dict[str, np.ndarray]:
        """
        Setup and save data splits.

        Returns:
            Dict with 'train', 'test', and optionally 'val' indices
        """
        log.info(f"\n{'='*60}")
        log.info("(A) Data Loading & Split Definition")
        log.info(f"{'='*60}\n")

        if hasattr(self.dm, 'full_dataset'):
            full_dataset = self.dm.full_dataset
        else:
            log.warning("⚠️  Using dm.train_set as full_dataset (may cause issues if already subsetted)")
            full_dataset = self.dm.train_set

        dataset_size = len(full_dataset)
        log.info(f"Full dataset size: {dataset_size}")

        stratify_labels = self.dm.get_labels_for_stratification(full_dataset)

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

        from torch.utils.data import Subset
        if hasattr(self.dm, 'full_dataset'):
            full_dataset = self.dm.full_dataset
        else:
            log.warning("⚠️  full_dataset not found, using train_set")
            full_dataset = self.dm.train_set

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
        import time

        log.info(f"\n{'='*60}")
        log.info(f"(C) Frozen Encoder - Extracting Embeddings at {resolution}px")
        log.info(f"{'='*60}\n")

        model = create_model(self.model_info, resolution=resolution)
        model = model.to(self.device)
        model.eval()

        freeze_backbone(model, self.model_info.get("type", "dinov3"))

        # Profile encoder efficiency metrics
        log.info(f"📊 Profiling encoder efficiency at {resolution}px...")

        encoder_gflops = profile_model(model, resolution)
        if encoder_gflops > 0:
            self.efficiency_metrics["encoder_gflops"] = encoder_gflops
            log.info(f"  Encoder GFLOPs: {encoder_gflops:.2f}")
        else:
            log.warning("  Could not compute encoder GFLOPs")

        encoder_latency = calculate_inference_latency(model, resolution)
        if encoder_latency > 0:
            self.efficiency_metrics["encoder_latency_ms"] = encoder_latency
            log.info(f"  Encoder latency: {encoder_latency:.2f} ms")
        else:
            log.warning("  Could not compute encoder latency")

        peak_memory = get_gpu_memory()
        if peak_memory > 0:
            self.efficiency_metrics["peak_gpu_memory_mb"] = peak_memory
            log.info(f"  Peak GPU memory: {peak_memory} MB")

        extraction_start = time.time()

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

        extraction_time = time.time() - extraction_start
        self.efficiency_metrics["embedding_extraction_time_s"] = extraction_time
        log.info(f"  Total embedding extraction time: {extraction_time:.2f}s")

        del model
        torch.cuda.empty_cache()

        return all_embeddings

    def _create_linear_classifier(self, embedding_dim: int, num_classes: int) -> torch.nn.Module:
        """Create a simple linear classifier for embeddings."""
        # Safely read dropout rate from model config which may be an OmegaConf DictConfig
        # or a plain dict/object
        dropout_source = self.cfg.model.get("config", {}) if hasattr(self.cfg, "model") else {}
        dropout_rate = float(self._cfg_get(dropout_source, "dropout_rate", 0.1))

        return torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(embedding_dim, num_classes),
        ).to(self.device)

    def _cfg_get(self, section: Any, key: str, default: Any = None) -> Any:
        """Safe config getter that handles DictConfig, dict or objects with attributes.

        This avoids AttributeError when a config section is a plain dict (which does not
        support attribute access) or an OmegaConf DictConfig (which may support both).
        """
        if section is None:
            return default

        # OmegaConf's DictConfig acts like a mapping but also supports attribute access.
        try:
            # dict-like access
            if isinstance(section, dict):
                return section.get(key, default)
        except Exception:
            pass

        # Try attribute access (works for DictConfig or simple objects)
        try:
            return getattr(section, key, default)
        except Exception:
            # Fallback to mapping-style get if present
            try:
                return section.get(key, default)  # type: ignore[attr-defined]
            except Exception:
                return default

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
                     These are ABSOLUTE indices into the original dataset.
                     Need to be converted to RELATIVE indices for embeddings.

        Returns:
            Tuple of (mean_metric, std_metric)
        """
        trial_cfg = self._apply_hyperparams_to_cfg(params)

        fold_metrics = []

        splits = self.split_manager.load_splits()
        train_indices = splits["train"]

        abs_to_rel = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(train_indices)}

        for fold, (train_fold_indices, val_fold_indices) in enumerate(cv_folds):
            log.info(f"  Fold {fold+1}/{len(cv_folds)}")

            train_fold_rel = np.array([abs_to_rel[idx] for idx in train_fold_indices])
            val_fold_rel = np.array([abs_to_rel[idx] for idx in val_fold_indices])

            train_dataset = SubsetEmbeddingDataset(
                train_embeddings, train_labels, torch.from_numpy(train_fold_rel)
            )
            val_dataset = SubsetEmbeddingDataset(
                train_embeddings, train_labels, torch.from_numpy(val_fold_rel)
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

            train_fold_labels = train_labels[train_fold_rel]
            fold_class_weights = self._compute_class_weights(train_fold_labels, num_classes=num_classes)

            loss_fn = cross_entropy_loss(
                label_smoothing=float(getattr(trial_cfg.loss, "label_smoothing", 0.0)),
                class_weight=fold_class_weights,
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

        # Filter out NaN values (from folds where AUROC couldn't be computed)
        valid_metrics = [m for m in fold_metrics if not np.isnan(m)]
        if len(valid_metrics) < len(fold_metrics):
            log.warning(
                f"  {len(fold_metrics) - len(valid_metrics)}/{len(fold_metrics)} folds had invalid metrics (NaN), "
                f"averaging over {len(valid_metrics)} valid folds"
            )

        if not valid_metrics:
            log.warning("  All folds had invalid metrics!")
            return float('nan'), float('nan')

        mean_metric = float(np.mean(valid_metrics))
        std_metric = float(np.std(valid_metrics))

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

        # Validate input embeddings and labels
        log.info(f"Validating inputs...")
        log.info(f"  Train embeddings shape: {train_embeddings.shape}")
        log.info(f"  Train labels shape: {train_labels.shape}")
        log.info(f"  Train embeddings device: {train_embeddings.device}")
        log.info(f"  Train labels device: {train_labels.device}")
        log.info(f"  Train labels dtype: {train_labels.dtype}")
        log.info(f"  Unique labels: {torch.unique(train_labels).tolist()}")

        assert train_embeddings.shape[0] == train_labels.shape[0], \
            f"Mismatch: {train_embeddings.shape[0]} embeddings vs {train_labels.shape[0]} labels"

        splits = self.split_manager.load_splits()
        train_indices = splits["train"]

        log.info(f"  Number of train indices: {len(train_indices)}")
        assert len(train_indices) == train_embeddings.shape[0], \
            f"Mismatch: {len(train_indices)} train indices vs {train_embeddings.shape[0]} embeddings"

        full_dataset = self.dm.full_dataset if hasattr(self.dm, 'full_dataset') else self.dm.train_set
        stratify_labels = self.dm.get_labels_for_stratification(full_dataset)

        cv_folds = self.split_manager.create_cv_folds(
            train_indices=train_indices,
            n_folds=self.k_folds,
            stratify_labels=stratify_labels,
            force_recompute=self.force_recompute,
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
                error_details = traceback.format_exc()
                log.error(f"✗ Trial {i} failed: {e}")
                log.error(f"Traceback:\n{error_details}")
                result_entry = {
                    "trial": i,
                    "params": params,
                    "mean_metric": float('-inf'),
                    "std_metric": 0.0,
                    "error": str(e),
                    "traceback": error_details,
                }
                results.append(result_entry)

        metric_key = str(getattr(self.cfg.train, "metric_key", "val_acc"))
        reverse = not metric_key.endswith("loss")
        valid_results = [r for r in results if r["mean_metric"] != float('-inf')]

        if not valid_results:
            error_summary = "\n".join([
                f"  Trial {r['trial']}: {r['params']} -> {r.get('error', 'Unknown error')}"
                for r in results if 'error' in r
            ])
            raise RuntimeError(
                f"All hyperparameter trials failed!\n"
                f"Total trials: {len(results)}\n"
                f"Errors:\n{error_summary}"
            )

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
                "resolution": self.current_resolution,
                "domain": self.domain,
                "model": self.model_name,
            },
            "best_result": self._to_serializable(best_result),
            "all_results": self._to_serializable(results),
        }

        output_path = os.path.join(self.search_dir, "hyperparam_search_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        best_params_output = {
            "best_hyperparameters": self._to_serializable(best_result["params"]),
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

        abs_output_path = os.path.abspath(output_path)
        abs_best_params_path = os.path.abspath(best_params_path)

        log.info(f"💾 Results saved to {abs_output_path}")
        log.info(f"💾 Best params saved to {abs_best_params_path}")

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
            force_recompute=self.force_recompute,
        )

        train_embeddings, train_labels = all_embeddings["train"]
        test_embeddings, test_labels = all_embeddings["test"]

        # Compute num_classes from all labels (train + test) to ensure we capture all classes
        all_labels = torch.cat([train_labels, test_labels])
        num_classes = int(all_labels.max().item()) + 1

        log.info("Computing class weights from training data...")
        self.class_weights = self._compute_class_weights(train_labels, num_classes=num_classes)

        self.loss_fn = cross_entropy_loss(
            label_smoothing=float(getattr(self.cfg.loss, "label_smoothing", 0.0)),
            class_weight=self.class_weights,
            ignore_index=int(getattr(self.cfg.loss, "ignore_index", -100)),
            reduction=str(getattr(self.cfg.loss, "reduction", "mean")),
        )

        if self.hyperparam_search_enabled:
            best_params, _ = self._run_hyperparam_search(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
            )

            self.cfg = self._apply_hyperparams_to_cfg(best_params)
            # Recreate loss function with updated label smoothing and class weights
            self.loss_fn = cross_entropy_loss(
                label_smoothing=float(getattr(self.cfg.loss, "label_smoothing", 0.0)),
                class_weight=self.class_weights,
                ignore_index=-100,
                reduction="mean",
            )

        elif self.pretuned_hyperparams:
            log.info("📌 Applying pre-tuned hyperparameters")
            self.cfg = self._apply_hyperparams_to_cfg(self.pretuned_hyperparams)
            self.loss_fn = cross_entropy_loss(
                label_smoothing=float(getattr(self.cfg.loss, "label_smoothing", 0.0)),
                class_weight=self.class_weights,
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

        self._log_efficiency_metrics()

        self._save_comprehensive_results(result)

        return result

    def _log_efficiency_metrics(self):
        """Log efficiency metrics to wandb."""
        metrics_to_log = {
            "efficiency/encoder_gflops": self.efficiency_metrics.get("encoder_gflops"),
            "efficiency/encoder_latency_ms": self.efficiency_metrics.get("encoder_latency_ms"),
            "efficiency/embedding_extraction_time_s": self.efficiency_metrics.get("embedding_extraction_time_s"),
            "efficiency/peak_gpu_memory_mb": self.efficiency_metrics.get("peak_gpu_memory_mb"),
            "efficiency/resolution": self.current_resolution,
        }

        metrics_to_log = {k: v for k, v in metrics_to_log.items() if v is not None}

        if metrics_to_log:
            self.wandb.log(metrics_to_log)
            log.info(f"\n📊 Efficiency metrics logged to wandb:")
            for k, v in metrics_to_log.items():
                log.info(f"  {k}: {v}")

    def _save_comprehensive_results(self, training_result: Dict[str, Any]):
        """Save comprehensive results including efficiency metrics to JSON."""
        best_metric = training_result.get("best_metric", None)
        history = training_result.get("history", {})

        final_val_auroc = history.get("val_auroc", [None])[-1] if history.get("val_auroc") else None
        final_val_acc = history.get("val_acc", [None])[-1] if history.get("val_acc") else None
        final_val_loss = history.get("val_loss", [None])[-1] if history.get("val_loss") else None

        comprehensive_results = {
            "experiment_info": {
                "resolution": self.current_resolution,
                "model_name": self.model_name,
                "domain": self.domain,
                "dataset": self.dataset_name,
            },
            "accuracy_metrics": {
                "best_metric": best_metric,
                "metric_key": str(getattr(self.cfg.train, "metric_key", "val_acc")),
                "final_val_auroc": final_val_auroc,
                "final_val_acc": final_val_acc,
                "final_val_loss": final_val_loss,
            },
            "efficiency_metrics": self.efficiency_metrics,
            "hyperparameters": {
                "lr": float(self.cfg.train.optimizer.lr),
                "weight_decay": float(self.cfg.train.optimizer.weight_decay),
                "batch_size": int(self.cfg.data.batch_size),
                "epochs": int(self.cfg.train.epochs),
            },
            "training_history": {
                "num_epochs": len(history.get("train_loss", [])),
                "final_train_loss": history.get("train_loss", [None])[-1] if history.get("train_loss") else None,
            },
        }

        results_path = os.path.join(self.run_dir, f"results_{self.model_name}_{self.current_resolution}px.json")
        with open(results_path, "w") as f:
            json.dump(comprehensive_results, f, indent=2)

        log.info(f"\n💾 Comprehensive results saved to: {os.path.abspath(results_path)}")

        summary_metrics = {
            "summary/best_metric": best_metric,
            "summary/final_val_auroc": final_val_auroc,
            "summary/final_val_acc": final_val_acc,
            "summary/resolution": self.current_resolution,
            "summary/encoder_gflops": self.efficiency_metrics.get("encoder_gflops"),
            "summary/encoder_latency_ms": self.efficiency_metrics.get("encoder_latency_ms"),
        }
        summary_metrics = {k: v for k, v in summary_metrics.items() if v is not None}
        self.wandb.log(summary_metrics)


def run(cfg: Any) -> Dict[str, Any]:
    """Entry point."""
    wrapper = ProbeTwoStageWrapper(cfg)
    return wrapper.run()
