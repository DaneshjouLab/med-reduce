# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
# SPDX-License-Identifier: MIT

# src/cli/train.py
# -*- coding: utf-8 -*-
# pylint: disable=import-error, broad-exception-caught
from __future__ import annotations

import os
import sys
import time
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import torch
import numpy as np
import hydra  # pylint: disable=import-error
from omegaconf import DictConfig, OmegaConf  # pylint: disable=import-error

# ---- Training wrappers (each provides run(cfg) -> dict of metrics)
from src.wrappers import probe as probe_wrapper  # pylint: disable=import-error
from src.wrappers import finetune as finetune_wrapper  # pylint: disable=import-error

# ---- Data pipeline
from src.data.datamodule import BaseDataModule  # pylint: disable=import-error
from src.transformations.transforms import (
    ResolutionReductionTransform,
)  # pylint: disable=import-error

# ---- Optional HF preprocessor (only needed when actually running a HF backbone)
try:
    from transformers import AutoImageProcessor  # noqa: F401
except ImportError:  # pragma: no cover
    # If transformers is not installed, we continue without it
    AutoImageProcessor = None  # type: ignore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _is_rank_zero() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return local_rank == 0


def _select_device(cfg: DictConfig) -> torch.device:
    if "train" in cfg and "device" in cfg.train and cfg.train.device:
        return torch.device(cfg.train.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _save_resolved_config(cfg: DictConfig, run_dir: Path) -> None:
    if not _is_rank_zero():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f.name)


def _print_run_header(cfg: DictConfig, run_dir: Path, device: torch.device) -> None:
    if not _is_rank_zero():
        return
    model_name = getattr(cfg.model, "name", getattr(cfg.model, "type", "N/A"))
    banner = (
        f"\n=== TRAIN START ===\n"
        f"mode       : {cfg.train.mode}\n"
        f"dataset    : {cfg.dataset.name}\n"
        f"model      : {model_name}\n"
        f"device     : {device}\n"
        f"seed       : {cfg.seed}\n"
        f"run_dir    : {str(run_dir)}\n"
        f"===================\n"
    )
    print(banner, flush=True)


def _dispatch_wrapper(cfg: DictConfig) -> Dict[str, Any]:
    mode = str(cfg.train.mode).lower()
    if mode == "probe":
        return probe_wrapper.run(cfg)
    if mode == "finetune":
        return finetune_wrapper.run(cfg)
    # if mode == "distill":
    #     return distill_wrapper.run(cfg)
    raise ValueError(
        f"Unknown train.mode='{cfg.train.mode}'. Expected one of: probe | finetune | distill"
    )


def _ensure_keys(d: DictConfig, keys_defaults: Dict[str, Any]) -> None:
    """Ensure keys exist in DictConfig with defaults (in-place)."""
    for k, v in keys_defaults.items():
        if k not in d or d[k] is None:
            d[k] = v


def _normalize_dataset_into_data(cfg: DictConfig) -> None:
    """
    Map cfg.dataset → cfg.data so wrappers and datamodules get a consistent schema.

    Expected cfg.dataset fields (typical):
      - name: str (e.g., "isic2019", "chexpert")
      - data_dir: str or null (some loaders use HF hub instead)
      - image_size: int (model input size, e.g., 224)
      - num_classes: int (optional; forwarded to model.config.num_labels)
      - batch_size: int
      - num_workers: int
      - pin_memory: bool (optional)
      - degradation: (optional group)
          target_resolution: int | [w,h]
          restore_original_size: bool
          # or
          reduction_factor: float in (0,1]

    After this, cfg.data contains:
      dataset_name, data_dir, image_size, batch_size, num_workers, pin_memory
    """
    if "dataset" not in cfg:
        raise ValueError("Config is missing 'dataset' group (cfg.dataset.*).")

    ds = cfg.dataset
    if "data" not in cfg or cfg.data is None:
        cfg.data = OmegaConf.create()

    cfg.data.dataset_name = str(ds.get("name"))
    cfg.data.data_dir = ds.get("data_dir")  # can be None for HF datasets
    cfg.data.image_size = int(ds.get("image_size", 224))
    cfg.data.batch_size = int(
        ds.get("batch_size", getattr(cfg.train, "batch_size", 64))
    )
    cfg.data.num_workers = int(ds.get("num_workers", 4))
    cfg.data.pin_memory = bool(ds.get("pin_memory", True))

    # Optional: propagate num_classes → model.config.num_labels
    num_classes = ds.get("num_classes", None)
    if num_classes is not None:
        if "model" not in cfg:
            cfg.model = OmegaConf.create()
        if "config" not in cfg.model or cfg.model.config is None:
            cfg.model.config = OmegaConf.create()
        cfg.model.config.num_labels = int(num_classes)

    if not cfg.data.dataset_name:
        raise ValueError("cfg.dataset.name must be set (e.g., 'isic2019').")


def _build_degradation_transform(
    degr_cfg: DictConfig,
) -> Optional[ResolutionReductionTransform]:
    """
    Build a ResolutionReductionTransform from a degradation config group.
    Supports:
      - target_resolution: int | [w,h]
      - restore_original_size: bool
      - reduction_factor: float
    """
    if not degr_cfg:
        return None

    # normalize target_resolution
    target_res: Optional[Union[int, Tuple[int, int]]] = getattr(
        degr_cfg, "target_resolution", None
    )
    restore: bool = bool(getattr(degr_cfg, "restore_original_size", False))
    reduction_factor = getattr(degr_cfg, "reduction_factor", None)

    if target_res is not None:
        if isinstance(target_res, int):
            target_res = (int(target_res), int(target_res))
        else:
            target_res = tuple(int(x) for x in target_res)
        return ResolutionReductionTransform(
            target_resolution=target_res, restore_original_size=restore
        )

    if reduction_factor is not None:
        return ResolutionReductionTransform(
            reduction_factor=float(reduction_factor), restore_original_size=restore
        )

    return None


def _build_datamodule(cfg: DictConfig) -> BaseDataModule:
    """
    Construct the BaseDataModule using normalized cfg.data and optional
    degradation transforms. Keeps HF preprocessor optional.
    """
    # Optional degradation transform
    degr_cfg = getattr(cfg.dataset, "degradation", None)
    transform = _build_degradation_transform(degr_cfg) if degr_cfg else None

    # Optional HF preprocessor (only if actually running a HF backbone)
    preproc = None
    model_id = getattr(cfg.model, "model_id", None)
    if model_id and AutoImageProcessor is not None:
        try:
            preproc = AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            preproc = None  # safe no-op path

    image_size = int(getattr(cfg.data, "image_size", 224))
    batch_size = int(getattr(cfg.data, "batch_size", 64))
    num_workers = int(getattr(cfg.data, "num_workers", 4))
    pin_memory = bool(getattr(cfg.data, "pin_memory", True))
    model_type = str(getattr(cfg.model, "type", "vit"))

    dm = BaseDataModule(
        cfg=cfg,
        dataset_name=str(cfg.data.dataset_name),
        data_dir=getattr(cfg.data, "data_dir", None),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        preprocessor=preproc,  # None -> ModelPreprocessor no-op
        resolution=image_size,  # model input size (e.g., 224)
        transform=transform,  # may be None
        model_type=model_type,
    )

    # Expose to wrappers
    cfg.runtime = dict(getattr(cfg, "runtime", {}))
    # cfg.runtime["datamodule"] = dm
    return dm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../configs", config_name="probe_two_stage", version_base=None)
def main(cfg: DictConfig):
    """Main training CLI entry point."""
    OmegaConf.set_struct(cfg, False)

    print("Hydra config:\n", OmegaConf.to_yaml(cfg), flush=True)

    # Normalize dataset selection into cfg.data for wrappers/datamodules
    _normalize_dataset_into_data(cfg)

    # Set up run directory using Hydra's output directory
    # Hydra creates a unique directory for each run (outputs/YYYY-MM-DD/HH-MM-SS by default)
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Device & seeding
    device = _select_device(cfg)
    _seed_everything(
        seed=int(cfg.seed),
        deterministic=bool(getattr(cfg.train, "deterministic", False)),
    )

    # Attach runtime info used by wrappers/engines
    cfg.runtime = {
        "device": str(device),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "rank_zero": _is_rank_zero(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }

    # Persist resolved config and print header
    _save_resolved_config(cfg, run_dir)
    _print_run_header(cfg, run_dir, device)

    # Kick off the selected training paradigm
    try:
        # Check if hyperparameter search is enabled from config
        hyperparam_cfg = getattr(cfg.train, "hyperparam_search", None)
        run_hyperparam_search = False
        
        if hyperparam_cfg is not None:
            run_hyperparam_search = getattr(hyperparam_cfg, "enabled", False)
        
        if run_hyperparam_search:
            if _is_rank_zero():
                print("\n🔍 Starting hyperparameter search...\n", flush=True)
            
            from src.wrappers.finetune_randomsearch import run as run_random_search
            
            # Extract search configuration
            n_samples = getattr(hyperparam_cfg, "n_samples", 15)
            search_subset_frac = getattr(hyperparam_cfg, "subset_frac", 0.3)
            use_cv_for_search = getattr(hyperparam_cfg, "use_cv", False)
            param_grid = getattr(hyperparam_cfg, "param_grid", {})
            
            # Validate param_grid
            if not param_grid:
                raise ValueError(
                    "hyperparam_search.param_grid must be specified when search is enabled"
                )
            
            if _is_rank_zero():
                print(f"  Strategy: {'K-Fold CV' if use_cv_for_search else 'Single Split'}")
                print(f"  Search data: {search_subset_frac*100:.0f}% of training set")
                print(f"  Configurations to try: {n_samples}")
                print(f"  Parameter grid: {param_grid}\n", flush=True)
            
            # Run hyperparameter search
            search_results = run_random_search(
                cfg=cfg,
                param_grid=param_grid,
                n_samples=n_samples,
                use_cv=use_cv_for_search,
                search_subset_frac=search_subset_frac,
            )
            
            # Apply best parameters to config
            best_params = search_results['best_params']
            if _is_rank_zero():
                print(f"\n🏆 Best hyperparameters found: {best_params}\n", flush=True)
            
            # Update config with best parameters
            if 'lr' in best_params:
                cfg.train.learning_rate = best_params['lr']
                if _is_rank_zero():
                    print(f"  ✓ Updated learning_rate: {best_params['lr']}")
            
            if 'weight_decay' in best_params:
                cfg.train.weight_decay = best_params['weight_decay']
                if _is_rank_zero():
                    print(f"  ✓ Updated weight_decay: {best_params['weight_decay']}")
            
            if 'batch_size' in best_params:
                cfg.data.batch_size = best_params['batch_size']
                if _is_rank_zero():
                    print(f"  ✓ Updated batch_size: {best_params['batch_size']}")
            
            if 'label_smoothing' in best_params:
                if hasattr(cfg, 'loss'):
                    cfg.loss.label_smoothing = best_params['label_smoothing']
                    if _is_rank_zero():
                        print(f"  ✓ Updated label_smoothing: {best_params['label_smoothing']}")
            
            if 'momentum' in best_params:
                if hasattr(cfg.train, 'momentum'):
                    cfg.train.momentum = best_params['momentum']
                    if _is_rank_zero():
                        print(f"  ✓ Updated momentum: {best_params['momentum']}")
            
            if 'dropout' in best_params:
                if hasattr(cfg.model, 'dropout'):
                    cfg.model.dropout = best_params['dropout']
                    if _is_rank_zero():
                        print(f"  ✓ Updated dropout: {best_params['dropout']}")
            
            # Save search results
            if _is_rank_zero():
                search_results_path = run_dir / "hyperparam_search_results.json"
                with open(search_results_path, "w", encoding="utf-8") as f:
                    # Convert to serializable format
                    serializable_results = {
                        "best_params": best_params,
                        "results": [
                            {
                                "trial": r.get("trial"),
                                "params": r.get("params"),
                                "mean_metric": float(r.get("mean_metric", 0)),
                                "std_metric": float(r.get("std_metric", 0)),
                                "fold_metrics": [float(m) for m in r.get("fold_metrics", [])],
                            }
                            for r in search_results.get("results", [])
                        ]
                    }
                    json.dump(serializable_results, f, indent=2)
                print(f"\n💾 Search results saved to {search_results_path}\n", flush=True)
            
            # Reset subset_frac to use full data for final training
            cfg.train.subset_frac = 1.0
            
            if _is_rank_zero():
                print("\n🚀 Running final K-Fold CV with best hyperparameters on full dataset...\n", flush=True)
        
        # Run K-Fold CV (either with searched params or original config)
        from src.wrappers.finetune_cv import run as run_cv
        metrics = run_cv(cfg)
        
    except KeyboardInterrupt:
        if _is_rank_zero():
            print("\n⚠️ Training interrupted by user.", flush=True)
        raise
    except Exception as e:
        if _is_rank_zero():
            print(f"\n❌ Training failed: {e}\n", flush=True)
            import traceback
            traceback.print_exc()
        raise

    # Save final metrics
    if _is_rank_zero():
        metrics = metrics or {}
        
        # Include best params if search was run
        if run_hyperparam_search:
            metrics['best_hyperparams'] = best_params
            metrics['search_config'] = {
                'n_samples': n_samples,
                'subset_frac': search_subset_frac,
                'use_cv': use_cv_for_search,
                'param_grid': param_grid,
            }
        
        final_metrics_path = run_dir / "final_metrics.json"
        with open(final_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Training complete! Final metrics written to {final_metrics_path}\n", flush=True)
        
        # Print summary
        if "mean_metric" in metrics:
            metric_name = getattr(cfg.train, "metric_key", "val_acc")
            print(f"📊 Final {metric_name}: {metrics['mean_metric']:.4f}", flush=True)
            if "std_metric" in metrics:
                print(f"   Std deviation: {metrics['std_metric']:.4f}", flush=True)
        
        if run_hyperparam_search and "best_hyperparams" in metrics:
            print(f"\n🏆 Best hyperparameters used:", flush=True)
            for param, value in metrics['best_hyperparams'].items():
                print(f"   {param}: {value}", flush=True)
        
        print("", flush=True)  # Final newline

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())
