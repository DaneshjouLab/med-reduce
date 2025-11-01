# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
# SPDX-License-Identifier: MIT

# src/cli/train.py
# -*- coding: utf-8 -*-
# pylint: disable=import-error, broad-exception-caught
"""
CLI entry point for training/evaluating models across paradigms (probe/finetune).
It normalizes dataset config → data config, builds a BaseDataModule using the
dataset factory, and dispatches to the selected training wrapper.

Usage examples:
  python -m src.cli.train train.mode=probe \
      dataset.name=isic2019 dataset.data_dir=/data/ISIC \
      dataset.image_size=224 dataset.batch_size=128 \
      model.type=vit model.model_id=google/vit-base-patch16-224

  # With controlled degradation: downsample to 112px, then pipeline resizes to 224
  python -m src.cli.train train.mode=probe \
      dataset.name=isic2019 dataset.data_dir=/data/ISIC \
      dataset.image_size=224 dataset.batch_size=128 \
      dataset.degradation.target_resolution=112 \
      model.type=vit model.model_id=google/vit-base-patch16-224

  # Random search (build-in basic sweep)
  python -m src.cli.train -m \
    hydra.sweeper=basic \
    hydra.sweeper.n_trials=20 \
    hydra.sweeper.params='optim.lr=log(1e-5,1e-3); optim.weight_decay=uniform(0,0.1); \
dataset.batch_size=choice(64,128)' \
    train.mode=probe \
    dataset.name=isic2019 dataset.data_dir=/data/ISIC \
    dataset.image_size=224 \
    model.type=vit model.model_id=google/vit-base-patch16-224

  OR

  python -m src.cli.train -m \
    train.mode=probe \
    dataset.name=isic2019 dataset.data_dir=/data/ISIC \
    dataset.image_size=224 dataset.batch_size=128 \
    model.type=vit model.model_id=google/vit-base-patch16-224 \
    optim.lr=1e-5,3e-5,1e-4,3e-4 \
    optim.weight_decay=0.0,0.01

"""

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
    cfg.runtime["datamodule"] = dm
    return dm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    """Main training CLI entry point."""
    OmegaConf.set_struct(cfg, False)

    # Normalize dataset selection into cfg.data for wrappers/datamodules
    _normalize_dataset_into_data(cfg)

    # Set up run directory (Hydra sets CWD to the unique run dir)
    run_dir = Path(os.getcwd())

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

    # Build & setup datamodule using dataset factory logic
    dm = _build_datamodule(cfg)
    dm.setup(stage="fit")  # prepares train/val (or splits train if no val split)

    # Persist resolved config and print header
    _save_resolved_config(cfg, run_dir)
    _print_run_header(cfg, run_dir, device)

    # Kick off the selected training paradigm
    try:
        from src.wrappers.finetune_randomsearch import run as run_random_search

        param_grid = {
            "lr": [1e-5, 1e-4, 5e-4],
            "weight_decay": [0.01, 0.05],
            "batch_size": [64, 128],
        }

        random_search_results = run_random_search(cfg, param_grid)
        
        print(random_search_results['best_params'], random_search_results['results'])

        best_params = random_search_results['best_params']
        if 'lr' in best_params:
            cfg.train.learning_rate = best_params['lr']
        if 'weight_decay' in best_params:
            cfg.train.weight_decay = best_params['weight_decay']
        if 'batch_size' in best_params:
            cfg.data.batch_size = best_params['batch_size']
            
        # Now, run the final CV using the optimized configuration
        from src.wrappers.finetune_cv import run as run_cv
        metrics = run_cv(cfg)
    except KeyboardInterrupt:
        if _is_rank_zero():
            print("\n⚠️ Training interrupted by user.", flush=True)
        raise
    except Exception as e:
        if _is_rank_zero():
            print(f"\n❌ Training failed: {e}\n", flush=True)
        raise

    # Save final metrics
    if _is_rank_zero():
        metrics = metrics or {}
        with open(run_dir / "final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(
            "✅ Train done. Final metrics written to final_metrics.json\n", flush=True
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())
