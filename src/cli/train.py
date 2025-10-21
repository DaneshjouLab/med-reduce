# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/cli/train.py

"""CLI entry point for training models with different paradigms (probe/finetune/distill).

Usage:
python -m src.cli.train train.mode=probe \
    dataset.name=isic2019 dataset.image_size=224 dataset.batch_size=128 \
    model.type=vit model.model_id=google/vit-base-patch16-224
"""

import os
import sys
import time
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# ---- Wrappers implement run(cfg) and return a dict of metrics
from src.wrappers import probe as probe_wrapper
from src.wrappers import finetune as finetune_wrapper

log = logging.getLogger("train")


# ------------------------- helpers -------------------------


def _is_rank_zero() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return local_rank == 0


def _select_device(cfg: DictConfig) -> torch.device:
    if "device" in cfg.train and cfg.train.device:
        return torch.device(cfg.train.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _save_resolved_config(cfg: DictConfig, run_dir: Path):
    if not _is_rank_zero():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f.name)


def _print_run_header(cfg: DictConfig, run_dir: Path, device: torch.device):
    if not _is_rank_zero():
        return
    banner = (
        f"\n=== TRAIN START ===\n"
        f"mode       : {cfg.train.mode}\n"
        f"dataset    : {cfg.dataset.name}\n"
        f"model      : {getattr(cfg.model, 'name', getattr(cfg.model, 'type', 'N/A'))}\n"
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
        f"Unknown train.mode='{cfg.train.mode}'. "
        f"Expected one of: probe | finetune | distill"
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
      - image_size: int
      - num_classes: int
      - batch_size: int
      - num_workers: int
      - pin_memory: bool (optional)

    After this, cfg.data will contain:
      dataset_name, data_dir, image_size, batch_size, num_workers, pin_memory
    """
    if "dataset" not in cfg:
        raise ValueError("Config is missing 'dataset' group (cfg.dataset.*).")

    ds = cfg.dataset
    # Create cfg.data if missing
    if "data" not in cfg or cfg.data is None:
        cfg.data = OmegaConf.create()

    # Copy/normalize fields
    cfg.data.dataset_name = str(ds.get("name"))
    cfg.data.data_dir = ds.get("data_dir")  # can be None for HF datasets
    cfg.data.image_size = int(ds.get("image_size", 224))
    cfg.data.batch_size = int(
        ds.get("batch_size", getattr(cfg.train, "batch_size", 64))
    )
    cfg.data.num_workers = int(ds.get("num_workers", 4))
    cfg.data.pin_memory = bool(ds.get("pin_memory", True))

    # Optional: enforce/propagate num_classes to model config
    num_classes = ds.get("num_classes", None)
    if num_classes is not None:
        if "model" not in cfg:
            cfg.model = OmegaConf.create()
        if "config" not in cfg.model or cfg.model.config is None:
            cfg.model.config = OmegaConf.create()
        # Many factories look for "num_labels"
        cfg.model.config.num_labels = int(num_classes)

    # Sanity checks
    if not cfg.data.dataset_name:
        raise ValueError("cfg.dataset.name must be set (e.g., 'isic2019').")


# ------------------------- main -------------------------

@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    """Main training CLI entry point."""
    # Allow wrappers to attach runtime fields if needed
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

    # Persist resolved config and print header
    _save_resolved_config(cfg, run_dir)
    _print_run_header(cfg, run_dir, device)

    # Kick off the selected training paradigm
    try:
        metrics = _dispatch_wrapper(cfg)
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
    # Example: python -m src.cli.train train.mode=probe dataset.name=isic2019
    # pylint: disable=no-value-for-parameter
    sys.exit(main())
