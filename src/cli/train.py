# src/cli/train.py
# -*- coding: utf-8 -*-
# pylint: disable=import-error, broad-exception-caught
from __future__ import annotations

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import hydra  # pylint: disable=import-error
from omegaconf import DictConfig, OmegaConf  # pylint: disable=import-error

# ---- Training wrappers (each provides run(cfg) -> dict of metrics)
from src.wrappers import probe_cv  # pylint: disable=import-error
from src.wrappers import segmentation_cv  # pylint: disable=import-error

# ---- Data pipeline
from src.data.datamodule import BaseDataModule  # pylint: disable=import-error
from src.transformations.transforms import (
    ResolutionReductionTransform,
)  # pylint: disable=import-error
from src.utils.reproducibility import (  # pylint: disable=import-error
    seed_everything,
    SeedTracker,
    log_reproducibility_info,
)

# ---- Optional HF preprocessor (only needed when actually running a HF backbone)
try:
    from transformers import AutoImageProcessor  # noqa: F401
except ImportError:  # pragma: no cover
    # If transformers is not installed, we continue without it
    AutoImageProcessor = None  # type: ignore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train")


def _is_rank_zero() -> bool:
    # Works for both torchrun (DDP) and single process
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return local_rank == 0


def _select_device(cfg: DictConfig) -> torch.device:
    # Honor cfg.train.device if present; otherwise auto
    if "device" in cfg.train and cfg.train.device:
        dev = cfg.train.device
        return torch.device(dev)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_resolved_config(cfg: DictConfig, run_dir: Path):
    if not _is_rank_zero():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f.name)


def _print_run_header(cfg: DictConfig, run_dir: Path, device: torch.device):
    if not _is_rank_zero():
        return
    banner = (
        f"\n=== TRAIN START ===\n"
        f"mode       : {cfg.train.mode}\n"
        f"dataset    : {cfg.dataset.name}\n"
        f"model      : {getattr(cfg.model, 'name', 'N/A')}\n"
        f"device     : {device}\n"
        f"seed       : {cfg.seed}\n"
        f"run_dir    : {str(run_dir)}\n"
        f"===================\n"
    )
    print(banner, flush=True)


def _dispatch_wrapper(cfg: DictConfig) -> Dict[str, Any]:
    mode = cfg.train.mode.lower()
    if mode == "probe":
        return probe_cv.run(cfg)
    elif mode == "segmentation":
        return segmentation_cv.run(cfg)
    raise ValueError(
        f"Unknown train.mode='{cfg.train.mode}'. Expected one of: probe | segmentation"
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
# Helper functions
# ---------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    """Convert OmegaConf objects to regular Python objects for JSON serialization."""
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    else:
        return obj


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

    # ---- Device & seeding
    device = _select_device(cfg)
    seed_value = int(cfg.seed)
    deterministic_mode = getattr(cfg.train, "deterministic", False)

    seed_everything(seed=seed_value, deterministic=deterministic_mode)

    seed_tracker = SeedTracker(base_seed=seed_value)
    seed_tracker.log_seed("main", seed_value, {"deterministic": deterministic_mode})

    log_reproducibility_info(output_dir=run_dir)

    # ---- Optional: attach runtime info for wrappers/engines
    cfg.runtime = {
        "device": str(device),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "rank_zero": _is_rank_zero(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "seed": seed_value,
    }

    # Persist resolved config and print header
    _save_resolved_config(cfg, run_dir)
    _print_run_header(cfg, run_dir, device)

    # Kick off the selected training paradigm
    try:
        # Dispatch to appropriate wrapper based on train.mode
        # Wrappers handle hyperparameter search internally if enabled in config
        metrics = _dispatch_wrapper(cfg)

    except KeyboardInterrupt:
        if _is_rank_zero():
            print("\n⚠️ Training interrupted by user.", flush=True)
        raise
    except Exception as e:
        # Surface a readable error at rank zero; still propagate for proper exit codes
        if _is_rank_zero():
            print(f"\n❌ Training failed: {e}\n", flush=True)
            import traceback
            traceback.print_exc()
        raise

    # ---- Persist final metrics
    if _is_rank_zero():
        metrics = metrics or {}

        final_metrics_path = run_dir / "final_metrics.json"
        with open(final_metrics_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(metrics), f, indent=2)

        print(f"\n✅ Training complete! Final metrics written to {final_metrics_path}\n", flush=True)

        # Print summary
        if "mean_metric" in metrics:
            metric_name = getattr(cfg.train, "metric_key", "val_acc")
            print(f"📊 Final {metric_name}: {metrics['mean_metric']:.4f}", flush=True)
            if "std_metric" in metrics:
                print(f"   Std deviation: {metrics['std_metric']:.4f}", flush=True)

        print("", flush=True)  # Final newline

if __name__ == "__main__":
    # Support `python -m src.cli.train train=probe dataset=isic2019`
    sys.exit(main())
