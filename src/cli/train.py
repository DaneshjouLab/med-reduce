# src/cli/train.py
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

# ---- Optional: import wrappers (these implement run(cfg) and return a dict of metrics)
from src.wrappers import probe as probe_wrapper
from src.wrappers import finetune as finetune_wrapper
from src.wrappers import distill as distill_wrapper

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
        return probe_wrapper.run(cfg)
    elif mode == "finetune":
        return finetune_wrapper.run(cfg)
    elif mode == "distill":
        return distill_wrapper.run(cfg)
    else:
        raise ValueError(
            f"Unknown train.mode='{cfg.train.mode}'. "
            f"Expected one of: probe | finetune | distill"
        )


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    # ---- Resolve and freeze config for safety
    OmegaConf.set_struct(cfg, False)  # allow wrappers to attach runtime fields if needed

    # ---- Set up run directory (Hydra changes CWD to a unique run dir automatically)
    run_dir = Path(os.getcwd())

    # ---- Device & seeding
    device = _select_device(cfg)
    _seed_everything(seed=int(cfg.seed), deterministic=getattr(cfg.train, "deterministic", False))

    # ---- Optional: attach runtime info for wrappers/engines
    cfg.runtime = {
        "device": str(device),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "rank_zero": _is_rank_zero(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }

    # ---- Save resolved config
    _save_resolved_config(cfg, run_dir)
    _print_run_header(cfg, run_dir, device)

    # ---- Kick off the selected training paradigm via wrapper
    try:
        metrics = _dispatch_wrapper(cfg)
    except KeyboardInterrupt:
        if _is_rank_zero():
            print("\n⚠️ Training interrupted by user.", flush=True)
        raise
    except Exception as e:
        # Surface a readable error at rank zero; still propagate for proper exit codes
        if _is_rank_zero():
            print(f"\n❌ Training failed: {e}\n", flush=True)
        raise

    # ---- Persist final metrics
    if _is_rank_zero():
        metrics = metrics or {}
        with open(run_dir / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("✅ Train done. Final metrics written to final_metrics.json\n", flush=True)


if __name__ == "__main__":
    # Support `python -m src.cli.train train=probe dataset=isic2019`
    sys.exit(main())
