# src/utils/logging.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Optional: Weights & Biases
_WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger format/level."""
    fmt = "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def get_logger(name: str) -> logging.Logger:
    """Get a module-specific logger."""
    return logging.getLogger(name)


class MetricAverager:
    """
    Track running averages of named scalars.
    Usage:
        ma = MetricAverager()
        ma.update(loss=0.1, acc=0.9)
        avgs = ma.averages()  # {"loss": ..., "acc": ...}
    """
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self.totals[k] = self.totals.get(k, 0.0) + float(v)
            self.counts[k] = self.counts.get(k, 0) + 1

    def averages(self) -> Dict[str, float]:
        return {k: (self.totals[k] / max(self.counts[k], 1)) for k in self.totals}

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()


class WandbLogger:
    """
    Thin W&B wrapper that is safe when wandb is not installed.
    """
    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Any = None,
        enabled: Optional[bool] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.enabled = (_WANDB_AVAILABLE if enabled is None else enabled)
        self.run = None
        if self.enabled:
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=_maybe_serialize_config(config),
                entity=entity,
                tags=tags or [],
                reinit=True,
            )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        if self.enabled and self.run is not None:
            wandb.log(metrics, step=step, commit=commit)

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        if self.enabled and self.run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)

    def save_artifact(self, path: str, name: Optional[str] = None, type_: str = "file") -> None:
        if self.enabled and self.run is not None and os.path.exists(path):
            art = wandb.Artifact(name or Path(path).name, type=type_)
            art.add_file(path)
            self.run.log_artifact(art)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


def _maybe_serialize_config(cfg: Any) -> Dict[str, Any]:
    """Best-effort: convert config objects to plain dicts."""
    try:
        from omegaconf import OmegaConf  # type: ignore
        if isinstance(cfg, dict):
            return cfg
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    except Exception:
        pass
    try:
        json.dumps(cfg)  # type: ignore
        return cfg  # type: ignore
    except Exception:
        return {}
