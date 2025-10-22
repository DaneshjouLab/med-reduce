# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
Logging and metrics utilities for machine learning experiments.

Provides functionality for:
- Standard Python logging setup
- Metric averaging for tracking training stats
- Optional Weights & Biases integration
"""

# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Optional: Weights & Biases
wandb_available = False  # Module-level flag for wandb availability
try:
    import wandb  # type: ignore  # pylint: disable=import-error
    wandb_available = True
except ImportError:
    # wandb is an optional dependency
    wandb_available = False


def setup_logging(level: int = logging.INFO) -> None:  # pylint: disable=no-member
    """Configure root logger format/level."""
    fmt = "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)  # pylint: disable=no-member


def get_logger(name: str) -> logging.Logger:  # pylint: disable=no-member
    """Get a module-specific logger."""
    return logging.getLogger(name)  # pylint: disable=no-member


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
        """
        Update metrics with new values.

        Args:
            **kwargs: Keyword arguments of metric names and values
        """
        for k, v in kwargs.items():
            self.totals[k] = self.totals.get(k, 0.0) + float(v)
            self.counts[k] = self.counts.get(k, 0) + 1

    def averages(self) -> Dict[str, float]:
        """
        Get current averages for all tracked metrics.

        Returns:
            Dictionary mapping metric names to their averages
        """
        return {k: (v / max(self.counts[k], 1)) for k, v in self.totals.items()}

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self.totals.clear()
        self.counts.clear()


class WandbLogger:
    """
    Thin W&B wrapper that is safe when wandb is not installed.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Any = None,
        enabled: Optional[bool] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the W&B logger.

        Args:
            project: W&B project name
            run_name: Optional name for this run
            config: Configuration to log (dict, OmegaConf or compatible)
            enabled: Whether to enable logging (defaults to wandb_available)
            entity: Optional W&B team/entity name
            tags: Optional list of tags for the run
        """
        self.enabled = (wandb_available if enabled is None else enabled)
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
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step/iteration number
            commit: Whether to immediately commit to W&B
        """
        if self.enabled and self.run is not None:
            wandb.log(metrics, step=step, commit=commit)

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """
        Watch a model's parameters and gradients.

        Args:
            model: PyTorch model to watch
            log: What to log ('gradients', 'parameters', 'all', or None)
            log_freq: How frequently to log
        """
        if self.enabled and self.run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)

    def save_artifact(self, path: str, name: Optional[str] = None, type_: str = "file") -> None:
        """
        Save a file as a W&B artifact.

        Args:
            path: Path to the file to save
            name: Optional artifact name (defaults to filename)
            type_: Artifact type
        """
        if self.enabled and self.run is not None and os.path.exists(path):
            art = wandb.Artifact(name or Path(path).name, type=type_)
            art.add_file(path)
            self.run.log_artifact(art)

    def finish(self) -> None:
        """Mark the W&B run as complete."""
        if self.enabled and self.run is not None:
            self.run.finish()


def _maybe_serialize_config(cfg: Any) -> Dict[str, Any]:
    """
    Best-effort: convert config objects to plain dicts.

    Args:
        cfg: Configuration object (dict, OmegaConf, or other)

    Returns:
        JSON-serializable dictionary
    """
    # If it's already a dict, return it
    if isinstance(cfg, dict):
        return cfg

    # Try OmegaConf conversion
    try:
        # pylint: disable=import-error,import-outside-toplevel
        from omegaconf import OmegaConf  # type: ignore
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    except ImportError:
        # OmegaConf not available
        pass

    # Try direct JSON serialization
    try:
        json.dumps(cfg)  # type: ignore
        return cfg  # type: ignore
    except (TypeError, ValueError):
        # Not JSON serializable
        return {}
