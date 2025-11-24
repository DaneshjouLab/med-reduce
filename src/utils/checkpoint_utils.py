# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""Utilities for loading and managing model checkpoints."""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
from torch import nn

# pylint: disable=import-error
from src.models.factory import create_model


def load_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Target device to load tensors to
        map_location: Alternative to device, passed to torch.load

    Returns:
        Dictionary containing checkpoint data:
            - model_state_dict: model weights
            - fold: fold number
            - metric: best metric value
            - model_config: model configuration
            - cfg: full training configuration
            - optimizer_state_dict: optimizer state (if saved)
    """
    if map_location is None and device is not None:
        map_location = str(device)

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load model to (default: cuda if available, else cpu)
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        Loaded model with weights restored
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(checkpoint_path, device=device)

    # Recreate model from saved config
    model_config = checkpoint["model_config"]
    image_size = checkpoint["cfg"].data.image_size

    model = create_model(model_config, resolution=image_size)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(device)
    model.eval()

    return model


def find_best_checkpoint(checkpoint_dir: str | Path, metric_key: str = "val_acc") -> Path:
    """
    Find the checkpoint with the best metric in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        metric_key: Metric to optimize ("val_acc" for max, "val_loss" for min)

    Returns:
        Path to the best checkpoint file
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Extract metrics from filenames (format: ..._metric0.9234.pt)
    maximize = not metric_key.endswith("loss")
    best_checkpoint = None
    best_metric = float("-inf") if maximize else float("inf")

    for ckpt_path in checkpoints:
        # Extract metric from filename
        try:
            metric_str = ckpt_path.stem.split("_metric")[-1]
            metric = float(metric_str)

            if maximize:
                if metric > best_metric:
                    best_metric = metric
                    best_checkpoint = ckpt_path
            else:
                if metric < best_metric:
                    best_metric = metric
                    best_checkpoint = ckpt_path
        except (ValueError, IndexError):
            continue

    if best_checkpoint is None:
        raise ValueError(f"Could not parse metric from checkpoint filenames in {checkpoint_dir}")

    return best_checkpoint


def load_all_fold_models(
    checkpoint_dir: str | Path,
    device: Optional[torch.device] = None,
) -> List[nn.Module]:
    """
    Load models from all fold checkpoints in a directory.

    Useful for ensemble predictions.

    Args:
        checkpoint_dir: Directory containing fold checkpoint files
        device: Device to load models to

    Returns:
        List of models, one per fold
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*_fold*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoint files found in {checkpoint_dir}")

    models = []
    for ckpt_path in checkpoints:
        model = load_model_from_checkpoint(ckpt_path, device=device)
        models.append(model)

    return models


def ensemble_predict(
    models: List[nn.Module],
    inputs: torch.Tensor,
    device: Optional[torch.device] = None,
    average_logits: bool = True,
) -> torch.Tensor:
    """
    Make ensemble predictions from multiple models.

    Args:
        models: List of trained models
        inputs: Input tensor (batch_size, channels, height, width)
        device: Device to run inference on
        average_logits: If True, average logits. If False, average probabilities.

    Returns:
        Ensemble predictions (batch_size, num_classes)
    """
    if device is None:
        device = next(models[0].parameters()).device

    inputs = inputs.to(device)
    predictions = []

    with torch.no_grad():
        for model in models:
            model.eval()
            output = model(inputs)

            # Handle different output formats
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, dict) and "logits" in output:
                logits = output["logits"]
            else:
                logits = output

            if average_logits:
                predictions.append(logits)
            else:
                predictions.append(torch.softmax(logits, dim=-1))

    # Average predictions
    ensemble_output = torch.stack(predictions).mean(dim=0)

    return ensemble_output
