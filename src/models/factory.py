# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/models/factory.py
# -*- coding: utf-8 -*-
"""Unified model factory: HF vision models (ViT/DINOv2), optional timm,
   plus matching preprocessors and helpers (freeze_backbone, save_model)."""

from __future__ import annotations
from typing import Dict, Any

import os
import json
import torch
from torch import nn
from PIL import Image

# --- Hugging Face ---
from transformers import (
    ViTForImageClassification,
    AutoModelForImageClassification,
    ViTFeatureExtractor,
    AutoImageProcessor,
)

# --- Optional timm ---
try:
    import timm  # type: ignore
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

# --- Project constants (optional) ---
try:
    from src.utils.constants import HF_MODELS  # e.g., {"vit", "dinov2"}
except ImportError:
    HF_MODELS = {"vit", "dinov2", "dinov3"}

from src.models.dinov3 import DINOv3ForImageClassification, DINOv3Config

# --- Pillow resampling constant (handles both new and old Pillow versions) ---
try:
    # Pillow ≥9.1 uses Image.Resampling
    RESAMPLING_LANCZOS = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:
    # Pillow <9.1 fallback; use getattr to avoid pylint false positives
    RESAMPLING_LANCZOS = getattr(Image, "LANCZOS", None) or getattr(Image, "BICUBIC", None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_model(model_info: Dict[str, Any], resolution: int = 224) -> nn.Module:
    """
    Factory function to create models based on type.

    Args:
        model_info: {"type": "vit"|"dinov2"|"timm", "model_id": str, "config": {...}}
        resolution: input image resolution (passed to HF heads when supported)

    Returns:
        nn.Module
    """
    model_type = model_info["type"]
    model_id = model_info["model_id"]
    config = model_info.get("config", {})

    # --- HuggingFace ViT ---
    if model_type == "vit":
        return ViTForImageClassification.from_pretrained(
            model_id,
            num_labels=config["num_labels"],
            ignore_mismatched_sizes=bool(config.get("ignore_mismatched_sizes", True)),
            image_size=resolution,
        )

    # --- HuggingFace DINOv2 (AutoModel) ---
    if model_type == "dinov2":
        return AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=config["num_labels"],
            ignore_mismatched_sizes=bool(config.get("ignore_mismatched_sizes", True)),
            image_size=resolution,
        )

    if model_type == "dinov3":
        dinov3_config = DINOv3Config(
            backbone_model_id=model_id,
            num_labels=config["num_labels"],
            hidden_size=config.get("hidden_size", 768),
            dropout_rate=config.get("dropout_rate", 0.1),
            use_quantization=config.get("use_quantization", False),
            use_safetensors=True
        )
        return DINOv3ForImageClassification(dinov3_config)

    # --- timm ---
    if model_type == "timm":
        if not _TIMM_AVAILABLE:
            raise RuntimeError("timm is not installed but model_type='timm' was requested.")
        num_classes = int(config.get("num_labels", 1000))
        pretrained = bool(config.get("pretrained", True))
        return timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)

    raise ValueError(f"Unknown model type: {model_type}")


def create_preprocessor(model_info: Dict[str, Any], resolution: int = 224):
    """
    Create appropriate preprocessor for model type.

    Args:
        model_info: Dictionary with model configuration
        resolution: Input image resolution

    Returns:
        HF preprocessor (FeatureExtractor/ImageProcessor) or None (for timm)
    """
    model_type = model_info["type"]
    model_id = model_info["model_id"]

    if model_type == "vit":
        return ViTFeatureExtractor.from_pretrained(
            model_id,
            size=resolution,
            do_resize=True,
            resample=RESAMPLING_LANCZOS,
            do_normalize=True,
            # ImageNet normalization statistics [red, green, blue]
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    if model_type == "dinov2":
        return AutoImageProcessor.from_pretrained(
            model_id,
            size=resolution,
            do_resize=True,
            resample=RESAMPLING_LANCZOS,
            do_normalize=True,
            # ImageNet normalization statistics [red, green, blue]
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    if model_type == "dinov3":
        return AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

    if model_type == "timm":
        # timm uses torchvision transforms; return None and build transforms in your datamodule
        return None

    raise ValueError(f"Unknown model type: {model_type}")


def freeze_backbone(model: nn.Module, model_type: str) -> None:
    """
    Freeze backbone parameters for transfer learning.
    For HF classifiers, keep 'classifier' or 'head' trainable; freeze the rest.

    Args:
        model: nn.Module
        model_type: 'vit' | 'dinov2' | 'timm'
    """
    if model_type in HF_MODELS:
        for name, param in model.named_parameters():
            # Leave classifier/head trainable, freeze others
            if ("classifier" in name) or ("head" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        return

    if model_type == "timm":
        for name, param in model.named_parameters():
            if ("classifier" in name) or ("fc" in name) or ("head" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        return

    raise ValueError(f"Unsupported model_type: {model_type}")


def save_model(
    model: nn.Module,
    model_info: Dict[str, Any],
    save_dir: str,
    preprocessor=None
) -> None:
    """
    Save model based on its type.

    Args:
        model: Model to save
        model_info: Model configuration
        save_dir: Directory to save to
        preprocessor: Optional HF preprocessor to save
    """
    os.makedirs(save_dir, exist_ok=True)
    model_type = model_info["type"]

    if model_type in HF_MODELS:
        model.save_pretrained(save_dir)
        if preprocessor is not None:
            preprocessor.save_pretrained(save_dir)
        return

    if model_type == "timm":
        # Torch-style checkpoint for timm models
        ckpt_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), ckpt_path)
        # Minimal config export
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "timm",
                    "model_id": model_info.get("model_id"),
                    "num_labels": model_info.get("config", {}).get("num_labels", None),
                },
                f,
                indent=2,
            )
        return

    raise ValueError(f"Unsupported model_type: {model_type}")
