# src/models/factory.py
# -*- coding: utf-8 -*-
"""Unified model factory: HF vision models (ViT/DINOv2), optional timm,
   plus matching preprocessors and helpers (freeze_backbone, save_model)."""

from typing import Dict, Any

import os
import torch.nn as nn
from PIL import Image

# --- Hugging Face ---
from transformers import (
    ViTForImageClassification,
    AutoModelForImageClassification,
    ViTImageProcessor,
    AutoImageProcessor,
)

# --- Optional timm ---
try:
    import timm  # type: ignore
    _TIMM_AVAILABLE = True
except Exception:
    _TIMM_AVAILABLE = False

# --- Project constants (small change from your code: avoid importing configs directly) ---
try:
    from src.utils.constants import HF_MODELS  # e.g., {"vit", "dinov2"}
except ImportError:
    HF_MODELS = {"vit", "dinov2", "dinov3"}

from src.models.dinov3 import DINOv3ForImageClassification, DINOv3Config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_model(model_info: Dict[str, Any], resolution: int = 224):
    """
    Factory function to create models based on type.

    Args:
        model_info: {"type": "vit"|"dinov2"|("timm"), "model_id": str, "config": {...}}
        resolution: input image resolution

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
            ignore_mismatched_sizes=config.get("ignore_mismatched_sizes", True),
            image_size=resolution,
        )

    # --- HuggingFace DINOv2 (AutoModel) ---
    elif model_type == "dinov2":
        return AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=config["num_labels"],
            ignore_mismatched_sizes=config.get("ignore_mismatched_sizes", True),
            image_size=resolution,
        )

    if model_type == "dinov3":
        dinov3_config = DINOv3Config(
            backbone_model_id=model_id,
            num_labels=config["num_labels"],
            hidden_size=config.get("hidden_size", 384),
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
        # If you need image_size-specific config, many timm models accept it via `img_size`
        return timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_preprocessor(model_info: Dict[str, Any], resolution: int = 224):
    """
    Create appropriate preprocessor for model type.

    Args:
        model_info: Dictionary with model configuration
        resolution: Input image resolution

    Returns:
        HF preprocessor (ImageProcessor) or None (for timm)
    """
    model_type = model_info["type"]
    model_id = model_info["model_id"]

    if model_type == "vit":
        return ViTImageProcessor.from_pretrained(
            model_id,
            size=resolution,
            do_resize=True,
            resample=Image.LANCZOS,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    elif model_type == "dinov2":
        return AutoImageProcessor.from_pretrained(
            model_id,
            size=resolution,
            do_resize=True,
            resample=Image.LANCZOS,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    if model_type == "dinov3":
        return AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

    if model_type == "timm":
        # timm uses torchvision transforms; return None and build transforms in your datamodule
        return None

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def freeze_backbone(model: nn.Module, model_type: str):
    """
    Freeze backbone parameters for transfer learning.
    For HF classifiers, keep 'classifier' or 'head' trainable; freeze the rest.
    For segmentation models, keep seg_conv, pre_head_norm, pre_head_dropout trainable.

    Args:
        model: nn.Module
        model_type: 'vit' | 'dinov2' | 'dinov3' | ('timm' if you wire it similarly)
    """
    # Patterns for trainable parameters (classification and segmentation heads)
    trainable_patterns = ("classifier", "head", "seg_conv", "pre_head_norm", "pre_head_dropout")

    if model_type in HF_MODELS:
        for name, param in model.named_parameters():
            # Leave classifier/head/segmentation layers trainable, freeze others
            if any(pattern in name for pattern in trainable_patterns):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif model_type == "timm":
        # Optional: implement project-specific rules (e.g., freeze all except last classifier)
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in trainable_patterns) or ("fc" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def save_model(model: nn.Module, model_info: Dict[str, Any], save_dir: str, preprocessor=None):
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
    elif model_type == "timm":
        # Torch-style checkpoint for timm models
        import torch
        ckpt_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), ckpt_path)
        # Minimal config export
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            import json
            json.dump(
                {
                    "model_type": "timm",
                    "model_id": model_info["model_id"],
                    "num_labels": model_info.get("config", {}).get("num_labels", None),
                },
                f,
                indent=2,
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
