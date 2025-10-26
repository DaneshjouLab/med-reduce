"""Model architectures and model-related utilities."""
import json
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import (
    ViTForImageClassification,
    AutoModelForImageClassification,
    ViTFeatureExtractor,
    AutoImageProcessor,
)
from dinov3 import DINOv3Config, DINOv3ForImageClassification
from PIL import Image

from src.config import HF_MODELS, NUM_FILTERED_CLASSES, FILTERED_CLASSES

id2label = {idx: label for idx, label in enumerate(FILTERED_CLASSES)}
label2id = {label: idx for idx, label in enumerate(FILTERED_CLASSES)}

def create_model(model_info: Dict[str, Any], resolution: int = 224):
    """
    Factory function to create models based on type.
    
    Args:
        model_info: Dictionary with model configuration
        resolution: Input image resolution
        
    Returns:
        Model instance
    """
    model_type = model_info["type"]
    model_id = model_info["model_id"]
    config = model_info["config"]
    
    if model_type == "vit":
        return ViTForImageClassification.from_pretrained(
            model_id,
            num_labels=config["num_labels"],
            ignore_mismatched_sizes=config.get("ignore_mismatched_sizes", True),
            image_size=resolution,
            use_safetensors=True
        )
    elif model_type == "dinov2":
        return AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=config["num_labels"],
            ignore_mismatched_sizes=config.get("ignore_mismatched_sizes", True),
            image_size=resolution,
            use_safetensors=True
        )
    elif model_type == "dinov3":
        dinov3_config = DINOv3Config(
            backbone_model_id=model_id,
            num_labels=config["num_labels"],
            hidden_size=config.get("hidden_size", 768),
            dropout_rate=config.get("dropout_rate", 0.1),
            use_quantization=config.get("use_quantization", False),
            use_safetensors=True
        )
        return DINOv3ForImageClassification(dinov3_config)
    elif model_type == "MedSigLIP":
        return AutoModelForImageClassification.from_pretrained(
            model_id,
            problem_type="multi_label_classification",
            num_labels=NUM_FILTERED_CLASSES,
            id2label=id2label,
            label2id=label2id,
            use_safetensors=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_preprocessor(model_info: Dict[str, Any], resolution: int = 224):
    """
    Create appropriate preprocessor for model type.
    
    Args:
        model_info: Dictionary with model configuration
        resolution: Input image resolution
        
    Returns:
        Preprocessor instance
    """
    model_type = model_info["type"]
    model_id = model_info["model_id"]
    
    if model_type == "vit":
        return ViTFeatureExtractor.from_pretrained(
            model_id,
            size=resolution,
            do_resize=True,
            resample=Image.LANCZOS,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    elif model_type == "dinov2":
        return AutoImageProcessor.from_pretrained(
            model_id,
            size=resolution,
            do_resize=True,
            resample=Image.LANCZOS,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    elif model_type == "dinov3":
        return AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    elif model_type == "MedSigLIP":
        return AutoImageProcessor.from_pretrained("google/medsiglip-448")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def freeze_backbone(model: nn.Module, model_type: str):
    """
    Freeze backbone parameters for transfer learning.
    
    Args:
        model: The model to freeze
        model_type: Type of model ('vit', 'dinov2')
    """
    if model_type in HF_MODELS:
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name:
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
        preprocessor: Optional preprocessor to save
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model_type = model_info["type"]
    
    if model_type in HF_MODELS:
        model.save_pretrained(save_dir)
        if preprocessor:
            preprocessor.save_pretrained(save_dir)