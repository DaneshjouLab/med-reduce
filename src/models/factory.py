"""Model factory for loading pre-trained encoders."""
import torch
import timm
from typing import Tuple, Any
from transformers import AutoImageProcessor, AutoModel

from src.config import UNIV2_CONFIG, DINOV3_CONFIG, SUPPORTED_ENCODERS
from src.transformation.transforms import get_univ2_transforms


def load_univ2() -> Tuple[torch.nn.Module, Any]:
    """
    Load UNI2 (Universal Vision Transformer) model.
    
    Returns:
        Tuple of (model, transforms)
    """
    weights_path = UNIV2_CONFIG["model_path"]
    
    timm_kwargs = {
        'img_size': UNIV2_CONFIG["img_size"],
        'patch_size': UNIV2_CONFIG["patch_size"],
        'depth': UNIV2_CONFIG["depth"],
        'num_heads': UNIV2_CONFIG["num_heads"],
        'init_values': UNIV2_CONFIG["init_values"],
        'embed_dim': UNIV2_CONFIG["embed_dim"],
        'mlp_ratio': UNIV2_CONFIG["mlp_ratio"],
        'num_classes': UNIV2_CONFIG["num_classes"],
        'no_embed_class': UNIV2_CONFIG["no_embed_class"],
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    
    model = timm.create_model(
        model_name='vit_giant_patch14_224',
        pretrained=False,
        **timm_kwargs
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    
    transforms = get_univ2_transforms()
    
    return model, transforms


def load_dinov3() -> Tuple[torch.nn.Module, Any]:
    """
    Load DINOv3 (Self-supervised Vision Transformer) model.
    
    Returns:
        Tuple of (model, transforms)
    """
    model_id = DINOV3_CONFIG["model_id"]
    
    transforms = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    
    return model, transforms


def get_patch_encoder(model_name: str) -> Tuple[torch.nn.Module, Any]:
    """
    Factory function to get patch encoder by name.
    
    Args:
        model_name: Name of the encoder ('univ2' or 'dinov3')
        
    Returns:
        Tuple of (model, transforms)
        
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in SUPPORTED_ENCODERS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from {SUPPORTED_ENCODERS}"
        )
    
    if model_name == "univ2":
        return load_univ2()
    elif model_name == "dinov3":
        return load_dinov3()
    else:
        raise ValueError(f"Unknown model: {model_name}")

