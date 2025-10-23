"""Image transformations for preprocessing WSI patches."""
from torchvision import transforms
from PIL import Image
from typing import Callable, List

from src.config import IMAGE_NORMALIZATION


def get_univ2_transforms() -> transforms.Compose:
    """
    Get standard UNI2 preprocessing transforms.
    
    Returns:
        Composed torchvision transforms
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGE_NORMALIZATION["mean"],
            std=IMAGE_NORMALIZATION["std"]
        ),
    ])


def get_segmentation_transforms() -> transforms.Compose:
    """
    Get preprocessing transforms for tissue segmentation.
    
    Returns:
        Composed torchvision transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGE_NORMALIZATION["mean"],
            std=IMAGE_NORMALIZATION["std"]
        )
    ])


def quarter_resolution(region_rgb: Image.Image) -> Image.Image:
    """
    Reduce image resolution by half using bilinear interpolation.
    
    Args:
        region_rgb: PIL Image
        
    Returns:
        Resized PIL Image at half resolution
    """
    w, h = region_rgb.size
    return region_rgb.resize((w // 2, h // 2), resample=Image.BILINEAR)


def get_custom_transforms(
    transforms_list: List[Callable],
    include_normalization: bool = True
) -> List[Callable]:
    """
    Create custom transformation pipeline.
    
    Args:
        transforms_list: List of transformation functions
        include_normalization: Whether to include standard normalization
        
    Returns:
        List of transformation functions
    """
    if include_normalization:
        normalize = transforms.Normalize(
            mean=IMAGE_NORMALIZATION["mean"],
            std=IMAGE_NORMALIZATION["std"]
        )
        transforms_list.append(normalize)
    
    return transforms_list

