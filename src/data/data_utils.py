"""Utility functions for data loading and processing."""
import os
import openslide
from torch.utils.data import DataLoader
from typing import List, Optional, Any, Tuple, Dict, Callable
from PIL import Image

from src.config import get_slide_dir
from .datasets import TCGASlides, TCGAPatches, TCGATissuePatches


def is_svs(path: str) -> bool:
    """
    Check if path is a valid SVS file.
    
    Args:
        path: File path
        
    Returns:
        True if path exists and has .svs extension
    """
    return os.path.exists(path) and path.endswith(".svs")


def get_patient_id(filename: str) -> str:
    """
    Extract patient ID from TCGA filename.
    
    Args:
        filename: TCGA filename or path
        
    Returns:
        Patient ID (e.g., 'TCGA-XX-XXXX')
    """
    f = filename.split("/")[-1]
    return "-".join(f.split("-")[:3])


def is_dx(filename: str) -> bool:
    """
    Check if filename is a DX (diagnostic) slide.
    
    Args:
        filename: TCGA filename or path
        
    Returns:
        True if slide is DX type
    """
    f = filename.split("/")[-1]
    return f.split("-")[5][:2] == "DX"


def single_sample_collate(batch: List[Any]) -> Any:
    """Collate function that returns single sample."""
    return batch[0]


def get_slides_loader(
    datasets: List[str],
    var: Optional[str] = None,
    verbose: bool = True
) -> DataLoader:
    """
    Create DataLoader for TCGA slides.
    
    Args:
        datasets: List of dataset names
        var: Optional variable for labeling
        verbose: Whether to print info
        
    Returns:
        DataLoader for slides
    """
    slides_dataset = TCGASlides(datasets, var=var)
    slides_loader = DataLoader(
        slides_dataset,
        collate_fn=single_sample_collate
    )
    
    if verbose:
        num_slides = len(slides_loader)
        dataset_names = ', '.join([d.upper() for d in datasets])
        print(f"Processing {num_slides} {dataset_names} slides ...")
    
    return slides_loader


def get_slide_by_path(slide_path: str) -> openslide.OpenSlide:
    """
    Load slide from path.
    
    Args:
        slide_path: Path to slide file
        
    Returns:
        OpenSlide object
        
    Raises:
        FileNotFoundError: If slide path doesn't exist
    """
    if os.path.exists(slide_path):
        return openslide.OpenSlide(slide_path)
    raise FileNotFoundError(f"{slide_path} could not be found")


def view_whole_slide(slide: openslide.OpenSlide) -> Image.Image:
    """
    Get thumbnail of whole slide at lowest resolution.
    
    Args:
        slide: OpenSlide object
        
    Returns:
        PIL Image of whole slide
    """
    max_level = slide.level_count - 1
    whole_slide = slide.read_region(
        (0, 0),
        max_level,
        slide.level_dimensions[max_level]
    )
    return whole_slide


def get_slide_path(slide_dir: str, short_name: str) -> str:
    """
    Get full slide path from directory and short name.
    
    Args:
        slide_dir: Directory containing slides
        short_name: Short identifier for slide
        
    Returns:
        Full path to slide
    """
    matching_files = [f for f in os.listdir(slide_dir) if f.startswith(short_name)]
    if not matching_files:
        raise FileNotFoundError(f"No slide found with prefix {short_name}")
    return os.path.join(slide_dir, matching_files[0])


def get_patches_loader(
    slide_path: str,
    level: int,
    transforms: Any,
    patch_len: int,
    batch_size: int,
    num_workers: int,
    verbose: bool = True
) -> DataLoader:
    """
    Create DataLoader for patches from a slide.
    
    Args:
        slide_path: Path to slide
        level: Pyramid level
        transforms: Transform pipeline
        patch_len: Patch size
        batch_size: Batch size
        num_workers: Number of workers
        verbose: Whether to print info
        
    Returns:
        DataLoader for patches
    """
    patches_dataset = TCGAPatches(slide_path, level, transforms, patch_len)
    patches_loader = DataLoader(
        patches_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return patches_loader


def get_tissue_patches_loader(
    slide_path: str,
    pkl: Dict[Tuple[int, int], float],
    level: int,
    encoder_transforms: Any,
    patch_transforms: List[Callable],
    threshold: float,
    patch_len: int,
    batch_size: int,
    num_workers: int,
    verbose: bool = True
) -> DataLoader:
    """
    Create DataLoader for tissue patches above threshold.
    
    Args:
        slide_path: Path to slide
        pkl: Dictionary of coords to tissue fractions
        level: Pyramid level
        encoder_transforms: Encoder transforms
        patch_transforms: Additional transforms
        threshold: Tissue fraction threshold
        patch_len: Patch size
        batch_size: Batch size
        num_workers: Number of workers
        verbose: Whether to print info
        
    Returns:
        DataLoader for tissue patches
    """
    tissue_patches_dataset = TCGATissuePatches(
        slide_path, pkl, level, encoder_transforms,
        patch_transforms, threshold, patch_len
    )
    tissue_patches_loader = DataLoader(
        tissue_patches_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return tissue_patches_loader

