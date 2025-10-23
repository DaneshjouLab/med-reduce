"""General utility functions for I/O operations."""
import os
import json
import pickle
from typing import Any, Dict
from PIL import Image


def save_pickle(obj: Any, path: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to pickle
        path: Path to save file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        path: Path to pickle file
        
    Returns:
        Unpickled object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filename: Path to save file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filename: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filename: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filename, "r") as f:
        return json.load(f)


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

