"""Configuration and constants for TCGA pipeline."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os

# Root directory
ROOT_DIR = os.getenv("TCGA_ROOT", "/oak/stanford/groups/roxanad/")

# Data directories
OUTPUTS_DIR = os.path.join(ROOT_DIR, "rpark23/outputs/")
CLINICAL_DIR = os.path.join(ROOT_DIR, "rpark23/clinical_data/tcga/")
WSI_DATASETS_DIR = os.path.join(ROOT_DIR, "wsi-datasets/tcga/")
CACHE_DIR = os.path.join(ROOT_DIR, "rpark23/cache/hub/")

# Model constants
SUPPORTED_ENCODERS = ["univ2", "dinov3"]

# Image normalization constants
IMAGE_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# Default image sizes
DEFAULT_PATCH_SIZE = 224
DEFAULT_WSI_PATCH_SIZE = 512

# UNI2 Model configuration
UNIV2_CONFIG = {
    "model_path": os.path.join(
        CACHE_DIR,
        "models--MahmoodLab--UNI2-h/snapshots/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin"
    ),
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
}

# DINOv3 Model configuration
DINOV3_CONFIG = {
    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

@dataclass
class SegmentationConfig:
    """Configuration for tissue segmentation."""
    confidence_thresh: float = 0.5
    patch_len: int = 512
    level: int = 0
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class EncodingConfig:
    """Configuration for patch encoding."""
    model_name: str = "univ2"
    level: int = 0
    patch_len: int = 224
    batch_size: int = 32
    num_workers: int = 4
    threshold: float = 0.5  # Tissue fraction threshold
    
    def __post_init__(self):
        if self.model_name not in SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder: {self.model_name}. "
                f"Choose from {SUPPORTED_ENCODERS}"
            )


@dataclass
class ClassificationConfig:
    """Configuration for linear classification."""
    input_dim: Optional[int] = None
    num_epochs: int = 20
    batch_size: int = 64
    num_workers: int = 4
    lr_range: tuple = (1e-6, 1e2)
    num_lr_steps: int = 33
    device: str = "cuda"  # Will be auto-detected if cuda not available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return self.__dict__


@dataclass
class DataSplitConfig:
    """Configuration for dataset splitting."""
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 < total < 1.01):  # Allow small floating point errors
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total}"
            )


def get_slide_dir(dataset: str) -> str:
    """Get slide directory for a specific dataset."""
    return os.path.join(WSI_DATASETS_DIR, dataset, "svs/")


def get_tissue_info_dir(dataset: str, level: int) -> str:
    """Get tissue info directory for a specific dataset and level."""
    return os.path.join(OUTPUTS_DIR, f"hest/tcga/{dataset}/level_{level}/")


def get_features_dir(model_name: str, dataset: str, level: int) -> str:
    """Get features directory for a specific model, dataset, and level."""
    return os.path.join(OUTPUTS_DIR, f"{model_name}/tcga/{dataset}/level_{level}/")

