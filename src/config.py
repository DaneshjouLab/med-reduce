# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""Configuration and constants."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Optional import - used for hardware detection
try:
    import torch  # pylint: disable=import-error
    CUDA_AVAILABLE = torch.cuda.is_available()
    BFLOAT16_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8
except ImportError:
    # If torch is not available, assume no CUDA/bfloat16
    CUDA_AVAILABLE = False
    BFLOAT16_AVAILABLE = False


# --- GLOBAL CONSTANTS ---

# Model constants
HF_MODELS = ["vit", "dinov2", "dinov3"]

# Dataset constants
NUM_CLASSES = 8
FILTERED_CLASSES = [0, 1]  # Changed to integers for standard ML use
NUM_FILTERED_CLASSES = len(FILTERED_CLASSES)

# Image constants
DEFAULT_IMAGE_SIZE = 224
IMAGE_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# --- MODEL REGISTRY ---
# Defines pre-configured models that ModelConfig will select from
MODEL_REGISTRY = [
    {
        "name": "vit",
        "model_id": "google/vit-base-patch16-224",
        "type": "vit",
        "config": {"num_labels": NUM_FILTERED_CLASSES, "ignore_mismatched_sizes": True},
    },
    {
        "name": "dinov2",
        "model_id": "facebook/dinov2-base",
        "type": "dinov2",
        "config": {"num_labels": NUM_FILTERED_CLASSES, "ignore_mismatched_sizes": True},
    },
    {
        "name": "dinov3",
        "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "type": "dinov3",
        "dtype": "bfloat16",  # Use string here, map to torch.dtype in factory
        "config": {
            "num_labels": NUM_FILTERED_CLASSES,
            "ignore_mismatched_sizes": True
        }
    },
]


# --- CONFIGURATION DATACLASSES ---

@dataclass
class ModelConfig:
    """Model selection configuration (replaces direct registry access)."""
    name: str = "vit" # Key used to look up settings in MODEL_REGISTRY
    checkpoint_path: Optional[str] = None # For loading custom weights

@dataclass
class DataConfig:
    """Dataset and DataLoader configuration (required by FinetuneWrapper)."""
    dataset_name: str = "flowers"
    data_dir: str = "./data/flowers"
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = CUDA_AVAILABLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return self.__dict__

@dataclass
class LossConfig:
    """Loss function configuration (required by FinetuneWrapper)."""
    label_smoothing: float = 0.0
    ignore_index: int = -100
    reduction: str = "mean"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return self.__dict__

@dataclass
class TrainingConfig:  # pylint: disable=too-many-instance-attributes
    """Training configuration."""
    
    # Core training loop params
    epochs: int = 3  # Merged num_epochs and epochs to one canonical name
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: Optional[float] = None
    mixed_precision: bool = BFLOAT16_AVAILABLE # Use bfloat16 if supported
    log_interval: int = 50
    metric_key: str = "val_acc"
    
    # Optimizer/Scheduler choice
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"

    # Data/Ablation/CV params (separated for clarity)
    num_train_images: int = 100
    proportion_per_transform: float = 0.2
    eval_steps: int = 10
    k_folds: int = 5
    subset_frac: float = 1.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return self.__dict__

    def to_wandb_config(self) -> Dict[str, Any]:
        """Create wandb configuration."""
        return {
            **self.to_dict(),
            "cuda_available": CUDA_AVAILABLE,
            "bfloat16_available": BFLOAT16_AVAILABLE,
        }

@dataclass
class RuntimeConfig:
    """Runtime and environment configuration."""
    
    run_dir: str = "./runs/finetune"
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return self.__dict__
    
@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""
    
    project: str = "resolution-aware-finetune"
    run_name: Optional[str] = None
    wandb_enabled: bool = True
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["finetune"])
    
    # UMAP embedding settings (used in FinetuneWrapper.train)
    save_umap_embeddings: bool = False
    umap_max_samples: Optional[int] = None 
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return self.__dict__

# --- MAIN CONFIGURATION ---

@dataclass
class MainConfig:
    """The root configuration object passed to the run function."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

# Instantiate the main config object that will be passed to run(cfg)
CONFIG = MainConfig()