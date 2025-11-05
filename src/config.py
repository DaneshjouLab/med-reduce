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


@dataclass
class HyperparamSearchConfig:
    """Hyperparameter search configuration."""
    enabled: bool = False
    n_samples: int = 15
    subset_frac: float = 0.3
    use_cv: bool = False
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "weight_decay": [0.0, 0.01, 0.05, 0.1],
        "batch_size": [32, 64, 128],
    })
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.enabled:
            if not self.param_grid:
                raise ValueError("param_grid must be specified when search is enabled")
            if not 0.0 < self.subset_frac <= 1.0:
                raise ValueError(f"subset_frac must be in (0, 1], got {self.subset_frac}")
            if self.n_samples < 1:
                raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "n_samples": self.n_samples,
            "subset_frac": self.subset_frac,
            "use_cv": self.use_cv,
            "param_grid": self.param_grid,
        }


@dataclass
class TrainingConfig:  # pylint: disable=too-many-instance-attributes
    """Training configuration."""
    
    # Core training loop params
    epochs: int = 3  # Merged num_epochs and epochs to one canonical name
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: Optional[float] = None
    mixed_precision: bool = BFLOAT16_AVAILABLE  # Use bfloat16 if supported
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
    deterministic: bool = False
    
    # Hyperparameter search
    hyperparam_search: HyperparamSearchConfig = field(default_factory=HyperparamSearchConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.hyperparam_search.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        base_dict = {k: v for k, v in self.__dict__.items() if k != "hyperparam_search"}
        base_dict["hyperparam_search"] = self.hyperparam_search.to_dict()
        return base_dict

    def to_wandb_config(self) -> Dict[str, Any]:
        """Create wandb configuration."""
        return {
            **self.to_dict(),
            "cuda_available": CUDA_AVAILABLE,
            "bfloat16_available": BFLOAT16_AVAILABLE,
        }