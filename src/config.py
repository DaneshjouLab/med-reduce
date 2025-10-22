# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""Configuration and constants."""
from dataclasses import dataclass
from typing import Dict, Any

# Optional import - used for hardware detection
try:
    import torch  # pylint: disable=import-error
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    # If torch is not available, assume no CUDA
    CUDA_AVAILABLE = False

# Model constants
HF_MODELS = ["vit", "dinov2"]

# Dataset constants
NUM_CLASSES = 8
FILTERED_CLASSES = ["0", "1"]
NUM_FILTERED_CLASSES = len(FILTERED_CLASSES)

# Image constants
DEFAULT_IMAGE_SIZE = 224
IMAGE_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


@dataclass
class TrainingConfig: # pylint: disable=too-many-instance-attributes
    """Training configuration."""

    num_train_images: int = 100
    proportion_per_transform: float = 0.2
    resolution: int = 224
    batch_size: int = 256
    num_epochs: int = 3
    eval_steps: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return self.__dict__

    def to_wandb_config(self) -> Dict[str, Any]:
        """Create wandb configuration."""
        return {
            **self.to_dict(),
            "gpu_available": CUDA_AVAILABLE,
        }


# Model configurations
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
]
