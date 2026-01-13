# src/utils/reproducibility.py
"""Reproducibility utilities for ensuring deterministic training.

This module provides functions for:
1. Setting random seeds across all libraries
2. Creating worker initialization functions for DataLoader
3. Logging and saving seed information
4. Validating reproducibility settings
"""
import random
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seed for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic mode for PyTorch
                      (slower but guarantees reproducibility)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info(f"Seed {seed} set with DETERMINISTIC mode enabled")
    else:
        torch.backends.cudnn.benchmark = True
        logger.info(f"Seed {seed} set with non-deterministic mode (faster but less reproducible)")


def get_worker_init_fn(base_seed: int):
    """Create worker initialization function for DataLoader.

    This ensures that each DataLoader worker has a different but reproducible seed.

    Args:
        base_seed: Base random seed

    Returns:
        Worker initialization function for DataLoader

    Usage:
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=4,
            worker_init_fn=get_worker_init_fn(seed)
        )
    """
    def worker_init_fn(worker_id: int):
        """Initialize worker with unique but deterministic seed."""
        # Each worker gets: base_seed + worker_id
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # PyTorch's dataloader workers inherit the parent process's random state
        # but we reset them here for clarity
        torch.manual_seed(worker_seed)

    return worker_init_fn


class SeedTracker:
    """Track and log seed information throughout training.

    This class helps maintain a record of all seeds used during an experiment
    for full reproducibility.
    """

    def __init__(self, base_seed: int):
        """Initialize seed tracker.

        Args:
            base_seed: Base random seed for the experiment
        """
        self.base_seed = base_seed
        self.seed_log: Dict[str, Any] = {
            "base_seed": base_seed,
            "components": {}
        }

    def log_seed(self, component: str, seed: int, metadata: Optional[Dict[str, Any]] = None):
        """Log a seed used by a specific component.

        Args:
            component: Name of the component (e.g., "dataloader", "split_manager", "augmentation")
            seed: Seed value used
            metadata: Optional additional metadata about how the seed is used
        """
        entry = {
            "seed": seed,
            "metadata": metadata or {}
        }
        self.seed_log["components"][component] = entry
        logger.debug(f"Logged seed for {component}: {seed}")

    def save(self, output_path: Path):
        """Save seed log to JSON file.

        Args:
            output_path: Path to save seed log (should end in .json)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.seed_log, f, indent=2)

        logger.info(f"Saved seed log to {output_path}")

    def get_summary(self) -> str:
        """Get human-readable summary of seed usage.

        Returns:
            Formatted string summary of all seeds
        """
        lines = [f"Base Seed: {self.base_seed}"]
        for component, info in self.seed_log["components"].items():
            lines.append(f"  {component}: {info['seed']}")
            if info['metadata']:
                for key, value in info['metadata'].items():
                    lines.append(f"    {key}: {value}")
        return "\n".join(lines)


def verify_reproducibility_settings() -> Dict[str, Any]:
    """Verify current reproducibility settings.

    Returns:
        Dictionary with current reproducibility configuration
    """
    settings = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "numpy_version": np.__version__,
        "python_hashseed": os.environ.get('PYTHONHASHSEED', 'not set'),
    }

    # Check if deterministic algorithms are enabled
    try:
        settings["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except AttributeError:
        settings["torch_deterministic_algorithms"] = "not available (torch < 1.8)"

    return settings


def log_reproducibility_info(output_dir: Optional[Path] = None):
    """Log comprehensive reproducibility information.

    Args:
        output_dir: Optional directory to save reproducibility info JSON
    """
    settings = verify_reproducibility_settings()

    logger.info("=== Reproducibility Settings ===")
    for key, value in settings.items():
        logger.info(f"  {key}: {value}")
    logger.info("================================")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        settings_file = output_dir / "reproducibility_settings.json"

        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)

        logger.info(f"Saved reproducibility settings to {settings_file}")

    return settings
