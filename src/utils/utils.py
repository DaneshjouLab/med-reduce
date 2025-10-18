# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""General utilities for environment, GPU, and I/O operations."""
import os
import json
import shutil
from typing import Dict, Any, Optional
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("pynvml not installed, GPU memory monitoring disabled.")

def env_path(key: str, default: str) -> str:
    """Get environment variable or default value."""
    return os.environ.get(key, default)

def setup_environment():
    """Setup cache paths and environment variables."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = os.getenv(
        "HF_HOME", "~/.cache/huggingface/transformers"
    )
    os.environ["HF_DATASETS_CACHE"] = os.getenv(
        "HF_DATASETS_CACHE", "~/.cache/huggingface/datasets"
    )
    os.environ["HF_HOME"] = os.getenv("HF_HOME", "~/.cache/huggingface")

def get_gpu_memory(device_id: int = 0) -> float:
    """
    Get GPU memory usage in MB.

    Returns:
        Memory usage in MB, or -1 if unavailable.
    """
    if not torch.cuda.is_available() or not PYNVML_AVAILABLE:
        return -1

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return -1

def check_disk_space(required_gb: float = 1.0) -> bool:
    """
    Check if sufficient disk space is available.

    Args:
        required_gb: Required space in GB

    Returns:
        True if sufficient space available
    """
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (2**30)

    print(f"Disk space: {free_gb:.2f} GB free")

    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space. Need {required_gb} GB, have {free_gb:.2f} GB"
        )
    return True

def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {filepath}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
