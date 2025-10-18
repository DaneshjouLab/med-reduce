# src/utils/training_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from typing import Optional

import torch

def env_path(key: str, default: str = ".") -> str:
    """Read an env var with a default; expand ~ and vars."""
    return os.path.expanduser(os.path.expandvars(os.getenv(key, default)))

def get_gpu_memory() -> int:
    """Return total used GPU memory (MB) across visible GPUs; 0 if unavailable."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        used = 0
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            used += mem.used
        pynvml.nvmlShutdown()
        return int(used / (1024 * 1024))
    except Exception:
        return 0

def profile_model(model: torch.nn.Module, resolution: int) -> float:
    """
    Estimate model FLOPs in GFLOPs using thop; returns -1 on failure.
    """
    try:
        from thop import profile  # type: ignore
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy = torch.randn(1, 3, resolution, resolution, device=device)
        model = model.to(device)
        flops, _ = profile(model, inputs=(dummy,))
        return float(flops) / 1e9
    except Exception as e:
        print(f"[profile_model] FLOP profiling failed: {e}")
        return -1.0
