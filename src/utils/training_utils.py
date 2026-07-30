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

    Profiles a CPU deepcopy of the model, NOT the live model. thop registers
    forward hooks and ``total_ops``/``total_params`` buffers on every submodule
    and does not reliably remove them; if run on the live encoder they linger and
    fire during embedding extraction, crashing with a cuda/cpu device mismatch.
    Copying isolates all of that and leaves the extraction model untouched.
    """
    try:
        import copy
        from thop import profile  # type: ignore
        model_copy = copy.deepcopy(model).to("cpu").eval()
        dummy = torch.randn(1, 3, resolution, resolution)
        flops, _ = profile(model_copy, inputs=(dummy,))
        del model_copy
        return float(flops) / 1e9
    except Exception as e:
        print(f"[profile_model] FLOP profiling failed: {e}")
        return -1.0

def calculate_inference_latency(
    model: torch.nn.Module,
    resolution: int,
    warmup_runs: int = 10,
    bench_runs: int = 100
) -> float:
    """
    Measure actual model inference latency in ms via repeated forward passes.
    Returns -1 on failure.

    Args:
        model: PyTorch model to benchmark
        resolution: Input image resolution (assumes square images)
        warmup_runs: Number of warmup iterations before timing
        bench_runs: Number of timed iterations for averaging
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy = torch.randn(1, 3, resolution, resolution, device=device)
        model = model.to(device)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy)

        # Synchronize before timing (critical for GPU)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        import time
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(bench_runs):
                _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()

        avg_latency_ms = ((end - start) / bench_runs) * 1000
        return float(avg_latency_ms)

    except Exception as e:
        print(f"[calculate_inference_latency] Latency benchmarking failed: {e}")
        return -1.0