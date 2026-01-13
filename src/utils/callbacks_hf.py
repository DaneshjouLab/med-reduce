# src/utils/callbacks_hf.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import csv
from typing import Dict, Any, Optional
from transformers import TrainerCallback  # type: ignore

from src.utils.training_utils import get_gpu_memory, profile_model, calculate_inference_latency

# Optional W&B; safe if not installed
try:
    import wandb  # type: ignore
    _WANDB = True
except Exception:
    _WANDB = False

class LossLoggerCallback(TrainerCallback):
    """Append step logs to a JSONL file (robust, HF-friendly)."""
    def __init__(self, log_dir: str, phase: str, model_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{model_name}_{phase}_log.jsonl")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload: Dict[str, Any] = {"step": state.global_step, **logs}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(payload) + "\n")

class WandbCallback(TrainerCallback):
    """W&B logger with model/phase, GPU memory, FLOPs, and inference latency."""
    def __init__(
        self,
        model_name: str,
        phase: str,
        model: Optional[Any] = None,
        image_size: Optional[int] = None,
        log_model_metrics: bool = True
    ):
        self.model_name = model_name
        self.phase = phase
        self.best_accuracy = 0.0
        self.model = model
        self.image_size = image_size
        self.log_model_metrics = log_model_metrics
        self._profiled = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Profile model FLOPs and latency once at training start."""
        if not _WANDB or not self.log_model_metrics or self._profiled:
            return
        if self.model is None or self.image_size is None:
            return

        self._profiled = True

        # Profile FLOPs
        gflops = profile_model(self.model, self.image_size)
        if gflops > 0:
            wandb.log({"model/gflops": gflops}, step=0)

        # Profile inference latency
        latency_ms = calculate_inference_latency(self.model, self.image_size)
        if latency_ms > 0:
            wandb.log({"model/inference_latency_ms": latency_ms}, step=0)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _WANDB or not logs:
            return
        logs = dict(logs)
        logs["model"] = self.model_name
        logs["phase"] = self.phase
        mem = get_gpu_memory()
        if mem > 0:
            logs["gpu_memory_mb"] = mem
        wandb.log(logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not _WANDB or not metrics:
            return
        if "eval_accuracy" in metrics:
            self.best_accuracy = max(self.best_accuracy, metrics["eval_accuracy"])
            metrics = dict(metrics, best_accuracy=self.best_accuracy)
        wandb.log(metrics)