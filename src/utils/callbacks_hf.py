# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/utils/callbacks_hf.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any
from transformers import TrainerCallback  # type: ignore

from src.utils.training_utils import get_gpu_memory

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
    """Minimal W&B logger that adds model/phase and GPU memory (if available)."""
    def __init__(self, model_name: str, phase: str):
        self.model_name = model_name
        self.phase = phase
        self.best_accuracy = 0.0

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
