# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/wrappers/probe.py
# -*- coding: utf-8 -*-
"""Linear probing wrapper for training classification heads on frozen backbones."""
from __future__ import annotations
from typing import Any, Dict

import os
import csv
import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate

# pylint: disable=import-error
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.utils.optim import make_optimizer_and_scheduler
from src.losses.classification import cross_entropy_loss
from src.models.factory import (
    create_model, create_preprocessor, freeze_backbone, save_model
)
from src.engines.linear_probe_engine import train_probe
from src.utils.training_utils import profile_model, calculate_inference_latency, get_gpu_memory
from src.evaluation.metrics_collector import MetricsCollector, compute_composite_efficiency

log = get_logger(__name__)


class ProbeWrapper:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Orchestrates linear probing:
      - builds model (+preprocessor if you need it for your datasets),
      - freezes backbone,
      - prepares dataloaders via DataModule,
      - creates optimizer/scheduler/loss,
      - calls the probe engine,
      - saves best model.
    """

    def __init__(self, cfg: Any):
        """
        Expected cfg fields (suggested):
          cfg.model: {type, model_id, config{num_labels,...}}
          cfg.data: {dataset_name, data_dir, image_size, batch_size, num_workers}
          cfg.train: {epochs, optimizer{...}, scheduler{...}, grad_clip, mixed_precision}
          cfg.loss: {label_smoothing, ignore_index, reduction}
          cfg.logging: {project, entity, run_name, wandb_enabled}
          cfg.runtime: {run_dir}
          cfg.metrics: {teacher_model, teacher_accuracy, teacher_flops, teacher_latency, teacher_memory} (optional)
        """
        self.cfg = cfg
        setup_logging()

        # Initialize metrics collector
        run_dir = getattr(cfg.runtime, "run_dir", "./runs/probe")
        experiment_name = getattr(cfg.logging, "run_name", "probe_experiment")
        self.metrics_collector = MetricsCollector(run_dir, experiment_name)

        # Build model
        self.model_info = cfg.model
        self.model = create_model(self.model_info, resolution=cfg.data.image_size)

        # Optionally create preprocessor (useful if datamodule needs it)
        try:
            self.preprocessor = create_preprocessor(
                self.model_info, resolution=cfg.data.image_size
            )
        except (ImportError, AttributeError, KeyError) as e:
            log.debug(f"Preprocessor creation failed: {e}")
            self.preprocessor = None

        # Freeze backbone for linear probing
        freeze_backbone(self.model, self.model_info["type"])

        # Data - Use datamodule from config if specified, otherwise create default
        if hasattr(cfg, 'datamodule') and cfg.datamodule is not None:
            self.dm = instantiate(cfg.datamodule)
        else:
            # Fallback for backward compatibility
            raise ValueError(
                "Config must specify 'datamodule' section with _target_. "
                "Example: datamodule._target_ = 'src.data.isic_datamodule.ISICDataModule'"
            )
        self.dm.setup("fit")

        # Optimizer & scheduler
        self.optimizer, (self.scheduler, self.sched_meta) = (
            make_optimizer_and_scheduler(cfg, self.model.parameters())
        )

        # Loss
        self.loss_fn = cross_entropy_loss(
            label_smoothing=float(getattr(cfg.loss, "label_smoothing", 0.0)),
            class_weight=None,
            ignore_index=int(getattr(cfg.loss, "ignore_index", -100)),
            reduction=str(getattr(cfg.loss, "reduction", "mean")),
        )

        # W&B
        self.wandb = WandbLogger(
            project=getattr(cfg.logging, "project", "resolution-aware-probe"),
            run_name=getattr(cfg.logging, "run_name", None),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            entity=getattr(cfg.logging, "entity", None),
            tags=getattr(cfg.logging, "tags", ["probe"]),
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _make_loaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for training and validation."""
        return {
            "train": self.dm.train_dataloader(),
            "val": self.dm.val_dataloader(),
            # "test": self.dm.test_dataloader(),  # optional
        }

    def train(self) -> Dict[str, Any]:
        """Run linear probing training."""
        log.info("Starting linear probe...")
        import time
        from datetime import datetime
        start_time = time.time()

        # Profile model metrics once
        log.info("Profiling model performance metrics...")
        gflops = profile_model(self.model, self.cfg.data.image_size)
        latency_ms = calculate_inference_latency(self.model, self.cfg.data.image_size)
        gpu_memory_mb = get_gpu_memory()

        # Compute composite efficiency
        composite_compute = compute_composite_efficiency(gflops, latency_ms, gpu_memory_mb)

        # Log to WandB
        if self.wandb:
            self.wandb.log({
                "model/gflops": gflops,
                "model/inference_latency_ms": latency_ms,
                "model/gpu_memory_mb": gpu_memory_mb,
                "model/composite_compute": composite_compute
            })

        # Legacy CSV for backward compatibility
        run_dir = getattr(self.cfg.runtime, "run_dir", "./runs/probe")
        os.makedirs(run_dir, exist_ok=True)
        metrics_csv = os.path.join(run_dir, f"{self.model_info.get('model_id', 'model')}_probe_metrics.csv")

        metrics_data = {
            "model_name": self.model_info.get("model_id", "unknown"),
            "phase": "probe",
            "gflops": gflops,
            "inference_latency_ms": latency_ms,
            "gpu_memory_mb": gpu_memory_mb,
        }

        with open(metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_data.keys()))
            writer.writeheader()
            writer.writerow(metrics_data)

        log.info(f"Model metrics saved to {metrics_csv}")

        # Run training
        results = train_probe(
            model=self.model,
            loaders=self._make_loaders(),
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=(self.scheduler, self.sched_meta),
            device=self.device,
            epochs=int(self.cfg.train.epochs),
            grad_clip=getattr(self.cfg.train, "grad_clip", None),
            mixed_precision=bool(
                getattr(self.cfg.train, "mixed_precision", True)
            ),
            log_interval=int(getattr(self.cfg.train, "log_interval", 50)),
            wandb_logger=self.wandb,
            metric_key=str(getattr(self.cfg.train, "metric_key", "val_acc")),
        )

        # Save best model (HF format if applicable)
        try:
            save_model(
                self.model,
                self.model_info,
                save_dir=run_dir,
                preprocessor=self.preprocessor
            )
        except (OSError, ValueError, AttributeError) as e:
            log.warning(f"Failed to save model in HF format; error: {e}")

        # Calculate training time
        training_time_hours = (time.time() - start_time) / 3600.0

        # Update legacy CSV with final results
        final_metrics = dict(metrics_data)
        final_metrics["best_metric"] = results.get("best_metric", None)
        final_metrics["final_gpu_memory_mb"] = get_gpu_memory()

        with open(metrics_csv, "a", newline="") as f:
            fieldnames = list(final_metrics.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(final_metrics)

        # Collect comprehensive metrics using new system
        best_metric = results.get("best_metric", 0.0)
        experiment_metrics = {
            "experiment_id": f"{self.model_info.get('model_id', 'unknown')}_{datetime.now().isoformat()}",
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_info.get("model_id", "unknown"),
            "model_family": self.model_info.get("type", "unknown"),
            "phase": "probe",
            "dataset": getattr(self.cfg.data, "dataset_name", "unknown"),
            "task": "classification",
            "num_classes": getattr(self.model_info.get("config", {}), "num_labels", 0),
            "image_resolution": self.cfg.data.image_size,
            "epochs": int(self.cfg.train.epochs),
            "batch_size": getattr(self.cfg.data, "batch_size", 0),
            "learning_rate": float(getattr(self.cfg.train.optimizer, "lr", 0.0)),
            "optimizer": getattr(self.cfg.train.optimizer, "_target_", "unknown").split(".")[-1],
            "top1_accuracy": float(best_metric * 100),  # Convert to percentage if needed
            "flops_giga": float(gflops),
            "inference_latency_ms": float(latency_ms),
            "peak_gpu_memory_mb": float(gpu_memory_mb),
            "training_time_hours": training_time_hours,
            "composite_compute": float(composite_compute),
            "config_path": getattr(self.cfg, "config_path", None),
            "checkpoint_path": run_dir,
        }

        # Add teacher model info if available
        if hasattr(self.cfg, "metrics"):
            experiment_metrics.update({
                "teacher_model": getattr(self.cfg.metrics, "teacher_model", None),
                "teacher_accuracy": getattr(self.cfg.metrics, "teacher_accuracy", None),
                "teacher_flops": getattr(self.cfg.metrics, "teacher_flops", None),
                "teacher_latency": getattr(self.cfg.metrics, "teacher_latency", None),
                "teacher_memory": getattr(self.cfg.metrics, "teacher_memory", None),
            })

            # Compute teacher composite compute if all metrics available
            if all(getattr(self.cfg.metrics, k, None) is not None
                   for k in ["teacher_flops", "teacher_latency", "teacher_memory"]):
                experiment_metrics["teacher_composite_compute"] = compute_composite_efficiency(
                    self.cfg.metrics.teacher_flops,
                    self.cfg.metrics.teacher_latency,
                    self.cfg.metrics.teacher_memory
                )

        # Store metrics using new collector
        self.metrics_collector.add_metrics_dict(experiment_metrics)
        log.info(f"Comprehensive metrics saved to {self.metrics_collector.json_path}")

        # Finish W&B
        if self.wandb:
            self.wandb.log({
                "best/metric": results.get("best_metric", None),
                "model/composite_compute": composite_compute,
                "training/time_hours": training_time_hours
            })
            if experiment_metrics.get("aet_score") is not None:
                self.wandb.log({"model/aet_score": experiment_metrics["aet_score"]})
            self.wandb.finish()

        log.info("Linear probe finished.")
        return results


def run(cfg: Any) -> Dict[str, Any]:
    """Convenience function to run from a CLI entrypoint."""
    wrapper = ProbeWrapper(cfg)
    return wrapper.train()
