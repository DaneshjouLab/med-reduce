# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/evaluation/metrics_collector.py
# -*- coding: utf-8 -*-
"""
Comprehensive metrics collection and storage for experimental analysis.
Tracks accuracy, computational efficiency, and derived metrics like AET scores.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
import os
import json
import csv
import pandas as pd
import numpy as np


@dataclass
class ExperimentMetrics:
    """
    Complete metrics record for a single experiment run.
    Stores all information needed for accuracy-efficiency trade-off analysis.
    """
    # Experiment identification
    experiment_id: str
    timestamp: str
    model_name: str
    model_family: str  # e.g., "dinov2", "resnet", "vit"
    phase: str  # "probe", "finetune", "distill"

    # Dataset & task info
    dataset: str
    task: str  # "classification", "segmentation", etc.
    num_classes: int
    image_resolution: int

    # Training configuration
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str

    # Accuracy metrics
    top1_accuracy: float
    top5_accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None

    # Efficiency metrics
    flops_giga: float  # GFLOPs for forward pass
    inference_latency_ms: float  # ms per image
    peak_gpu_memory_mb: float  # Peak GPU memory during training
    training_time_hours: Optional[float] = None

    # Model size metrics
    num_parameters_millions: Optional[float] = None
    model_size_mb: Optional[float] = None

    # Composite efficiency metric (C = FLOPs × latency × memory)
    composite_compute: Optional[float] = None

    # Additional metadata
    notes: str = ""
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Teacher model info (for distillation/comparison)
    teacher_model: Optional[str] = None
    teacher_accuracy: Optional[float] = None
    teacher_flops: Optional[float] = None
    teacher_latency: Optional[float] = None
    teacher_memory: Optional[float] = None
    teacher_composite_compute: Optional[float] = None

    # Normalized metrics (relative to teacher)
    normalized_accuracy: Optional[float] = None
    normalized_compute: Optional[float] = None
    aet_score: Optional[float] = None  # Accuracy-Efficiency Trade-off

    # Cross-validation metrics (if applicable)
    cv_fold: Optional[int] = None
    cv_mean_accuracy: Optional[float] = None
    cv_std_accuracy: Optional[float] = None

    def __post_init__(self):
        """Compute derived metrics after initialization."""
        self._compute_composite_metrics()

    def _compute_composite_metrics(self):
        """Calculate composite efficiency and normalized metrics."""
        # Composite compute: C = FLOPs × latency × memory
        if all(x is not None and x > 0 for x in [self.flops_giga, self.inference_latency_ms, self.peak_gpu_memory_mb]):
            self.composite_compute = self.flops_giga * self.inference_latency_ms * self.peak_gpu_memory_mb

        # If teacher info provided, compute normalized metrics
        if self.teacher_accuracy is not None and self.teacher_accuracy > 0:
            self.normalized_accuracy = self.top1_accuracy / self.teacher_accuracy

        if self.teacher_composite_compute is not None and self.teacher_composite_compute > 0:
            if self.composite_compute is not None:
                self.normalized_compute = self.composite_compute / self.teacher_composite_compute

        # AET score: ratio of normalized accuracy to normalized compute
        if self.normalized_accuracy is not None and self.normalized_compute is not None and self.normalized_compute > 0:
            self.aet_score = self.normalized_accuracy / self.normalized_compute

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary suitable for CSV export."""
        return {k: v for k, v in asdict(self).items()}


class MetricsCollector:
    """
    Central hub for collecting, storing, and managing experimental metrics.
    Supports multiple storage backends: JSON, CSV, and pandas DataFrames.
    """

    def __init__(self, output_dir: str, experiment_name: str = "experiment"):
        """
        Initialize metrics collector.

        Args:
            output_dir: Directory to save metrics files
            experiment_name: Name prefix for output files
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)

        # Storage paths
        self.json_path = os.path.join(output_dir, f"{experiment_name}_metrics.json")
        self.csv_path = os.path.join(output_dir, f"{experiment_name}_metrics.csv")
        self.summary_path = os.path.join(output_dir, f"{experiment_name}_summary.json")

        # In-memory storage
        self.metrics: List[ExperimentMetrics] = []

        # Load existing metrics if available
        self._load_existing_metrics()

    def _load_existing_metrics(self):
        """Load existing metrics from JSON file if it exists."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                    self.metrics = [ExperimentMetrics(**m) for m in data]
                print(f"Loaded {len(self.metrics)} existing experiment records from {self.json_path}")
            except Exception as e:
                print(f"Warning: Could not load existing metrics: {e}")

    def add_metrics(self, metrics: ExperimentMetrics):
        """
        Add new experimental metrics to the collection.

        Args:
            metrics: ExperimentMetrics object containing all measurements
        """
        self.metrics.append(metrics)
        self._save_all()

    def add_metrics_dict(self, metrics_dict: Dict[str, Any]) -> ExperimentMetrics:
        """
        Add metrics from a dictionary.

        Args:
            metrics_dict: Dictionary containing metric fields

        Returns:
            Created ExperimentMetrics object
        """
        # Set defaults for required fields if missing
        if "timestamp" not in metrics_dict:
            metrics_dict["timestamp"] = datetime.now().isoformat()
        if "experiment_id" not in metrics_dict:
            metrics_dict["experiment_id"] = f"{metrics_dict.get('model_name', 'unknown')}_{metrics_dict['timestamp']}"

        metrics = ExperimentMetrics(**metrics_dict)
        self.add_metrics(metrics)
        return metrics

    def _save_all(self):
        """Save metrics to all storage backends."""
        self._save_json()
        self._save_csv()
        self._save_summary()

    def _save_json(self):
        """Save metrics as JSON (preserves full structure)."""
        with open(self.json_path, 'w') as f:
            json.dump([m.to_dict() for m in self.metrics], f, indent=2)

    def _save_csv(self):
        """Save metrics as CSV (flattened for easy analysis)."""
        if not self.metrics:
            return

        df = pd.DataFrame([m.to_flat_dict() for m in self.metrics])
        df.to_csv(self.csv_path, index=False)

    def _save_summary(self):
        """Save summary statistics across all experiments."""
        if not self.metrics:
            return

        df = self.to_dataframe()

        summary = {
            "total_experiments": len(self.metrics),
            "unique_models": df['model_name'].nunique(),
            "unique_datasets": df['dataset'].nunique(),
            "accuracy_range": {
                "min": float(df['top1_accuracy'].min()),
                "max": float(df['top1_accuracy'].max()),
                "mean": float(df['top1_accuracy'].mean()),
                "std": float(df['top1_accuracy'].std())
            },
            "efficiency_stats": {
                "flops_range_giga": {
                    "min": float(df['flops_giga'].min()),
                    "max": float(df['flops_giga'].max()),
                    "mean": float(df['flops_giga'].mean())
                },
                "latency_range_ms": {
                    "min": float(df['inference_latency_ms'].min()),
                    "max": float(df['inference_latency_ms'].max()),
                    "mean": float(df['inference_latency_ms'].mean())
                }
            },
            "best_accuracy": {
                "model": df.loc[df['top1_accuracy'].idxmax(), 'model_name'],
                "accuracy": float(df['top1_accuracy'].max()),
                "flops": float(df.loc[df['top1_accuracy'].idxmax(), 'flops_giga'])
            },
            "best_efficiency": {
                "model": df.loc[df['flops_giga'].idxmin(), 'model_name'],
                "accuracy": float(df.loc[df['flops_giga'].idxmin(), 'top1_accuracy']),
                "flops": float(df['flops_giga'].min())
            },
            "timestamp": datetime.now().isoformat()
        }

        # Add AET statistics if available
        if 'aet_score' in df.columns and df['aet_score'].notna().any():
            aet_scores = df['aet_score'].dropna()
            summary["aet_stats"] = {
                "mean": float(aet_scores.mean()),
                "std": float(aet_scores.std()),
                "best_model": df.loc[aet_scores.idxmax(), 'model_name'],
                "best_score": float(aet_scores.max())
            }

        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame for analysis."""
        return pd.DataFrame([m.to_flat_dict() for m in self.metrics])

    def filter_by(self, **kwargs) -> List[ExperimentMetrics]:
        """
        Filter metrics by field values.

        Example:
            collector.filter_by(model_family="dinov2", phase="probe")
        """
        filtered = self.metrics
        for key, value in kwargs.items():
            filtered = [m for m in filtered if getattr(m, key, None) == value]
        return filtered

    def get_pareto_frontier(self,
                           accuracy_key: str = "top1_accuracy",
                           efficiency_key: str = "flops_giga",
                           minimize_efficiency: bool = True) -> pd.DataFrame:
        """
        Find Pareto-optimal models (best accuracy-efficiency trade-offs).

        Args:
            accuracy_key: Metric to maximize (e.g., "top1_accuracy")
            efficiency_key: Metric to optimize (e.g., "flops_giga")
            minimize_efficiency: If True, prefer lower efficiency values (e.g., FLOPs)

        Returns:
            DataFrame containing only Pareto-optimal experiments
        """
        df = self.to_dataframe()

        if df.empty:
            return df

        # Sort by efficiency
        df = df.sort_values(efficiency_key, ascending=minimize_efficiency)

        # Find Pareto frontier
        pareto_mask = np.zeros(len(df), dtype=bool)
        best_accuracy = -np.inf

        for idx, row in df.iterrows():
            if row[accuracy_key] > best_accuracy:
                pareto_mask[idx] = True
                best_accuracy = row[accuracy_key]

        return df[pareto_mask]

    def export_for_paper(self, output_path: Optional[str] = None) -> str:
        """
        Export formatted results table for LaTeX/paper.

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{self.experiment_name}_paper_table.csv")

        df = self.to_dataframe()

        # Select key columns for paper
        paper_cols = [
            'model_name', 'model_family', 'image_resolution',
            'top1_accuracy', 'flops_giga', 'inference_latency_ms',
            'peak_gpu_memory_mb', 'aet_score'
        ]

        # Filter to existing columns
        available_cols = [c for c in paper_cols if c in df.columns]
        paper_df = df[available_cols].copy()

        # Round numeric columns for readability
        numeric_cols = paper_df.select_dtypes(include=[np.number]).columns
        paper_df[numeric_cols] = paper_df[numeric_cols].round(3)

        # Sort by AET score if available, otherwise by accuracy
        if 'aet_score' in paper_df.columns:
            paper_df = paper_df.sort_values('aet_score', ascending=False)
        else:
            paper_df = paper_df.sort_values('top1_accuracy', ascending=False)

        paper_df.to_csv(output_path, index=False)
        return output_path


def compute_aet_score(
    model_accuracy: float,
    model_compute: float,
    teacher_accuracy: float,
    teacher_compute: float
) -> float:
    """
    Compute Accuracy-Efficiency Trade-off (AET) score.

    AET = (A_model / A_teacher) / (C_model / C_teacher)

    An AET > 1 indicates better accuracy-per-compute than the teacher.

    Args:
        model_accuracy: Model's accuracy (e.g., Top-1)
        model_compute: Model's composite compute (FLOPs × latency × memory)
        teacher_accuracy: Teacher's accuracy
        teacher_compute: Teacher's composite compute

    Returns:
        AET score
    """
    if teacher_accuracy <= 0 or teacher_compute <= 0:
        raise ValueError("Teacher metrics must be positive")

    normalized_accuracy = model_accuracy / teacher_accuracy
    normalized_compute = model_compute / teacher_compute

    if normalized_compute <= 0:
        raise ValueError("Normalized compute must be positive")

    return normalized_accuracy / normalized_compute


def compute_composite_efficiency(flops_giga: float,
                                 latency_ms: float,
                                 memory_mb: float) -> float:
    """
    Compute composite efficiency metric.

    C = FLOPs × latency × memory

    Args:
        flops_giga: Forward pass FLOPs in billions
        latency_ms: Inference latency in milliseconds
        memory_mb: Peak GPU memory in megabytes

    Returns:
        Composite efficiency score
    """
    return flops_giga * latency_ms * memory_mb
