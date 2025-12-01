# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/evaluation/analyze_results.py
# -*- coding: utf-8 -*-
"""
Analysis and visualization tools for experimental results.
Generates plots for accuracy-efficiency trade-offs and Pareto frontiers.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics_collector import MetricsCollector


class ResultsAnalyzer:
    """Analyze and visualize experimental results for paper figures."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize analyzer with a metrics collector.

        Args:
            collector: MetricsCollector containing experimental results
        """
        self.collector = collector
        self.df = collector.to_dataframe()

        # Set plotting style
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10

    def plot_accuracy_vs_compute(
        self,
        output_path: Optional[str] = None,
        accuracy_metric: str = "top1_accuracy",
        compute_metric: str = "flops_giga",
        group_by: str = "model_family",
        show_pareto: bool = True,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """
        Create scatter plot of accuracy vs computational cost.

        Args:
            output_path: Where to save the figure
            accuracy_metric: Column name for accuracy metric
            compute_metric: Column name for compute metric (FLOPs, latency, etc.)
            group_by: Column to color points by (e.g., "model_family")
            show_pareto: Whether to highlight Pareto frontier
            figsize: Figure size in inches

        Returns:
            Path to saved figure
        """
        if output_path is None:
            output_path = os.path.join(
                self.collector.output_dir,
                f"accuracy_vs_{compute_metric}.png"
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Get data
        df = self.df.copy()

        # Remove rows with missing values
        df = df.dropna(subset=[accuracy_metric, compute_metric])

        if df.empty:
            raise ValueError("No valid data for plotting")

        # Color by group if specified
        if group_by in df.columns:
            groups = df[group_by].unique()
            palette = sns.color_palette("husl", len(groups))
            color_map = dict(zip(groups, palette))

            for group in groups:
                group_df = df[df[group_by] == group]
                ax.scatter(
                    group_df[compute_metric],
                    group_df[accuracy_metric],
                    label=group,
                    alpha=0.7,
                    s=100,
                    color=color_map[group]
                )
        else:
            ax.scatter(
                df[compute_metric],
                df[accuracy_metric],
                alpha=0.7,
                s=100
            )

        # Add Pareto frontier
        if show_pareto:
            pareto_df = self.collector.get_pareto_frontier(
                accuracy_key=accuracy_metric,
                efficiency_key=compute_metric,
                minimize_efficiency=True
            )

            if not pareto_df.empty:
                pareto_df = pareto_df.sort_values(compute_metric)
                ax.plot(
                    pareto_df[compute_metric],
                    pareto_df[accuracy_metric],
                    'r--',
                    linewidth=2,
                    label='Pareto Frontier',
                    alpha=0.8
                )

                # Annotate Pareto points
                for _, row in pareto_df.iterrows():
                    ax.annotate(
                        row['model_name'],
                        (row[compute_metric], row[accuracy_metric]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7
                    )

        ax.set_xlabel(self._format_label(compute_metric))
        ax.set_ylabel(self._format_label(accuracy_metric))
        ax.set_title('Accuracy vs. Computational Cost')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_aet_scores(
        self,
        output_path: Optional[str] = None,
        group_by: str = "model_family",
        figsize: Tuple[int, int] = (12, 6)
    ) -> str:
        """
        Create bar plot of AET scores across models.

        Args:
            output_path: Where to save the figure
            group_by: Column to group bars by
            figsize: Figure size in inches

        Returns:
            Path to saved figure
        """
        if output_path is None:
            output_path = os.path.join(
                self.collector.output_dir,
                "aet_scores.png"
            )

        df = self.df.dropna(subset=['aet_score']).copy()

        if df.empty:
            print("No AET scores available for plotting")
            return output_path

        # Sort by AET score
        df = df.sort_values('aet_score', ascending=False)

        fig, ax = plt.subplots(figsize=figsize)

        # Color by group
        if group_by in df.columns:
            groups = df[group_by].unique()
            palette = sns.color_palette("husl", len(groups))
            color_map = dict(zip(groups, palette))
            colors = [color_map[g] for g in df[group_by]]
        else:
            colors = 'steelblue'

        bars = ax.bar(
            range(len(df)),
            df['aet_score'],
            color=colors,
            alpha=0.7
        )

        # Add horizontal line at AET = 1 (break-even with teacher)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Teacher Baseline')

        ax.set_xlabel('Model')
        ax.set_ylabel('AET Score (Accuracy/Compute)')
        ax.set_title('Accuracy-Efficiency Trade-off (AET) Scores')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_multi_metric_comparison(
        self,
        output_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> str:
        """
        Create multi-panel figure comparing various metrics.

        Args:
            output_path: Where to save the figure
            metrics: List of metric pairs to plot (accuracy vs compute)
            figsize: Figure size in inches

        Returns:
            Path to saved figure
        """
        if output_path is None:
            output_path = os.path.join(
                self.collector.output_dir,
                "multi_metric_comparison.png"
            )

        if metrics is None:
            metrics = [
                ("top1_accuracy", "flops_giga"),
                ("top1_accuracy", "inference_latency_ms"),
                ("top1_accuracy", "peak_gpu_memory_mb"),
                ("top1_accuracy", "composite_compute")
            ]

        # Filter metrics to those available in data
        available_metrics = [
            (acc, comp) for acc, comp in metrics
            if acc in self.df.columns and comp in self.df.columns
        ]

        if not available_metrics:
            print("No valid metrics found for comparison")
            return output_path

        n_plots = len(available_metrics)
        n_rows = (n_plots + 1) // 2
        n_cols = 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]

        for idx, (acc_metric, comp_metric) in enumerate(available_metrics):
            ax = axes[idx]
            df = self.df.dropna(subset=[acc_metric, comp_metric])

            # Scatter plot
            for family in df['model_family'].unique():
                family_df = df[df['model_family'] == family]
                ax.scatter(
                    family_df[comp_metric],
                    family_df[acc_metric],
                    label=family,
                    alpha=0.7,
                    s=80
                )

            # Pareto frontier
            pareto_df = self.collector.get_pareto_frontier(
                accuracy_key=acc_metric,
                efficiency_key=comp_metric
            )

            if not pareto_df.empty:
                pareto_df = pareto_df.sort_values(comp_metric)
                ax.plot(
                    pareto_df[comp_metric],
                    pareto_df[acc_metric],
                    'r--',
                    linewidth=1.5,
                    alpha=0.6
                )

            ax.set_xlabel(self._format_label(comp_metric))
            ax.set_ylabel(self._format_label(acc_metric))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Remove extra subplots
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        return output_path

    def generate_summary_table(
        self,
        output_path: Optional[str] = None,
        group_by: Optional[str] = "model_family"
    ) -> str:
        """
        Generate LaTeX-formatted summary table.

        Args:
            output_path: Where to save the table
            group_by: Column to group results by

        Returns:
            Path to saved table
        """
        if output_path is None:
            output_path = os.path.join(
                self.collector.output_dir,
                "summary_table.tex"
            )

        df = self.df.copy()

        # Select key columns
        cols = [
            'model_name', 'image_resolution', 'top1_accuracy',
            'flops_giga', 'inference_latency_ms', 'peak_gpu_memory_mb', 'aet_score'
        ]

        # Filter to available columns
        cols = [c for c in cols if c in df.columns]
        summary_df = df[cols].copy()

        # Round numeric columns
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        summary_df[numeric_cols] = summary_df[numeric_cols].round(3)

        # Sort by AET score or accuracy
        if 'aet_score' in summary_df.columns:
            summary_df = summary_df.sort_values('aet_score', ascending=False)
        else:
            summary_df = summary_df.sort_values('top1_accuracy', ascending=False)

        # Convert to LaTeX
        latex_str = summary_df.to_latex(
            index=False,
            column_format='l' + 'r' * (len(cols) - 1),
            caption="Model performance comparison across resolution and efficiency metrics",
            label="tab:model_comparison",
            float_format="%.3f"
        )

        with open(output_path, 'w') as f:
            f.write(latex_str)

        return output_path

    def generate_all_visualizations(self, output_dir: Optional[str] = None):
        """
        Generate all standard visualizations for paper.

        Args:
            output_dir: Directory to save all figures
        """
        if output_dir is None:
            output_dir = os.path.join(self.collector.output_dir, "figures")

        os.makedirs(output_dir, exist_ok=True)

        print("Generating visualizations...")

        # 1. Accuracy vs FLOPs
        try:
            path = self.plot_accuracy_vs_compute(
                output_path=os.path.join(output_dir, "accuracy_vs_flops.png"),
                compute_metric="flops_giga"
            )
            print(f"✓ Saved accuracy vs FLOPs: {path}")
        except Exception as e:
            print(f"✗ Failed to generate accuracy vs FLOPs: {e}")

        # 2. Accuracy vs Latency
        try:
            path = self.plot_accuracy_vs_compute(
                output_path=os.path.join(output_dir, "accuracy_vs_latency.png"),
                compute_metric="inference_latency_ms"
            )
            print(f"✓ Saved accuracy vs latency: {path}")
        except Exception as e:
            print(f"✗ Failed to generate accuracy vs latency: {e}")

        # 3. AET scores
        try:
            path = self.plot_aet_scores(
                output_path=os.path.join(output_dir, "aet_scores.png")
            )
            print(f"✓ Saved AET scores: {path}")
        except Exception as e:
            print(f"✗ Failed to generate AET scores: {e}")

        # 4. Multi-metric comparison
        try:
            path = self.plot_multi_metric_comparison(
                output_path=os.path.join(output_dir, "multi_metric_comparison.png")
            )
            print(f"✓ Saved multi-metric comparison: {path}")
        except Exception as e:
            print(f"✗ Failed to generate multi-metric comparison: {e}")

        # 5. Summary table
        try:
            path = self.generate_summary_table(
                output_path=os.path.join(output_dir, "summary_table.tex")
            )
            print(f"✓ Saved LaTeX summary table: {path}")
        except Exception as e:
            print(f"✗ Failed to generate summary table: {e}")

        print(f"\nAll visualizations saved to: {output_dir}")

    @staticmethod
    def _format_label(metric_name: str) -> str:
        """Format metric name for axis labels."""
        label_map = {
            "top1_accuracy": "Top-1 Accuracy (%)",
            "top5_accuracy": "Top-5 Accuracy (%)",
            "flops_giga": "FLOPs (GFLOPs)",
            "inference_latency_ms": "Inference Latency (ms)",
            "peak_gpu_memory_mb": "Peak GPU Memory (MB)",
            "composite_compute": "Composite Compute (C)",
            "aet_score": "AET Score",
            "num_parameters_millions": "Parameters (M)"
        }
        return label_map.get(metric_name, metric_name.replace('_', ' ').title())


def load_and_analyze(
    metrics_dir: str,
    experiment_name: str = "experiment"
) -> ResultsAnalyzer:
    """
    Convenience function to load metrics and create analyzer.

    Args:
        metrics_dir: Directory containing metrics files
        experiment_name: Name of experiment

    Returns:
        ResultsAnalyzer instance
    """
    collector = MetricsCollector(metrics_dir, experiment_name)
    return ResultsAnalyzer(collector)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Directory containing metrics JSON/CSV files"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="Name of experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations"
    )

    args = parser.parse_args()

    # Load and analyze
    analyzer = load_and_analyze(args.metrics_dir, args.experiment_name)
    analyzer.generate_all_visualizations(args.output_dir)
