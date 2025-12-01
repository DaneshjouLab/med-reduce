#!/usr/bin/env python3
"""
Example script demonstrating comprehensive metrics analysis workflow.

This script shows how to:
1. Load experimental metrics
2. Compute AET scores
3. Find Pareto-optimal models
4. Generate publication-ready visualizations

Usage:
    python examples/analyze_experiment_results.py --metrics_dir ./runs/my_experiment
"""

import argparse
from pathlib import Path
import pandas as pd

from src.evaluation.metrics_collector import MetricsCollector
from src.evaluation.analyze_results import ResultsAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experimental results and generate visualizations"
    )
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
        help="Name of experiment (prefix for metrics files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: metrics_dir/figures)"
    )
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Print summary statistics"
    )
    parser.add_argument(
        "--export_paper_table",
        action="store_true",
        help="Export formatted table for paper"
    )

    args = parser.parse_args()

    # Load metrics
    print(f"\n{'='*60}")
    print(f"Loading metrics from: {args.metrics_dir}")
    print(f"{'='*60}\n")

    collector = MetricsCollector(args.metrics_dir, args.experiment_name)

    if not collector.metrics:
        print("❌ No metrics found. Have you run any experiments yet?")
        print(f"   Expected file: {collector.json_path}")
        return

    print(f"✓ Loaded {len(collector.metrics)} experiment records\n")

    # Convert to DataFrame for analysis
    df = collector.to_dataframe()

    # Show summary statistics
    if args.show_stats or True:  # Always show basic stats
        print(f"{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}\n")

        print(f"Total experiments: {len(df)}")
        print(f"Unique models: {df['model_name'].nunique()}")
        print(f"Model families: {', '.join(df['model_family'].unique())}")
        print(f"Datasets: {', '.join(df['dataset'].unique())}")
        print(f"Phases: {', '.join(df['phase'].unique())}")

        print(f"\n{'='*60}")
        print("ACCURACY METRICS")
        print(f"{'='*60}\n")

        print(f"Top-1 Accuracy:")
        print(f"  Range:  {df['top1_accuracy'].min():.2f}% - {df['top1_accuracy'].max():.2f}%")
        print(f"  Mean:   {df['top1_accuracy'].mean():.2f}%")
        print(f"  Median: {df['top1_accuracy'].median():.2f}%")
        print(f"  Std:    {df['top1_accuracy'].std():.2f}%")

        print(f"\n{'='*60}")
        print("EFFICIENCY METRICS")
        print(f"{'='*60}\n")

        print(f"FLOPs (GFLOPs):")
        print(f"  Range: {df['flops_giga'].min():.2f} - {df['flops_giga'].max():.2f}")
        print(f"  Mean:  {df['flops_giga'].mean():.2f}")

        print(f"\nInference Latency (ms):")
        print(f"  Range: {df['inference_latency_ms'].min():.2f} - {df['inference_latency_ms'].max():.2f}")
        print(f"  Mean:  {df['inference_latency_ms'].mean():.2f}")

        print(f"\nPeak GPU Memory (MB):")
        print(f"  Range: {df['peak_gpu_memory_mb'].min():.0f} - {df['peak_gpu_memory_mb'].max():.0f}")
        print(f"  Mean:  {df['peak_gpu_memory_mb'].mean():.0f}")

        # AET scores if available
        if 'aet_score' in df.columns and df['aet_score'].notna().any():
            aet_df = df.dropna(subset=['aet_score'])
            print(f"\n{'='*60}")
            print("AET SCORES (Accuracy-Efficiency Trade-off)")
            print(f"{'='*60}\n")

            print(f"Range: {aet_df['aet_score'].min():.3f} - {aet_df['aet_score'].max():.3f}")
            print(f"Mean:  {aet_df['aet_score'].mean():.3f}")

            best_aet_idx = aet_df['aet_score'].idxmax()
            best_model = aet_df.loc[best_aet_idx]
            print(f"\n🏆 Best AET Score:")
            print(f"   Model:    {best_model['model_name']}")
            print(f"   Score:    {best_model['aet_score']:.3f}")
            print(f"   Accuracy: {best_model['top1_accuracy']:.2f}%")
            print(f"   FLOPs:    {best_model['flops_giga']:.2f} G")
        else:
            print(f"\n⚠️  No AET scores available (teacher metrics missing)")

    # Find Pareto-optimal models
    print(f"\n{'='*60}")
    print("PARETO-OPTIMAL MODELS (Accuracy vs. FLOPs)")
    print(f"{'='*60}\n")

    pareto_df = collector.get_pareto_frontier(
        accuracy_key="top1_accuracy",
        efficiency_key="flops_giga",
        minimize_efficiency=True
    )

    if not pareto_df.empty:
        print(f"Found {len(pareto_df)} Pareto-optimal models:\n")
        for idx, row in pareto_df.iterrows():
            print(f"  • {row['model_name']:<30} "
                  f"Acc: {row['top1_accuracy']:>6.2f}%  "
                  f"FLOPs: {row['flops_giga']:>6.2f}G  "
                  f"Latency: {row['inference_latency_ms']:>6.2f}ms")
    else:
        print("No Pareto-optimal models found (need at least 2 models)")

    # Best models by different criteria
    print(f"\n{'='*60}")
    print("BEST MODELS BY CRITERIA")
    print(f"{'='*60}\n")

    best_acc_idx = df['top1_accuracy'].idxmax()
    best_acc = df.loc[best_acc_idx]
    print(f"🎯 Highest Accuracy:")
    print(f"   {best_acc['model_name']}: {best_acc['top1_accuracy']:.2f}%")

    best_flops_idx = df['flops_giga'].idxmin()
    best_flops = df.loc[best_flops_idx]
    print(f"\n⚡ Lowest FLOPs:")
    print(f"   {best_flops['model_name']}: {best_flops['flops_giga']:.2f}G")

    best_latency_idx = df['inference_latency_ms'].idxmin()
    best_latency = df.loc[best_latency_idx]
    print(f"\n🚀 Fastest Inference:")
    print(f"   {best_latency['model_name']}: {best_latency['inference_latency_ms']:.2f}ms")

    # Generate visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")

    analyzer = ResultsAnalyzer(collector)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(args.metrics_dir) / "figures"

    try:
        analyzer.generate_all_visualizations(str(output_dir))
        print(f"\n✓ All visualizations saved to: {output_dir}")
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")

    # Export paper table
    if args.export_paper_table:
        print(f"\n{'='*60}")
        print("EXPORTING PAPER TABLE")
        print(f"{'='*60}\n")

        try:
            table_path = collector.export_for_paper()
            print(f"✓ Paper table saved to: {table_path}")

            # Also generate LaTeX version
            latex_path = analyzer.generate_summary_table(
                output_path=str(Path(output_dir) / "summary_table.tex")
            )
            print(f"✓ LaTeX table saved to: {latex_path}")
        except Exception as e:
            print(f"❌ Error exporting table: {e}")

    # Print file locations
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print(f"{'='*60}\n")

    print("Metrics data:")
    print(f"  • JSON: {collector.json_path}")
    print(f"  • CSV:  {collector.csv_path}")
    print(f"  • Summary: {collector.summary_path}")

    print(f"\nVisualizations:")
    print(f"  • Directory: {output_dir}")
    print(f"  • accuracy_vs_flops.png")
    print(f"  • accuracy_vs_latency.png")
    print(f"  • aet_scores.png")
    print(f"  • multi_metric_comparison.png")
    print(f"  • summary_table.tex")

    print(f"\n{'='*60}")
    print("✓ Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
