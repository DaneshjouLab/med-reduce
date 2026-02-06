#!/usr/bin/env python3
"""
Aggregate results across multiple seeds and generate summary tables.

Generates tables with mean ± SD for:
- Accuracy metrics: Top-1 %, AUROC, Macro F1 %
- Efficiency metrics: GFLOPs, Peak GPU Memory (MB), Inference Latency (ms)

Usage:
    python -m src.evaluation.aggregate_results \
        --results-dir /path/to/runs/probe_two_stage \
        --seeds 42 123 456 \
        --resolutions 512 256 128 64 \
        --model dinov3 \
        --output results_table
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.utils.logging_core import get_logger

log = get_logger(__name__)


@dataclass
class ResolutionMetrics:
    """Metrics for a single resolution across all seeds."""
    resolution: int
    seeds: List[int] = field(default_factory=list)

    # Accuracy metrics (per seed)
    top1_acc: List[float] = field(default_factory=list)
    auroc: List[float] = field(default_factory=list)
    macro_f1: List[float] = field(default_factory=list)

    # Efficiency metrics (per seed)
    gflops: List[float] = field(default_factory=list)
    peak_gpu_memory_mb: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)

    # Hyperparameters (should be same across seeds)
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    batch_size: Optional[int] = None


def load_results_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load a single results JSON file."""
    if not filepath.exists():
        log.warning(f"Results file not found: {filepath}")
        return None

    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log.error(f"Error parsing JSON from {filepath}: {e}")
        return None


def collect_results(
    results_dir: str,
    seeds: List[int],
    resolutions: List[int],
    model_name: str = "dinov3",
) -> Dict[int, ResolutionMetrics]:
    """
    Collect results from all seed directories for each resolution.

    Expected directory structure:
        results_dir/
            seed_42/
                results_{model}_{resolution}px.json
            seed_123/
                ...

    Args:
        results_dir: Base directory containing seed subdirectories
        seeds: List of seed values to collect
        resolutions: List of resolutions to collect
        model_name: Model name used in results filename

    Returns:
        Dict mapping resolution -> ResolutionMetrics
    """
    results_dir = Path(results_dir)
    metrics_by_resolution: Dict[int, ResolutionMetrics] = {}

    for resolution in resolutions:
        metrics = ResolutionMetrics(resolution=resolution)

        for seed in seeds:
            seed_dir = results_dir / f"seed_{seed}"
            results_file = seed_dir / f"results_{model_name}_{resolution}px.json"

            data = load_results_file(results_file)
            if data is None:
                log.warning(f"Skipping seed {seed} for resolution {resolution}px (file not found)")
                continue

            metrics.seeds.append(seed)

            # Extract accuracy metrics
            acc_metrics = data.get("accuracy_metrics", {})
            if acc_metrics.get("final_val_acc") is not None:
                metrics.top1_acc.append(acc_metrics["final_val_acc"] * 100)  # Convert to %
            if acc_metrics.get("final_val_auroc") is not None:
                metrics.auroc.append(acc_metrics["final_val_auroc"] * 100)  # Convert to %
            # Note: F1 is not currently saved in results - would need to add to training
            if acc_metrics.get("final_val_f1") is not None:
                metrics.macro_f1.append(acc_metrics["final_val_f1"] * 100)

            # Extract efficiency metrics
            eff_metrics = data.get("efficiency_metrics", {})
            if eff_metrics.get("encoder_gflops") is not None:
                metrics.gflops.append(eff_metrics["encoder_gflops"])
            if eff_metrics.get("peak_gpu_memory_mb") is not None:
                metrics.peak_gpu_memory_mb.append(eff_metrics["peak_gpu_memory_mb"])
            if eff_metrics.get("encoder_latency_ms") is not None:
                metrics.latency_ms.append(eff_metrics["encoder_latency_ms"])

            # Extract hyperparameters (same for all seeds)
            hyperparams = data.get("hyperparameters", {})
            if metrics.lr is None:
                metrics.lr = hyperparams.get("lr")
                metrics.weight_decay = hyperparams.get("weight_decay")
                metrics.batch_size = hyperparams.get("batch_size")

        if metrics.seeds:
            metrics_by_resolution[resolution] = metrics
            log.info(f"Collected {len(metrics.seeds)} seeds for {resolution}px")
        else:
            log.warning(f"No results found for {resolution}px")

    return metrics_by_resolution


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return float('nan'), float('nan')
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))


def format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    """Format mean ± std as string."""
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def generate_table(
    metrics_by_resolution: Dict[int, ResolutionMetrics],
    resolutions: List[int],
) -> Dict[str, Any]:
    """
    Generate summary statistics table.

    Returns dict with:
        - 'data': List of row dicts
        - 'headers': List of column headers
    """
    headers = [
        "Resolution",
        "Top-1 Acc (%)",
        "AUROC (%)",
        "Macro F1 (%)",
        "GFLOPs",
        "Peak GPU (MB)",
        "Latency (ms)",
        "Seeds",
    ]

    rows = []
    for resolution in sorted(resolutions, reverse=True):
        metrics = metrics_by_resolution.get(resolution)
        if metrics is None:
            continue

        top1_mean, top1_std = compute_mean_std(metrics.top1_acc)
        auroc_mean, auroc_std = compute_mean_std(metrics.auroc)
        f1_mean, f1_std = compute_mean_std(metrics.macro_f1)
        gflops_mean, gflops_std = compute_mean_std(metrics.gflops)
        mem_mean, mem_std = compute_mean_std(metrics.peak_gpu_memory_mb)
        lat_mean, lat_std = compute_mean_std(metrics.latency_ms)

        row = {
            "resolution": resolution,
            "top1_acc": format_mean_std(top1_mean, top1_std),
            "auroc": format_mean_std(auroc_mean, auroc_std),
            "macro_f1": format_mean_std(f1_mean, f1_std),
            "gflops": format_mean_std(gflops_mean, gflops_std),
            "peak_gpu_mb": format_mean_std(mem_mean, mem_std, precision=0),
            "latency_ms": format_mean_std(lat_mean, lat_std),
            "n_seeds": len(metrics.seeds),
            # Raw values for programmatic access
            "_raw": {
                "top1_acc": (top1_mean, top1_std),
                "auroc": (auroc_mean, auroc_std),
                "macro_f1": (f1_mean, f1_std),
                "gflops": (gflops_mean, gflops_std),
                "peak_gpu_mb": (mem_mean, mem_std),
                "latency_ms": (lat_mean, lat_std),
            }
        }
        rows.append(row)

    return {"headers": headers, "data": rows}


def to_markdown(table: Dict[str, Any], title: str = "Results Summary") -> str:
    """Convert table to markdown format."""
    lines = [
        f"## {title}",
        "",
        "| Resolution | Top-1 Acc (%) | AUROC (%) | Macro F1 (%) | GFLOPs | Peak GPU (MB) | Latency (ms) | Seeds |",
        "|------------|---------------|-----------|--------------|--------|---------------|--------------|-------|",
    ]

    for row in table["data"]:
        lines.append(
            f"| {row['resolution']}px | {row['top1_acc']} | {row['auroc']} | {row['macro_f1']} | "
            f"{row['gflops']} | {row['peak_gpu_mb']} | {row['latency_ms']} | {row['n_seeds']} |"
        )

    return "\n".join(lines)


def to_latex(table: Dict[str, Any], title: str = "Results Summary") -> str:
    """Convert table to LaTeX format."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{title}}}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Resolution & Top-1 (\\%) & AUROC (\\%) & F1 (\\%) & GFLOPs & GPU (MB) & Latency (ms) & Seeds \\\\",
        "\\midrule",
    ]

    for row in table["data"]:
        lines.append(
            f"{row['resolution']}px & {row['top1_acc']} & {row['auroc']} & {row['macro_f1']} & "
            f"{row['gflops']} & {row['peak_gpu_mb']} & {row['latency_ms']} & {row['n_seeds']} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def to_csv(table: Dict[str, Any]) -> str:
    """Convert table to CSV format."""
    lines = ["resolution,top1_acc_mean,top1_acc_std,auroc_mean,auroc_std,f1_mean,f1_std,gflops_mean,gflops_std,peak_gpu_mb_mean,peak_gpu_mb_std,latency_ms_mean,latency_ms_std,n_seeds"]

    for row in table["data"]:
        raw = row["_raw"]
        lines.append(
            f"{row['resolution']},"
            f"{raw['top1_acc'][0]:.4f},{raw['top1_acc'][1]:.4f},"
            f"{raw['auroc'][0]:.4f},{raw['auroc'][1]:.4f},"
            f"{raw['macro_f1'][0]:.4f},{raw['macro_f1'][1]:.4f},"
            f"{raw['gflops'][0]:.4f},{raw['gflops'][1]:.4f},"
            f"{raw['peak_gpu_mb'][0]:.1f},{raw['peak_gpu_mb'][1]:.1f},"
            f"{raw['latency_ms'][0]:.4f},{raw['latency_ms'][1]:.4f},"
            f"{row['n_seeds']}"
        )

    return "\n".join(lines)


def save_tables(
    table: Dict[str, Any],
    output_path: str,
    title: str = "Results Summary",
) -> None:
    """Save table in multiple formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = output_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(to_markdown(table, title))
    log.info(f"Saved markdown table to {md_path}")

    # LaTeX
    tex_path = output_path.with_suffix(".tex")
    with open(tex_path, "w") as f:
        f.write(to_latex(table, title))
    log.info(f"Saved LaTeX table to {tex_path}")

    # CSV
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write(to_csv(table))
    log.info(f"Saved CSV table to {csv_path}")

    # JSON (full data)
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        # Remove _raw for cleaner JSON output
        clean_data = []
        for row in table["data"]:
            clean_row = {k: v for k, v in row.items() if k != "_raw"}
            clean_row["raw_values"] = row["_raw"]
            clean_data.append(clean_row)
        json.dump({"headers": table["headers"], "data": clean_data}, f, indent=2)
    log.info(f"Saved JSON data to {json_path}")


def print_table(table: Dict[str, Any]) -> None:
    """Print table to console."""
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY (mean ± SD across seeds)")
    print("=" * 100)
    print()
    print(f"{'Resolution':<12} {'Top-1 (%)':<16} {'AUROC (%)':<16} {'Macro F1 (%)':<16} "
          f"{'GFLOPs':<14} {'GPU (MB)':<14} {'Latency (ms)':<14} {'Seeds':<6}")
    print("-" * 100)

    for row in table["data"]:
        print(f"{row['resolution']}px{'':<8} {row['top1_acc']:<16} {row['auroc']:<16} {row['macro_f1']:<16} "
              f"{row['gflops']:<14} {row['peak_gpu_mb']:<14} {row['latency_ms']:<14} {row['n_seeds']:<6}")

    print("=" * 100)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results across seeds and generate summary tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Base directory containing seed subdirectories (e.g., runs/probe_two_stage)",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Seeds to aggregate (default: 42 123 456)",
    )

    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[512, 256, 128, 64],
        help="Resolutions to include (default: 512 256 128 64)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dinov3",
        help="Model name (default: dinov3)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (without extension). If not provided, only prints to console.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Multi-Resolution Linear Probing Results",
        help="Title for the table",
    )

    args = parser.parse_args()

    # Collect results
    log.info(f"Collecting results from: {args.results_dir}")
    log.info(f"Seeds: {args.seeds}")
    log.info(f"Resolutions: {args.resolutions}")

    metrics = collect_results(
        results_dir=args.results_dir,
        seeds=args.seeds,
        resolutions=args.resolutions,
        model_name=args.model,
    )

    if not metrics:
        log.error("No results found!")
        return

    # Generate table
    table = generate_table(metrics, args.resolutions)

    # Print to console
    print_table(table)

    # Save to files
    if args.output:
        save_tables(table, args.output, title=args.title)


if __name__ == "__main__":
    main()
