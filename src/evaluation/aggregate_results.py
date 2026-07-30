#!/usr/bin/env python3
"""
Aggregate results across domains, models, seeds, and resolutions.

Scans the results/ directory tree to auto-discover all experiment outputs,
computes mean +/- SD across seeds, and generates summary tables (Markdown,
LaTeX, CSV, JSON) and a consolidated DataFrame for downstream visualization.

Directory layout assumed
------------------------
results/
  med-reduce-{derm,path,rad}-results/
    runs/probe_two_stage/
      seed_{42,123,456}/
        results_{dataset}_{model}_{resolution}px.json

Usage:
    # Auto-discover everything under results/
    python -m src.evaluation.aggregate_results --results-root results/

    # Restrict to specific domains / models
    python -m src.evaluation.aggregate_results \
        --results-root results/ \
        --domains dermatology radiology \
        --models dinov3 resnet50_distilled

    # Save tables
    python -m src.evaluation.aggregate_results \
        --results-root results/ \
        --output results/summary
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_core import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain registry: maps directory-name fragment -> (domain_label, dataset_prefix)
# ---------------------------------------------------------------------------
DOMAIN_REGISTRY = {
    "derm": ("dermatology", "images"),
    "path": ("pathology", "tcga"),
    "rad": ("radiology", "combined_train_valid_chexpert_v1.0"),
}

# Pathology tasks (order matters for display)
PATHOLOGY_TASKS = [
    "luad_vs_lusc",
    "lgg_vs_gbm",
    "kras",
    "tp53",
    "egfr",
]

PATHOLOGY_TASK_LABELS = {
    "luad_vs_lusc": "LUAD vs LUSC",
    "lgg_vs_gbm": "LGG vs GBM",
    "kras": "KRAS",
    "tp53": "TP53",
    "egfr": "EGFR",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ResolutionMetrics:
    """Metrics for a single (domain, task, model, resolution) across seeds."""
    domain: str
    task: str
    model: str
    resolution: int
    seeds: List[int] = field(default_factory=list)

    # Per-seed accuracy
    auroc: List[float] = field(default_factory=list)
    top1_acc: List[float] = field(default_factory=list)
    macro_f1: List[float] = field(default_factory=list)

    # Per-seed efficiency
    gflops: List[float] = field(default_factory=list)
    peak_gpu_memory_mb: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)

    # Hyperparameters (shared across seeds)
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    batch_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return None


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    a = np.array(values)
    return float(np.mean(a)), float(np.std(a))


def _fmt(mean: float, std: float, precision: int = 2) -> str:
    if np.isnan(mean):
        return "—"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------
def _detect_domain(dir_name: str) -> Optional[Tuple[str, str]]:
    """Return (domain_label, dataset_prefix) from a results directory name."""
    for key, val in DOMAIN_REGISTRY.items():
        if key in dir_name:
            return val
    return None


def _parse_result_filename(
    filename: str, dataset_prefix: str
) -> Optional[Dict[str, str]]:
    """
    Parse a results filename into its components.

    Examples
    --------
    results_images_dinov3_512px.json
    results_tcga_kras_resnet50_distilled_128px.json
    results_combined_train_valid_chexpert_v1.0_dinov3_64px.json
    """
    stem = filename.replace("results_", "").replace(".json", "")

    # Strip dataset prefix
    if stem.startswith(dataset_prefix + "_"):
        stem = stem[len(dataset_prefix) + 1:]
    elif stem.startswith(dataset_prefix):
        stem = stem[len(dataset_prefix):]

    # Extract resolution from the end: _<N>px
    m = re.search(r"_(\d+)px$", stem)
    if not m:
        return None
    resolution = m.group(1)
    stem = stem[: m.start()]

    # For pathology: next token is the task name
    task = "default"
    if dataset_prefix == "tcga":
        # Try each known task name (longest first to handle lgg_vs_gbm before kras)
        for t in sorted(PATHOLOGY_TASKS, key=len, reverse=True):
            if stem.startswith(t + "_"):
                task = t
                stem = stem[len(t) + 1:]
                break
            elif stem == t:
                # Edge case: task is the only remaining token
                task = t
                stem = ""
                break

    model = stem if stem else "unknown"
    return {"task": task, "model": model, "resolution": resolution}


def discover_results(
    results_root: Path,
    domains: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    resolutions: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Walk *results_root* and build a DataFrame of all experiments.

    Each row contains the raw metrics from one (domain, task, model,
    resolution, seed) results JSON.
    """
    records: List[Dict[str, Any]] = []

    for domain_dir in sorted(results_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        detected = _detect_domain(domain_dir.name)
        if detected is None:
            continue
        domain_label, dataset_prefix = detected

        if domains and domain_label not in domains:
            continue

        probe_dir = domain_dir / "runs" / "probe_two_stage"
        if not probe_dir.is_dir():
            log.warning("No probe_two_stage dir in %s", domain_dir)
            continue

        for seed_dir in sorted(probe_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            m = re.match(r"seed_(\d+)", seed_dir.name)
            if not m:
                continue
            seed = int(m.group(1))
            if seeds and seed not in seeds:
                continue

            for json_file in sorted(seed_dir.glob("results_*.json")):
                # Skip backup files
                if "_backup_" in json_file.name:
                    continue

                parsed = _parse_result_filename(json_file.name, dataset_prefix)
                if parsed is None:
                    continue

                if models and parsed["model"] not in models:
                    continue
                res = int(parsed["resolution"])
                if resolutions and res not in resolutions:
                    continue

                data = load_json(json_file)
                if data is None:
                    continue

                acc = data.get("accuracy_metrics", {})
                eff = data.get("efficiency_metrics", {})
                hp = data.get("hyperparameters", {})

                records.append(
                    {
                        "domain": domain_label,
                        "task": parsed["task"],
                        "model": parsed["model"],
                        "resolution": res,
                        "seed": seed,
                        "auroc": acc.get("best_metric")
                        or acc.get("final_val_auroc"),
                        "top1_acc": acc.get("final_val_acc"),
                        "macro_f1": acc.get("final_val_f1"),
                        "per_class_auroc": acc.get("per_class_auroc"),
                        "gflops": eff.get("encoder_gflops"),
                        "peak_gpu_memory_mb": eff.get("peak_gpu_memory_mb"),
                        "latency_ms": eff.get("encoder_latency_ms"),
                        "lr": hp.get("lr"),
                        "weight_decay": hp.get("weight_decay"),
                        "batch_size": hp.get("batch_size"),
                        "source_file": str(json_file),
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        log.warning("No results discovered under %s", results_root)
    else:
        log.info(
            "Discovered %d result files across %d domain(s)",
            len(df),
            df["domain"].nunique(),
        )
    return df


# ---------------------------------------------------------------------------
# Aggregation (mean ± SD across seeds)
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (domain, task, model, resolution) and compute mean ± SD.

    Returns a DataFrame with one row per group and columns like
    auroc_mean, auroc_std, top1_acc_mean, etc.
    """
    if df.empty:
        return df

    group_cols = ["domain", "task", "model", "resolution"]
    metric_cols = [
        "auroc",
        "top1_acc",
        "macro_f1",
        "gflops",
        "peak_gpu_memory_mb",
        "latency_ms",
    ]

    agg_dict = {col: ["mean", "std", "count"] for col in metric_cols if col in df.columns}
    agg_dict["seed"] = "count"

    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict)
    grouped.columns = [
        f"{col}_{stat}" if col != "seed" else "n_seeds"
        for col, stat in grouped.columns
    ]

    # Drop duplicate count columns — keep only n_seeds
    count_cols = [c for c in grouped.columns if c.endswith("_count") and c != "n_seeds"]
    grouped.drop(columns=count_cols, inplace=True, errors="ignore")

    grouped.reset_index(inplace=True)

    # Sort: domain, task, model, resolution desc
    grouped.sort_values(
        ["domain", "task", "model", "resolution"],
        ascending=[True, True, True, False],
        inplace=True,
    )

    return grouped


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------
def _section_title(domain: str, task: str) -> str:
    if task == "default":
        return domain.capitalize()
    label = PATHOLOGY_TASK_LABELS.get(task, task)
    return f"{domain.capitalize()} — {label}"


def to_markdown(agg: pd.DataFrame, title: str = "Cross-Domain Results Summary") -> str:
    """Render aggregated results as a Markdown table."""
    lines = [f"## {title}", ""]
    header = (
        "| Domain | Task | Model | Res | AUROC (%) | Top-1 (%) | "
        "F1 (%) | GFLOPs | GPU (MB) | Latency (ms) | Seeds |"
    )
    sep = "|".join([""] + ["-" * 10] * 11 + [""])
    lines += [header, sep]

    for _, r in agg.iterrows():
        task_label = PATHOLOGY_TASK_LABELS.get(r["task"], r["task"])
        auroc = _fmt(r.get("auroc_mean", float("nan")),
                     r.get("auroc_std", float("nan")))
        acc = _fmt(r.get("top1_acc_mean", float("nan")),
                   r.get("top1_acc_std", float("nan")))
        f1 = _fmt(r.get("macro_f1_mean", float("nan")),
                  r.get("macro_f1_std", float("nan")))
        gf = _fmt(r.get("gflops_mean", float("nan")),
                  r.get("gflops_std", float("nan")))
        mem = _fmt(r.get("peak_gpu_memory_mb_mean", float("nan")),
                   r.get("peak_gpu_memory_mb_std", float("nan")), precision=0)
        lat = _fmt(r.get("latency_ms_mean", float("nan")),
                   r.get("latency_ms_std", float("nan")))
        n = int(r.get("n_seeds", 0))
        lines.append(
            f"| {r['domain']} | {task_label} | {r['model']} | "
            f"{r['resolution']}px | {auroc} | {acc} | {f1} | "
            f"{gf} | {mem} | {lat} | {n} |"
        )

    return "\n".join(lines)


def to_latex(agg: pd.DataFrame, title: str = "Cross-Domain Results") -> str:
    """Render aggregated results as a LaTeX longtable."""
    lines = [
        "\\begin{longtable}{llllccccccc}",
        f"\\caption{{{title}}} \\\\",
        "\\toprule",
        "Domain & Task & Model & Res & AUROC (\\%) & Top-1 (\\%) & "
        "F1 (\\%) & GFLOPs & GPU (MB) & Latency (ms) & Seeds \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Domain & Task & Model & Res & AUROC (\\%) & Top-1 (\\%) & "
        "F1 (\\%) & GFLOPs & GPU (MB) & Latency (ms) & Seeds \\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]

    prev_domain = None
    for _, r in agg.iterrows():
        if r["domain"] != prev_domain and prev_domain is not None:
            lines.append("\\midrule")
        prev_domain = r["domain"]

        task_label = PATHOLOGY_TASK_LABELS.get(r["task"], r["task"])
        auroc = _fmt(r.get("auroc_mean", float("nan")),
                     r.get("auroc_std", float("nan")))
        acc = _fmt(r.get("top1_acc_mean", float("nan")),
                   r.get("top1_acc_std", float("nan")))
        f1 = _fmt(r.get("macro_f1_mean", float("nan")),
                  r.get("macro_f1_std", float("nan")))
        gf = _fmt(r.get("gflops_mean", float("nan")),
                  r.get("gflops_std", float("nan")))
        mem = _fmt(r.get("peak_gpu_memory_mb_mean", float("nan")),
                   r.get("peak_gpu_memory_mb_std", float("nan")), precision=0)
        lat = _fmt(r.get("latency_ms_mean", float("nan")),
                   r.get("latency_ms_std", float("nan")))
        n = int(r.get("n_seeds", 0))
        lines.append(
            f"{r['domain']} & {task_label} & {r['model']} & "
            f"{r['resolution']} & {auroc} & {acc} & {f1} & "
            f"{gf} & {mem} & {lat} & {n} \\\\"
        )

    lines.append("\\end{longtable}")
    return "\n".join(lines)


def to_csv(agg: pd.DataFrame) -> str:
    """Render aggregated results as CSV."""
    return agg.to_csv(index=False)


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------
def print_table(agg: pd.DataFrame) -> None:
    """Pretty-print the aggregated table to stdout."""
    print("\n" + "=" * 120)
    print("CROSS-DOMAIN RESULTS SUMMARY  (mean ± SD across seeds)")
    print("=" * 120)

    prev_domain = None
    for _, r in agg.iterrows():
        if r["domain"] != prev_domain:
            if prev_domain is not None:
                print("-" * 120)
            prev_domain = r["domain"]
            print(f"\n  {r['domain'].upper()}")
            print(
                f"  {'Task':<18} {'Model':<28} {'Res':<6} "
                f"{'AUROC (%)':<16} {'Top-1 (%)':<16} {'F1 (%)':<16} "
                f"{'GFLOPs':<14} {'GPU (MB)':<12} {'Lat (ms)':<14} {'#':<4}"
            )
            print("  " + "-" * 116)

        task_label = PATHOLOGY_TASK_LABELS.get(r["task"], r["task"])
        auroc = _fmt(r.get("auroc_mean", float("nan")),
                     r.get("auroc_std", float("nan")))
        acc = _fmt(r.get("top1_acc_mean", float("nan")),
                   r.get("top1_acc_std", float("nan")))
        f1 = _fmt(r.get("macro_f1_mean", float("nan")),
                  r.get("macro_f1_std", float("nan")))
        gf = _fmt(r.get("gflops_mean", float("nan")),
                  r.get("gflops_std", float("nan")))
        mem = _fmt(r.get("peak_gpu_memory_mb_mean", float("nan")),
                   r.get("peak_gpu_memory_mb_std", float("nan")), precision=0)
        lat = _fmt(r.get("latency_ms_mean", float("nan")),
                   r.get("latency_ms_std", float("nan")))
        n = int(r.get("n_seeds", 0))
        print(
            f"  {task_label:<18} {r['model']:<28} {r['resolution']:<6} "
            f"{auroc:<16} {acc:<16} {f1:<16} "
            f"{gf:<14} {mem:<12} {lat:<14} {n:<4}"
        )

    print("=" * 120 + "\n")


# ---------------------------------------------------------------------------
# File savers
# ---------------------------------------------------------------------------
def save_tables(
    agg: pd.DataFrame,
    output_stem: str,
    title: str = "Cross-Domain Results",
) -> None:
    """Write Markdown, LaTeX, CSV, and JSON to *output_stem*.{ext}."""
    out = Path(output_stem)
    out.parent.mkdir(parents=True, exist_ok=True)

    md_path = out.with_suffix(".md")
    md_path.write_text(to_markdown(agg, title))
    log.info("Saved Markdown → %s", md_path)

    tex_path = out.with_suffix(".tex")
    tex_path.write_text(to_latex(agg, title))
    log.info("Saved LaTeX   → %s", tex_path)

    csv_path = out.with_suffix(".csv")
    csv_path.write_text(to_csv(agg))
    log.info("Saved CSV     → %s", csv_path)

    json_path = out.with_suffix(".json")
    # Convert to JSON-safe dict (handle NaN)
    json_records = json.loads(agg.to_json(orient="records"))
    json_path.write_text(json.dumps(json_records, indent=2))
    log.info("Saved JSON    → %s", json_path)


# ---------------------------------------------------------------------------
# Legacy single-domain helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------
def collect_results(
    results_dir: str,
    seeds: List[int],
    resolutions: List[int],
    model_name: str = "dinov3",
) -> Dict[int, ResolutionMetrics]:
    """Collect results for a single domain directory (legacy API)."""
    results_dir = Path(results_dir)
    metrics_by_resolution: Dict[int, ResolutionMetrics] = {}

    for resolution in resolutions:
        metrics = ResolutionMetrics(
            domain="unknown", task="default", model=model_name, resolution=resolution,
        )

        for seed in seeds:
            seed_dir = results_dir / f"seed_{seed}"
            # Try common filename patterns
            for pattern in [
                f"results_{model_name}_{resolution}px.json",
                f"results_*_{model_name}_{resolution}px.json",
            ]:
                matches = list(seed_dir.glob(pattern))
                if matches:
                    break

            if not matches:
                log.warning("No results for seed %d at %dpx", seed, resolution)
                continue

            data = load_json(matches[0])
            if data is None:
                continue

            metrics.seeds.append(seed)
            acc = data.get("accuracy_metrics", {})
            eff = data.get("efficiency_metrics", {})
            hp = data.get("hyperparameters", {})

            if acc.get("final_val_acc") is not None:
                metrics.top1_acc.append(acc["final_val_acc"] * 100)
            if acc.get("final_val_auroc") is not None:
                metrics.auroc.append(acc["final_val_auroc"] * 100)
            if acc.get("final_val_f1") is not None:
                metrics.macro_f1.append(acc["final_val_f1"] * 100)
            if eff.get("encoder_gflops") is not None:
                metrics.gflops.append(eff["encoder_gflops"])
            if eff.get("peak_gpu_memory_mb") is not None:
                metrics.peak_gpu_memory_mb.append(eff["peak_gpu_memory_mb"])
            if eff.get("encoder_latency_ms") is not None:
                metrics.latency_ms.append(eff["encoder_latency_ms"])
            if metrics.lr is None:
                metrics.lr = hp.get("lr")
                metrics.weight_decay = hp.get("weight_decay")
                metrics.batch_size = hp.get("batch_size")

        if metrics.seeds:
            metrics_by_resolution[resolution] = metrics

    return metrics_by_resolution


def generate_table(
    metrics_by_resolution: Dict[int, ResolutionMetrics],
    resolutions: List[int],
) -> Dict[str, Any]:
    """Generate legacy-format table dict from ResolutionMetrics."""
    headers = [
        "Resolution", "Top-1 Acc (%)", "AUROC (%)", "Macro F1 (%)",
        "GFLOPs", "Peak GPU (MB)", "Latency (ms)", "Seeds",
    ]
    rows = []
    for res in sorted(resolutions, reverse=True):
        m = metrics_by_resolution.get(res)
        if m is None:
            continue
        t_m, t_s = _mean_std(m.top1_acc)
        a_m, a_s = _mean_std(m.auroc)
        f_m, f_s = _mean_std(m.macro_f1)
        g_m, g_s = _mean_std(m.gflops)
        mem_m, mem_s = _mean_std(m.peak_gpu_memory_mb)
        l_m, l_s = _mean_std(m.latency_ms)
        rows.append({
            "resolution": res,
            "top1_acc": _fmt(t_m, t_s), "auroc": _fmt(a_m, a_s),
            "macro_f1": _fmt(f_m, f_s), "gflops": _fmt(g_m, g_s),
            "peak_gpu_mb": _fmt(mem_m, mem_s, 0), "latency_ms": _fmt(l_m, l_s),
            "n_seeds": len(m.seeds),
            "_raw": {
                "top1_acc": (t_m, t_s), "auroc": (a_m, a_s),
                "macro_f1": (f_m, f_s), "gflops": (g_m, g_s),
                "peak_gpu_mb": (mem_m, mem_s), "latency_ms": (l_m, l_s),
            },
        })
    return {"headers": headers, "data": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results across domains, models, seeds, and resolutions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-root", type=str, default="results/",
        help="Root directory containing per-domain result folders (default: results/)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="(Legacy) Single domain results dir (e.g. runs/probe_two_stage).",
    )
    parser.add_argument(
        "--domains", type=str, nargs="*", default=None,
        help="Restrict to these domains (e.g. dermatology radiology).",
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None,
        help="Restrict to these model names (e.g. dinov3 resnet50_distilled).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seeds to aggregate (default: all discovered).",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=None,
        help="Resolutions to include (default: all discovered).",
    )
    parser.add_argument(
        "--model", type=str, default="dinov3",
        help="(Legacy) Model name for single-domain mode.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path stem (without extension). Tables saved as .md/.tex/.csv/.json.",
    )
    parser.add_argument(
        "--title", type=str, default="Cross-Domain Results",
        help="Title for tables.",
    )

    args = parser.parse_args()

    # Legacy single-domain mode
    if args.results_dir:
        seeds = args.seeds or [42, 123, 456]
        resolutions = args.resolutions or [512, 256, 128, 64]
        metrics = collect_results(
            args.results_dir, seeds, resolutions, model_name=args.model,
        )
        if not metrics:
            log.error("No results found!")
            return
        table = generate_table(metrics, resolutions)
        # Print to console
        print("\n" + "=" * 100)
        print("RESULTS SUMMARY (mean ± SD across seeds)")
        print("=" * 100)
        for row in table["data"]:
            print(
                f"  {row['resolution']}px  AUROC={row['auroc']}  "
                f"Acc={row['top1_acc']}  F1={row['macro_f1']}"
            )
        print("=" * 100 + "\n")
        if args.output:
            save_tables(
                aggregate(discover_results(Path(args.results_dir).parent.parent.parent)),
                args.output, args.title,
            )
        return

    # Cross-domain auto-discovery mode
    root = Path(args.results_root)
    if not root.is_dir():
        log.error("Results root not found: %s", root)
        return

    raw_df = discover_results(
        root,
        domains=args.domains,
        models=args.models,
        seeds=args.seeds,
        resolutions=args.resolutions,
    )
    if raw_df.empty:
        return

    agg_df = aggregate(raw_df)
    print_table(agg_df)

    if args.output:
        save_tables(agg_df, args.output, args.title)
        # Also save the raw per-seed dataframe for downstream analysis
        raw_csv = Path(args.output).with_name(
            Path(args.output).stem + "_raw"
        ).with_suffix(".csv")
        raw_df.drop(columns=["source_file", "per_class_auroc"], errors="ignore").to_csv(
            raw_csv, index=False,
        )
        log.info("Saved raw per-seed data → %s", raw_csv)


if __name__ == "__main__":
    main()
