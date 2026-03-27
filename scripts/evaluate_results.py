#!/usr/bin/env python3
"""
Evaluate and visualize linear probing results across domains, models, and resolutions.

Auto-discovers result JSON files, aggregates across seeds (mean ± SD),
and generates publication-ready plots and tables.

Usage:
    python scripts/evaluate_results.py [--results-dir results/] [--output-dir results/]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESOLUTIONS = [512, 256, 128, 64]
SEEDS = ["seed_42", "seed_123", "seed_456"]

MODEL_DISPLAY_NAMES = {
    "dinov3": "DINOv3 (Teacher)",
    "resnet18_distilled": "ResNet18-D",
    "resnet50_distilled": "ResNet50-D",
    "tiny_vit_21m_224_distilled": "TinyViT-21M-D",
}

MODEL_ORDER = ["dinov3", "resnet50_distilled", "resnet18_distilled", "tiny_vit_21m_224_distilled"]

DOMAIN_DISPLAY = {
    "dermatology": "Dermatology",
    "pathology": "Pathology",
    "radiology": "Radiology",
}

TASK_DISPLAY = {
    "luad_vs_lusc": "LUAD vs LUSC",
    "lgg_vs_gbm": "LGG vs GBM",
    "kras": "KRAS",
    "tp53": "TP53",
    "egfr": "EGFR",
}

# Color palette for models
MODEL_COLORS = {
    "dinov3": "#2c7bb6",
    "resnet50_distilled": "#d7191c",
    "resnet18_distilled": "#fdae61",
    "tiny_vit_21m_224_distilled": "#018571",
}

MODEL_MARKERS = {
    "dinov3": "o",
    "resnet50_distilled": "s",
    "resnet18_distilled": "D",
    "tiny_vit_21m_224_distilled": "^",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def discover_result_files(results_dir: Path) -> list[Path]:
    """Recursively find all result JSON files, excluding backups."""
    files = sorted(results_dir.rglob("results_*.json"))
    return [f for f in files if "backup" not in f.name]


def parse_result_file(path: Path) -> dict | None:
    """Load a result JSON and return a flat record, or None on failure."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    info = data.get("experiment_info", {})
    acc = data.get("accuracy_metrics", {})
    eff = data.get("efficiency_metrics", {})

    # Extract seed from path (e.g., .../seed_42/...)
    seed = None
    for part in path.parts:
        if part.startswith("seed_"):
            seed = int(part.split("_")[1])
            break

    if seed is None:
        return None

    # Determine task for pathology
    dataset = info.get("dataset", "")
    task = None
    if dataset.startswith("tcga_"):
        task = dataset.replace("tcga_", "")

    record = {
        "domain": info.get("domain", "unknown"),
        "model": info.get("model_name", "unknown"),
        "resolution": info.get("resolution", 0),
        "seed": seed,
        "dataset": dataset,
        "task": task,
        "auroc": acc.get("final_val_auroc"),
        "accuracy": acc.get("final_val_acc"),
        "f1": acc.get("final_val_f1"),
        "loss": acc.get("final_val_loss"),
        "gflops": eff.get("encoder_gflops"),
        "latency_ms": eff.get("encoder_latency_ms"),
        "embed_time_s": eff.get("embedding_extraction_time_s"),
        "peak_gpu_mb": eff.get("peak_gpu_memory_mb"),
        "per_class_auroc": acc.get("per_class_auroc"),
        "file": str(path),
    }
    return record


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Discover and load all results into a DataFrame."""
    files = discover_result_files(results_dir)
    records = []
    for f in files:
        rec = parse_result_file(f)
        if rec is not None:
            records.append(rec)

    if not records:
        print(f"No result files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    # Store per_class_auroc separately (not great in DataFrame)
    print(f"Loaded {len(df)} result records from {len(files)} files")
    print(f"  Domains: {sorted(df['domain'].unique())}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Resolutions: {sorted(df['resolution'].unique(), reverse=True)}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt(values, precision=3) -> str:
    """Format as mean ± SD."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return "—"
    mean = arr.mean()
    if len(arr) == 1:
        return f"{mean:.{precision}f}"
    sd = arr.std(ddof=1)
    return f"{mean:.{precision}f} ± {sd:.{precision}f}"


def fmt_latex(values, precision=3) -> str:
    """Format as mean ± SD for LaTeX."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return "—"
    mean = arr.mean()
    if len(arr) == 1:
        return f"{mean:.{precision}f}"
    sd = arr.std(ddof=1)
    return f"${mean:.{precision}f} \\pm {sd:.{precision}f}$"


def model_display(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    """Aggregate a metric across seeds, returning mean and SD."""
    grouped = df.groupby(group_cols)[metric].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = group_cols + [f"{metric}_mean", f"{metric}_sd", f"{metric}_n"]
    grouped[f"{metric}_sd"] = grouped[f"{metric}_sd"].fillna(0)
    return grouped


# ---------------------------------------------------------------------------
# Plot setup
# ---------------------------------------------------------------------------
def setup_plot_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def save_fig(fig, output_dir: Path, name: str):
    """Save figure as both PNG and PDF."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png")
    fig.savefig(fig_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved: figures/{name}.png, figures/{name}.pdf")


# ---------------------------------------------------------------------------
# Plot 1: AUROC vs Resolution (per domain)
# ---------------------------------------------------------------------------
def plot_auroc_vs_resolution(df: pd.DataFrame, output_dir: Path):
    """Line plot of AUROC vs resolution for each domain, one line per model."""
    domains = sorted(df["domain"].unique())

    for domain in domains:
        ddf = df[df["domain"] == domain]

        # For pathology, average across tasks first
        if domain == "pathology":
            group_cols = ["model", "resolution", "seed"]
            ddf = ddf.groupby(group_cols)["auroc"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(5, 3.5))
        models = [m for m in MODEL_ORDER if m in ddf["model"].unique()]

        for model in models:
            mdf = ddf[ddf["model"] == model]
            agg = mdf.groupby("resolution")["auroc"].agg(["mean", "std"]).reindex(RESOLUTIONS)
            agg["std"] = agg["std"].fillna(0)

            ax.errorbar(
                agg.index, agg["mean"], yerr=agg["std"],
                marker=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                label=model_display(model),
                linewidth=1.5, markersize=6, capsize=3,
            )
            ax.fill_between(
                agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                alpha=0.1, color=MODEL_COLORS[model],
            )

        ax.set_xlabel("Resolution (px)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{DOMAIN_DISPLAY.get(domain, domain)} — AUROC vs Resolution")
        ax.set_xticks(RESOLUTIONS)
        ax.set_xticklabels([str(r) for r in RESOLUTIONS])
        ax.invert_xaxis()
        ax.legend(loc="best", framealpha=0.9)
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.02))
        fig.tight_layout()
        save_fig(fig, output_dir, f"auroc_vs_resolution_{domain}")


# ---------------------------------------------------------------------------
# Plot 2: Efficiency vs AUROC scatter
# ---------------------------------------------------------------------------
def plot_efficiency_vs_auroc(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of GFLOPs vs AUROC, colored by model, sized by resolution."""
    # Aggregate across seeds
    group_cols = ["model", "resolution", "domain"]
    agg = df.groupby(group_cols).agg(
        auroc_mean=("auroc", "mean"),
        gflops_mean=("gflops", "mean"),
    ).reset_index()

    # Average across domains for a single overview plot
    agg2 = agg.groupby(["model", "resolution"]).agg(
        auroc_mean=("auroc_mean", "mean"),
        gflops_mean=("gflops_mean", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    res_sizes = {512: 120, 256: 80, 128: 50, 64: 30}

    models = [m for m in MODEL_ORDER if m in agg2["model"].unique()]
    for model in models:
        mdf = agg2[agg2["model"] == model]
        ax.scatter(
            mdf["gflops_mean"], mdf["auroc_mean"],
            s=[res_sizes.get(r, 50) for r in mdf["resolution"]],
            c=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            label=model_display(model),
            edgecolors="white", linewidth=0.5, zorder=3,
        )
        # Annotate resolution next to each point
        for _, row in mdf.iterrows():
            ax.annotate(
                f"{int(row['resolution'])}",
                (row["gflops_mean"], row["auroc_mean"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 2), textcoords="offset points",
            )

    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("AUROC")
    ax.set_title("Accuracy–Efficiency Tradeoff")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, output_dir, "efficiency_vs_auroc")


# ---------------------------------------------------------------------------
# Plot 3: Per-class AUROC heatmap (dermatology)
# ---------------------------------------------------------------------------
def plot_perclass_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of per-class AUROC for dermatology, rows=model×resolution, cols=classes."""
    derm = df[df["domain"] == "dermatology"].copy()
    if derm.empty:
        print("  Skipping per-class heatmap: no dermatology data")
        return

    # Collect per-class data
    records = []
    for _, row in derm.iterrows():
        pc = row.get("per_class_auroc")
        if pc and isinstance(pc, dict):
            for cls, val in pc.items():
                records.append({
                    "model": row["model"],
                    "resolution": row["resolution"],
                    "seed": row["seed"],
                    "class": cls,
                    "auroc": val,
                })

    if not records:
        print("  Skipping per-class heatmap: no per_class_auroc data")
        return

    pcdf = pd.DataFrame(records)
    # Average across seeds
    pivot = pcdf.groupby(["model", "resolution", "class"])["auroc"].mean().reset_index()
    pivot["label"] = pivot.apply(
        lambda r: f"{model_display(r['model'])}\n{int(r['resolution'])}px", axis=1
    )

    # Order rows
    row_order = []
    models = [m for m in MODEL_ORDER if m in pivot["model"].unique()]
    for model in models:
        for res in RESOLUTIONS:
            lbl = f"{model_display(model)}\n{res}px"
            if lbl in pivot["label"].values:
                row_order.append(lbl)

    classes = sorted(pivot["class"].unique())
    heat_data = pivot.pivot_table(index="label", columns="class", values="auroc")
    heat_data = heat_data.reindex(index=row_order, columns=classes)

    fig, ax = plt.subplots(figsize=(max(4, len(classes) * 1.2 + 1), max(4, len(row_order) * 0.4 + 1)))
    sns.heatmap(
        heat_data, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=0.5, vmax=1.0,
        cbar_kws={"label": "AUROC"},
    )
    ax.set_title("Dermatology — Per-Class AUROC")
    ax.set_xlabel("Class")
    ax.set_ylabel("")
    fig.tight_layout()
    save_fig(fig, output_dir, "perclass_heatmap_dermatology")


# ---------------------------------------------------------------------------
# Plot 4: Pathology task comparison
# ---------------------------------------------------------------------------
def plot_pathology_tasks(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart: AUROC per task per model at 512px."""
    path_df = df[(df["domain"] == "pathology") & (df["resolution"] == 512)].copy()
    if path_df.empty:
        # Fall back to highest available resolution
        path_df = df[df["domain"] == "pathology"].copy()
        if path_df.empty:
            print("  Skipping pathology task plot: no pathology data")
            return
        max_res = path_df["resolution"].max()
        path_df = path_df[path_df["resolution"] == max_res]
        res_label = f"{max_res}px"
    else:
        res_label = "512px"

    agg = path_df.groupby(["task", "model"])["auroc"].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0)

    tasks = [t for t in TASK_DISPLAY if t in agg["task"].unique()]
    models = [m for m in MODEL_ORDER if m in agg["model"].unique()]
    n_tasks = len(tasks)
    n_models = len(models)

    if n_tasks == 0 or n_models == 0:
        return

    fig, ax = plt.subplots(figsize=(max(5, n_tasks * 1.2 + 1), 4))
    bar_width = 0.8 / n_models
    x = np.arange(n_tasks)

    for i, model in enumerate(models):
        mdf = agg[agg["model"] == model].set_index("task")
        means = [mdf.loc[t, "mean"] if t in mdf.index else 0 for t in tasks]
        stds = [mdf.loc[t, "std"] if t in mdf.index else 0 for t in tasks]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means, bar_width, yerr=stds,
            label=model_display(model), color=MODEL_COLORS[model],
            capsize=2, edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Task")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Pathology — AUROC per Task ({res_label})")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_DISPLAY.get(t, t) for t in tasks], rotation=15, ha="right")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.05))
    fig.tight_layout()
    save_fig(fig, output_dir, "pathology_task_comparison")


# ---------------------------------------------------------------------------
# Plot 5: Summary bar chart
# ---------------------------------------------------------------------------
def plot_summary_bars(df: pd.DataFrame, output_dir: Path):
    """AUROC at 512px (or max res) for all models across all domains side-by-side."""
    # Use highest resolution available
    max_res = df["resolution"].max()
    sub = df[df["resolution"] == max_res].copy()

    # For pathology, average across tasks per seed first
    records = []
    for domain in sub["domain"].unique():
        ddf = sub[sub["domain"] == domain]
        if domain == "pathology":
            ddf = ddf.groupby(["model", "seed"])["auroc"].mean().reset_index()
        for model in ddf["model"].unique():
            mdf = ddf[ddf["model"] == model]
            records.append({
                "domain": DOMAIN_DISPLAY.get(domain, domain),
                "model": model,
                "auroc_mean": mdf["auroc"].mean(),
                "auroc_std": mdf["auroc"].std(ddof=1) if len(mdf) > 1 else 0,
            })

    agg = pd.DataFrame(records)
    domains = sorted(agg["domain"].unique())
    models = [m for m in MODEL_ORDER if m in agg["model"].unique()]
    n_domains = len(domains)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(5, n_domains * 2), 4))
    bar_width = 0.8 / n_models
    x = np.arange(n_domains)

    for i, model in enumerate(models):
        mdf = agg[agg["model"] == model].set_index("domain")
        means = [mdf.loc[d, "auroc_mean"] if d in mdf.index else 0 for d in domains]
        stds = [mdf.loc[d, "auroc_std"] if d in mdf.index else 0 for d in domains]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means, bar_width, yerr=stds,
            label=model_display(model), color=MODEL_COLORS[model],
            capsize=2, edgecolor="white", linewidth=0.5,
        )

    ax.set_ylabel("AUROC")
    ax.set_title(f"Summary — AUROC at {max_res}px")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.05))
    fig.tight_layout()
    save_fig(fig, output_dir, "summary_auroc")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def generate_main_tables(df: pd.DataFrame, output_dir: Path):
    """Generate main results tables (one per domain) in LaTeX, Markdown, and CSV."""
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    domains = sorted(df["domain"].unique())

    for domain in domains:
        ddf = df[df["domain"] == domain]

        # For pathology, average across tasks per (model, resolution, seed)
        if domain == "pathology":
            ddf = ddf.groupby(["model", "resolution", "seed"]).agg(
                auroc=("auroc", "mean"),
                accuracy=("accuracy", "mean"),
                f1=("f1", "mean"),
                gflops=("gflops", "first"),
                latency_ms=("latency_ms", "first"),
            ).reset_index()

        models = [m for m in MODEL_ORDER if m in ddf["model"].unique()]
        resolutions = sorted(ddf["resolution"].unique(), reverse=True)

        # Build table rows
        rows_md = []
        rows_latex = []
        rows_csv = []

        for model in models:
            for res in resolutions:
                sub = ddf[(ddf["model"] == model) & (ddf["resolution"] == res)]
                if sub.empty:
                    continue
                row = {
                    "Model": model_display(model),
                    "Res": f"{res}px",
                    "AUROC": fmt(sub["auroc"], 3),
                    "Acc": fmt(sub["accuracy"], 3),
                    "F1": fmt(sub["f1"], 3),
                    "GFLOPs": fmt(sub["gflops"], 2),
                    "Latency (ms)": fmt(sub["latency_ms"], 2),
                }
                rows_md.append(row)

                row_latex = {
                    "Model": model_display(model),
                    "Res": f"{res}",
                    "AUROC": fmt_latex(sub["auroc"], 3),
                    "Acc": fmt_latex(sub["accuracy"], 3),
                    "F1": fmt_latex(sub["f1"], 3),
                    "GFLOPs": fmt_latex(sub["gflops"], 2),
                    "Latency (ms)": fmt_latex(sub["latency_ms"], 2),
                }
                rows_latex.append(row_latex)

                row_csv = {
                    "Model": model_display(model),
                    "Resolution": res,
                    "AUROC": fmt(sub["auroc"], 4),
                    "Accuracy": fmt(sub["accuracy"], 4),
                    "F1": fmt(sub["f1"], 4),
                    "GFLOPs": fmt(sub["gflops"], 3),
                    "Latency_ms": fmt(sub["latency_ms"], 3),
                    "N_seeds": len(sub),
                }
                rows_csv.append(row_csv)

        domain_label = DOMAIN_DISPLAY.get(domain, domain)
        _save_markdown_table(table_dir, f"main_{domain}", domain_label, rows_md)
        _save_latex_table(table_dir, f"main_{domain}", domain_label, rows_latex)
        _save_csv_table(table_dir, f"main_{domain}", rows_csv)


def generate_pathology_task_table(df: pd.DataFrame, output_dir: Path):
    """Per-task table for pathology: rows=task, cols=model AUROC at each resolution."""
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    path_df = df[df["domain"] == "pathology"]
    if path_df.empty:
        print("  Skipping pathology task table: no data")
        return

    models = [m for m in MODEL_ORDER if m in path_df["model"].unique()]
    resolutions = sorted(path_df["resolution"].unique(), reverse=True)
    tasks = [t for t in TASK_DISPLAY if t in path_df["task"].unique()]

    # Build rows
    rows_md = []
    rows_latex = []
    rows_csv = []

    for task in tasks:
        for res in resolutions:
            row_md = {"Task": TASK_DISPLAY.get(task, task), "Res": f"{res}px"}
            row_latex = {"Task": TASK_DISPLAY.get(task, task), "Res": f"{res}"}
            row_csv = {"Task": task, "Resolution": res}

            for model in models:
                sub = path_df[(path_df["task"] == task) & (path_df["model"] == model) & (path_df["resolution"] == res)]
                col_name = model_display(model)
                row_md[col_name] = fmt(sub["auroc"], 3)
                row_latex[col_name] = fmt_latex(sub["auroc"], 3)
                row_csv[col_name] = fmt(sub["auroc"], 4)

            rows_md.append(row_md)
            rows_latex.append(row_latex)
            rows_csv.append(row_csv)

    _save_markdown_table(table_dir, "pathology_tasks", "Pathology — Per-Task AUROC", rows_md)
    _save_latex_table(table_dir, "pathology_tasks", "Pathology — Per-Task AUROC", rows_latex)
    _save_csv_table(table_dir, "pathology_tasks", rows_csv)


def _save_markdown_table(table_dir: Path, name: str, title: str, rows: list[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines.append("")

    path = table_dir / f"{name}.md"
    path.write_text("\n".join(lines))
    print(f"  Saved: tables/{name}.md")


def _save_latex_table(table_dir: Path, name: str, title: str, rows: list[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    col_spec = "l" * len(cols)
    lines = [
        f"% {title}",
        f"\\begin{{table}}[htbp]",
        f"  \\centering",
        f"  \\caption{{{title}}}",
        f"  \\label{{tab:{name}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        f"    \\toprule",
        "    " + " & ".join(cols) + " \\\\",
        f"    \\midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(str(row[c]) for c in cols) + " \\\\")
    lines += [
        f"    \\bottomrule",
        f"  \\end{{tabular}}",
        f"\\end{{table}}",
    ]

    path = table_dir / f"{name}.tex"
    path.write_text("\n".join(lines))
    print(f"  Saved: tables/{name}.tex")


def _save_csv_table(table_dir: Path, name: str, rows: list[dict]):
    if not rows:
        return
    csv_df = pd.DataFrame(rows)
    path = table_dir / f"{name}.csv"
    csv_df.to_csv(path, index=False)
    print(f"  Saved: tables/{name}.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize linear probing results."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Root directory containing result JSON files (default: results/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Output directory for figures/ and tables/ (default: results/)",
    )
    args = parser.parse_args()

    setup_plot_style()

    print("=" * 60)
    print("  Linear Probing Evaluation — Results Aggregation")
    print("=" * 60)

    df = load_all_results(args.results_dir)
    print()

    # --- Plots ---
    print("Generating plots...")
    plot_auroc_vs_resolution(df, args.output_dir)
    plot_efficiency_vs_auroc(df, args.output_dir)
    plot_perclass_heatmap(df, args.output_dir)
    plot_pathology_tasks(df, args.output_dir)
    plot_summary_bars(df, args.output_dir)
    print()

    # --- Tables ---
    print("Generating tables...")
    generate_main_tables(df, args.output_dir)
    generate_pathology_task_table(df, args.output_dir)
    print()

    print("=" * 60)
    print("  Done! Check results/figures/ and results/tables/")
    print("=" * 60)


if __name__ == "__main__":
    main()
