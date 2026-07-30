#!/usr/bin/env python3
"""
Cross-domain visualization suite for med-reduce experiments.

Generates publication-quality figures from the results/ directory tree:
  1. AUROC vs resolution curves  (per domain, all models on one plot)
  2. Cross-domain summary panel  (3×1 grid: derm / path / rad)
  3. Model comparison bar charts (teacher vs students at each resolution)
  4. Efficiency scatter plots    (AUROC vs FLOPs / latency / memory)
  5. Per-class AUROC heatmaps    (radiology 8-class, pathology tasks)
  6. Resolution degradation delta (AUROC drop from 512px baseline)

Usage:
    python -m src.evaluation.analyze_results \
        --results-root results/ \
        --output-dir results/figures

    python -m src.evaluation.analyze_results \
        --results-root results/ \
        --output-dir results/figures \
        --domains dermatology radiology \
        --models dinov3 resnet50_distilled
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.aggregate_results import (
    PATHOLOGY_TASK_LABELS,
    aggregate,
    discover_results,
)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    "dinov3": "DINOv3 (Teacher)",
    "resnet50_distilled": "ResNet-50 (Distilled)",
    "tiny_vit_21m_224_distilled": "TinyViT-21M (Distilled)",
}

MODEL_COLORS = {
    "dinov3": "#2176AE",
    "resnet50_distilled": "#E8572A",
    "tiny_vit_21m_224_distilled": "#57A773",
}

MODEL_MARKERS = {
    "dinov3": "o",
    "resnet50_distilled": "s",
    "tiny_vit_21m_224_distilled": "^",
}

DOMAIN_DISPLAY = {
    "dermatology": "Dermatology (ISIC)",
    "pathology": "Pathology (TCGA)",
    "radiology": "Radiology (CheXpert)",
}


def _style():
    """Apply consistent publication style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def _sync_axes(axes, sync_y: bool = True, sync_x: bool = False) -> None:
    """Set all axes in *axes* to the same x/y limits (global min/max)."""
    # Flatten if needed (handles both 1-D lists and 2-D numpy arrays)
    flat = []
    for a in axes:
        if hasattr(a, "__iter__"):
            flat.extend(a)
        else:
            flat.append(a)
    # Filter out deleted axes
    flat = [a for a in flat if a.get_visible() and a.has_data()]
    if not flat:
        return

    if sync_y:
        ymin = min(a.get_ylim()[0] for a in flat)
        ymax = max(a.get_ylim()[1] for a in flat)
        for a in flat:
            a.set_ylim(ymin, ymax)

    if sync_x:
        xmin = min(a.get_xlim()[0] for a in flat)
        xmax = max(a.get_xlim()[1] for a in flat)
        for a in flat:
            a.set_xlim(xmin, xmax)


def _model_label(m: str) -> str:
    return MODEL_DISPLAY.get(m, m)


def _model_color(m: str) -> str:
    return MODEL_COLORS.get(m, "#888888")


def _model_marker(m: str) -> str:
    return MODEL_MARKERS.get(m, "D")


# ---------------------------------------------------------------------------
# 1. AUROC vs Resolution (single domain/task)
# ---------------------------------------------------------------------------
def plot_auroc_vs_resolution(
    raw: pd.DataFrame,
    domain: str,
    task: str = "default",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Line plot of AUROC vs resolution for each model, with error bands."""
    subset = raw[(raw["domain"] == domain) & (raw["task"] == task)].copy()
    if subset.empty:
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    for model in sorted(subset["model"].unique()):
        mdf = subset[subset["model"] == model]
        stats = (
            mdf.groupby("resolution")["auroc"]
            .agg(["mean", "std", "count"])
            .sort_index()
        )
        x = stats.index.values
        y = stats["mean"].values
        yerr = stats["std"].values

        ax.plot(
            x, y,
            marker=_model_marker(model),
            color=_model_color(model),
            label=_model_label(model),
            linewidth=2,
            markersize=7,
        )
        ax.fill_between(
            x, y - yerr, y + yerr,
            alpha=0.15, color=_model_color(model),
        )

    if title is None:
        task_label = PATHOLOGY_TASK_LABELS.get(task, task)
        domain_label = DOMAIN_DISPLAY.get(domain, domain)
        title = f"{domain_label}" if task == "default" else f"{domain_label} — {task_label}"

    ax.set_title(title)
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("AUROC")
    ax.set_xticks(sorted(subset["resolution"].unique()))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.legend(loc="lower left")
    ax.set_ylim(bottom=max(0.4, ax.get_ylim()[0] - 0.02))

    return ax


# ---------------------------------------------------------------------------
# 2. Cross-domain summary panel
# ---------------------------------------------------------------------------
def plot_cross_domain_panel(
    raw: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """
    3-column figure: dermatology | pathology (macro-avg across tasks) | radiology.
    Each column shows AUROC vs resolution for all models.
    """
    _style()
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    domains_order = ["dermatology", "pathology", "radiology"]

    for idx, domain in enumerate(domains_order):
        ddf = raw[raw["domain"] == domain].copy()
        if ddf.empty:
            axes[idx].set_title(DOMAIN_DISPLAY.get(domain, domain))
            axes[idx].text(0.5, 0.5, "No data", ha="center", va="center",
                           transform=axes[idx].transAxes, fontsize=16, color="gray")
            continue

        # For pathology, macro-average AUROC across tasks
        if domain == "pathology":
            ddf = (
                ddf.groupby(["model", "resolution", "seed"])["auroc"]
                .mean()
                .reset_index()
            )
            ddf["task"] = "default"
            ddf["domain"] = domain

        plot_auroc_vs_resolution(
            ddf, domain, task="default", ax=axes[idx],
            title=DOMAIN_DISPLAY.get(domain, domain),
        )

    # Unify y-axis limits across all subplots
    _sync_axes(axes, sync_y=True, sync_x=True)

    fig.suptitle("AUROC vs Resolution Across Domains", fontsize=20, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"  Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 3. Per-task pathology panel
# ---------------------------------------------------------------------------
def plot_pathology_tasks(
    raw: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 8),
) -> Optional[plt.Figure]:
    """Grid of AUROC vs resolution for each pathology task."""
    _style()
    path_df = raw[raw["domain"] == "pathology"]
    if path_df.empty:
        return None

    tasks = sorted(path_df["task"].unique())
    n = len(tasks)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, task in enumerate(tasks):
        r, c = divmod(i, ncols)
        plot_auroc_vs_resolution(raw, "pathology", task=task, ax=axes[r][c])

    # Hide unused axes
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        fig.delaxes(axes[r][c])

    # Unify axes across all task subplots
    active_axes = [axes[divmod(i, ncols)] for i in range(n)]
    _sync_axes(active_axes, sync_y=True, sync_x=True)

    fig.suptitle("Pathology: AUROC vs Resolution by Task", fontsize=20, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"  Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 4. Resolution degradation delta
# ---------------------------------------------------------------------------
def plot_degradation_delta(
    raw: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Bar chart showing AUROC drop (delta) from 512px baseline per model/domain.
    """
    _style()

    # Compute per-group mean AUROC
    # For pathology, macro-avg across tasks first
    frames = []
    for domain in raw["domain"].unique():
        ddf = raw[raw["domain"] == domain].copy()
        if domain == "pathology":
            ddf = (
                ddf.groupby(["model", "resolution", "seed"])["auroc"]
                .mean()
                .reset_index()
            )
        ddf["domain"] = domain
        frames.append(ddf)

    combined = pd.concat(frames, ignore_index=True)
    stats = (
        combined.groupby(["domain", "model", "resolution"])["auroc"]
        .mean()
        .reset_index()
    )

    # Compute delta from 512px
    baseline = stats[stats["resolution"] == 512][["domain", "model", "auroc"]].rename(
        columns={"auroc": "auroc_512"}
    )
    merged = stats.merge(baseline, on=["domain", "model"])
    merged["delta"] = merged["auroc"] - merged["auroc_512"]
    merged = merged[merged["resolution"] != 512]

    domains = sorted(merged["domain"].unique())
    fig, axes = plt.subplots(1, len(domains), figsize=figsize, sharey=True)
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        ddf = merged[merged["domain"] == domain]
        models = sorted(ddf["model"].unique())
        resolutions = sorted(ddf["resolution"].unique())
        n_models = len(models)
        bar_width = 0.8 / max(n_models, 1)
        x = np.arange(len(resolutions))

        for j, model in enumerate(models):
            mdf = ddf[ddf["model"] == model].sort_values("resolution")
            vals = [
                mdf[mdf["resolution"] == r]["delta"].values[0]
                if r in mdf["resolution"].values
                else 0
                for r in resolutions
            ]
            ax.bar(
                x + j * bar_width - (n_models - 1) * bar_width / 2,
                vals,
                bar_width,
                label=_model_label(model),
                color=_model_color(model),
                alpha=0.85,
            )

        ax.set_title(DOMAIN_DISPLAY.get(domain, domain))
        ax.set_xlabel("Resolution (px)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in resolutions])
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if ax == axes[0]:
            ax.set_ylabel("AUROC Change from 512px")
        ax.legend(fontsize=11)

    # Unify y-axis across domain panels
    _sync_axes(axes, sync_y=True)

    fig.suptitle("Resolution Degradation: AUROC Drop from Baseline", fontsize=20, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"  Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 5. Efficiency scatter: AUROC vs GFLOPs
# ---------------------------------------------------------------------------
def plot_efficiency_scatter(
    raw: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
) -> plt.Figure:
    """Scatter plot of mean AUROC vs GFLOPs (log scale) with Pareto frontier."""
    _style()
    fig, ax = plt.subplots(figsize=figsize)

    # Mean over seeds
    stats = (
        raw.groupby(["domain", "model", "resolution"])
        .agg(auroc=("auroc", "mean"), gflops=("gflops", "mean"))
        .reset_index()
    )

    # For pathology, average across tasks first
    path_raw = raw[raw["domain"] == "pathology"]
    if not path_raw.empty:
        path_avg = (
            path_raw.groupby(["model", "resolution", "seed"])
            .agg(auroc=("auroc", "mean"), gflops=("gflops", "first"))
            .reset_index()
            .groupby(["model", "resolution"])
            .agg(auroc=("auroc", "mean"), gflops=("gflops", "mean"))
            .reset_index()
        )
        path_avg["domain"] = "pathology"
        stats = stats[stats["domain"] != "pathology"]
        stats = pd.concat([stats, path_avg], ignore_index=True)

    # Drop rows with missing GFLOPs
    stats = stats.dropna(subset=["gflops", "auroc"])

    # --- Pareto frontier (maximize AUROC, minimize GFLOPs) ---
    pareto_pts = []
    sorted_by_flops = stats.sort_values("gflops", ascending=True)
    best_auroc = -np.inf
    for _, row in sorted_by_flops.iterrows():
        if row["auroc"] > best_auroc:
            pareto_pts.append(row)
            best_auroc = row["auroc"]
    if pareto_pts:
        pareto_df = pd.DataFrame(pareto_pts).sort_values("gflops")
        # Extend to plot edges for a step-like frontier
        ax.plot(
            pareto_df["gflops"], pareto_df["auroc"],
            color="#CC3333", linewidth=2, linestyle="--",
            alpha=0.7, zorder=5, label="Pareto frontier",
        )
        # Shade the dominated region lightly
        gf_vals = pareto_df["gflops"].values
        au_vals = pareto_df["auroc"].values
        ax.fill_between(
            gf_vals, au_vals, y2=ax.get_ylim()[0] if ax.get_ylim()[0] < min(au_vals) else 0.4,
            alpha=0.04, color="#CC3333",
        )

    # --- Scatter points ---
    res_sizes = {64: 50, 128: 100, 256: 170, 512: 260}
    domain_shapes = {"dermatology": "o", "pathology": "s", "radiology": "^"}

    for model in sorted(stats["model"].unique()):
        mdf = stats[stats["model"] == model]
        for domain in sorted(mdf["domain"].unique()):
            ddf = mdf[mdf["domain"] == domain]
            sizes = [res_sizes.get(int(r), 120) for r in ddf["resolution"]]
            marker = domain_shapes.get(domain, "D")
            ax.scatter(
                ddf["gflops"], ddf["auroc"],
                s=sizes,
                c=_model_color(model),
                marker=marker,
                alpha=0.80,
                edgecolors="white",
                linewidths=0.6,
                zorder=10,
            )
            # Annotate each point: resolution
            for _, row in ddf.iterrows():
                ax.annotate(
                    f"{int(row['resolution'])}",
                    (row["gflops"], row["auroc"]),
                    fontsize=8, alpha=0.55,
                    xytext=(5, 5), textcoords="offset points",
                )

    # --- Legend: combine model color + domain shape ---
    from matplotlib.lines import Line2D
    legend_handles = []
    # Model colors
    for model in sorted(stats["model"].unique()):
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_model_color(model), markersize=9,
                   label=_model_label(model))
        )
    # Domain shapes
    for domain, marker in domain_shapes.items():
        if domain in stats["domain"].values:
            legend_handles.append(
                Line2D([0], [0], marker=marker, color="w",
                       markerfacecolor="#888888", markersize=9,
                       label=DOMAIN_DISPLAY.get(domain, domain))
            )
    # Pareto line
    legend_handles.append(
        Line2D([0], [0], color="#CC3333", linewidth=2, linestyle="--",
               label="Pareto frontier")
    )
    # Resolution size key
    for res, sz in sorted(res_sizes.items()):
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#BBBBBB", markersize=np.sqrt(sz) / 2,
                   label=f"{res}px")
        )

    ax.legend(handles=legend_handles, loc="lower right", fontsize=11,
              ncol=2, framealpha=0.9)

    # --- Axes ---
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("GFLOPs (log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs Computational Cost")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
        print(f"  Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 6. Model comparison grouped bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison_bars(
    raw: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Grouped bar chart: for each domain, compare models at 512px and 64px.
    """
    _style()

    # Average pathology across tasks
    frames = []
    for domain in raw["domain"].unique():
        ddf = raw[raw["domain"] == domain].copy()
        if domain == "pathology":
            ddf = (
                ddf.groupby(["model", "resolution", "seed"])["auroc"]
                .mean()
                .reset_index()
            )
        ddf["domain"] = domain
        frames.append(ddf)

    combined = pd.concat(frames, ignore_index=True)
    # Focus on 512 and 64
    combined = combined[combined["resolution"].isin([512, 64])]
    stats = (
        combined.groupby(["domain", "model", "resolution"])["auroc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    domains = sorted(stats["domain"].unique())
    fig, axes = plt.subplots(1, len(domains), figsize=figsize, sharey=False)
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        ddf = stats[stats["domain"] == domain]
        models = sorted(ddf["model"].unique())
        resolutions = sorted(ddf["resolution"].unique(), reverse=True)

        x = np.arange(len(models))
        n_res = len(resolutions)
        bw = 0.35

        for i, res in enumerate(resolutions):
            rdf = ddf[ddf["resolution"] == res].set_index("model")
            vals = [rdf.loc[m, "mean"] if m in rdf.index else 0 for m in models]
            errs = [rdf.loc[m, "std"] if m in rdf.index else 0 for m in models]
            offset = (i - (n_res - 1) / 2) * bw
            bars = ax.bar(
                x + offset, vals, bw,
                yerr=errs, capsize=3,
                label=f"{res}px",
                alpha=0.85,
            )

        ax.set_title(DOMAIN_DISPLAY.get(domain, domain))
        ax.set_xticks(x)
        ax.set_xticklabels([_model_label(m) for m in models], rotation=20, ha="right", fontsize=11)
        ax.set_ylabel("AUROC")
        ax.legend(fontsize=11)

    # Unify y-axis across domain panels, with a sensible lower bound
    _sync_axes(axes, sync_y=True)
    global_min = min(a.get_ylim()[0] for a in axes)
    floor = max(global_min, 0.4)  # never go below 0.4 for AUROC bar charts
    for a in axes:
        a.set_ylim(bottom=floor)

    fig.suptitle("Model Comparison: 512px vs 64px", fontsize=20, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"  Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 7. Best hyperparameters table
# ---------------------------------------------------------------------------
def generate_hyperparams_table(
    raw: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Table of best hyperparameters (from 512px, seed 42) per domain/task/model.
    Saves CSV and LaTeX.
    """
    # Use 512px results at seed 42 as the canonical HP source
    hp_df = raw[
        (raw["resolution"] == 512)
        & raw["lr"].notna()
    ][["domain", "task", "model", "seed", "lr", "weight_decay", "batch_size"]].copy()

    # Take seed 42 if available, otherwise first seed
    hp_df = hp_df.sort_values("seed")
    hp_df = hp_df.drop_duplicates(subset=["domain", "task", "model"], keep="first")
    hp_df = hp_df.drop(columns=["seed"]).sort_values(["domain", "task", "model"])

    # Pretty task labels
    hp_df["task_label"] = hp_df["task"].map(
        lambda t: PATHOLOGY_TASK_LABELS.get(t, t if t != "default" else "—")
    )
    hp_df["model_label"] = hp_df["model"].map(_model_label)

    # CSV
    csv_path = output_dir / "best_hyperparameters.csv"
    hp_df[["domain", "task_label", "model_label", "lr", "weight_decay", "batch_size"]].rename(
        columns={
            "task_label": "task",
            "model_label": "model",
        }
    ).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # LaTeX
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Best hyperparameters selected at 512px via 5-fold cross-validation.}",
        "\\label{tab:best_hyperparams}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{llllcc}",
        "\\toprule",
        "Domain & Task & Model & LR & Weight Decay & Batch Size \\\\",
        "\\midrule",
    ]
    prev_domain = None
    for _, r in hp_df.iterrows():
        if r["domain"] != prev_domain and prev_domain is not None:
            lines.append("\\midrule")
        prev_domain = r["domain"]

        lr_str = f"${r['lr']:.0e}$".replace("e-0", "\\times10^{-") + "}" if r["lr"] else "—"
        # Fix formatting
        lr_str = f"${r['lr']:.1e}$".replace("e-0", " \\times 10^{-").replace("e-", " \\times 10^{-")
        if "10^" in lr_str:
            lr_str = lr_str.rstrip("$") + "}$"

        wd = r["weight_decay"]
        if wd is None or (isinstance(wd, float) and wd == 0.0):
            wd_str = "$0$"
        else:
            wd_str = f"${wd:.1e}$".replace("e-0", " \\times 10^{-").replace("e-", " \\times 10^{-")
            if "10^" in wd_str:
                wd_str = wd_str.rstrip("$") + "}$"

        bs = int(r["batch_size"]) if r["batch_size"] else "—"

        lines.append(
            f"{r['domain'].capitalize()} & {r['task_label']} & "
            f"{r['model_label']} & {lr_str} & {wd_str} & {bs} \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path = output_dir / "best_hyperparameters.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved: {tex_path}")

    # Console
    print("\n  BEST HYPERPARAMETERS (at 512px)")
    print(f"  {'Domain':<14} {'Task':<16} {'Model':<28} {'LR':<12} {'WD':<12} {'BS':<6}")
    print("  " + "-" * 88)
    for _, r in hp_df.iterrows():
        print(
            f"  {r['domain']:<14} {r['task_label']:<16} {r['model_label']:<28} "
            f"{r['lr']:<12.1e} {r['weight_decay']:<12.1e} {int(r['batch_size']):<6}"
        )
    print()


# ---------------------------------------------------------------------------
# 8. AUROC / log(GFLOPs) efficiency table
# ---------------------------------------------------------------------------
def generate_efficiency_table(
    raw: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Table of AUROC / log10(GFLOPs) per domain/task/model/resolution.
    Higher = better accuracy per unit of compute on log scale.
    Saves CSV and LaTeX.
    """
    df = raw.dropna(subset=["auroc", "gflops"]).copy()
    df = df[df["gflops"] > 0]

    # For pathology, keep per-task granularity
    stats = (
        df.groupby(["domain", "task", "model", "resolution"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            gflops_mean=("gflops", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    stats["log_gflops"] = np.log2(stats["gflops_mean"])
    stats["efficiency"] = stats["auroc_mean"] / stats["log_gflops"]

    stats = stats.sort_values(
        ["domain", "task", "model", "resolution"],
        ascending=[True, True, True, False],
    )

    # Pretty labels
    stats["task_label"] = stats["task"].map(
        lambda t: PATHOLOGY_TASK_LABELS.get(t, t if t != "default" else "—")
    )
    stats["model_label"] = stats["model"].map(_model_label)

    # CSV
    out = stats[
        ["domain", "task_label", "model_label", "resolution",
         "auroc_mean", "auroc_std", "gflops_mean", "log_gflops", "efficiency"]
    ].rename(columns={"task_label": "task", "model_label": "model"})
    out = out.round(4)
    csv_path = output_dir / "auroc_per_log_gflops.csv"
    out.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # LaTeX — show one row per (domain, task, model) at 512px and 64px
    focus = stats[stats["resolution"].isin([512, 64])].copy()
    focus = focus.sort_values(
        ["domain", "task", "model", "resolution"],
        ascending=[True, True, True, False],
    )

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{AUROC per $\\log_2$(GFLOPs): higher values indicate better accuracy "
        "per unit of computational cost on a logarithmic scale.}",
        "\\label{tab:auroc_per_log_gflops}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lllrcccr}",
        "\\toprule",
        "Domain & Task & Model & Res & AUROC & GFLOPs & $\\log_2$(GF) "
        "& AUROC/$\\log_2$(GF) \\\\",
        "\\midrule",
    ]
    prev_domain = None
    for _, r in focus.iterrows():
        if r["domain"] != prev_domain and prev_domain is not None:
            lines.append("\\midrule")
        prev_domain = r["domain"]
        auroc_str = f"{r['auroc_mean']:.3f}"
        gf_str = f"{r['gflops_mean']:.2f}"
        lgf_str = f"{r['log_gflops']:.2f}"
        eff_str = f"{r['efficiency']:.3f}"
        lines.append(
            f"{r['domain'].capitalize()} & {r['task_label']} & "
            f"{r['model_label']} & {r['resolution']} & "
            f"{auroc_str} & {gf_str} & {lgf_str} & {eff_str} \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path = output_dir / "auroc_per_log_gflops.tex"
    tex_path.write_text("\n".join(lines))
    print(f"  Saved: {tex_path}")

    # Console — compact view: best efficiency per domain/task (across models & resolutions)
    best = stats.loc[stats.groupby(["domain", "task"])["efficiency"].idxmax()]
    print("\n  BEST AUROC / log2(GFLOPs) PER DOMAIN & TASK")
    print(
        f"  {'Domain':<14} {'Task':<16} {'Model':<28} {'Res':<6} "
        f"{'AUROC':<8} {'GFLOPs':<10} {'log(GF)':<10} {'Efficiency':<10}"
    )
    print("  " + "-" * 102)
    for _, r in best.iterrows():
        print(
            f"  {r['domain']:<14} {r['task_label']:<16} {r['model_label']:<28} "
            f"{r['resolution']:<6} {r['auroc_mean']:<8.3f} {r['gflops_mean']:<10.2f} "
            f"{r['log_gflops']:<10.2f} {r['efficiency']:<10.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# Generate all
# ---------------------------------------------------------------------------
def generate_all(
    results_root: Path,
    output_dir: Path,
    domains: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
) -> None:
    """Discover results and generate all visualizations."""
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering results in {results_root} ...")
    raw = discover_results(results_root, domains=domains, models=models)
    if raw.empty:
        print("No results found.")
        return

    print(
        f"Found {len(raw)} results: "
        f"{raw['domain'].nunique()} domains, "
        f"{raw['model'].nunique()} models, "
        f"{raw['seed'].nunique()} seeds"
    )

    # 1. Cross-domain summary panel
    try:
        plot_cross_domain_panel(raw, str(output_dir / "cross_domain_auroc.png"))
    except Exception as e:
        print(f"  SKIP cross_domain_auroc: {e}")

    # 2. Per-task pathology grid
    try:
        fig = plot_pathology_tasks(raw, str(output_dir / "pathology_tasks_auroc.png"))
        if fig is None:
            print("  SKIP pathology_tasks_auroc: no pathology data")
    except Exception as e:
        print(f"  SKIP pathology_tasks_auroc: {e}")

    # 3. Individual domain plots
    for domain in raw["domain"].unique():
        ddf = raw[raw["domain"] == domain]
        tasks = ddf["task"].unique()
        for task in tasks:
            safe_name = f"{domain}_{task}_auroc.png" if task != "default" else f"{domain}_auroc.png"
            try:
                fig, ax = plt.subplots(figsize=(7, 5))
                plot_auroc_vs_resolution(raw, domain, task, ax=ax)
                fig.tight_layout()
                fig.savefig(str(output_dir / safe_name), dpi=300)
                plt.close(fig)
                print(f"  Saved: {safe_name}")
            except Exception as e:
                print(f"  SKIP {safe_name}: {e}")

    # 4. Degradation delta
    try:
        plot_degradation_delta(raw, str(output_dir / "degradation_delta.png"))
    except Exception as e:
        print(f"  SKIP degradation_delta: {e}")

    # 5. Efficiency scatter
    try:
        plot_efficiency_scatter(raw, str(output_dir / "efficiency_scatter.png"))
    except Exception as e:
        print(f"  SKIP efficiency_scatter: {e}")

    # 6. Model comparison bars
    try:
        plot_model_comparison_bars(raw, str(output_dir / "model_comparison_bars.png"))
    except Exception as e:
        print(f"  SKIP model_comparison_bars: {e}")

    # 7. Save aggregated tables
    from src.evaluation.aggregate_results import aggregate as agg_fn, save_tables
    agg_df = agg_fn(raw)
    save_tables(agg_df, str(output_dir / "summary_table"), title="Cross-Domain Results")

    # 8. Best hyperparameters table
    try:
        generate_hyperparams_table(raw, output_dir)
    except Exception as e:
        print(f"  SKIP best_hyperparameters: {e}")

    # 9. AUROC / log(GFLOPs) efficiency table
    try:
        generate_efficiency_table(raw, output_dir)
    except Exception as e:
        print(f"  SKIP auroc_per_log_gflops: {e}")

    plt.close("all")
    print(f"\nAll outputs saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-domain visualizations from experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-root", type=str, default="results/",
        help="Root directory containing per-domain result folders.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/figures",
        help="Directory to save all figures and tables.",
    )
    parser.add_argument(
        "--domains", type=str, nargs="*", default=None,
        help="Restrict to these domains.",
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None,
        help="Restrict to these model names.",
    )
    # Legacy compatibility
    parser.add_argument("--metrics_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    output = args.output_dir or args.output_dir or "results/figures"
    root = args.metrics_dir or args.results_root

    generate_all(
        Path(root),
        Path(output),
        domains=args.domains,
        models=args.models,
    )


if __name__ == "__main__":
    main()
