#!/usr/bin/env python3
"""Per-abnormality resolution-sensitivity analysis (reviewer R2.3).

Reviewer #2 asked whether resolution affects some abnormalities more than others.
The linear-probe engine already records per-class / per-label AUROC in every
``results_*.json`` (``accuracy_metrics.per_class_auroc``), so this requires **no
retraining** — we just aggregate the existing runs.

For each domain it produces AUROC-vs-resolution broken down by finding:
  * radiology (CheXpert): one row per observation label,
  * dermatology (ISIC):   one row per lesion class,
  * pathology (TCGA):     one row per task (each task is a separate binary file),
plus the AUROC change from the highest to the lowest resolution (the sensitivity
Δ), sorted most-sensitive first. Outputs CSV, Markdown and LaTeX tables, and an
optional grouped-bar figure.

Usage:
  python scripts/per_abnormality_resolution.py \
      --results-dir /path/to/runs [/more/run/dirs ...] \
      --model dinov3 \
      --output-dir Med_REDUCE_Paper/analysis_r2_3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESOLUTIONS = [512, 256, 128, 64]


# ---------------------------------------------------------------------------
# Discovery / parsing
# ---------------------------------------------------------------------------
def discover(results_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in results_dirs:
        files.extend(sorted(Path(d).rglob("results_*.json")))
    return files


def _seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("_")[1])
            except (IndexError, ValueError):
                return None
    return None


def parse(path: Path) -> list[dict]:
    """Return long-form rows: {domain, dataset, model, resolution, seed, finding, auroc}."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    info = data.get("experiment_info", {})
    acc = data.get("accuracy_metrics", {})

    model = info.get("model_name", "unknown")
    domain = info.get("domain", "unknown")
    dataset = info.get("dataset", domain)
    resolution = int(info.get("resolution", 0) or 0)
    seed = info.get("seed", _seed_from_path(path))

    rows: list[dict] = []
    base = {
        "domain": domain,
        "dataset": dataset,
        "model": model,
        "resolution": resolution,
        "seed": seed,
    }

    per_class = acc.get("per_class_auroc")
    if isinstance(per_class, dict) and per_class:
        # Per-class / per-label breakdown (radiology labels, dermatology classes).
        for finding, auroc in per_class.items():
            if auroc is None:
                continue
            rows.append({**base, "finding": str(finding), "auroc": float(auroc)})
        return rows

    # No per-class breakdown → binary task (pathology). The task-level AUROC IS
    # the per-abnormality value; label it by the dataset (e.g. tcga_kras -> KRAS).
    overall = acc.get("final_val_auroc")
    if overall is None:
        overall = acc.get("best_metric")
    if overall is not None and not (isinstance(overall, float) and np.isnan(overall)):
        task_name = str(dataset).replace("tcga_", "").upper() if str(dataset).startswith("tcga_") else str(dataset)
        rows.append({**base, "finding": task_name, "auroc": float(overall)})

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """finding × resolution table of mean AUROC (± SD across seeds) with Δ sensitivity."""
    agg = (
        df.groupby(["finding", "resolution"])["auroc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    mean_pivot = agg.pivot(index="finding", columns="resolution", values="mean")
    std_pivot = agg.pivot(index="finding", columns="resolution", values="std")

    cols = [r for r in RESOLUTIONS if r in mean_pivot.columns]
    mean_pivot = mean_pivot[cols]
    std_pivot = std_pivot.reindex(columns=cols)

    hi, lo = cols[0], cols[-1]
    out = pd.DataFrame(index=mean_pivot.index)
    for r in cols:
        out[f"AUROC@{r}"] = mean_pivot[r]
        out[f"SD@{r}"] = std_pivot[r]
    # Sensitivity: drop from highest to lowest resolution (positive = degrades).
    out["delta_hi_lo"] = mean_pivot[hi] - mean_pivot[lo]
    out = out.sort_values("delta_hi_lo", ascending=False)
    return out


def to_markdown(table: pd.DataFrame, cols: list[int]) -> str:
    header = "| Finding | " + " | ".join(f"{r}px" for r in cols) + " | Δ(hi→lo) |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [header, sep]
    for finding, row in table.iterrows():
        cells = []
        for r in cols:
            m = row.get(f"AUROC@{r}", np.nan)
            s = row.get(f"SD@{r}", np.nan)
            cells.append(f"{m:.3f}±{s:.3f}" if pd.notna(s) else (f"{m:.3f}" if pd.notna(m) else "—"))
        lines.append(f"| {finding} | " + " | ".join(cells) + f" | {row['delta_hi_lo']:+.3f} |")
    return "\n".join(lines)


def to_latex(table: pd.DataFrame, cols: list[int], caption: str, label: str) -> str:
    colspec = "|l|" + "c|" * (len(cols) + 1)
    head = " & ".join(["\\textbf{Finding}"] + [f"\\textbf{{{r}}}" for r in cols] + ["\\textbf{$\\Delta$}"])
    lines = [
        "\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\renewcommand{\\arraystretch}{1.15}", f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
        head + " \\\\", "\\hline",
    ]
    for finding, row in table.iterrows():
        cells = []
        for r in cols:
            m = row.get(f"AUROC@{r}", np.nan)
            s = row.get(f"SD@{r}", np.nan)
            cells.append(f"${m:.3f}\\pm{s:.3f}$" if pd.notna(s) else (f"${m:.3f}$" if pd.notna(m) else "--"))
        safe = str(finding).replace("_", "\\_")
        lines.append(f"{safe} & " + " & ".join(cells) + f" & ${row['delta_hi_lo']:+.3f}$ \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def plot(table: pd.DataFrame, cols: list[int], domain: str, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable; skipping figure")
        return
    findings = list(table.index)
    x = np.arange(len(findings))
    width = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(findings)), 4.5))
    for i, r in enumerate(cols):
        means = table[f"AUROC@{r}"].values
        errs = table[f"SD@{r}"].values
        ax.bar(x + i * width, means, width, yerr=np.nan_to_num(errs), capsize=2, label=f"{r}px")
    ax.set_xticks(x + width * (len(cols) - 1) / 2)
    ax.set_xticklabels(findings, rotation=40, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Per-finding AUROC vs. resolution — {domain}")
    ax.legend(title="Resolution", fontsize=8)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    out = out_dir / f"per_abnormality_{domain}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", nargs="+", required=True, type=Path,
                    help="One or more run directories containing results_*.json")
    ap.add_argument("--model", default=None,
                    help="Only include this model_name (e.g. dinov3). Default: all models.")
    ap.add_argument("--output-dir", type=Path, default=Path("analysis_r2_3"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    files = discover(args.results_dir)
    if not files:
        print(f"No results_*.json under {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = [r for path in files for r in parse(path)]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No usable AUROC rows parsed.", file=sys.stderr)
        sys.exit(1)
    if args.model:
        df = df[df["model"] == args.model]
        if df.empty:
            print(f"No rows for model={args.model}. Available: {sorted(set(r['model'] for r in rows))}", file=sys.stderr)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Parsed {len(df)} rows across models={sorted(df['model'].unique())}, "
          f"domains={sorted(df['domain'].unique())}")

    for domain, ddf in df.groupby("domain"):
        table = build_table(ddf)
        cols = [r for r in RESOLUTIONS if f"AUROC@{r}" in table.columns]
        if not cols:
            continue
        stem = args.output_dir / f"per_abnormality_{domain}"
        table.to_csv(f"{stem}.csv")
        (Path(f"{stem}.md")).write_text(
            f"### {domain} — per-finding AUROC vs. resolution (mean±SD over seeds)\n\n"
            + to_markdown(table, cols) + "\n"
        )
        (Path(f"{stem}.tex")).write_text(
            to_latex(
                table, cols,
                caption=f"Per-finding AUROC vs.\\ resolution for {domain}. "
                        f"$\\Delta$ is the AUROC change from {cols[0]} to {cols[-1]} pixels "
                        f"(larger = more resolution-sensitive).",
                label=f"tab:per_abnormality_{domain}",
            )
        )
        print(f"\n[{domain}] wrote {stem}.csv/.md/.tex")
        print(to_markdown(table, cols))
        if not args.no_plot:
            plot(table, cols, domain, args.output_dir)


if __name__ == "__main__":
    main()
