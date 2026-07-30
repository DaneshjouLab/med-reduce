#!/usr/bin/env python3
"""
Summarize LP baseline results across domains, resolutions, and seeds.

Reads results JSON files from results/<domain>/runs/probe_two_stage/seed_*/
and prints tables with mean +/- SD for accuracy and efficiency metrics.

Usage:
    python scripts/summarize_lp_results.py
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("results")
RESOLUTIONS = [512, 256, 128, 64]
SEEDS = ["seed_42", "seed_123", "seed_456"]

# Runs now live under results/<teacher>/<domain>/runs/. Set TEACHER to switch
# between dinov3 and biomedclip result trees.
TEACHER = "dinov3"

DOMAINS = {
    "Dermatology": {
        "dir": f"{TEACHER}/dermatology",
        "pattern": "results_images_dinov3_{res}px.json",
    },
    "Radiology": {
        "dir": f"{TEACHER}/radiology",
        "pattern": "results_combined_train_valid_chexpert_v1.0_dinov3_{res}px.json",
    },
    "Pathology": {
        "dir": f"{TEACHER}/pathology",
        "tasks": ["luad_vs_lusc", "lgg_vs_gbm", "kras", "tp53", "egfr"],
        "pattern": "results_tcga_{task}_dinov3_{res}px.json",
    },
}

# Metrics to extract
ACCURACY_METRICS = [
    ("val_auroc", "final_val_auroc"),
    ("val_acc", "final_val_acc"),
    ("val_f1", "final_val_f1"),
    ("val_loss", "final_val_loss"),
]

EFFICIENCY_METRICS = [
    ("GFLOPs", "encoder_gflops"),
    ("Latency (ms)", "encoder_latency_ms"),
    ("Embed Time (s)", "embedding_extraction_time_s"),
    ("Peak GPU (MB)", "peak_gpu_memory_mb"),
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(values: list[float], precision: int = 4) -> str:
    """Format as mean +/- SD."""
    if not values:
        return "—"
    arr = np.array(values)
    mean = arr.mean()
    if len(arr) == 1:
        return f"{mean:.{precision}f}"
    sd = arr.std(ddof=1)
    return f"{mean:.{precision}f} ± {sd:.{precision}f}"


def print_table(title: str, headers: list[str], rows: list[tuple[str, list[str]]]):
    """Print a formatted table."""
    col_widths = [max(len(h), 20) for h in headers]
    label_width = max(len(r[0]) for r in rows) if rows else 20
    label_width = max(label_width, 10)

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Header
    header_line = f"{'Metric':<{label_width}}"
    for h, w in zip(headers, col_widths):
        header_line += f"  {h:>{w}}"
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for label, vals in rows:
        line = f"{label:<{label_width}}"
        for v, w in zip(vals, col_widths):
            line += f"  {v:>{w}}"
        print(line)
    print()


def summarize_domain(domain_name: str, config: dict):
    """Summarize a single domain (possibly with multiple tasks)."""
    tasks = config.get("tasks", [None])
    domain_dir = RESULTS_ROOT / config["dir"] / "runs" / "probe_two_stage"

    for task in tasks:
        if task:
            task_label = f"{domain_name} — {task}"
        else:
            task_label = domain_name

        # Collect metrics: metric_name -> resolution -> [values across seeds]
        acc_data = defaultdict(lambda: defaultdict(list))
        eff_data = defaultdict(lambda: defaultdict(list))

        for seed in SEEDS:
            for res in RESOLUTIONS:
                if task:
                    fname = config["pattern"].format(task=task, res=res)
                else:
                    fname = config["pattern"].format(res=res)
                path = domain_dir / seed / fname
                data = load_json(path)
                if data is None:
                    continue

                for display_name, key in ACCURACY_METRICS:
                    val = data.get("accuracy_metrics", {}).get(key)
                    if val is not None:
                        acc_data[display_name][res].append(val)

                for display_name, key in EFFICIENCY_METRICS:
                    val = data.get("efficiency_metrics", {}).get(key)
                    if val is not None:
                        eff_data[display_name][res].append(val)

        # Build and print accuracy table
        headers = [f"{r}px" for r in RESOLUTIONS]
        acc_rows = []
        for display_name, _ in ACCURACY_METRICS:
            vals = [fmt(acc_data[display_name].get(r, []), precision=3) for r in RESOLUTIONS]
            acc_rows.append((display_name, vals))
        if acc_rows:
            print_table(f"{task_label} — Accuracy Metrics", headers, acc_rows)

        # Build and print efficiency table
        eff_rows = []
        for display_name, _ in EFFICIENCY_METRICS:
            prec = 2 if "GFLOPs" in display_name or "Latency" in display_name else 1
            vals = [fmt(eff_data[display_name].get(r, []), precision=prec) for r in RESOLUTIONS]
            eff_rows.append((display_name, vals))
        if eff_rows:
            print_table(f"{task_label} — Efficiency Metrics", headers, eff_rows)


def main():
    import sys
    import io

    # Capture all output
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print("LP Baseline Results Summary (DINOv3 teacher)")
    print(f"Seeds: {', '.join(SEEDS)}")
    print(f"Resolutions: {RESOLUTIONS}")

    for domain_name, config in DOMAINS.items():
        summarize_domain(domain_name, config)

    sys.stdout = old_stdout
    output = buf.getvalue()

    # Print to terminal
    print(output, end="")

    # Save to file
    out_path = RESULTS_ROOT / "lp_baseline_summary.txt"
    out_path.write_text(output)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
