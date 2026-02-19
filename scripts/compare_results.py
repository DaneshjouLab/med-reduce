#!/usr/bin/env python3
"""Compare accuracy and efficiency metrics across models, tasks, and seeds.

Usage:
    python scripts/compare_results.py results/derm-results
    python scripts/compare_results.py results/path-results
    python scripts/compare_results.py results/derm-results --resolutions 512 256
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456]
RESOLUTIONS = [512, 256, 128, 64]

# Model -> which run directory to look in
MODEL_RUN_DIRS = {
    "dinov3":   "runs/probe_two_stage",
    "resnet18": "runs/probe_distilled",
    "tiny_vit": "runs/probe_distilled",
}

ACCURACY_KEYS = [
    "final_val_auroc",
    "final_val_acc",
    "final_val_f1",
    "final_val_loss",
]

EFFICIENCY_KEYS = [
    "encoder_gflops",
    "encoder_latency_ms",
    "embedding_extraction_time_s",
    "peak_gpu_memory_mb",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(values: list, precision: int = 4) -> str:
    if not values:
        return "—"
    mean, std = np.mean(values), np.std(values)
    if np.isnan(mean):
        return "—"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def fmt_eff(values: list, key: str) -> str:
    if not values:
        return "—"
    mean = np.mean(values)
    if np.isnan(mean):
        return "—"
    if "gflops" in key:
        return f"{mean:.3f}"
    if "latency" in key:
        return f"{mean:.2f}"
    if "memory" in key:
        return f"{int(mean)}"
    if "time" in key:
        return f"{mean:.1f}"
    return f"{mean:.4f}"


def discover_results(base_dir: str):
    """Auto-discover all result files and extract (task, model, resolution, seed).

    Filename patterns:
        results_{dataset}_{model}_{res}px.json          (derm/rad)
        results_images_{model}_{res}px.json              (derm alternate)
        results_tcga_{task}_{model}_{res}px.json         (pathology)

    Returns list of dicts with keys: path, model, resolution, seed, task
    """
    entries = []
    # Pattern: results[_*]_{model}_{res}px.json  (skip *_backup_*)
    pattern = re.compile(
        r"^results_(?P<prefix>.+)_(?P<model>dinov3|resnet18|tiny_vit)_(?P<res>\d+)px\.json$"
    )

    for model_name, run_dir in MODEL_RUN_DIRS.items():
        for seed in SEEDS:
            seed_dir = os.path.join(base_dir, run_dir, f"seed_{seed}")
            if not os.path.isdir(seed_dir):
                continue
            for fname in os.listdir(seed_dir):
                if "backup" in fname:
                    continue
                m = pattern.match(fname)
                if not m:
                    continue
                if m.group("model") != model_name:
                    continue

                prefix = m.group("prefix")
                res = int(m.group("res"))

                # Determine task from prefix
                # "tcga_luad_vs_lusc" -> task = "luad_vs_lusc"
                # "images" or "combined_train_valid_chexpert_v1.0" -> task = None (single-task domain)
                if prefix.startswith("tcga_"):
                    task = prefix[5:]  # strip "tcga_"
                else:
                    task = None

                entries.append({
                    "path": os.path.join(seed_dir, fname),
                    "model": model_name,
                    "resolution": res,
                    "seed": seed,
                    "task": task,
                })

    return entries


def collect_results(entries: list, resolutions: list[int]):
    """Group entries into: task -> model -> resolution -> {accuracy: {key: [vals]}, efficiency: ...}"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"accuracy": defaultdict(list), "efficiency": defaultdict(list), "n_seeds": 0}
    )))

    for e in entries:
        if e["resolution"] not in resolutions:
            continue

        data = load_json(e["path"])
        if data is None:
            continue

        task = e["task"] or "_default"
        bucket = results[task][e["model"]][e["resolution"]]
        bucket["n_seeds"] += 1

        acc = data.get("accuracy_metrics", {})
        eff = data.get("efficiency_metrics", {})

        for key in ACCURACY_KEYS:
            val = acc.get(key)
            if val is not None:
                bucket["accuracy"][key].append(val)

        for key in EFFICIENCY_KEYS:
            val = eff.get(key)
            if val is not None:
                bucket["efficiency"][key].append(val)

        # Per-class AUROC
        per_class = acc.get("per_class_auroc")
        if per_class and isinstance(per_class, dict):
            for cls_name, val in per_class.items():
                bucket["accuracy"][f"auroc_{cls_name}"].append(val)

    return results


# ── Printing ─────────────────────────────────────────────────────────────────

def print_task_results(task_name: str, task_data: dict, resolutions: list[int]):
    """Print accuracy and efficiency tables for one task."""
    models_present = sorted(task_data.keys(), key=lambda m: ["dinov3", "resnet18", "tiny_vit"].index(m) if m in ["dinov3", "resnet18", "tiny_vit"] else 99)

    label = task_name if task_name != "_default" else "(single-task)"
    print(f"\n{'#' * 100}")
    print(f"  TASK: {label}")
    print(f"{'#' * 100}")

    # ── Accuracy table ──
    print(f"\n  ACCURACY (mean ± std across seeds)")
    header = f"  {'Model':<12} {'Res':>4} {'n':>2}  {'AUROC':>18}  {'Accuracy':>18}  {'F1':>18}  {'Loss':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model in models_present:
        for res in resolutions:
            bucket = task_data[model].get(res)
            if bucket is None or bucket["n_seeds"] == 0:
                continue
            acc = bucket["accuracy"]
            n = bucket["n_seeds"]
            print(f"  {model:<12} {res:>4} {n:>2}"
                  f"  {fmt(acc.get('final_val_auroc', [])):>18}"
                  f"  {fmt(acc.get('final_val_acc', [])):>18}"
                  f"  {fmt(acc.get('final_val_f1', [])):>18}"
                  f"  {fmt(acc.get('final_val_loss', [])):>18}")
        print()

    # ── Efficiency table ──
    print(f"  EFFICIENCY")
    header = f"  {'Model':<12} {'Res':>4}  {'GFLOPs':>10}  {'Latency(ms)':>12}  {'Extract(s)':>12}  {'GPU Mem(MB)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model in models_present:
        for res in resolutions:
            bucket = task_data[model].get(res)
            if bucket is None or bucket["n_seeds"] == 0:
                continue
            eff = bucket["efficiency"]
            print(f"  {model:<12} {res:>4}"
                  f"  {fmt_eff(eff.get('encoder_gflops', []), 'gflops'):>10}"
                  f"  {fmt_eff(eff.get('encoder_latency_ms', []), 'latency'):>12}"
                  f"  {fmt_eff(eff.get('embedding_extraction_time_s', []), 'time'):>12}"
                  f"  {fmt_eff(eff.get('peak_gpu_memory_mb', []), 'memory'):>12}")
        print()

    # ── Per-class AUROC (if available) ──
    has_per_class = False
    for model in models_present:
        for res in resolutions:
            bucket = task_data[model].get(res)
            if bucket is None:
                continue
            if any(k.startswith("auroc_") for k in bucket["accuracy"]):
                has_per_class = True
                break

    if has_per_class:
        print(f"  PER-CLASS AUROC")
        for model in models_present:
            for res in resolutions:
                bucket = task_data[model].get(res)
                if bucket is None or bucket["n_seeds"] == 0:
                    continue
                class_keys = sorted(k for k in bucket["accuracy"] if k.startswith("auroc_"))
                if not class_keys:
                    continue
                print(f"    {model} @ {res}px:")
                for k in class_keys:
                    cls_name = k[6:]  # strip "auroc_"
                    print(f"      {cls_name:<30} {fmt(bucket['accuracy'][k])}")
        print()

    # ── Delta vs DINOv3 ──
    if "dinov3" in task_data:
        print(f"  AUROC DELTA vs DINOv3")
        header = f"  {'Model':<12} {'Res':>4}  {'DINOv3 AUROC':>14}  {'Student AUROC':>14}  {'Delta':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for model in models_present:
            if model == "dinov3":
                continue
            for res in resolutions:
                dino_bucket = task_data["dinov3"].get(res)
                student_bucket = task_data[model].get(res)
                if not dino_bucket or not student_bucket:
                    continue
                dino_vals = dino_bucket["accuracy"].get("final_val_auroc", [])
                student_vals = student_bucket["accuracy"].get("final_val_auroc", [])
                if not dino_vals or not student_vals:
                    continue
                dino_mean = np.mean(dino_vals)
                student_mean = np.mean(student_vals)
                delta = student_mean - dino_mean
                sign = "+" if delta >= 0 else ""
                print(f"  {model:<12} {res:>4}"
                      f"  {dino_mean:>14.4f}"
                      f"  {student_mean:>14.4f}"
                      f"  {sign}{delta:>9.4f}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Compare LP results across models, tasks, and seeds")
    parser.add_argument("results_dir", help="Path to results directory (e.g., results/derm-results)")
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS,
                        help="Resolutions to compare (default: 512 256 128 64)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to include (default: 42 123 456)")
    args = parser.parse_args()

    if args.seeds:
        global SEEDS
        SEEDS = args.seeds

    base_dir = args.results_dir
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Results directory: {os.path.abspath(base_dir)}")
    print(f"Seeds: {SEEDS}")
    print(f"Resolutions: {args.resolutions}")

    entries = discover_results(base_dir)
    if not entries:
        print("No result files found!", file=sys.stderr)
        sys.exit(1)

    # Summarize what was found
    tasks_found = sorted(set(e["task"] or "_default" for e in entries))
    models_found = sorted(set(e["model"] for e in entries))
    print(f"Found {len(entries)} result files: models={models_found}, tasks={tasks_found}")

    results = collect_results(entries, args.resolutions)

    # Print per task
    for task in sorted(results.keys(), key=lambda t: (t == "_default", t)):
        print_task_results(task, results[task], args.resolutions)


if __name__ == "__main__":
    main()
