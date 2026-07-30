#!/usr/bin/env python3
"""
Summarize dataset sizes (total, train, test) per domain from split metadata.

Usage:
    python scripts/dataset_summary.py
"""

import json
from pathlib import Path

RESULTS_ROOT = Path("results")

# Splits now live in a single shared results/splits/ (teacher-agnostic), keyed by
# the dataset identifier the pipeline uses.
DATASETS = {
    "Dermatology": {
        "splits": {"ISIC (images)": "images"},
    },
    "Radiology": {
        "splits": {"CheXpert": "combined_train_valid_chexpert_v1.0"},
    },
    "Pathology": {
        "splits": {
            "LUAD vs LUSC": "tcga_luad_vs_lusc",
            "LGG vs GBM": "tcga_lgg_vs_gbm",
            "KRAS": "tcga_kras",
            "TP53": "tcga_tp53",
            "EGFR": "tcga_egfr",
        },
    },
}

SEED = "seed_42"  # Splits are the same across seeds


def main():
    rows = []

    for domain, config in DATASETS.items():
        domain_dir = RESULTS_ROOT / "splits"
        for display_name, split_name in config["splits"].items():
            meta_path = domain_dir / split_name / SEED / "metadata.json"
            if not meta_path.exists():
                rows.append((domain, display_name, "—", "—", "—"))
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            total = meta["dataset_size"]
            train = meta["split_sizes"]["train"]
            test = meta["split_sizes"]["test"]
            rows.append((domain, display_name, total, train, test))

    # Print table
    headers = ("Domain", "Dataset", "Total", "Train", "Test")
    col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

    header_line = "  ".join(f"{h:<{col_w[i]}}" for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        line = "  ".join(f"{str(v):>{col_w[i]}}" if i >= 2 else f"{str(v):<{col_w[i]}}" for i, v in enumerate(row))
        print(line)

    # Save
    out_path = RESULTS_ROOT / "dataset_summary.txt"
    with open(out_path, "w") as f:
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        for row in rows:
            line = "  ".join(f"{str(v):>{col_w[i]}}" if i >= 2 else f"{str(v):<{col_w[i]}}" for i, v in enumerate(row))
            f.write(line + "\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
