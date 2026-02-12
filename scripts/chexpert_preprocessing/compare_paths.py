"""Compare Path overlap across all pairs of CSV files in this directory.

  - train.csv vs train_cheXbert.csv — 100% overlap (all 223,414 paths match). They're the same images, just with different path prefixes.
  - test_chex.csv is fully contained in both train files (2000 shared).
  - valid.csv has zero overlap with any other file — it's a completely separate split.

  """

import glob
import os
from itertools import combinations

import pandas as pd

CSV_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load paths from every CSV ────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))

LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

file_paths: dict[str, set[str]] = {}
file_dfs: dict[str, pd.DataFrame] = {}
for f in csv_files:
    df = pd.read_csv(f)
    name = os.path.basename(f)
    df["_key"] = (
        df["Path"].str.lower().str.extract(r"(patient\d+/.*)")[0]
    )
    df = df.dropna(subset=["_key"])
    file_paths[name] = set(df["_key"])
    file_dfs[name] = df.set_index("_key")
    total = len(df)
    unique = len(file_paths[name])
    dup = total - unique
    print(f"{name}: {total} rows, {unique} unique paths, {dup} internal duplicates")

print()

# ── Pairwise comparison ──────────────────────────────────────────────────────
print(f"{'Pair':<55} {'Shared':>8}  {'Only A':>8}  {'Only B':>8}")
print("-" * 85)

for (name_a, set_a), (name_b, set_b) in combinations(file_paths.items(), 2):
    shared = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    label = f"{name_a}  vs  {name_b}"
    print(f"{label:<55} {len(shared):>8}  {len(only_a):>8}  {len(only_b):>8}")

    # Compare labels for shared paths
    if shared:
        df_a = file_dfs[name_a]
        df_b = file_dfs[name_b]
        common_cols = [c for c in LABEL_COLS if c in df_a.columns and c in df_b.columns]
        if common_cols:
            shared_idx = sorted(shared)
            a_vals = df_a.loc[shared_idx, common_cols]
            b_vals = df_b.loc[shared_idx, common_cols]
            diff_mask = a_vals != b_vals
            # Count per-column disagreements (treating NaN==NaN as agree)
            both_nan = a_vals.isna() & b_vals.isna()
            diff_mask = diff_mask & ~both_nan
            n_diff = diff_mask.sum()
            total_shared = len(shared_idx)
            cols_with_diff = n_diff[n_diff > 0]
            if cols_with_diff.empty:
                print(f"  -> Labels: all {len(common_cols)} columns IDENTICAL for {total_shared} shared paths")
            else:
                print(f"  -> Label differences across {total_shared} shared paths:")
                for col, cnt in cols_with_diff.items():
                    print(f"     {col:<30} {cnt:>6} differ ({cnt/total_shared*100:.1f}%)")
            print()
