"""Build a combined CSV from train.csv and valid.csv.

Steps:
  1. Load train.csv and valid.csv
  2. Concatenate and drop duplicate paths
  3. Keep only frontal images (path contains 'frontal')
  4. Keep one image per patient (first occurrence)
  5. Save to train_valid_combined.csv
"""

import os

import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load ──────────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DIR, "train.csv"))
valid = pd.read_csv(os.path.join(DIR, "valid.csv"))
print(f"train: {len(train)} rows")
print(f"valid: {len(valid)} rows")

# ── 2. Combine & deduplicate ─────────────────────────────────────────────────
df = pd.concat([train, valid], ignore_index=True)
df = df.drop_duplicates(subset="Path")
print(f"Combined (unique paths): {len(df)} rows")

# ── 3. Frontal only ──────────────────────────────────────────────────────────
df = df[df["Path"].str.lower().str.contains("frontal")]
print(f"After frontal filter: {len(df)} rows")

# ── 4. One per patient (keep first) ──────────────────────────────────────────
df["_patient"] = df["Path"].str.lower().str.extract(r"(patient\d+)")[0]
df = df.drop_duplicates(subset="_patient", keep="first").drop(columns="_patient")
print(f"After one-per-patient: {len(df)} rows")

# ── 5. Add image_id (flat filename) ───────────────────────────────────────────
df["image_id"] = (
    df["Path"]
    .str.lower()
    .str.extract(r"(patient\d+/.*)")[0]
    .str.replace("/", "_", regex=False)
)

# ── 6. Save ──────────────────────────────────────────────────────────────────
out = os.path.join(DIR, "train_valid_combined.csv")
df.to_csv(out, index=False)
print(f"Saved to {out}")
