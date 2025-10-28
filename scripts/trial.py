# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=all
"""
Trial: load ISIC (HF, 224×224), then save original vs. 112×112 downsampled.
"""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.data.isic_loader import ISICBaseDataset  # HF-backed
from src.transformation.transforms import ResolutionReductionTransform

def main():
    out_dir = Path("outputs/trial_isic112")
    out_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: no transform here, so we get true originals from HF (224×224)
    ds = ISICBaseDataset(
        repo_id="MKZuziak/ISIC_2019_224",
        split="train",
        transform=None,
    )
    print(f"Loaded {len(ds)} samples")

    # Build the reduction transform to 112×112
    reduce112 = ResolutionReductionTransform(
        target_resolution=(112, 112),
        restore_original_size=False,
    )

    for i in range(5):
        sample = ds[i]
        img = sample["image"]
        lbl = sample["label"]

        orig_path = out_dir / f"original_{i}.png"
        img.save(orig_path)

        reduced = reduce112(img)
        red_path = out_dir / f"reduced_{i}.png"
        reduced.save(red_path)

        print(f"[{i}] label={lbl} | original={img.size} → reduced={reduced.size} | "
              f"saved: {orig_path.name}, {red_path.name}")

    print(f"\n✅ Done. Check: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
