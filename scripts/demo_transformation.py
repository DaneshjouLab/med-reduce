# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-exception-caught,wrong-import-position,import-error

"""
Apply ResolutionReductionTransform to a few ISIC images and save:
- original_<idx>.png
- resolution_reduced_<idx>.png  (reduced-size, not upsampled)
"""

import sys
from pathlib import Path
from typing import Iterable, Optional

# Add src to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from src.data.isic_loader import ISICBaseDataset
from src.transformation.transforms import ResolutionReductionTransform


def load_isic_dataset() -> Optional[ISICBaseDataset]:
    """Load ISIC dataset from HuggingFace."""
    print("🔄 Loading ISIC dataset from HuggingFace...")
    try:
        hf_dataset = load_dataset("MKZuziak/ISIC_2019_224", split="train")
        ds = ISICBaseDataset(hf_dataset)
        print(f"✅ Loaded {len(ds)} samples")
        return ds
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("Make sure you have 'datasets' installed: pip install datasets")
        return None


def process_images(
    dataset: ISICBaseDataset,
    indices: Iterable[int],
    transform: ResolutionReductionTransform,
    output_dir: Path,
    save_prefix: str = "resolution_reduced",
) -> int:
    """
    For each index: save original and one reduced image using `transform`.
    Assumes transform returns the reduced-size image (no upsample).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")

    processed = 0
    for idx in indices:
        try:
            sample = dataset[idx]
            original = sample["image"]
            label = sample.get("label", None)
            print(f"\n🖼️  Image {idx} | Original size: {original.size} | Label: {label}")

            # Save original
            orig_path = output_dir / f"original_{idx}.png"
            original.save(orig_path)
            print(f"  💾 Saved original → {orig_path.name}")

            # Save reduced (actual transformed size)
            reduced = transform(original)
            out_path = output_dir / f"{save_prefix}_{idx}.png"
            reduced.save(out_path)
            print(f"  🔧 Reduced to: {reduced.size}")
            print(f"  💾 Saved reduced → {out_path.name}")

            processed += 1

        except Exception as e:
            print(f"  ❌ Error at index {idx}: {e}")

    return processed


def main():
    ds = load_isic_dataset()
    if ds is None:
        return

    output_dir = Path("outputs")
    num_images = min(5, len(ds))
    indices = range(num_images)

    # Use target size and DO NOT upsample back in the transform
    # Ensure your class has restore_original_size=False (default) as we discussed.
    resolution_transform = ResolutionReductionTransform(
        target_resolution=(54, 54),
        restore_original_size=False
    )

    print("\n▶️ Saving originals and reduced versions for first 5 images...")
    n = process_images(
        dataset=ds,
        indices=indices,
        transform=resolution_transform,
        output_dir=output_dir,
        save_prefix="resolution_reduced",
    )
    print(f"\n✅ Done. Saved {n} reduced images. Check: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
