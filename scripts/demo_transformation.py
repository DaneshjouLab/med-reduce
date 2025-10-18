#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-exception-caught

"""
Demo script to apply ResolutionReductionTransform to ISIC dataset images and save results.
"""

import sys
from pathlib import Path
from PIL import Image

# Add src to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from src.data.isic_loader import ISICBaseDataset
from src.transformation.transforms import ResolutionReductionTransform

def main():
    print("🔄 Loading ISIC dataset from HuggingFace...")
    try:
        hf_dataset = load_dataset("MKZuziak/ISIC_2019_224", split="train")
        isic_dataset = ISICBaseDataset(hf_dataset)
        print(f"✅ Loaded {len(isic_dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("Make sure you have 'datasets' installed: pip install datasets")
        return

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")

    print("\nApplying ResolutionReductionTransform to first 5 images...")

    # Create the transform
    resolution_transform = ResolutionReductionTransform()  # Random reduction factor

    # Process first 5 images
    num_images = min(5, len(isic_dataset))

    for i in range(num_images):
        print(f"\nProcessing image {i+1}/{num_images}...")

        try:
            # Get the original image
            sample = isic_dataset[i]
            original_image = sample["image"]
            label = sample["label"]

            print(f"Original size: {original_image.size}, Label: {label}")

            # Save original image
            original_path = output_dir / f"original_{i}.png"
            original_image.save(original_path)
            print(f"  💾 Saved original: {original_path}")

            # Apply resolution reduction transform
            transformed_image = resolution_transform(original_image)
            print(f"  🔄 Transformed size: {transformed_image.size}")

            # Save transformed image
            transformed_path = output_dir / f"resolution_reduced_{i}.png"
            transformed_image.save(transformed_path)
            print(f"  💾 Saved transformed: {transformed_path}")

        except Exception as e:
            print(f"  ❌ Error processing image {i}: {e}")
            continue

    print(f"\n✅ All images saved to: {output_dir.absolute()}")

    # Show what reduction factors were used (they're random)
    print("\nTesting with fixed reduction factors...")
    for factor in [0.25, 0.5, 0.75]:
        print(f"\nTesting reduction factor: {factor}")
        try:
            fixed_transform = ResolutionReductionTransform(reduction_factor=factor)

            # Use first image for this demo
            sample = isic_dataset[0]
            original_image = sample["image"]

            transformed = fixed_transform(original_image)
            output_path = output_dir / f"fixed_reduction_{factor}_{0}.png"
            transformed.save(output_path)
            print(f"  💾 Saved: {output_path}")

        except Exception as e:
            print(f"  ❌ Error with factor {factor}: {e}")

    print(f"\n🎉 Demo completed! Check {output_dir.absolute()} for results.")

if __name__ == "__main__":
    main()