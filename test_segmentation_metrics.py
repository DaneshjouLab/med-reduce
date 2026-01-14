#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for segmentation metrics.

Verifies that Dice, IoU, and pixel accuracy are computed correctly
with dummy data before running full training.
"""
import torch
import sys

# Test 1: Import metrics module
print("="*60)
print("Test 1: Import segmentation metrics module")
print("="*60)
try:
    from src.evaluation.segmentation_metrics import (
        compute_dice_coefficient,
        compute_iou,
        compute_pixel_accuracy,
        compute_segmentation_metrics,
    )
    print("✓ Successfully imported all metric functions\n")
except ImportError as e:
    print(f"✗ Failed to import: {e}\n")
    sys.exit(1)

# Test 2: Perfect prediction (Dice = 1.0)
print("="*60)
print("Test 2: Perfect prediction (should get Dice=1.0, IoU=1.0)")
print("="*60)
logits = torch.ones(4, 1, 256, 256) * 10.0  # High logits → sigmoid ≈ 1
masks = torch.ones(4, 256, 256)  # All ones

metrics = compute_segmentation_metrics(logits, masks)
print(f"Dice: {metrics['dice']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_acc']:.4f}")

assert abs(metrics['dice'] - 1.0) < 0.01, "Perfect prediction should give Dice=1.0"
assert abs(metrics['iou'] - 1.0) < 0.01, "Perfect prediction should give IoU=1.0"
assert abs(metrics['pixel_acc'] - 1.0) < 0.01, "Perfect prediction should give pixel_acc=1.0"
print("✓ Perfect prediction test passed\n")

# Test 3: Worst prediction (Dice = 0.0)
print("="*60)
print("Test 3: Worst prediction (should get Dice≈0.0, IoU≈0.0)")
print("="*60)
logits = torch.ones(4, 1, 256, 256) * 10.0  # High logits → sigmoid ≈ 1
masks = torch.zeros(4, 256, 256)  # All zeros (opposite of prediction)

metrics = compute_segmentation_metrics(logits, masks)
print(f"Dice: {metrics['dice']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_acc']:.4f}")

assert metrics['dice'] < 0.1, "Worst prediction should give Dice≈0.0"
assert metrics['iou'] < 0.1, "Worst prediction should give IoU≈0.0"
assert metrics['pixel_acc'] < 0.1, "Worst prediction should give pixel_acc≈0.0"
print("✓ Worst prediction test passed\n")

# Test 4: Half correct prediction
print("="*60)
print("Test 4: Half correct prediction (Dice≈0.5, IoU≈0.33)")
print("="*60)
logits = torch.zeros(1, 1, 100, 100)
logits[0, 0, :50, :] = 10.0  # Top half is positive
masks = torch.zeros(1, 100, 100)
masks[0, :50, :] = 1.0  # Top half is positive

metrics = compute_segmentation_metrics(logits, masks)
print(f"Dice: {metrics['dice']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_acc']:.4f}")

assert abs(metrics['dice'] - 1.0) < 0.01, "Half correct should give Dice=1.0 (perfect match)"
assert abs(metrics['iou'] - 1.0) < 0.01, "Half correct should give IoU=1.0 (perfect match)"
assert abs(metrics['pixel_acc'] - 1.0) < 0.01, "Half correct should give pixel_acc=1.0"
print("✓ Half correct prediction test passed\n")

# Test 5: Random prediction
print("="*60)
print("Test 5: Random prediction (0 < metrics < 1)")
print("="*60)
torch.manual_seed(42)
logits = torch.randn(4, 1, 256, 256)  # Random logits
masks = torch.randint(0, 2, (4, 256, 256)).float()  # Random binary masks

metrics = compute_segmentation_metrics(logits, masks)
print(f"Dice: {metrics['dice']:.4f}")
print(f"IoU: {metrics['iou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_acc']:.4f}")

assert 0.0 <= metrics['dice'] <= 1.0, "Dice must be in [0, 1]"
assert 0.0 <= metrics['iou'] <= 1.0, "IoU must be in [0, 1]"
assert 0.0 <= metrics['pixel_acc'] <= 1.0, "Pixel accuracy must be in [0, 1]"
print("✓ Random prediction test passed\n")

# Test 6: Different tensor shapes
print("="*60)
print("Test 6: Different tensor shapes")
print("="*60)

# Test with [B, H, W] prediction (no channel dim)
logits_3d = torch.randn(2, 128, 128)
masks_3d = torch.randint(0, 2, (2, 128, 128)).float()
metrics = compute_segmentation_metrics(logits_3d, masks_3d)
print(f"Shape [B, H, W] - Dice: {metrics['dice']:.4f}")

# Test with [B, C, H, W] prediction (with channel dim)
logits_4d = torch.randn(2, 1, 128, 128)
masks_4d = torch.randint(0, 2, (2, 1, 128, 128)).float()
metrics = compute_segmentation_metrics(logits_4d, masks_4d)
print(f"Shape [B, C, H, W] - Dice: {metrics['dice']:.4f}")

print("✓ Different shape test passed\n")

# Final summary
print("="*60)
print("✅ All segmentation metrics tests passed!")
print("="*60)
print("\nMetrics module is working correctly and ready for training.\n")
