# src/evaluation/segmentation_visualization.py
# -*- coding: utf-8 -*-
"""
Visualization tools for segmentation predictions.

Provides functions to visualize predicted segmentation masks overlaid on images,
helping to understand model performance and debug issues.
"""
from __future__ import annotations
from typing import Optional, Union
from pathlib import Path

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from PIL import Image


def _denormalize_image(image: torch.Tensor) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization.

    Args:
        image: Normalized image tensor [3, H, W]

    Returns:
        Denormalized image as numpy array [H, W, 3] in [0, 255]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert to numpy and transpose to [H, W, 3]
    img_np = image.cpu().numpy().transpose(1, 2, 0)

    # Denormalize
    img_np = img_np * std + mean

    # Clip to [0, 1] and convert to [0, 255]
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    return img_np


def _prepare_mask(mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Prepare mask for visualization.

    Args:
        mask: Mask tensor [1, H, W] or [H, W]
        threshold: Threshold for binarization (default: 0.5)

    Returns:
        Binary mask as numpy array [H, W] in {0, 1}
    """
    # Remove channel dimension if present
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    # Convert to numpy
    mask_np = mask.cpu().numpy()

    # Binarize
    mask_binary = (mask_np > threshold).astype(np.uint8)

    return mask_binary


def save_segmentation_overlay(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    filename: Union[str, Path],
    threshold: float = 0.5,
) -> None:
    """
    Save side-by-side comparison of image, prediction, and ground truth.

    Creates a 4-panel figure:
    1. Original image
    2. Ground truth mask overlay
    3. Predicted mask overlay
    4. Error visualization (FP in red, FN in blue, TP in green)

    Args:
        image: Input image [3, H, W] (normalized)
        pred_mask: Predicted mask [1, H, W] or [H, W] (probabilities)
        gt_mask: Ground truth mask [1, H, W] or [H, W] (binary 0/1)
        filename: Output filename
        threshold: Threshold for binarizing predictions (default: 0.5)
    """
    # Denormalize image
    img_np = _denormalize_image(image)

    # Prepare masks
    pred_np = _prepare_mask(pred_mask, threshold)
    gt_np = _prepare_mask(gt_mask, threshold)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    # 2. Ground truth overlay
    axes[1].imshow(img_np)
    axes[1].imshow(gt_np, alpha=0.5, cmap='Greens', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis('off')

    # 3. Prediction overlay
    axes[2].imshow(img_np)
    axes[2].imshow(pred_np, alpha=0.5, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis('off')

    # 4. Error visualization
    # FP (false positive) = predicted but not in GT → red
    # FN (false negative) = in GT but not predicted → blue
    # TP (true positive) = both predicted and in GT → green
    error_map = np.zeros((*pred_np.shape, 3), dtype=np.uint8)

    # True positives (green)
    tp_mask = (pred_np == 1) & (gt_np == 1)
    error_map[tp_mask] = [0, 255, 0]

    # False positives (red)
    fp_mask = (pred_np == 1) & (gt_np == 0)
    error_map[fp_mask] = [255, 0, 0]

    # False negatives (blue)
    fn_mask = (pred_np == 0) & (gt_np == 1)
    error_map[fn_mask] = [0, 0, 255]

    axes[3].imshow(img_np)
    axes[3].imshow(error_map, alpha=0.5)
    axes[3].set_title("Errors (FP=Red, FN=Blue, TP=Green)", fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_segmentation_grid(
    images: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    filename: Union[str, Path],
    max_samples: int = 16,
    threshold: float = 0.5,
) -> None:
    """
    Save a grid of multiple segmentation examples.

    Creates a grid where each row shows:
    [Image | Ground Truth | Prediction]

    Args:
        images: Batch of images [B, 3, H, W]
        pred_masks: Batch of predicted masks [B, 1, H, W] or [B, H, W]
        gt_masks: Batch of ground truth masks [B, 1, H, W] or [B, H, W]
        filename: Output filename
        max_samples: Maximum number of samples to visualize (default: 16)
        threshold: Threshold for binarizing predictions (default: 0.5)
    """
    batch_size = min(images.size(0), max_samples)

    # Create figure
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    # Handle single sample case
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        # Get current sample
        img = images[i]
        pred = pred_masks[i]
        gt = gt_masks[i]

        # Denormalize image
        img_np = _denormalize_image(img)

        # Prepare masks
        pred_np = _prepare_mask(pred, threshold)
        gt_np = _prepare_mask(gt, threshold)

        # Column 1: Original image
        axes[i, 0].imshow(img_np)
        if i == 0:
            axes[i, 0].set_title("Image", fontsize=12)
        axes[i, 0].axis('off')

        # Column 2: Ground truth overlay
        axes[i, 1].imshow(img_np)
        axes[i, 1].imshow(gt_np, alpha=0.5, cmap='Greens', vmin=0, vmax=1)
        if i == 0:
            axes[i, 1].set_title("Ground Truth", fontsize=12)
        axes[i, 1].axis('off')

        # Column 3: Prediction overlay
        axes[i, 2].imshow(img_np)
        axes[i, 2].imshow(pred_np, alpha=0.5, cmap='Blues', vmin=0, vmax=1)
        if i == 0:
            axes[i, 2].set_title("Prediction", fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_prediction_errors(
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    filename: Union[str, Path],
    threshold: float = 0.5,
) -> None:
    """
    Visualize prediction errors with detailed breakdown.

    Creates a visualization highlighting:
    - True Positives (green): Correctly predicted foreground
    - False Positives (red): Incorrectly predicted as foreground
    - False Negatives (blue): Missed foreground regions
    - True Negatives (gray): Correctly predicted background

    Args:
        image: Input image [3, H, W]
        pred_mask: Predicted mask [1, H, W] or [H, W]
        gt_mask: Ground truth mask [1, H, W] or [H, W]
        filename: Output filename
        threshold: Threshold for binarizing predictions (default: 0.5)
    """
    # Denormalize image
    img_np = _denormalize_image(image)

    # Prepare masks
    pred_np = _prepare_mask(pred_mask, threshold)
    gt_np = _prepare_mask(gt_mask, threshold)

    # Create error map
    error_map = np.zeros((*pred_np.shape, 3), dtype=np.uint8)

    # True positives (green)
    tp_mask = (pred_np == 1) & (gt_np == 1)
    error_map[tp_mask] = [0, 255, 0]

    # False positives (red)
    fp_mask = (pred_np == 1) & (gt_np == 0)
    error_map[fp_mask] = [255, 0, 0]

    # False negatives (blue)
    fn_mask = (pred_np == 0) & (gt_np == 1)
    error_map[fn_mask] = [0, 0, 255]

    # True negatives (light gray)
    tn_mask = (pred_np == 0) & (gt_np == 0)
    error_map[tn_mask] = [200, 200, 200]

    # Compute statistics
    tp_count = tp_mask.sum()
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    tn_count = tn_mask.sum()
    total_pixels = pred_np.size

    # Compute metrics
    precision = tp_count / (tp_count + fp_count + 1e-6)
    recall = tp_count / (tp_count + fn_count + 1e-6)
    dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count + 1e-6)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    # 2. Error map overlay
    axes[1].imshow(img_np)
    axes[1].imshow(error_map, alpha=0.6)
    axes[1].set_title("Error Overlay", fontsize=14)
    axes[1].axis('off')

    # 3. Pure error map
    axes[2].imshow(error_map)
    axes[2].set_title("Error Map", fontsize=14)
    axes[2].axis('off')

    # Add legend and statistics
    legend_text = (
        f"TP (Green): {tp_count:,} ({100*tp_count/total_pixels:.1f}%)\n"
        f"FP (Red): {fp_count:,} ({100*fp_count/total_pixels:.1f}%)\n"
        f"FN (Blue): {fn_count:,} ({100*fn_count/total_pixels:.1f}%)\n"
        f"TN (Gray): {tn_count:,} ({100*tn_count/total_pixels:.1f}%)\n\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Dice: {dice:.4f}"
    )
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
