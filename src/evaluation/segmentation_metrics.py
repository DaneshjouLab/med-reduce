# src/evaluation/segmentation_metrics.py
# -*- coding: utf-8 -*-
"""
Segmentation-specific metrics for evaluating mask predictions.

Provides metrics commonly used in medical image segmentation:
- Dice coefficient (F1 for segmentation)
- Intersection over Union (IoU / Jaccard index)
- Pixel accuracy

All metrics operate on binary or multi-class segmentation masks.
"""
from __future__ import annotations
from typing import Dict, Union

import torch


def compute_dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    per_class: bool = False,
    threshold: float = 0.5,
) -> Union[float, torch.Tensor]:
    """
    Compute Dice coefficient (F1 score for segmentation).

    The Dice coefficient measures overlap between predicted and ground truth masks:
        Dice = 2 * |pred ∩ target| / (|pred| + |target|)

    Args:
        preds: Predicted probabilities [B, num_classes, H, W] or [B, H, W]
        targets: Ground truth masks [B, num_classes, H, W] or [B, H, W]
               Values should be 0 or 1 (binary masks)
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
        per_class: If True, return per-class Dice scores [C]. If False, return mean (default: False)
        threshold: Threshold for binarizing predictions (default: 0.5)

    Returns:
        Dice coefficient as float (mean across batch and classes) or
        Tensor of shape [C] if per_class=True
    """
    # Ensure 4D tensors [B, C, H, W]
    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    # Threshold predictions to binary
    preds_binary = (preds > threshold).float()

    # Flatten spatial dimensions [B, C, H*W]
    preds_flat = preds_binary.view(preds_binary.size(0), preds_binary.size(1), -1)
    targets_flat = targets.view(targets.size(0), targets.size(1), -1)

    # Compute intersection and union
    intersection = (preds_flat * targets_flat).sum(dim=2)  # [B, C]
    union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2)  # [B, C]

    # Compute Dice coefficient per sample and class
    dice = (2.0 * intersection + smooth) / (union + smooth)  # [B, C]

    if per_class:
        return dice.mean(dim=0)  # [C] - average across batch
    else:
        return dice.mean().item()  # scalar - average across batch and classes


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    per_class: bool = False,
    threshold: float = 0.5,
) -> Union[float, torch.Tensor]:
    """
    Compute Intersection over Union (IoU / Jaccard index).

    IoU measures overlap between predicted and ground truth masks:
        IoU = |pred ∩ target| / |pred ∪ target|

    Args:
        preds: Predicted probabilities [B, num_classes, H, W] or [B, H, W]
        targets: Ground truth masks [B, num_classes, H, W] or [B, H, W]
               Values should be 0 or 1 (binary masks)
        smooth: Smoothing factor to avoid division by zero (default: 1e-6)
        per_class: If True, return per-class IoU scores [C]. If False, return mean (default: False)
        threshold: Threshold for binarizing predictions (default: 0.5)

    Returns:
        IoU as float (mean across batch and classes) or
        Tensor of shape [C] if per_class=True
    """
    # Ensure 4D tensors [B, C, H, W]
    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    # Threshold predictions to binary
    preds_binary = (preds > threshold).float()

    # Flatten spatial dimensions [B, C, H*W]
    preds_flat = preds_binary.view(preds_binary.size(0), preds_binary.size(1), -1)
    targets_flat = targets.view(targets.size(0), targets.size(1), -1)

    # Compute intersection and union
    intersection = (preds_flat * targets_flat).sum(dim=2)  # [B, C]
    union = preds_flat.sum(dim=2) + targets_flat.sum(dim=2) - intersection  # [B, C]

    # Compute IoU per sample and class
    iou = (intersection + smooth) / (union + smooth)  # [B, C]

    if per_class:
        return iou.mean(dim=0)  # [C] - average across batch
    else:
        return iou.mean().item()  # scalar - average across batch and classes


def compute_pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute overall pixel-wise accuracy.

    Pixel accuracy measures the percentage of correctly classified pixels:
        Accuracy = (# correct pixels) / (# total pixels)

    Args:
        preds: Predicted probabilities [B, num_classes, H, W] or [B, H, W]
        targets: Ground truth masks [B, num_classes, H, W] or [B, H, W]
               Values should be 0 or 1 (binary masks)
        threshold: Threshold for binarizing predictions (default: 0.5)

    Returns:
        Pixel accuracy as float in [0, 1]
    """
    # Threshold predictions to binary
    preds_binary = (preds > threshold).float()

    # Compute correct predictions
    correct = (preds_binary == targets).float()

    # Return mean accuracy across all pixels
    return correct.mean().item()


def compute_segmentation_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Unified function to compute all segmentation metrics.

    This is the main function to use for evaluating segmentation models.
    It applies sigmoid activation to logits and computes all standard metrics.

    Args:
        logits: Raw model outputs [B, num_classes, H, W] - logits before sigmoid
        masks: Ground truth masks [B, H, W] or [B, num_classes, H, W]
              Values should be 0 or 1 (binary masks)
        threshold: Threshold for binarizing predictions (default: 0.5)

    Returns:
        Dictionary with keys:
            - "dice": Dice coefficient (F1 score)
            - "iou": Intersection over Union (Jaccard index)
            - "pixel_acc": Pixel-wise accuracy

    Example:
        >>> outputs = model(images, labels=masks)
        >>> metrics = compute_segmentation_metrics(outputs.logits, masks)
        >>> print(f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
    """
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Compute all metrics
    dice = compute_dice_coefficient(probs, masks, threshold=threshold)
    iou = compute_iou(probs, masks, threshold=threshold)
    pixel_acc = compute_pixel_accuracy(probs, masks, threshold=threshold)

    return {
        "dice": dice,
        "iou": iou,
        "pixel_acc": pixel_acc,
    }
