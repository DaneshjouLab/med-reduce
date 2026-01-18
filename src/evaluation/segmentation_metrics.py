# src/evaluation/segmentation_metrics.py
# -*- coding: utf-8 -*-
"""
Segmentation metrics for evaluating semantic segmentation models.

Provides efficient GPU-accelerated computation of:
- Dice coefficient (F1 score for segmentation)
- Intersection over Union (IoU / Jaccard index)
- Pixel accuracy

All metrics support batch-wise computation to avoid memory issues
with large validation sets.
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
from torch import Tensor


def compute_dice_coefficient(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice coefficient (F1 score for segmentation).

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        logits: Predicted logits [N, C, H, W] or [N, 1, H, W] for binary
        targets: Ground truth masks [N, C, H, W] or [N, 1, H, W] or [N, H, W]
        threshold: Threshold for converting probabilities to binary predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient (scalar float)
    """
    # Convert logits to probabilities and threshold
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Handle different target shapes
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    # Flatten spatial dimensions for computation
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def compute_iou(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard index).

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        logits: Predicted logits [N, C, H, W] or [N, 1, H, W] for binary
        targets: Ground truth masks [N, C, H, W] or [N, 1, H, W] or [N, H, W]
        threshold: Threshold for converting probabilities to binary predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU score (scalar float)
    """
    # Convert logits to probabilities and threshold
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Handle different target shapes
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    # Flatten spatial dimensions for computation
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_pixel_accuracy(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute pixel-wise accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        logits: Predicted logits [N, C, H, W] or [N, 1, H, W] for binary
        targets: Ground truth masks [N, C, H, W] or [N, 1, H, W] or [N, H, W]
        threshold: Threshold for converting probabilities to binary predictions

    Returns:
        Pixel accuracy (scalar float)
    """
    # Convert logits to probabilities and threshold
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Handle different target shapes
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    # Flatten and compute accuracy
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    correct = (preds_flat == targets_flat).sum()
    total = targets_flat.numel()

    accuracy = correct.float() / total
    return accuracy.item()


def compute_segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics efficiently.

    This function computes Dice, IoU, and pixel accuracy in a single pass
    to avoid redundant computations.

    Args:
        logits: Predicted logits [N, C, H, W] or [N, 1, H, W] for binary
        targets: Ground truth masks [N, C, H, W] or [N, 1, H, W] or [N, H, W]
        threshold: Threshold for converting probabilities to binary predictions

    Returns:
        Dictionary with keys: 'dice', 'iou', 'pixel_acc'
    """
    smooth = 1e-6

    # Convert logits to probabilities and threshold
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Handle different target shapes
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    # Flatten spatial dimensions for computation
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # Compute intersection and sums (shared across metrics)
    intersection = (preds_flat * targets_flat).sum()
    preds_sum = preds_flat.sum()
    targets_sum = targets_flat.sum()

    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (preds_sum + targets_sum + smooth)

    # IoU
    union = preds_sum + targets_sum - intersection
    iou = (intersection + smooth) / (union + smooth)

    # Pixel accuracy
    correct = (preds_flat == targets_flat).sum()
    total = targets_flat.numel()
    pixel_acc = correct.float() / total

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "pixel_acc": pixel_acc.item(),
    }


class RunningSegmentationMetrics:
    """
    Accumulator for computing segmentation metrics incrementally.

    Use this to compute metrics batch-by-batch without storing all
    predictions in memory. Much more memory-efficient for large
    validation sets.

    Example:
        metrics = RunningSegmentationMetrics()
        for batch in val_loader:
            logits = model(batch['images'])
            metrics.update(logits, batch['masks'])
        results = metrics.compute()
    """

    def __init__(self, threshold: float = 0.5, device: Optional[torch.device] = None):
        """
        Initialize running metrics accumulator.

        Args:
            threshold: Threshold for converting probabilities to binary predictions
            device: Device to store accumulators on (defaults to CPU)
        """
        self.threshold = threshold
        self.device = device or torch.device('cpu')
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.intersection_sum = torch.tensor(0.0, device=self.device)
        self.preds_sum = torch.tensor(0.0, device=self.device)
        self.targets_sum = torch.tensor(0.0, device=self.device)
        self.correct_pixels = torch.tensor(0, device=self.device, dtype=torch.long)
        self.total_pixels = torch.tensor(0, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def update(self, logits: Tensor, targets: Tensor):
        """
        Update running statistics with a new batch.

        Args:
            logits: Predicted logits [N, C, H, W]
            targets: Ground truth masks [N, C, H, W] or [N, H, W]
        """
        # Move to accumulator device if needed
        logits = logits.to(self.device)
        targets = targets.to(self.device)

        # Convert logits to predictions
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        # Handle different target shapes
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # Flatten and accumulate
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        self.intersection_sum += (preds_flat * targets_flat).sum()
        self.preds_sum += preds_flat.sum()
        self.targets_sum += targets_flat.sum()
        self.correct_pixels += (preds_flat == targets_flat).sum()
        self.total_pixels += targets_flat.numel()

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated statistics.

        Returns:
            Dictionary with keys: 'dice', 'iou', 'pixel_acc'
        """
        smooth = 1e-6

        # Dice coefficient
        dice = (2.0 * self.intersection_sum + smooth) / (
            self.preds_sum + self.targets_sum + smooth
        )

        # IoU
        union = self.preds_sum + self.targets_sum - self.intersection_sum
        iou = (self.intersection_sum + smooth) / (union + smooth)

        # Pixel accuracy
        pixel_acc = self.correct_pixels.float() / self.total_pixels

        return {
            "dice": dice.item(),
            "iou": iou.item(),
            "pixel_acc": pixel_acc.item(),
        }
