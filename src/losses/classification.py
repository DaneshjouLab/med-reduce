# src/losses/classification.py
# -*- coding: utf-8 -*-
from typing import Optional
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_loss(
    label_smoothing: float = 0.0,
    class_weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    """
    Standard cross-entropy with optional label smoothing and class weights.

    Args:
        label_smoothing: in [0,1). 0 = vanilla CE.
        class_weight: shape [C] tensor of per-class weights (on same device).
        ignore_index: targets with this index are ignored.
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Callable loss(logits [B,C], targets [B])
    """
    def _loss(logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    return _loss


def bce_with_logits_loss(
    pos_weight: Optional[Tensor] = None,
    reduction: str = "mean",
    ignore_value: float = -1.0,
):
    """
    Binary cross-entropy with logits for multi-label classification.

    Labels equal to ``ignore_value`` (default -1) are masked out so they
    contribute zero loss.  This handles CheXpert-style uncertain labels.

    Args:
        pos_weight: shape [C] tensor of per-label positive weights.
        reduction: 'none' | 'mean' | 'sum'
        ignore_value: Label value to ignore (default -1.0 for CheXpert uncertain).

    Returns:
        Callable loss(logits [B,C], targets [B,C])
    """
    def _loss(logits: Tensor, targets: Tensor) -> Tensor:
        # Mask: 1 where valid, 0 where uncertain
        mask = (targets != ignore_value).float()
        # Replace ignored values with 0 so BCE doesn't produce NaN
        safe_targets = targets.clamp(min=0.0)
        # Per-element loss
        per_element = F.binary_cross_entropy_with_logits(
            logits, safe_targets, pos_weight=pos_weight, reduction="none",
        )
        # Zero out loss for ignored labels
        masked_loss = per_element * mask
        if reduction == "mean":
            # Average over valid entries only
            return masked_loss.sum() / mask.sum().clamp(min=1.0)
        elif reduction == "sum":
            return masked_loss.sum()
        return masked_loss
    return _loss
