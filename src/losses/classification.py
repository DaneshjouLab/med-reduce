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
):
    """
    Binary cross-entropy with logits for multi-label classification.

    Args:
        pos_weight: shape [C] tensor of per-label positive weights.
        reduction: 'none' | 'mean' | 'sum'

    Returns:
        Callable loss(logits [B,C], targets [B,C])
    """
    def _loss(logits: Tensor, targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction=reduction,
        )
    return _loss
