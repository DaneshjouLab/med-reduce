# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
Classification loss functions for model training.

This module provides common loss functions used for classification tasks,
including cross-entropy loss with various options like label smoothing
and class weighting.
"""

# src/losses/classification.py
# -*- coding: utf-8 -*-
from typing import Optional
# pylint: disable=import-error
import torch.nn.functional as F
from torch import Tensor  # pylint: disable=import-error


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
