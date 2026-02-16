# src/losses/distillation.py
# -*- coding: utf-8 -*-
"""Embedding distillation loss for knowledge distillation."""

import torch.nn.functional as F
from torch import Tensor


def embedding_distillation_loss(alpha: float = 0.5):
    """
    Combined MSE + cosine similarity loss for embedding distillation.

    Loss = alpha * MSE(student, teacher) + (1 - alpha) * (1 - cosine_similarity(student, teacher)).mean()

    Args:
        alpha: Balance between MSE (alpha=1) and cosine (alpha=0).

    Returns:
        Callable loss(student_emb [B,D], teacher_emb [B,D]) -> scalar Tensor
    """
    def _loss(student_emb: Tensor, teacher_emb: Tensor) -> Tensor:
        mse = F.mse_loss(student_emb, teacher_emb)
        cosine = (1 - F.cosine_similarity(student_emb, teacher_emb)).mean()
        return alpha * mse + (1 - alpha) * cosine
    return _loss
