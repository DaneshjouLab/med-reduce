# src/losses/distillation.py
# -*- coding: utf-8 -*-
from typing import Dict
import torch.nn.functional as F
from torch import Tensor


def cosine_loss(reduction: str = "mean"):
    """
    Cosine embedding loss for feature alignment.
    Expects student and teacher embeddings with shape [B, D].
    """
    def _loss(s_embed: Tensor, t_embed: Tensor) -> Tensor:
        s_norm = F.normalize(s_embed, dim=-1)
        t_norm = F.normalize(t_embed, dim=-1)
        sim = (s_norm * t_norm).sum(dim=-1)
        loss = 1.0 - sim  # minimize (1 - cos)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss
    return _loss


def kl_divergence_loss(temperature: float = 1.0, reduction: str = "batchmean"):
    """
    KL divergence on logits with temperature scaling (Hinton et al., 2015).
    """
    T = float(temperature)
    T2 = T * T

    def _loss(s_logits: Tensor, t_logits: Tensor) -> Tensor:
        log_p_s = F.log_softmax(s_logits / T, dim=-1)
        p_t = F.softmax(t_logits / T, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction=reduction)
        return kl * T2
    return _loss


def hybrid_distillation_loss(
    alpha: float = 0.5,          # cosine(embeds)
    beta: float = 0.5,           # KL(logits)
    gamma: float = 0.0,          # optional CE to ground truth
    temperature: float = 2.0,
    ce_loss_fn=None,
):
    """
    Combined distillation objective:
        L = alpha * (1 - cos(s_embed, t_embed))
          + beta * KL(softmax(s/T), softmax(t/T)) * T^2
          + gamma * CE(s_logits, y)

    Returns:
        Callable loss_dict(outputs) where outputs = {
          "s_logits", "t_logits", "s_embed", "t_embed", "targets"
        }
    """
    cos_fn = cosine_loss(reduction="mean")
    kl_fn = kl_divergence_loss(temperature=temperature, reduction="batchmean")

    def _loss(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        s_logits = outputs["s_logits"]
        t_logits = outputs["t_logits"]
        s_embed  = outputs["s_embed"]
        t_embed  = outputs["t_embed"]

        l_cos = cos_fn(s_embed, t_embed) if alpha > 0 else s_logits.new_zeros(())
        l_kl  = kl_fn(s_logits, t_logits) if beta  > 0 else s_logits.new_zeros(())

        if gamma > 0 and ce_loss_fn is not None:
            targets = outputs["targets"]
            l_ce = ce_loss_fn(s_logits, targets)
        else:
            l_ce = s_logits.new_zeros(())

        total = alpha * l_cos + beta * l_kl + gamma * l_ce
        return {
            "loss_total": total,
            "loss_cos": l_cos.detach(),
            "loss_kl": l_kl.detach(),
            "loss_ce": l_ce.detach(),
        }
    return _loss
