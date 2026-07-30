# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""BiomedCLIP vision-tower wrapper used as an alternative (non-DINO, medical
vision-language) teacher for the Med-REDUCE framework.

Added in response to reviewer requests (R1.1 / R2.6) to test whether the
resolution-robustness and embedding-distillation findings generalize beyond a
DINO-style teacher and to compare against a medical foundation model.

Design notes
------------
* BiomedCLIP's vision tower is a ViT-B/16 pretrained at 224x224. Its positional
  embeddings are tied to the 14x14 = 196 patch grid, so inputs must be 224x224.
  The Med-REDUCE degradation protocol controls *information* via the downsample
  target R and then upsamples back to a native encoder-input resolution; for
  BiomedCLIP that native resolution is 224 (set ``data.native_resolution: 224``
  in the configs). As a safety net this wrapper also resizes any non-224 input
  to 224 internally, so it stays correct even if run with the default 512 native
  resolution.
* Normalization: the Med-REDUCE data pipeline feeds every encoder identical
  ImageNet-normalized inputs (see ``TabularDataModulePersistent.setup``). We keep
  BiomedCLIP on the same normalization for protocol consistency rather than
  swapping in CLIP normalization — the comparison is "same inputs, different
  encoder". This mirrors how the DINOv3 teacher is fed in the original paper.
* Output: a single global image embedding of dimension 512 (the projected
  CLIP image feature), consumed by the linear-probe and distillation pipelines
  exactly like the DINOv3 CLS embedding.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default open_clip hub id for BiomedCLIP.
BIOMEDCLIP_HF_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMEDCLIP_INPUT_SIZE = 224
BIOMEDCLIP_EMBED_DIM = 512


class _EmbedDimConfig:
    """Lightweight stand-in for a HF ``config`` object.

    Some call sites (e.g. ``get_embedding_dim``) probe ``model.config.hidden_size``.
    Exposing it keeps BiomedCLIP compatible with that code path.
    """

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class BiomedCLIPVisionEncoder(nn.Module):
    """Frozen BiomedCLIP vision tower exposed as a plain feature extractor.

    ``forward(pixel_values)`` maps ``[B, 3, H, W]`` -> ``[B, 512]``.
    """

    def __init__(
        self,
        model_id: str = BIOMEDCLIP_HF_ID,
        embed_dim: int = BIOMEDCLIP_EMBED_DIM,
        input_size: int = BIOMEDCLIP_INPUT_SIZE,
        normalize: bool = False,
    ):
        super().__init__()
        try:
            import open_clip  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "open_clip_torch is required to use the BiomedCLIP teacher. "
                "Install it with `pip install open_clip_torch`."
            ) from exc

        # ``create_model_from_pretrained`` downloads the checkpoint and returns
        # (model, preprocess). We only keep the vision tower.
        clip_model, _preprocess = open_clip.create_model_from_pretrained(model_id)
        self.visual = clip_model.visual

        self.embed_dim = int(embed_dim)
        self.input_size = int(input_size)
        self.normalize = bool(normalize)
        self.config = _EmbedDimConfig(self.embed_dim)

    @property
    def hidden_size(self) -> int:
        return self.embed_dim

    def forward(self, pixel_values: torch.Tensor, **_: Optional[object]) -> torch.Tensor:
        x = pixel_values
        # Enforce the 224x224 positional-grid constraint of the ViT-B/16 tower.
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        feats = self.visual(x)
        # open_clip vision towers may return a tensor or a (tokens, pooled) tuple.
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        if self.normalize:
            feats = F.normalize(feats, dim=-1)
        return feats
