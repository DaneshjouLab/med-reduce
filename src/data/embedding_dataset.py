# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""
Dataset classes for loading pre-computed embeddings (Stage 2 of linear probing).

This is used for the two-stage approach:
Stage 1: Extract and cache embeddings at each resolution
Stage 2: Train linear probe on cached embeddings (this module)
"""
from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.logging_core import get_logger

log = get_logger(__name__)


class EmbeddingDataset(Dataset):
    """
    Dataset that loads pre-computed embeddings instead of images.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Args:
            embeddings: Pre-computed embeddings [N, D]
            labels: Corresponding labels [N]
        """
        assert len(embeddings) == len(labels), \
            f"Embeddings and labels must have same length, got {len(embeddings)} vs {len(labels)}"

        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]

    @classmethod
    def from_cache(
        cls,
        cache_dir: str,
        dataset_name: str,
        model_name: str,
        resolution: int,
        split: str,
    ) -> "EmbeddingDataset":
        """
        Args:
            cache_dir: Root cache directory
            dataset_name: Name of dataset
            model_name: Name of model
            resolution: Image resolution
            split: 'train', 'val', or 'test'

        Returns:
            EmbeddingDataset instance
        """
        # Construct path
        cache_path = Path(cache_dir) / dataset_name / model_name / f"{resolution}px" / f"{split}_embeddings.pt"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {cache_path}. "
                f"Run embedding extraction first."
            )

        # Load embeddings
        data = torch.load(cache_path, map_location="cpu")
        embeddings = data["embeddings"]
        labels = data["labels"]

        log.info(f"Loaded {len(embeddings)} embeddings from {cache_path}")

        return cls(embeddings, labels)


class SubsetEmbeddingDataset(Dataset):
    """
    Subset of an EmbeddingDataset (for CV folds).

    Similar to torch.utils.data.Subset but optimized for embeddings.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Initialize subset embedding dataset.

        Args:
            embeddings: Full embeddings tensor [N, D]
            labels: Full labels tensor [N]
            indices: Indices to include in subset
        """
        self.embeddings = embeddings[indices]
        self.labels = labels[indices]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]
