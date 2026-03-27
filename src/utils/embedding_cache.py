# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""
Embedding cache system for storing DINOv3 embeddings at different resolutions.

1. Extract and cache embeddings at each resolution R
2. Load cached embeddings for linear probing

The cache is organized as:
  cache_dir/
    {dataset_name}/
      {model_name}/
        seed_{seed}/
          {resolution}px/
            train_embeddings.pt
            val_embeddings.pt  (optional)
            test_embeddings.pt
            metadata.json
"""
from __future__ import annotations
import os
import json
import hashlib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logging_core import get_logger

log = get_logger(__name__)


class EmbeddingCache:
    def __init__(
        self,
        cache_dir: str,
        dataset_name: str,
        model_name: str,
        seed: int = 42,
        device: torch.device = None,
    ):
        """
        Args:
            cache_dir: Root directory for embedding cache
            dataset_name: Name of dataset (e.g., 'isic', 'chexpert')
            model_name: Name of model (e.g., 'dinov3-vits16')
            seed: Random seed for reproducibility (used in path)
            device: Device to use for extraction
        """
        self.cache_dir = Path(cache_dir)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_dir = self.cache_dir / dataset_name / model_name / f"seed_{seed}"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def _extract_embeddings_from_model(self, model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from different model architectures.

        Handles:
        - DINOv3: model.backbone (Dinov2Model)
        - ViT (HuggingFace): model.vit (ViTModel)
        - Generic models with forward returning embeddings

        Args:
            model: The encoder model
            images: Input images tensor

        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        # DINOv3 models have a 'backbone' attribute
        if hasattr(model, 'backbone'):
            outputs = model.backbone(pixel_values=images)
            embeddings = outputs.pooler_output
            if embeddings is None and hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[:, 0, :]
            return embeddings

        # HuggingFace ViT models have a 'vit' attribute
        if hasattr(model, 'vit'):
            outputs = model.vit(pixel_values=images)
            embeddings = outputs.pooler_output
            if embeddings is None and hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[:, 0, :]
            return embeddings

        # HuggingFace DINOv2 models may have different attribute names
        if hasattr(model, 'dinov2'):
            outputs = model.dinov2(pixel_values=images)
            embeddings = outputs.pooler_output
            if embeddings is None and hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[:, 0, :]
            return embeddings

        # timm models: use forward_features to get pre-classifier embeddings
        if hasattr(model, 'forward_features'):
            feats = model.forward_features(images)
            if feats.dim() == 4:
                # CNN-style: [B, D, H, W] -> global average pool
                return feats.mean(dim=[2, 3])
            elif feats.dim() == 3:
                # Transformer-style: [B, tokens, D] -> global average pool
                return feats.mean(dim=1)
            return feats

        # Try base_model for other HuggingFace models
        if hasattr(model, 'base_model'):
            outputs = model.base_model(pixel_values=images)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state[:, 0, :]

        # Fallback: call model directly and try to extract embeddings
        outputs = model(images)
        if isinstance(outputs, dict):
            embeddings = outputs.get('pooler_output', outputs.get('last_hidden_state'))
            if embeddings is not None and embeddings.dim() == 3:
                embeddings = embeddings[:, 0, :]
            return embeddings
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0, :]

        # If outputs is a tensor, assume it's already embeddings
        if isinstance(outputs, torch.Tensor):
            if outputs.dim() == 3:
                return outputs[:, 0, :]
            return outputs

        raise ValueError(f"Could not extract embeddings from model type {type(model).__name__}")

    def _get_resolution_dir(self, resolution: int) -> Path:
        """Get directory for specific resolution."""
        return self.dataset_dir / f"{resolution}px"

    def _get_embedding_path(self, resolution: int, split: str) -> Path:
        """Get path for embedding file."""
        res_dir = self._get_resolution_dir(resolution)
        return res_dir / f"{split}_embeddings.pt"

    def _get_metadata_path(self, resolution: int) -> Path:
        """Get path for metadata file."""
        res_dir = self._get_resolution_dir(resolution)
        return res_dir / "metadata.json"

    def exists(self, resolution: int, split: str) -> bool:
        """Check if embeddings exist for given resolution and split."""
        embedding_path = self._get_embedding_path(resolution, split)
        metadata_path = self._get_metadata_path(resolution)
        return embedding_path.exists() and metadata_path.exists()

    def get_cache_key(self, model_info: Dict[str, Any], resolution: int) -> str:
        """Generate cache key from model configuration and resolution."""
        key_dict = {
            "model_id": model_info.get("model_id", ""),
            "model_type": model_info.get("type", ""),
            "resolution": resolution,
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:8]

    def extract_and_cache(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        resolution: int,
        split: str,
        model_info: Dict[str, Any],
        mixed_precision: bool = True,
        force_recompute: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            model: Frozen DINOv3 model (only backbone used)
            dataloader: DataLoader for the split
            resolution: Image resolution
            split: 'train', 'val', or 'test'
            model_info: Model configuration dict
            mixed_precision: Whether to use mixed precision
            force_recompute: Force recomputation even if cache exists

        Returns:
            Tuple of (embeddings, labels)
        """
        embedding_path = self._get_embedding_path(resolution, split)
        metadata_path = self._get_metadata_path(resolution)

        if not force_recompute and self.exists(resolution, split):
            log.info(f"✓ Loading cached embeddings from {embedding_path}")
            return self.load(resolution, split)

        log.info(f"🔄 Extracting embeddings at {resolution}px for {split} split...")

        res_dir = self._get_resolution_dir(resolution)
        res_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        emb_chunks = []
        label_chunks = []

        try:
            from torch.amp import autocast
        except ImportError:
            from torch.cuda.amp import autocast

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting {split} embeddings"):
                if isinstance(batch, (tuple, list)):
                    if len(batch) == 2:
                        images, labels = batch
                    else:
                        images = batch[0]
                        labels = batch[1] if len(batch) > 1 else torch.zeros(images.size(0))
                elif isinstance(batch, dict):
                    images = batch.get("pixel_values", batch.get("image"))
                    labels = batch.get("labels", batch.get("label"))
                else:
                    images = batch
                    labels = torch.zeros(images.size(0))

                images = images.to(self.device)

                with autocast(device_type=self.device.type, enabled=mixed_precision):
                    embeddings = self._extract_embeddings_from_model(model, images)

                emb_chunks.append(embeddings.cpu().float())
                if labels.ndim > 1:
                    label_chunks.append(labels.cpu().float())   # [B, C] multi-label
                else:
                    label_chunks.append(labels.cpu().long())    # [B] single-label

        embeddings = torch.cat(emb_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        del emb_chunks, label_chunks  # free the chunk lists

        torch.save(
            {"embeddings": embeddings, "labels": labels},
            embedding_path
        )

        # Convert model_info to plain dict if it's a DictConfig (from Hydra/OmegaConf)
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(model_info, DictConfig):
                model_info_dict = OmegaConf.to_container(model_info, resolve=True)
            else:
                model_info_dict = dict(model_info) if hasattr(model_info, '__iter__') else {}
        except ImportError:
            model_info_dict = dict(model_info) if hasattr(model_info, '__iter__') else {}

        metadata = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "model_info": model_info_dict,
            "resolution": resolution,
            "split": split,
            "num_samples": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "cache_key": self.get_cache_key(model_info, resolution),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"💾 Cached {len(embeddings)} embeddings to {embedding_path}")
        log.info(f"   Embedding shape: {embeddings.shape}")

        return embeddings, labels

    def load(self, resolution: int, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load cached embeddings.

        Args:
            resolution: Image resolution
            split: 'train', 'val', or 'test'

        Returns:
            Tuple of (embeddings, labels)
        """
        embedding_path = self._get_embedding_path(resolution, split)

        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {embedding_path}. "
                f"Run extract_and_cache() first."
            )

        data = torch.load(embedding_path, map_location="cpu", weights_only=True)
        embeddings = data["embeddings"]
        labels = data["labels"]

        log.info(f"📥 Loaded {len(embeddings)} embeddings from {embedding_path}")

        return embeddings, labels

    def get_metadata(self, resolution: int) -> Dict[str, Any]:
        """Load metadata for a resolution."""
        metadata_path = self._get_metadata_path(resolution)

        if not metadata_path.exists():
            return {}

        with open(metadata_path, "r") as f:
            return json.load(f)

    def list_cached_resolutions(self) -> list[int]:
        """List all resolutions with cached embeddings."""
        resolutions = []
        for item in self.dataset_dir.iterdir():
            if item.is_dir() and item.name.endswith("px"):
                try:
                    res = int(item.name[:-2])
                    resolutions.append(res)
                except ValueError:
                    continue
        return sorted(resolutions)

    def clear_cache(self, resolution: Optional[int] = None):
        """
        Clear cached embeddings.

        Args:
            resolution: If specified, only clear this resolution. Otherwise clear all.
        """
        if resolution is not None:
            res_dir = self._get_resolution_dir(resolution)
            if res_dir.exists():
                import shutil
                shutil.rmtree(res_dir)
                log.info(f"🗑️  Cleared cache for {resolution}px")
        else:
            if self.dataset_dir.exists():
                import shutil
                shutil.rmtree(self.dataset_dir)
                log.info(f"🗑️  Cleared all cache for {self.dataset_name}/{self.model_name}")
