# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""
Teacher embedding cache for knowledge distillation.

This module provides functionality to extract and cache embeddings from a teacher model
on clean, full-resolution images. The cached embeddings can then be used for distillation
during student model training.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import hashlib
import json

from src.utils.logging_core import get_logger
from src.models.factory import create_model

log = get_logger(__name__)


class TeacherEmbeddingCache:
    """
    Manages caching of teacher model embeddings for distillation.

    The cache stores embeddings extracted from a teacher model at full resolution
    for every training image. This allows efficient distillation without repeatedly
    running the teacher model during student training.

    Cache format:
        {dataset_hash}/
            ├── metadata.json         # Cache metadata (model, resolution, etc.)
            ├── embeddings.pt         # Tensor of all embeddings [N, D]
            ├── labels.pt             # Tensor of all labels [N]
            ├── image_ids.json        # List of unique image identifiers [N]
            ├── sample_ids.pt         # Legacy: Tensor of sample indices (for backward compat)
            └── cache.pt              # Single-file atomic cache (preferred for loading)
    """

    def __init__(
        self,
        cache_dir: str,
        teacher_model_info: Dict[str, Any],
        full_resolution: int = 224,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize teacher embedding cache.

        Args:
            cache_dir: Root directory for caching embeddings
            teacher_model_info: Model configuration dict (same format as cfg.model)
            full_resolution: Full resolution to use for teacher embeddings
            device: Device to run teacher model on
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.teacher_model_info = teacher_model_info
        self.full_resolution = full_resolution
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"Teacher embedding cache initialized at {cache_dir}")
        log.info(f"Teacher model: {teacher_model_info.get('model_id', 'unknown')}")
        log.info(f"Full resolution: {full_resolution}px")

    def _get_cache_hash(self, dataset_name: str, split: str) -> str:
        """Generate unique hash for this dataset/model/resolution combination."""
        cache_key = f"{dataset_name}_{split}_{self.teacher_model_info.get('model_id')}_{self.full_resolution}"
        return hashlib.md5(cache_key.encode()).hexdigest()[:16]

    def _get_cache_path(self, dataset_name: str, split: str) -> Path:
        """Get path to cache directory for this dataset."""
        cache_hash = self._get_cache_hash(dataset_name, split)
        return self.cache_dir / cache_hash

    def exists(self, dataset_name: str, split: str = "train") -> bool:
        """Check if embeddings are already cached for this dataset."""
        cache_path = self._get_cache_path(dataset_name, split)

        has_new_format = (
            (cache_path / "embeddings.pt").exists() and
            (cache_path / "labels.pt").exists() and
            (cache_path / "image_ids.json").exists() and
            (cache_path / "metadata.json").exists()
        )
        has_legacy_format = (
            (cache_path / "embeddings.pt").exists() and
            (cache_path / "labels.pt").exists() and
            (cache_path / "sample_ids.pt").exists() and
            (cache_path / "metadata.json").exists()
        )
        return has_new_format or has_legacy_format

    def _extract_teacher_embeddings(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Extract embeddings from teacher model.

        Args:
            model: Teacher model
            dataloader: DataLoader providing clean full-resolution images
            max_samples: Optional limit on number of samples

        Returns:
            Tuple of (embeddings, labels, image_ids)
            - embeddings: Tensor [N, D] of teacher embeddings
            - labels: Tensor [N] of labels
            - image_ids: List[str] of unique image identifiers
        """
        model.eval()
        emb_chunks = []
        label_chunks = []
        image_ids = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting teacher embeddings")):
                # Handle different batch formats
                if isinstance(batch, dict):
                    pixel_values = batch['pixel_values'].to(self.device)
                    batch_labels = batch['label'].to(self.device)
                    batch_image_ids = batch.get('image_id', None)
                else:
                    pixel_values, batch_labels = batch[0].to(self.device), batch[1].to(self.device)
                    batch_image_ids = None

                # Get embeddings before classifier
                if hasattr(model, 'visual') and hasattr(model, 'embed_dim'):
                    # BiomedCLIP-style feature extractor: forward -> [B, D]
                    emb = model(pixel_values)
                elif hasattr(model, 'backbone'):
                    # DINOv3 wrapper
                    outputs = model.backbone(pixel_values=pixel_values)
                    emb = outputs.pooler_output
                elif hasattr(model, 'vit'):
                    # ViT models
                    outputs = model.vit(pixel_values=pixel_values)
                    emb = outputs.last_hidden_state[:, 0]  # CLS token
                elif hasattr(model, 'dinov2'):
                    # DINOv2 models
                    outputs = model.dinov2(pixel_values=pixel_values)
                    emb = outputs.last_hidden_state[:, 0]
                else:
                    # Fallback: use model's forward but extract features
                    outputs = model(pixel_values=pixel_values, output_hidden_states=True)
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        emb = outputs.hidden_states[-1][:, 0]
                    else:
                        raise ValueError("Cannot extract embeddings from this model")

                emb_chunks.append(emb.cpu())
                label_chunks.append(batch_labels.cpu())

                # Extract image IDs if available, otherwise use fallback indices
                batch_size = emb.shape[0]
                if batch_image_ids is not None:
                    if isinstance(batch_image_ids, torch.Tensor):
                        batch_image_ids = batch_image_ids.tolist()
                    image_ids.extend([str(img_id) for img_id in batch_image_ids])
                else:
                    start_idx = batch_idx * dataloader.batch_size
                    fallback_ids = [f"sample_{start_idx + i}" for i in range(batch_size)]
                    image_ids.extend(fallback_ids)
                    if batch_idx == 0:
                        log.warning(
                            "No 'image_id' field found in batch. Using fallback sequential IDs. "
                            "For proper distillation, ensure your dataset returns 'image_id' in each sample."
                        )

                # Early stop if max_samples reached
                if max_samples and len(emb_chunks) * emb.shape[0] >= max_samples:
                    break

        embeddings = torch.cat(emb_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        del emb_chunks, label_chunks  # free the chunk lists

        if max_samples:
            embeddings = embeddings[:max_samples]
            labels = labels[:max_samples]
            image_ids = image_ids[:max_samples]

        return embeddings, labels, image_ids

    def cache_embeddings(
        self,
        dataloader: DataLoader,
        dataset_name: str,
        split: str = "train",
        force_recompute: bool = False,
        max_samples: Optional[int] = None,
    ) -> Path:
        """
        Cache teacher embeddings for a dataset.

        Args:
            dataloader: DataLoader providing clean full-resolution images
            dataset_name: Name of the dataset
            split: Dataset split (train/val/test)
            force_recompute: If True, recompute even if cache exists
            max_samples: Optional limit on number of samples

        Returns:
            Path to cache directory
        """
        cache_path = self._get_cache_path(dataset_name, split)

        # Check if already cached
        if self.exists(dataset_name, split) and not force_recompute:
            log.info(f"Teacher embeddings already cached at {cache_path}")
            return cache_path

        log.info(f"Caching teacher embeddings for {dataset_name} ({split})")
        cache_path.mkdir(parents=True, exist_ok=True)

        # Load teacher model
        log.info(f"Loading teacher model: {self.teacher_model_info.get('model_id')}")
        teacher_model = create_model(
            self.teacher_model_info,
            resolution=self.full_resolution
        ).to(self.device)
        teacher_model.eval()

        # Extract embeddings
        embeddings, labels, sample_ids = self._extract_teacher_embeddings(
            teacher_model,
            dataloader,
            max_samples=max_samples,
        )

        # Save to cache (keep legacy per-file layout for compatibility)
        log.info(f"Saving {len(embeddings)} embeddings to cache...")
        torch.save(embeddings, cache_path / "embeddings.pt")
        torch.save(labels, cache_path / "labels.pt")

        with open(cache_path / "image_ids.json", "w") as f:
            json.dump(sample_ids, f)

        try:
            numeric_ids = [int(x.split('_')[-1]) if isinstance(x, str) and x.startswith('sample_') else hash(x) for x in sample_ids]
            torch.save(torch.tensor(numeric_ids), cache_path / "sample_ids.pt")
        except:
            torch.save(torch.zeros(len(sample_ids), dtype=torch.long), cache_path / "sample_ids.pt")

        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "split": split,
            "teacher_model": self.teacher_model_info.get('model_id'),
            "resolution": self.full_resolution,
            "num_samples": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "cache_version": "1.0",
        }
        with open(cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        try:
            cache_dict = {
                "embeddings": embeddings,
                "labels": labels,
                "image_ids": sample_ids,
                "metadata": metadata,
            }
            tmp_path = cache_path / "cache.pt.tmp"
            dest_path = cache_path / "cache.pt"
            torch.save(cache_dict, tmp_path)
            # Atomic replace
            os.replace(str(tmp_path), str(dest_path))
            log.info(f"Saved single-file cache to {dest_path}")
        except Exception as e:  # pragma: no cover - best-effort enhancement
            log.warning(f"Failed to write single-file cache.pt: {e}")

        log.info(f"✓ Teacher embeddings cached successfully at {cache_path}")
        log.info(f"  - Samples: {len(embeddings)}")
        log.info(f"  - Embedding dim: {embeddings.shape[1]}")

        # Clean up teacher model
        del teacher_model
        torch.cuda.empty_cache()

        return cache_path

    def load_embeddings(
        self,
        dataset_name: str,
        split: str = "train",
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load cached teacher embeddings.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split (train/val/test)
            device: Device to load tensors to (default: same as cache device)

        Returns:
            Dictionary with keys:
            - embeddings: Tensor [N, D] of teacher embeddings
            - labels: Tensor [N] of labels
            - image_ids: List[str] of unique image identifiers
            - metadata: Dict with cache metadata
        """
        if not self.exists(dataset_name, split):
            raise FileNotFoundError(
                f"No cached embeddings found for {dataset_name} ({split}). "
                "Call cache_embeddings() first."
            )

        cache_path = self._get_cache_path(dataset_name, split)
        target_device = device or self.device

        log.info(f"Loading teacher embeddings from {cache_path}")

        # Prefer the single-file atomic cache if present
        cache_pt = cache_path / "cache.pt"
        if cache_pt.exists():
            try:
                data = torch.load(cache_pt, map_location=target_device, weights_only=False)
                embeddings = data.get("embeddings")
                labels = data.get("labels")
                image_ids = data.get("image_ids")
                metadata = data.get("metadata", {})

                log.info(f"✓ Loaded {len(embeddings) if embeddings is not None else 'unknown'} teacher embeddings from cache.pt")

                return {
                    "embeddings": embeddings,
                    "labels": labels,
                    "image_ids": image_ids,
                    "metadata": metadata,
                }
            except Exception as e:  # pragma: no cover - fallback behavior
                log.warning(f"Failed to load cache.pt ({e}), falling back to per-file layout")

        # Fallback to legacy per-file layout
        embeddings = torch.load(cache_path / "embeddings.pt", map_location=target_device)
        labels = torch.load(cache_path / "labels.pt", map_location=target_device)

        image_ids_json = cache_path / "image_ids.json"
        if image_ids_json.exists():
            with open(image_ids_json, "r") as f:
                image_ids = json.load(f)
        else:
            sample_ids = torch.load(cache_path / "sample_ids.pt", map_location="cpu")
            image_ids = [f"sample_{int(idx)}" for idx in sample_ids.tolist()]
            log.warning(
                "Loaded legacy sample_ids.pt. For proper distillation, "
                "consider regenerating the cache with actual image IDs."
            )

        with open(cache_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        log.info(f"✓ Loaded {len(embeddings)} teacher embeddings")

        return {
            "embeddings": embeddings,
            "labels": labels,
            "image_ids": image_ids,
            "metadata": metadata,
        }


class TeacherEmbeddingLookup:
    """
    Efficient lookup table for teacher embeddings by image_id.

    Use this during distillation training to quickly retrieve teacher embeddings
    for each batch based on image IDs.

    Example:
        cache = TeacherEmbeddingCache(...)
        data = cache.load_embeddings("isic2019", "train")
        lookup = TeacherEmbeddingLookup(data)

        # During training:
        for batch in dataloader:
            image_ids = batch['image_id']
            teacher_embs = lookup.get_embeddings(image_ids)
            # Use teacher_embs for distillation loss
    """

    def __init__(self, cache_data: Dict[str, Any]):
        """
        Initialize lookup table from cached data.

        Args:
            cache_data: Dictionary returned by TeacherEmbeddingCache.load_embeddings()
        """
        self.embeddings = cache_data['embeddings']
        self.labels = cache_data['labels']
        self.image_ids = cache_data['image_ids']

        # Build lookup dict: image_id -> index
        self.id_to_idx = {img_id: idx for idx, img_id in enumerate(self.image_ids)}

        log.info(f"Built embedding lookup table with {len(self.id_to_idx)} entries")

    def get_embeddings(self, image_ids: list) -> torch.Tensor:
        """
        Get teacher embeddings for a list of image IDs.

        Args:
            image_ids: List of image identifiers (strings)

        Returns:
            Tensor [B, D] of teacher embeddings

        Raises:
            KeyError: If any image_id is not found in the cache
        """
        indices = []
        for img_id in image_ids:
            if img_id not in self.id_to_idx:
                raise KeyError(
                    f"Image ID '{img_id}' not found in teacher embedding cache. "
                    f"Available IDs: {len(self.id_to_idx)}. "
                    "Ensure the same dataset is used for caching and training."
                )
            indices.append(self.id_to_idx[img_id])

        return self.embeddings[indices]

    def get_labels(self, image_ids: list) -> torch.Tensor:
        """Get labels for a list of image IDs."""
        indices = [self.id_to_idx[img_id] for img_id in image_ids]
        return self.labels[indices]

    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self.image_ids)


class MultiResolutionTeacherLookup:
    """Teacher embedding lookup that stores embeddings at multiple resolutions.

    During distillation each view targets a specific resolution.  The training
    loop calls ``get_embeddings(image_ids, resolution)`` to retrieve the teacher
    embedding extracted at that resolution.

    Example::

        lookup = MultiResolutionTeacherLookup()
        lookup.add_resolution(512, data_512)
        lookup.add_resolution(256, data_256)

        teacher_emb = lookup.get_embeddings(image_ids, resolution=256)
    """

    def __init__(self):
        # resolution -> TeacherEmbeddingLookup
        self._lookups: Dict[int, TeacherEmbeddingLookup] = {}
        self.resolutions: list[int] = []

    def add_resolution(self, resolution: int, cache_data: Dict[str, Any]):
        """Register embeddings for a resolution."""
        self._lookups[resolution] = TeacherEmbeddingLookup(cache_data)
        self.resolutions = sorted(self._lookups.keys(), reverse=True)
        log.info(f"  Added teacher embeddings at {resolution}px ({len(cache_data['image_ids'])} samples)")

    def get_embeddings(self, image_ids: list, resolution: int) -> torch.Tensor:
        """Retrieve teacher embeddings for *image_ids* at *resolution*."""
        if resolution not in self._lookups:
            raise KeyError(
                f"Resolution {resolution}px not in teacher lookup. "
                f"Available: {self.resolutions}"
            )
        return self._lookups[resolution].get_embeddings(image_ids)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (same across all resolutions)."""
        first = next(iter(self._lookups.values()))
        return first.embeddings.shape[1]

    def __len__(self) -> int:
        first = next(iter(self._lookups.values()))
        return len(first)


def create_clean_image_dataloader(
    dataset_name: str,
    data_dir: Optional[str],
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 8,
    image_size: int = 224,
    **kwargs
) -> DataLoader:
    """
    Create a dataloader for clean, full-resolution images.

    This loader provides images WITHOUT degradation transforms,
    at full resolution for teacher embedding extraction.

    Args:
        dataset_name: HuggingFace dataset name or local path
        data_dir: Cache directory for HF datasets
        split: Dataset split (train/val/test)
        batch_size: Batch size for loading
        num_workers: Number of dataloader workers
        image_size: Target image size (should be full resolution, e.g., 224)

    Returns:
        DataLoader with clean full-resolution images
    """
    from torchvision import transforms
    from src.data.isic_loader import ISICHFRawSplit

    # Transform for clean images at full resolution
    clean_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load clean dataset
    dataset = ISICHFRawSplit(
        repo_id=dataset_name,
        split=split,
        cache_dir=data_dir,
        transform=clean_transform,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for cache alignment
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Don't drop samples
    )

    return dataloader
