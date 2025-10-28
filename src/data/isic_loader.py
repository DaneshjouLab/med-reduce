# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""ISIC dataset loader implementation for dermatology image datasets.

This module provides a Hugging Face–backed loader for ISIC that returns
dicts shaped like:
    {"image": PIL.Image, "label": int}

Typical usage (HF-only, no DataModule required):
    from src.data.isic_loader import ISICHFRawSplit
    ds = ISICHFRawSplit(repo_id="MKZuziak/ISIC_2019_224", split="train")
    sample = ds[0]
    image, label = sample["image"], sample["label"]

You may pass a PIL->PIL transform (e.g., ResolutionReductionTransform) via `transform=...`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
from torch.utils.data import Dataset  # pylint: disable=import-error
from PIL import Image

try:
    # Hugging Face datasets is required for this loader
    from datasets import load_dataset, Image as HFImageFeature  # type: ignore
except Exception as _e:  # pragma: no cover
    load_dataset = None
    HFImageFeature = None
    _HF_IMPORT_ERROR = _e


def _to_pil(x: Any) -> Image.Image:
    """Best-effort conversion to PIL.Image."""
    if isinstance(x, Image.Image):
        return x
    try:
        # HF Image feature usually yields PIL already; if not, try convert
        return x.convert("RGB")
    except Exception:
        import numpy as np
        return Image.fromarray(np.asarray(x)).convert("RGB")


class ISICHFRawSplit(Dataset):
    """
    Hugging Face–backed ISIC split. Returns dicts:
        {"image": PIL.Image, "label": int}

    Parameters
    ----------
    repo_id : str
        HF dataset repo id. Default: "MKZuziak/ISIC_2019_224"
    split : str
        Split name to load (e.g., "train"). Depends on the repo.
    cache_dir : Optional[str]
        Local HF cache directory.
    image_column : str
        Column name for image. Default: "image"
    label_column : str
        Column name for label. Default: "label"
    transform : Optional[callable]
        Optional PIL->PIL transform (e.g., ResolutionReductionTransform).
    filter_fn : Optional[callable]
        Optional row-level filter: receives an item dict and returns True/False.
        If provided, the dataset builds an index of rows where filter_fn(item) is True.
    keep_indices : Optional[Sequence[int]]
        Optional explicit list of indices to keep (applied after filter_fn if both are given).

    Notes
    -----
    - Model-specific preprocessing (tensor conversion, normalization) is intentionally not applied here.
      That belongs in your model-ready pipeline/DataModule.
    """

    def __init__(
        self,
        *,
        repo_id: str = "MKZuziak/ISIC_2019_224",
        split: str = "train",
        cache_dir: Optional[str] = None,
        image_column: str = "image",
        label_column: str = "label",
        transform: Optional[Any] = None,
        filter_fn: Optional[Any] = None,
        keep_indices: Optional[Sequence[int]] = None,
    ):
        if load_dataset is None:  # pragma: no cover
            raise ImportError(
                "Hugging Face `datasets` is required for ISICHFRawSplit. "
                "Install via `pip install datasets`. "
                f"Original import error: {_HF_IMPORT_ERROR}"
            )

        self.repo_id = repo_id
        self.split = split
        self.cache_dir = cache_dir
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform

        # Load HF dataset split
        self.ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)

        # Validate image column if possible
        try:
            feats = self.ds.features
            if image_column in feats and HFImageFeature is not None:
                # If it's an HF Image feature, decoding to PIL happens on access.
                pass
        except Exception:
            pass  # proceed; we'll coerce per-sample in __getitem__

        # Build index (filter + keep_indices)
        idx = list(range(len(self.ds)))
        if filter_fn is not None:
            filtered = []
            for i in idx:
                try:
                    if filter_fn(self.ds[i]):
                        filtered.append(i)
                except Exception:
                    # Skip rows that fail the filter function
                    continue
            idx = filtered
        if keep_indices is not None:
            # Intersect in original order
            keep_set = set(int(k) for k in keep_indices)
            idx = [i for i in idx if i in keep_set]

        self._indices = idx

        self.class_names = None
        try:
            label_feat = self.ds.features.get(label_column, None)
            names = getattr(label_feat, "names", None)
            if names:
                self.class_names = tuple(names)
        except Exception:
            self.class_names = None

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = int(self._indices[idx])
        item = self.ds[real_idx]

        image = _to_pil(item[self.image_column])
        label = int(item[self.label_column])

        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "label": label}


class ISICBaseDataset(ISICHFRawSplit):  # type: ignore[misc]
    """
    Backwards-compatible alias for the previous ISICBaseDataset, now backed by HF.

    Usage (unchanged import path):
        from src.data.isic_loader import ISICBaseDataset
        ds = ISICBaseDataset(repo_id="MKZuziak/ISIC_2019_224", split="train")
    """
    pass