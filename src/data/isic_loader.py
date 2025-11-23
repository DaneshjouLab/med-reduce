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
import os
import numpy as np
from datasets import load_dataset, Dataset, Image as HFImageFeature

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
    
    if isinstance(x, (list, tuple)):
        x = x[0]
    
    # Handle numpy arrays with extra dimensions
    if isinstance(x, np.ndarray):
        while x.ndim > 3 and x.shape[0] == 1:
            x = x.squeeze(0)
    
    try:
        return x.convert("RGB")
    except Exception:
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
        self._split = split
        self.cache_dir = cache_dir
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform

        # Load HF dataset split
        ds = load_dataset(
            repo_id, split=split, cache_dir=cache_dir
        )

        self._original_length = len(ds)

        if filter_fn is not None:
            f = filter_fn
            idx = [i for i, row in enumerate(ds) if f(row)]
        else:
            idx = list(range(self._original_length))

        if keep_indices is not None:
            keep = set(int(k) for k in keep_indices)
            idx = [i for i in idx if i in keep]

        self._indices = sorted(idx)

        if len(self._indices) != len(ds):
            ds = ds.select(self._indices)

        self.ds = ds

        try:
            names = getattr(ds.features.get(label_column, None), "names", None)
            self.class_names = tuple(names) if names else None
        except Exception:
            self.class_names = None

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        img = row[self.image_column]
        label = row[self.label_column]
        
        if isinstance(label, (list, tuple)):
            label = label[0]
        
        if not isinstance(img, Image.Image):
            img = _to_pil(img)

        if self.transform:
            img = self.transform(img)

        return {"pixel_values": img, "label": int(label)}
    
    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

    @property
    def hf_dataset(self):
        return self.ds


class ISICBaseDataset(ISICHFRawSplit):  # type: ignore[misc]
    """
    Backwards-compatible alias for the previous ISICBaseDataset, now backed by HF.

    Usage (unchanged import path):
        from src.data.isic_loader import ISICBaseDataset
        ds = ISICBaseDataset(repo_id="MKZuziak/ISIC_2019_224", split="train")
    """
    pass

from typing import Optional, Sequence, Any, Dict
import os
from datasets import load_dataset, Dataset
from datasets import Image as HFImageFeature
from .isic_loader import ISICHFRawSplit

class ISICHFRawSplitLocal(ISICHFRawSplit):
    """
    Hugging Face-backed ISIC split for local files (CSV + Image folder).

    Parameters
    ----------
    data_dir : str
        The local root directory containing the data (e.g., '/my_classification_data').
    label_file : str
        The name of the CSV file (e.g., 'labels.csv').
    image_column : str
        The column in the CSV containing relative paths to images (e.g., 'file_name').
    label_column : str
        The column in the CSV containing the integer labels (e.g., 'label').
    """

    def __init__(
        self,
        *,
        data_dir: str,
        split: str = "train",
        label_file: str = "labels.csv",
        image_column: str = "file_name",
        label_column: str = "label",
        filter_fn: Optional[Any] = None,
        keep_indices: Optional[Sequence[int]] = None,
        **kwargs
    ):
        csv_path = os.path.join(data_dir, label_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label file not found: {csv_path}")

        ds: Dataset = load_dataset("csv", data_files=csv_path, split="train")

        if filter_fn:
            idx = [i for i, row in enumerate(ds) if filter_fn(row)]
        else:
            idx = list(range(len(ds)))

        if keep_indices is not None:
            keep = set(int(k) for k in keep_indices)
            idx = [i for i in idx if i in keep]

        ds = ds.select(idx)

        def map_to_full_path(example: Dict[str, Any]):
            example["image"] = os.path.join(data_dir, example[image_column])
            return example

        ds = ds.map(map_to_full_path, remove_columns=[image_column])
        ds = ds.cast_column("image", HFImageFeature())

        try:
            names = getattr(ds.features.get(label_column, None), "names", None)
            class_names = tuple(names) if names else None
        except Exception:
            class_names = None

        self.ds = ds
        self.repo_id = data_dir
        self.image_column = "image"  
        self.label_column = label_column
        self.class_names = class_names

        # Call parent constructor last
        super().__init__(
            repo_id=self.repo_id,
            split=split,
            **kwargs
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        image = _to_pil(item[self.image_column])
        label = int(item[self.label_column])
        return {"pixel_values": image, "label": label}

    @property
    def hf_dataset(self):
        return self.ds

class ISICSegRawSplit(Dataset):
    """Hugging Face-backed segmentation split (image + mask)."""

    def __init__(
        self,
        *,
        data_dir: str,
        label_file: str = "labels_seg.csv",
        image_column: str = "image_path",
        mask_column: str = "mask_path",
        transform: Optional[Any] = None, # Should accept image AND mask
        **kwargs
    ):
        # 1. Load the CSV file
        csv_path = os.path.join(data_dir, label_file)
        raw_ds = load_dataset('csv', data_files=csv_path)['train'] # Assume one split

        # 2. Map the file paths to full paths
        def map_files(example: Dict[str, Any]):
            example['image'] = os.path.join(data_dir, example[image_column])
            example['mask'] = os.path.join(data_dir, example[mask_column])
            return example

        self.ds = raw_ds.map(map_files, remove_columns=[image_column, mask_column])
        
        # 3. Cast both columns to the HF Image feature (for PIL decoding)
        self.ds = self.ds.cast_column('image', HFImageFeature())
        self.ds = self.ds.cast_column('mask', HFImageFeature())
        
        self.transform = transform
        self._indices = list(range(len(self.ds))) # Simple index

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]
        item = self.ds[real_idx]

        image = _to_pil(item['image'])
        mask = _to_pil(item['mask']) # Mask is also loaded as a PIL Image

        if self.transform is not None:
            # The transform must be designed to handle both the image and the mask
            # For segmentation, image and mask transforms must be coordinated!
            image, mask = self.transform(image, mask)

        # Return the two main components for segmentation models
        return {"pixel_values": image, "mask_target": mask}