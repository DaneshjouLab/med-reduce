# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""ISIC dataset loader implementation for dermatology image datasets.

This module provides a Hugging Face–backed loader

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

        # Try to get image_id if available in the dataset
        image_id = None
        if 'image_id' in row:
            image_id = row['image_id']

        result = {"pixel_values": img, "label": int(label)}
        if image_id is not None:
            result["image_id"] = str(image_id)

        return result
    
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

class ISICHFRawSplitLocal(Dataset):
    """
    Hugging Face-backed ISIC split for local files (CSV + Image folder).

    Parameters
    ----------
    data_dir : str
        The local root directory containing the images (e.g., '/path/to/images').
    label_file : str
        Path to the CSV file (can be absolute or relative to data_dir).
    image_id_column : str
        The column in the CSV containing image IDs (e.g., 'image_id').
        Image files are expected to be named as {image_id}.jpg in data_dir.
    label_column : str or list[str]
        The column(s) in the CSV containing the labels.
        - If str: Uses that column directly as integer label
        - If list[str]: Converts multi-label columns to integer (argmax for single label)
    image_extension : str
        File extension for images (default: '.jpg')
    transform : Optional[callable]
        Optional PIL->PIL transform.
    filter_fn : Optional[callable]
        Optional row-level filter function.
    keep_indices : Optional[Sequence[int]]
        Optional explicit list of indices to keep.
    """

    def __init__(
        self,
        *,
        data_dir: str = "/scratch/groups/roxanad/datasets/isic/challenges/2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data",
        label_file: str = "/scratch/groups/roxanad/datasets/isic/challenges/2017/ISIC-2017_Training_Part3_GroundTruth.csv",
        image_id_column: str = "image_id",
        label_column: str | Sequence[str] = "label",
        image_extension: str = ".jpg",
        transform: Optional[Any] = None,
        filter_fn: Optional[Any] = None,
        keep_indices: Optional[Sequence[int]] = None,
    ):
        # Load CSV
        if os.path.isabs(label_file):
            csv_path = label_file
        else:
            csv_path = os.path.join(data_dir, label_file)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label file not found: {csv_path}")

        ds: Dataset = load_dataset("csv", data_files=csv_path, split="train")

        # Apply filter_fn
        if filter_fn:
            idx = [i for i, row in enumerate(ds) if filter_fn(row)]
        else:
            idx = list(range(len(ds)))

        # Apply keep_indices
        if keep_indices is not None:
            keep = set(int(k) for k in keep_indices)
            idx = [i for i in idx if i in keep]

        ds = ds.select(idx)

        # Map image IDs to full paths and handle labels
        def map_to_full_path(example: Dict[str, Any]):
            image_id = example[image_id_column]
            example["image"] = os.path.join(data_dir, f"{image_id}{image_extension}")

            if image_id_column != "image_id":
                example["image_id"] = image_id

            # Handle label columns
            if isinstance(label_column, (list, tuple)):
                # Multi-label case: convert to single integer label
                label_values = [float(example[col]) for col in label_column]
                example["label"] = int(np.argmax(label_values))
            else:
                # Single label column
                example["label"] = int(float(example[label_column]))

            return example

        # Remove original columns we don't need (but keep image_id)
        cols_to_remove = []
        if image_id_column != "image_id":
            cols_to_remove.append(image_id_column)
        if isinstance(label_column, (list, tuple)):
            cols_to_remove.extend(label_column)
        elif label_column != "label":
            cols_to_remove.append(label_column)

        cols_to_remove = [c for c in cols_to_remove if c in ds.column_names]
        ds = ds.map(map_to_full_path, remove_columns=cols_to_remove)

        # Cast image column to HF Image feature for automatic PIL loading
        ds = ds.cast_column("image", HFImageFeature())

        # Try to get class names if available
        try:
            names = getattr(ds.features.get("label", None), "names", None)
            self.class_names = tuple(names) if names else None
        except Exception:
            self.class_names = None

        # If we have multi-label columns, create class names from them
        if isinstance(label_column, (list, tuple)):
            self.class_names = tuple(label_column)

        self.ds = ds
        self.data_dir = data_dir
        self.image_column = "image"
        self.label_column = "label"
        self.transform = transform
        self._indices = list(range(len(self.ds)))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        image = _to_pil(item[self.image_column])
        label = int(item[self.label_column])

        if self.transform:
            image = self.transform(image)

        # Include image_id if available in the dataset
        result = {"pixel_values": image, "label": label}
        if 'image_id' in item:
            result["image_id"] = str(item['image_id'])

        return result

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

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


class ISICSegRawSplitLocal(Dataset):
    """
    Local directory-based segmentation dataset.

    Expects two directories:
    - image_dir: containing images (e.g., img1.jpg, img2.jpg)
    - mask_dir: containing corresponding masks (e.g., img1_segmentation.png, img2_segmentation.png)

    Parameters
    ----------
    image_dir : str
        Directory containing the images
    mask_dir : str
        Directory containing the corresponding masks
    image_extension : str
        File extension for images (default: '.jpg')
    mask_extension : str
        File extension for masks (default: '.png')
    mask_suffix : str
        Suffix to append to image filename to get mask filename (default: '_segmentation')
        e.g., if image is 'ISIC_0000000.jpg' and mask_suffix='_segmentation',
        it will look for 'ISIC_0000000_segmentation.png'
    transform : Optional[callable]
        Transform that takes (image, mask) and returns (transformed_image, transformed_mask)
    """

    def __init__(
        self,
        *,
        image_dir: str,
        mask_dir: str,
        image_extension: str = ".jpg",
        mask_extension: str = ".png",
        mask_suffix: str = "_segmentation",
        transform: Optional[Any] = None,
        filter_fn: Optional[Any] = None,
        keep_indices: Optional[Sequence[int]] = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.mask_suffix = mask_suffix
        self.transform = transform

        # Get all image files
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        # List all image files
        all_files = os.listdir(image_dir)
        image_files = [f for f in all_files if f.endswith(image_extension)]

        # Create dataset as list of (image_path, mask_path) tuples
        data_pairs = []
        for img_file in image_files:
            # Get base name without extension
            base_name = os.path.splitext(img_file)[0]

            # Construct mask filename
            mask_file = f"{base_name}{mask_suffix}{mask_extension}"

            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # Only add if mask exists
            if os.path.exists(mask_path):
                data_pairs.append({
                    'image': img_path,
                    'mask': mask_path,
                    'image_id': base_name
                })

        if len(data_pairs) == 0:
            raise ValueError(
                f"No matching image-mask pairs found.\n"
                f"Image dir: {image_dir}\n"
                f"Mask dir: {mask_dir}\n"
                f"Expected mask pattern: {{image_name}}{mask_suffix}{mask_extension}"
            )

        # Convert to HF Dataset
        self.ds = Dataset.from_list(data_pairs)

        # Cast to Image features for automatic PIL loading
        self.ds = self.ds.cast_column('image', HFImageFeature())
        self.ds = self.ds.cast_column('mask', HFImageFeature())

        # Apply filter_fn if provided
        if filter_fn:
            idx = [i for i, row in enumerate(self.ds) if filter_fn(row)]
        else:
            idx = list(range(len(self.ds)))

        # Apply keep_indices if provided
        if keep_indices is not None:
            keep = set(int(k) for k in keep_indices)
            idx = [i for i in idx if i in keep]

        if len(idx) != len(self.ds):
            self.ds = self.ds.select(idx)

        self._indices = list(range(len(self.ds)))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self._indices[idx]
        item = self.ds[real_idx]

        image = _to_pil(item['image'])
        mask = _to_pil(item['mask'])

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {"pixel_values": image, "mask_target": mask}

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

    @property
    def hf_dataset(self):
        return self.ds
