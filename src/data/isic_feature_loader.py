# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""ISIC dataset loader for dermoscopic feature detection task."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from datasets import Dataset as HFDataset
from datasets import Image as HFImageFeature


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


class ISICFeatureDetectionDataset(Dataset):
    """
    Local directory-based dataset for dermoscopic feature detection.

    Expects three directories:
    - image_dir: containing images (e.g., ISIC_0000000.jpg)
    - superpixel_dir: containing superpixel masks (e.g., ISIC_0000000_superpixels.png)
    - annotation_dir: containing feature annotations (e.g., ISIC_0000000.json)

    The JSON annotations contain four features:
    - "network": Pigment Network
    - "negative network": Negative Network
    - "streaks": Streaks
    - "milia-like cysts": Milia-like Cysts

    Each feature array contains binary labels (0 or 1) for each superpixel ID.

    Parameters
    ----------
    image_dir : str
        Directory containing the dermoscopic images
    superpixel_dir : str
        Directory containing the superpixel masks (PNG files with encoded superpixel IDs)
    annotation_dir : str
        Directory containing the JSON feature annotations
    image_extension : str
        File extension for images (default: '.jpg')
    superpixel_extension : str
        File extension for superpixel masks (default: '.png')
    annotation_extension : str
        File extension for annotations (default: '.json')
    superpixel_suffix : str
        Suffix to append to image filename to get superpixel filename (default: '_superpixels')
    transform : Optional[callable]
        Transform that takes (image, superpixel_mask) and returns transformed versions
    """

    # Feature names in the order they appear in the JSON
    FEATURE_NAMES = ["network", "negative network", "streaks", "milia-like cysts"]

    def __init__(
        self,
        *,
        image_dir: str,
        superpixel_dir: str,
        annotation_dir: str,
        image_extension: str = ".jpg",
        superpixel_extension: str = ".png",
        annotation_extension: str = ".json",
        superpixel_suffix: str = "_superpixels",
        transform: Optional[Any] = None,
        filter_fn: Optional[Any] = None,
        keep_indices: Optional[Sequence[int]] = None,
    ):
        import json

        self.image_dir = image_dir
        self.superpixel_dir = superpixel_dir
        self.annotation_dir = annotation_dir
        self.image_extension = image_extension
        self.superpixel_extension = superpixel_extension
        self.annotation_extension = annotation_extension
        self.superpixel_suffix = superpixel_suffix
        self.transform = transform

        # Validate directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(superpixel_dir):
            raise FileNotFoundError(f"Superpixel directory not found: {superpixel_dir}")
        if not os.path.exists(annotation_dir):
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

        # List all image files
        all_files = os.listdir(image_dir)
        image_files = [f for f in all_files if f.endswith(image_extension)]

        # Create dataset as list of (image_path, superpixel_path, annotation_path) tuples
        data_triplets = []
        for img_file in image_files:
            # Get base name without extension
            base_name = os.path.splitext(img_file)[0]

            # Construct superpixel and annotation filenames
            superpixel_file = f"{base_name}{superpixel_suffix}{superpixel_extension}"
            annotation_file = f"{base_name}{annotation_extension}"

            img_path = os.path.join(image_dir, img_file)
            superpixel_path = os.path.join(superpixel_dir, superpixel_file)
            annotation_path = os.path.join(annotation_dir, annotation_file)

            # Only add if both superpixel and annotation exist
            if os.path.exists(superpixel_path) and os.path.exists(annotation_path):
                data_triplets.append({
                    'image': img_path,
                    'superpixel': superpixel_path,
                    'annotation': annotation_path,
                    'image_id': base_name
                })

        if len(data_triplets) == 0:
            raise ValueError(
                f"No matching image-superpixel-annotation triplets found.\n"
                f"Image dir: {image_dir}\n"
                f"Superpixel dir: {superpixel_dir}\n"
                f"Annotation dir: {annotation_dir}\n"
                f"Expected patterns:\n"
                f"  - Superpixel: {{image_name}}{superpixel_suffix}{superpixel_extension}\n"
                f"  - Annotation: {{image_name}}{annotation_extension}"
            )

        # Convert to HF Dataset
        self.ds = HFDataset.from_list(data_triplets)

        # Cast image and superpixel columns to Image features for automatic PIL loading
        self.ds = self.ds.cast_column('image', HFImageFeature())
        self.ds = self.ds.cast_column('superpixel', HFImageFeature())

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

    @staticmethod
    def decode_superpixel_mask(superpixel_image: Image.Image) -> np.ndarray:
        """
        Decode superpixel mask from RGB-encoded PNG to integer IDs.

        The superpixel IDs are encoded in the RGB channels using bit-shifting:
        ID = R + (G << 8) + (B << 16)

        Parameters
        ----------
        superpixel_image : PIL.Image
            RGB-encoded superpixel mask

        Returns
        -------
        np.ndarray
            Integer array of shape [H, W] with superpixel IDs
        """
        # Convert to numpy array
        sp_array = np.array(superpixel_image)

        # Decode RGB to integer ID
        # ID = R + (G << 8) + (B << 16)
        if sp_array.ndim == 3 and sp_array.shape[2] >= 3:
            r = sp_array[:, :, 0].astype(np.int32)
            g = sp_array[:, :, 1].astype(np.int32)
            b = sp_array[:, :, 2].astype(np.int32)
            superpixel_ids = r + (g << 8) + (b << 16)
        else:
            # If grayscale, assume it's already decoded
            superpixel_ids = sp_array.astype(np.int32)

        return superpixel_ids

    @staticmethod
    def parse_annotation_json(annotation_path: str) -> Dict[str, list]:
        """
        Parse JSON annotation file.

        Parameters
        ----------
        annotation_path : str
            Path to JSON annotation file

        Returns
        -------
        dict
            Dictionary with feature names as keys and binary label arrays as values
        """
        import json

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        return annotations

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch

        real_idx = self._indices[idx]
        item = self.ds[real_idx]

        # Load image and superpixel mask
        image = _to_pil(item['image'])
        superpixel_img = _to_pil(item['superpixel'])

        # Decode superpixel mask to integer IDs
        superpixel_mask = self.decode_superpixel_mask(superpixel_img)

        # Load and parse annotations
        annotations = self.parse_annotation_json(item['annotation'])

        # Convert annotations to tensor format [N, 4]
        # where N is the number of superpixels
        num_superpixels = max(
            len(annotations.get("network", [])),
            len(annotations.get("negative network", [])),
            len(annotations.get("streaks", [])),
            len(annotations.get("milia-like cysts", []))
        )

        # Build feature tensor
        feature_labels = np.zeros((num_superpixels, len(self.FEATURE_NAMES)), dtype=np.float32)
        for feat_idx, feat_name in enumerate(self.FEATURE_NAMES):
            if feat_name in annotations:
                labels = annotations[feat_name]
                feature_labels[:len(labels), feat_idx] = labels

        # Apply transforms if provided
        if self.transform is not None:
            image, superpixel_mask = self.transform(image, superpixel_mask)
        else:
            # Default: just convert to tensor
            from torchvision import transforms as T
            image = T.ToTensor()(image)
            superpixel_mask = torch.from_numpy(superpixel_mask).long()

        # Convert feature labels to tensor
        feature_labels = torch.from_numpy(feature_labels)

        return {
            "pixel_values": image,
            "superpixel_mask": superpixel_mask,
            "features": feature_labels,
            "image_id": item['image_id'],
            "num_superpixels": num_superpixels
        }

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

    @property
    def hf_dataset(self):
        return self.ds
