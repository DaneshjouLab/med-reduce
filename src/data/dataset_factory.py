# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""Dataset implementations and data utilities."""
# Standard library imports
from typing import Optional, List, Dict, Any, Union

# Third-party imports
import numpy as np  # pylint: disable=import-error
import torch  # pylint: disable=import-error
from PIL import Image  # pylint: disable=import-error
from torch.utils.data import Dataset, ConcatDataset, Subset  # pylint: disable=import-error
from datasets import Dataset as HFDataset

# Local imports
# pylint: disable=import-error,relative-beyond-top-level
from src.utils.constants import HF_MODELS, DEFAULT_IMAGE_SIZE

# ============================================================================
# DATA LOADING
# ============================================================================

class DatasetWrapper(Dataset):
    """Simple wrapper that handles dataset/subset access patterns."""

    def __init__(self, dataset: Union[Dataset, Subset]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item from dataset, satisfying Dataset abstract method."""
        return self.get_raw_item(idx)

    def get_raw_item(self, idx: int) -> Dict[str, Any]:
        """Get raw item from dataset, handling both Dataset and Subset."""
        # Convert numpy types to Python int
        if isinstance(idx, (np.integer, np.int64)):
            idx = int(idx)

        # Handle both direct dataset and Subset access
        if hasattr(self.dataset, 'dataset'):
            # This is a Subset
            subset_idx = int(self.dataset.indices[idx])
            return self.dataset.dataset[subset_idx]
        # Direct dataset access
        return self.dataset[idx]


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

class ImageProcessor:
    """Handles image preprocessing operations."""

    def __init__(self, resolution: int = DEFAULT_IMAGE_SIZE):
        self.resolution = resolution

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to target resolution."""
        return image.resize(
            (self.resolution, self.resolution),
            Image.Resampling.LANCZOS
        )

    def apply_transforms(self, image: Image.Image, transform: Optional[Any]) -> Image.Image:
        """Apply optional transformations to image."""
        if transform:
            return transform(image)
        return image


# ============================================================================
# MODEL-SPECIFIC PREPROCESSING
# ============================================================================

class ModelPreprocessor:
    """Handles model-specific preprocessing."""
    # pylint: disable=too-few-public-methods

    def __init__(self, preprocessor: Optional[Any] = None, model_type: str = "vit"):
        self.preprocessor = preprocessor
        self.model_type = model_type

        if self.preprocessor is None:
            # Will no-op and return a tensorized fallback later
            return

        if self.model_type not in HF_MODELS:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def preprocess(self, image: Image.Image, resolution: int) -> torch.Tensor:
        """Apply model-specific preprocessing; no-op if preprocessor is None."""
        if self.preprocessor is None:
            return self._convert_pil_to_tensor(image)

        # HuggingFace path
        if hasattr(self.preprocessor, "size") and self.preprocessor.size != resolution:
            try:
                self.preprocessor.size = resolution
            except (AttributeError, TypeError) as e: # pylint: disable=unused-variable
                # Some processors use tuples or different APIs for size
                pass

        encoding = self.preprocessor(images=image, return_tensors="pt")
        return encoding["pixel_values"].squeeze(0)

    def _convert_pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor with proper format."""
        # Minimal safety: convert PIL -> tensor [C,H,W] in [0,1]
        arr = np.asarray(image).astype(np.float32) / 255.0
        if arr.ndim == 2:  # grayscale -> [H,W] -> [H,W,1]
            arr = arr[:, :, None]
        # HW(C) -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

# ============================================================================
# DATASET CREATION UTILITIES
# ============================================================================

def split_dataset_for_transforms(
    dataset: Dataset,
    transforms_list: List[Any],
    proportion_per_transform: float
) -> List[Subset]:
    """Split dataset into subsets for applying different transforms."""
    num_images = len(dataset)
    images_per_transform = int(num_images * proportion_per_transform)

    # Shuffle indices
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    subsets = []
    used_indices = []

    # Create subset for each transform
    for i, _ in enumerate(transforms_list):
        start_idx = i * images_per_transform
        end_idx = start_idx + images_per_transform
        subset_indices = indices[start_idx:end_idx]
        used_indices.extend(subset_indices)
        subsets.append(Subset(dataset, subset_indices))

    # Add remaining samples
    remaining_indices = np.setdiff1d(indices, used_indices)
    if len(remaining_indices) > 0:
        subsets.append(Subset(dataset, remaining_indices))

    return subsets

# ============================================================================
# DATASET BALANCING
# ============================================================================

def get_class_distribution(dataset: Dataset, filtered_classes: List[str]) -> Dict[str, List[int]]:
    """Get class distribution and indices."""
    filtered_classes = [str(c) for c in filtered_classes]
    is_hf = isinstance(dataset, HFDataset)

    if is_hf:
        label_col = "label" if "label" in dataset.column_names else "labels"

        labels = np.array(dataset[label_col]).astype(str)

        class_indices = {
            cls: np.where(labels == cls)[0]
            for cls in filtered_classes
        }

        class_counts = {cls: len(class_indices[cls]) for cls in filtered_classes}
        print(f"Class counts before balancing: {class_counts}")

        return class_indices

    class_indices = {cls: [] for cls in filtered_classes}

    for idx in range(len(dataset)):
        item = dataset[idx]
        label_str = str(item["label"])
        if label_str in class_indices:
            class_indices[label_str].append(idx)

    class_indices = {cls: np.array(idxs) for cls, idxs in class_indices.items()}

    class_counts = {cls: len(class_indices[cls]) for cls in filtered_classes}
    print(f"Class counts before balancing: {class_counts}")

    return class_indices


def sample_balanced_indices(
    class_indices: Dict[str, List[int]],
    num_train_images: int,
    seed: int = 42
) -> List[int]:
    """Sample balanced indices from each class."""
    np.random.seed(seed)

    # Calculate samples per class
    num_classes = len(class_indices)
    min_class_size = min(len(indices) for indices in class_indices.values())
    images_per_class = min(num_train_images // num_classes, min_class_size)

    print(f"Sampling {images_per_class} images per class")

    # Sample from each class
    balanced_indices = []
    for _label, indices in class_indices.items():
        sampled = np.random.choice(indices, images_per_class, replace=False)
        balanced_indices.extend(sampled)

    np.random.shuffle(balanced_indices)
    return balanced_indices


def balance_dataset(
    dataset,
    filtered_classes,
    num_train_images,
    seed: int = 42
):
    """
    Balances dataset by sampling equal counts per class.
    Works with HF Datasets and PyTorch Datasets.
    """

    is_hf = isinstance(dataset, HFDataset)
    rng = np.random.default_rng(seed)

    # ============================================================
    # INDEX EXTRACTION
    # ============================================================
    if is_hf:
        if hasattr(dataset, "column_names"):
            label_col = "label" if "label" in dataset.column_names else "labels"
        elif hasattr(dataset, "ds") and hasattr(dataset.ds, "column_names"):
            label_col = "label" if "label" in dataset.ds.column_names else "labels"
        else:
            # fallback for PyTorch-style datasets
            label_col = "label"

        labels = dataset[label_col]  # zero-copy memoryview
        labels = np.array(labels)

        # Convert filtered_classes to match label dtype
        # If labels are int and filtered_classes are strings, convert filtered_classes to int
        print(f"Label dtype: {labels.dtype}, filtered_classes: {filtered_classes}")
        if labels.dtype.kind in ('i', 'u', 'f'):  # integer or float
            # Try to convert filtered_classes to numeric
            try:
                filtered_classes = [int(float(cls)) for cls in filtered_classes]
                print(f"Converted filtered_classes to numeric: {filtered_classes}")
            except (ValueError, TypeError):
                pass  # Keep as-is if conversion fails
        else:
            # Labels are strings or objects, convert to string
            labels = labels.astype(str)
            filtered_classes = [str(cls) for cls in filtered_classes]
            print(f"Converted labels and filtered_classes to strings")

        class_indices = {
            cls: np.where(labels == cls)[0]
            for cls in filtered_classes
        }

        # Debug: Print class distribution
        for cls, indices in class_indices.items():
            print(f"  Class {cls}: {len(indices)} samples")

        if all(len(indices) == 0 for indices in class_indices.values()):
            unique_labels = np.unique(labels)
            raise ValueError(
                f"No samples found for any filtered class! "
                f"Filtered classes: {filtered_classes}, "
                f"Unique labels in dataset: {unique_labels[:20]}"  # Show first 20
            )

    else:
        class_indices = {cls: [] for cls in filtered_classes}

        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label in class_indices:
                class_indices[label].append(idx)

        class_indices = {k: np.array(v) for k, v in class_indices.items()}

    # ============================================================
    # BALANCED SAMPLING (NUMPY RNG)
    # ============================================================
    per_class = num_train_images // len(filtered_classes)
    min_samples = min(len(arr) for arr in class_indices.values())
    per_class = min(per_class, min_samples)

    balanced_indices = []
    for cls in filtered_classes:
        arr = class_indices[cls]
        if len(arr) < per_class:
            raise ValueError(f"Class {cls} only has {len(arr)} samples.")

        chosen = rng.choice(arr, size=per_class, replace=False)
        balanced_indices.append(chosen)

    balanced_indices = np.concatenate(balanced_indices)

    if is_hf:
        balanced_indices.sort()
        return dataset.select(balanced_indices.tolist())

    return Subset(dataset, balanced_indices.tolist())