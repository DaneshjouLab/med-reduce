"""Dataset implementations and data utilities."""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset
from typing import Optional, List, Dict, Any, Union

from src.config import HF_MODELS, DEFAULT_IMAGE_SIZE

# ============================================================================
# DATA LOADING
# ============================================================================

class DatasetWrapper(Dataset):
    """Simple wrapper that handles dataset/subset access patterns."""

    def __init__(self, dataset: Union[Dataset, Subset]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

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
        else:
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

    def __init__(self, preprocessor: Optional[Any] = None, model_type: str = "vit"):
        self.preprocessor = preprocessor
        self.model_type = model_type

        if self.model_type not in HF_MODELS:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def preprocess(self, image: Image.Image, resolution: int) -> torch.Tensor:
        """Apply model-specific preprocessing."""
        # For HuggingFace models
        if hasattr(self.preprocessor, 'size'):
            self.preprocessor.size = resolution

        encoding = self.preprocessor(images=image, return_tensors="pt")
        return encoding["pixel_values"].squeeze(0)


# ============================================================================
# COMBINED DATASET
# ============================================================================

class ISICDataset(Dataset):
    """ISIC dataset that combines data loading, image processing, and model preprocessing."""

    def __init__(
        self,
        dataset: Union[Dataset, Subset],
        preprocessor: Optional[Any] = None,
        resolution: int = DEFAULT_IMAGE_SIZE,
        transform: Optional[Any] = None,
        model_type: str = "vit",
    ):
        # Data loading
        self.data_wrapper = DatasetWrapper(dataset)

        # Image processing
        self.image_processor = ImageProcessor(resolution)

        # Model preprocessing
        self.model_preprocessor = ModelPreprocessor(preprocessor, model_type)

        # Transformation
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_wrapper)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Load raw data
        item = self.data_wrapper.get_raw_item(idx)
        image = item["image"]
        label = item["label"]

        # 2. Process image
        image = self.image_processor.resize_image(image)
        image = self.image_processor.apply_transforms(image, self.transform)

        # 3. Apply model-specific preprocessing
        pixel_values = self.model_preprocessor.preprocess(
            image, self.image_processor.resolution
        )

        # 4. Prepare output
        label = torch.tensor(label, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label}

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


def create_transformed_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    transforms_list: List[Any],
    proportion_per_transform: float,
    preprocessor: Optional[Any],
    resolution: int,
    model_type: str
) -> tuple[Dataset, Dataset]:
    """Create train and validation datasets with transformations."""

    # Split training data into subsets
    train_subsets = split_dataset_for_transforms(
        train_dataset, transforms_list, proportion_per_transform
    )

    # Create datasets with transforms
    transformed_datasets = []

    # Apply each transform to corresponding subset
    for i, (subset, transform) in enumerate(zip(train_subsets[:-1], transforms_list)):
        transformed_ds = ISICDataset(
            subset, preprocessor, resolution, transform, model_type
        )
        transformed_datasets.append(transformed_ds)

    # Add untransformed subset (if any remaining)
    if len(train_subsets) > len(transforms_list):
        untransformed_ds = ISICDataset(
            train_subsets[-1], preprocessor, resolution, None, model_type
        )
        transformed_datasets.append(untransformed_ds)

    # Combine all training datasets
    train_ds = ConcatDataset(transformed_datasets)

    # Create validation dataset (no transformations)
    val_ds = ISICDataset(val_dataset, preprocessor, resolution, None, model_type)

    return train_ds, val_ds

# ============================================================================
# DATASET BALANCING
# ============================================================================

def get_class_distribution(dataset: Dataset, filtered_classes: List[str]) -> Dict[str, List[int]]:
    """Get class distribution and indices."""
    class_counts = {label: 0 for label in filtered_classes}
    class_indices = {label: [] for label in filtered_classes}

    for i, item in enumerate(dataset):
        label_str = str(item["label"])
        if label_str in filtered_classes:
            class_counts[label_str] += 1
            class_indices[label_str].append(i)

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
    for label, indices in class_indices.items():
        sampled = np.random.choice(indices, images_per_class, replace=False)
        balanced_indices.extend(sampled)

    np.random.shuffle(balanced_indices)
    return balanced_indices


def balance_dataset(
    dataset: Dataset,
    filtered_classes: List[str],
    num_train_images: int,
    seed: int = 42
) -> Dataset:
    """Balance dataset by sampling equal numbers from each class."""
    # Get class distribution
    class_indices = get_class_distribution(dataset, filtered_classes)

    # Sample balanced indices
    balanced_indices = sample_balanced_indices(class_indices, num_train_images, seed)

    # Return balanced dataset
    return dataset.select(balanced_indices)