"""Dataset implementations and data utilities."""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision import transforms
from typing import Optional, List, Dict, Any, Union

from src.config import HF_MODELS, DEFAULT_IMAGE_SIZE
from src.transformation.transforms import ResolutionReductionTransform

class ISICDataset(Dataset):
    """ISIC dataset with support for multiple model types and transformations."""

    def __init__(
        self,
        dataset: Union[Dataset, Subset],
        preprocessor: Optional[Any] = None,
        resolution: int = DEFAULT_IMAGE_SIZE,
        transform: Optional[transforms.Compose] = None,
        model_type: str = "vit",
        jpeg_quality: Optional[int] = None,
    ):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.resolution = resolution
        self.transform = transform
        self.model_type = model_type
        self.jpeg_quality = jpeg_quality

        # Create base preprocessing
        self.base_preprocessor = transforms.Compose([
            transforms.Resize((resolution, resolution), Image.LANCZOS),
            transforms.ToTensor(),
        ])

        self.model_preprocessor = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Convert numpy types to Python int
        if isinstance(idx, (np.integer, np.int64)):
            idx = int(idx)

        # Handle both direct dataset and Subset access
        if hasattr(self.dataset, 'dataset'):
            # This is a Subset
            subset_idx = int(self.dataset.indices[idx])
            item = self.dataset.dataset[subset_idx]
        else:
            # Direct dataset access
            item = self.dataset[idx]

        image = item["image"]
        label = item["label"]

        # Resize to target resolution
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Apply optional transformations
        if self.transform:
            image = self.transform(image)

        if self.jpeg_quality is not None:
            image = JPEGCompressionTransform(self.jpeg_quality)(image)

        # Apply model-specific preprocessing
        if self.model_type in HF_MODELS:
            # For HuggingFace models
            if hasattr(self.preprocessor, 'size'):
                self.preprocessor.size = self.resolution
            encoding = self.preprocessor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        label = torch.tensor(label, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label}

def create_transformed_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    transforms_list: List,
    proportion_per_transform: float,
    preprocessor: Optional[Any],
    resolution: int,
    model_type: str
) -> tuple[Dataset, Dataset]:
    """
    Create train and validation datasets with transformations.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        transforms_list: List of transforms to apply
        proportion_per_transform: Proportion of data for each transform
        preprocessor: Model preprocessor
        resolution: Image resolution
        model_type: Type of model

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    num_images = len(train_dataset)
    images_per_transform = int(num_images * proportion_per_transform)

    # Shuffle indices
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    transformed_datasets = []
    used_indices = []

    # Apply each transform to a subset
    for i, transform in enumerate(transforms_list):
        start_idx = i * images_per_transform
        end_idx = start_idx + images_per_transform
        subset_indices = indices[start_idx:end_idx]
        used_indices.extend(subset_indices)

        subset = Subset(train_dataset, subset_indices)
        transform_compose = transforms.Compose([transform])

        transformed_ds = ISICDataset(
            subset,
            preprocessor,
            resolution,
            transform_compose,
            model_type
        )
        transformed_datasets.append(transformed_ds)

    # Add remaining samples without transformation
    remaining_indices = np.setdiff1d(indices, used_indices)
    if len(remaining_indices) > 0:
        remaining_subset = Subset(train_dataset, remaining_indices)
        untransformed_ds = ISICDataset(
            remaining_subset,
            preprocessor,
            resolution,
            None,
            model_type
        )
        transformed_datasets.append(untransformed_ds)

    # Combine all training datasets
    train_ds = ConcatDataset(transformed_datasets)

    # Create validation dataset (no transformations)
    val_ds = ISICDataset(
        val_dataset,
        preprocessor,
        resolution,
        model_type=model_type,
    )

    return train_ds, val_ds

def balance_dataset(dataset: Dataset, filtered_classes: List[str], num_train_images: int, seed: int = 42):
    """
    Balance dataset by sampling equal numbers from each class.

    Args:
        dataset: Input dataset
        filtered_classes: Classes to keep
        num_train_images: Total number of training images
        seed: Random seed

    Returns:
        Balanced dataset
    """
    # Get class counts
    class_counts = {label: 0 for label in filtered_classes}
    class_indices = {label: [] for label in filtered_classes}

    for i, item in enumerate(dataset):
        label_str = str(item["label"])
        if label_str in filtered_classes:
            class_counts[label_str] += 1
            class_indices[label_str].append(i)

    print(f"Class counts before balancing: {class_counts}")

    # Calculate samples per class
    min_class_size = min(class_counts.values())
    images_per_class = min(num_train_images // len(filtered_classes), min_class_size)

    # Sample from each class
    np.random.seed(seed)
    balanced_indices = []

    for label in filtered_classes:
        indices = class_indices[label]
        sampled = np.random.choice(indices, images_per_class, replace=False)
        balanced_indices.extend(sampled)

    np.random.shuffle(balanced_indices)
    return dataset.select(balanced_indices)