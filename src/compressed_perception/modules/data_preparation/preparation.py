# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import io
from datasets import ClassLabel

def filter_and_cast_dataset(dataset, filtered_classes, num_classes):
    """
    Filter dataset by class labels and cast label column.
    """
    filtered_indices = [
        i for i, label in enumerate(dataset["label"])
        if str(label) in filtered_classes
    ]
    dataset = dataset.select(filtered_indices)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=num_classes))
    return dataset

def balance_dataset(dataset, num_train_images, filtered_classes):
    """
    Balance the dataset by sampling an equal number of images per class.
    """
    class_counts = {label: 0 for label in filtered_classes}
    for label in dataset["label"]:
        class_counts[str(label)] += 1

    min_class_size = min(class_counts.values())
    images_per_class = min(num_train_images // len(filtered_classes), min_class_size)

    np.random.seed(42)
    balanced_indices = []
    for label in filtered_classes:
        class_indices = [i for i, l in enumerate(dataset["label"]) if str(l) == label]
        sampled_indices = np.random.choice(class_indices, images_per_class, replace=False)
        balanced_indices.extend(sampled_indices)

    np.random.shuffle(balanced_indices)
    return dataset.select(balanced_indices)

def split_dataset(dataset, test_size=0.2, stratify_by_column="label", seed=42):
    """
    Split dataset into train and validation sets.
    """
    return dataset.train_test_split(test_size=test_size, stratify_by_column=stratify_by_column, seed=seed)

def get_default_transforms(resolution, apply_transforms=False):
    """
    Get torchvision transform pipeline.
    """
    transform_list = [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    if apply_transforms:
        from src.compressed_perception.modules.data_transformation.image_transformation import (
            JPEGCompressionTransform,
            GaussianBlurTransform,
            ColorQuantizationTransform
        )
        transform_list.extend([
            JPEGCompressionTransform(quality=75),
            GaussianBlurTransform(p=0.5),
            ColorQuantizationTransform(p=0.5),
        ])

    return transforms.Compose(transform_list)

class TorchDataset(Dataset):
    """
    PyTorch Dataset wrapper for Hugging Face datasets.
    """
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = Image.open(io.BytesIO(item["image"])).convert("RGB")
        image = self.transform(image) if self.transform else image
        return {"pixel_values": image, "labels": int(item["label"])}

def prepare_datasets(dataset, transform, split_ratio=0.8):
    """
    Prepare PyTorch-compatible train and val datasets.
    """
    train_size = int(split_ratio * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    return TorchDataset(train_dataset, transform), TorchDataset(val_dataset, transform)
