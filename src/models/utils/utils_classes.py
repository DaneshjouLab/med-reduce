"""
Utility classes for model training and data handling.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import TrainerCallback

from .constants import HF_MODELS, NUM_FILTERED_CLASSES, SSL_MODEL
from .transforms import JPEGCompressionTransform


class ISICDataset(Dataset):
    def __init__(
        self,
        dataset,
        preprocessor=None,
        resolution=224,
        transform=None,
        model_type="vit",
        jpeg_quality=None,
    ):
        """
        Dataset class for handling ISIC image data.

        Args:
            dataset: The dataset to load.
            preprocessor: Preprocessing function for Hugging Face models.
            resolution: Target image resolution.
            transform: Additional transformations to apply.
            model_type: Type of model (e.g., "vit", "ssl").
            jpeg_quality: JPEG compression quality (if applicable).
        """
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.resolution = resolution
        self.transform = transform
        self.model_type = model_type
        self.jpeg_quality = jpeg_quality

        # Base preprocessing pipeline for resizing and tensor conversion
        self.base_preprocessor = transforms.Compose([
            transforms.Resize((resolution, resolution), Image.LANCZOS),
            transforms.ToTensor(),
        ])

        # Preprocessor for SSL models
        if model_type == SSL_MODEL:
            self.preprocessor = transforms.Compose([
                transforms.Resize((resolution, resolution), Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert numpy.int64 to Python int if necessary
        if isinstance(idx, (np.integer, np.int64)):
            idx = int(idx)

        # Handle both direct dataset access and Subset access
        if hasattr(self.dataset, 'dataset'):
            # This is a Subset
            subset_idx = int(self.dataset.indices[idx])
            item = self.dataset.dataset[subset_idx]
        else:
            # This is a direct dataset
            item = self.dataset[idx]

        image = item["image"]
        label = item["label"]

        # Always resize to target resolution first
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Apply additional transformations if provided
        if self.transform:
            image = self.transform(image)

        # Apply JPEG compression if specified
        if self.jpeg_quality is not None:
            image = JPEGCompressionTransform(self.jpeg_quality)(image)

        # Preprocessing for Hugging Face models
        if self.model_type in HF_MODELS:
            # Ensure the preprocessor doesn't resize again
            if hasattr(self.preprocessor, 'size'):
                self.preprocessor.size = self.resolution
            encoding = self.preprocessor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)
        elif self.model_type == SSL_MODEL:
            pixel_values = self.preprocessor(image)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        label = torch.tensor(label, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label}


class SimCLRForClassification(nn.Module):
    def __init__(self, backbone, num_classes=NUM_FILTERED_CLASSES):
        """
        SimCLR-based classification model.

        Args:
            backbone: The backbone model (e.g., ResNet).
            num_classes: Number of output classes.
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, pixel_values, labels=None):
        """
        Forward pass for the model.

        Args:
            pixel_values: Input image tensors.
            labels: Ground truth labels (optional).

        Returns:
            dict: Dictionary containing logits and loss (if labels are provided).
        """
        features = self.backbone(pixel_values)
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return (
            {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}
        )


class LossLoggerCallback(TrainerCallback):
    """
    Logs each training step's loss and other metrics to a structured JSON Lines file.
    """

    def __init__(self, log_dir: str, phase: str, model_name: str):
        """
        Initialize the callback.

        Args:
            log_dir: Directory to save the log file.
            phase: Training phase (e.g., "finetune").
            model_name: Name of the model.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, f"{model_name}_{phase}_log.jsonl"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Log metrics to a JSON Lines file.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            logs: Metrics to log.
        """
        if logs is None:
            return
        with open(self.log_file, "a") as f:
            json.dump({"step": state.global_step, **logs}, f)
            f.write("\n")