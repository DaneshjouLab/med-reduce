# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
Utility classes for model training and data handling.
"""

import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import TrainerCallback

from src.compressed_perception.models.training.constants import HF_MODELS, NUM_FILTERED_CLASSES, SSL_MODEL
from src.compressed_perception.modules.data_transformation.image_transformation import JPEGCompressionTransform

# Compatibility for LANCZOS resampling
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS # pylint: disable=no-member


from transformers import TrainerCallback

class WandbCallback(TrainerCallback):
    """
    Custom callback for logging metrics and evaluation results to Weights & Biases.
    Tracks best accuracy and GPU memory usage if available.
    """
    MODEL_KEY = "model"
    PHASE_KEY = "phase"
    BEST_ACCURACY_KEY = "best_accuracy"
    EVAL_ACCURACY_KEY = "eval_accuracy"
    GPU_MEMORY_KEY = "gpu_memory_mb"

    def __init__(self, model_name, phase):
        self.model_name = model_name
        self.phase = phase
        self.best_accuracy = 0.0

    def on_log(self, _args, _state, _control, logs=None, **_kwargs):
        if logs is not None:
            logs[self.MODEL_KEY] = self.model_name
            logs[self.PHASE_KEY] = self.phase
            try:
                from src.compressed_perception.models.training.utils_methods import GPU_AVAILABLE, get_gpu_memory
                if GPU_AVAILABLE:
                    logs[self.GPU_MEMORY_KEY] = get_gpu_memory()
            except ImportError:
                pass
            import wandb
            wandb.log(logs)

    def on_evaluate(self, _args, _state, _control, metrics=None, **_kwargs):
        if metrics is not None:
            if self.EVAL_ACCURACY_KEY in metrics:
                self.best_accuracy = max(self.best_accuracy, metrics[self.EVAL_ACCURACY_KEY])
                metrics[self.BEST_ACCURACY_KEY] = self.best_accuracy
            import wandb
            wandb.log(metrics)

def get_trainer_callbacks(name):
    """Get callbacks for the Trainer."""
    return [
        LossLoggerCallback(
            log_dir=os.environ.get("LOG_DIR", "./logs"),
            phase="finetune",
            model_name=name,
        ),
        WandbCallback(name, "finetune"),
    ]
           
class ISICDataset(Dataset):
    """
    Dataset class for handling ISIC image data with optional transformations.
    """
    def __init__(self, dataset, config=None):
        """
        Args:
            dataset: The dataset to load.
            config (dict, optional): Configuration dictionary with keys:
                - preprocessor
                - resolution
                - transform
                - model_type
                - jpeg_quality
        """
        self.dataset = dataset
        config = config or {}
        self.preprocessor = config.get("preprocessor", None)
        self.resolution = config.get("resolution", 224)
        self.transform = config.get("transform", None)
        self.model_type = config.get("model_type", "vit")
        self.jpeg_quality = config.get("jpeg_quality", None)

        # Base preprocessing pipeline for resizing and tensor conversion
        self.base_preprocessor = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), LANCZOS),
            transforms.ToTensor(),
        ])

        # Preprocessor for SSL models
        if self.model_type == SSL_MODEL:
            self.preprocessor = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution), LANCZOS),
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
        image = image.resize((self.resolution, self.resolution), LANCZOS)

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


class SimCLRForClassification(nn.Module): # pylint: disable=too-few-public-methods
    """
    SimCLR-based classification model.
    """
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


class LossLoggerCallback(TrainerCallback): # pylint: disable=too-few-public-methods
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

    def on_log(self, _args, state, _control, logs=None, **_kwargs):
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
        with open(self.log_file, "a", encoding="utf-8") as f:
            json.dump({"step": state.global_step, **logs}, f)
            f.write("\n")
