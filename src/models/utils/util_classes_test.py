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
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.resolution = resolution
        self.transform = transform
        self.model_type = model_type
        self.jpeg_quality = jpeg_quality
        
        # Create a base preprocessing pipeline that always resizes to the target resolution
        self.base_preprocessor = transforms.Compose([
            transforms.Resize((resolution, resolution), Image.LANCZOS),
            transforms.ToTensor(),
        ])
        
        if model_type == SSL_MODEL:
            self.preprocessor = transforms.Compose([
                transforms.Resize((resolution, resolution), Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        if self.transform:
            image = self.transform(image)

        if self.jpeg_quality is not None:
            image = JPEGCompressionTransform(self.jpeg_quality)(image)

        if self.model_type in HF_MODELS:
            # For HF models, ensure the preprocessor doesn't resize again
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
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, pixel_values, labels=None):
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
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, f"{model_name}_{phase}_log.jsonl"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with open(self.log_file, "a") as f:
            json.dump({"step": state.global_step, **logs}, f)
            f.write("\n")
