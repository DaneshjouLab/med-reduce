"""
This script is a baseline for comparing different image classification models
at three different image compression levels, in comparison to the original.
It supports fine-tuning, linear probing, and optional degradation transforms.

Purpose: General pipeline for comparing image classification models (ViT, DINOv2, SimCLR) with optional image degradation transforms.
Features:
- Supports fine-tuning and linear probing.
- Balances dataset across classes.
- Applies optional transforms (JPEG, blur, quantization).
- Uses a fixed model list (currently ViT).
- No command-line argument parsing; hyperparameters are set in the main() function.
- Designed for baseline model comparison across compression levels.
"""

# Environment Setup
import os
import io
import json
import numpy as np
from PIL import Image

# PyTorch & Torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

# Hugging Face Transformers & Datasets
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTForImageClassification,
)
from datasets import load_dataset

# Weights & Biases
import wandb

# Model Profiling & Vision Backbones
import timm
from thop import profile

# Local Application Imports
from utils.constants import SSL_MODEL, SIMCLR_BACKBONE, FILTERED_CLASSES, NUM_FILTERED_CLASSES
from utils.transforms import JPEGCompressionTransform, GaussianBlurTransform, ColorQuantizationTransform
from utils.util_classes import SimCLRForClassification, LossLoggerCallback
from utils.util_methods import env_path, compute_metrics, get_gpu_memory, freeze_backbone

# GPU Memory Monitoring (optional)
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("pynvml not installed, GPU memory monitoring disabled.")

# Cache paths
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "~/.cache/huggingface/transformers")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", "~/.cache/huggingface/datasets")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "~/.cache/huggingface")


class WandbCallback(TrainerCallback):
    def __init__(self, model_name, phase):
        self.model_name = model_name
        self.phase = phase
        self.best_accuracy = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["model"] = self.model_name
            logs["phase"] = self.phase
            if GPU_AVAILABLE:
                logs["gpu_memory_mb"] = get_gpu_memory()
            wandb.log(logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            if "eval_accuracy" in metrics:
                self.best_accuracy = max(self.best_accuracy, metrics["eval_accuracy"])
                metrics["best_accuracy"] = self.best_accuracy
            wandb.log(metrics)


def initialize_model_and_preprocessor(model_info, resolution):
    name, model_id, typ, config = (
        model_info["name"],
        model_info["model_id"],
        model_info["type"],
        model_info["config"],
    )

    if typ == "vit":
        preprocessor = ViTFeatureExtractor.from_pretrained(
            model_id,
            size=resolution,
            do_resize=True,
            resample=Image.LANCZOS,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
        model = ViTForImageClassification.from_pretrained(
            model_id,
            num_labels=NUM_FILTERED_CLASSES,
            ignore_mismatched_sizes=True,
            image_size=resolution,
        )
    elif typ == "dinov2":
        preprocessor = AutoImageProcessor.from_pretrained(
            model_id,
            size=resolution,
            do_resize=True,
            resample=Image.LANCZOS,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
        model = AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=NUM_FILTERED_CLASSES,
            ignore_mismatched_sizes=True,
            image_size=resolution,
        )
    elif typ == SSL_MODEL:
        backbone = timm.create_model(
            SIMCLR_BACKBONE,
            pretrained=True,
            num_classes=0,
        )
        model = SimCLRForClassification(backbone, NUM_FILTERED_CLASSES)
        freeze_backbone(model, SSL_MODEL)
        preprocessor = None
    else:
        raise ValueError(f"Unsupported model type: {typ}")

    return model, preprocessor


def balance_dataset(dataset, num_train_images, filtered_classes):
    print("Balancing dataset...")
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


def prepare_datasets(dataset, preprocessor, resolution, proportion_per_transform, apply_transforms=False):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if apply_transforms:
        transform = transforms.Compose([
            transform,
            JPEGCompressionTransform(quality=75),
            GaussianBlurTransform(p=0.5),
            ColorQuantizationTransform(p=0.5),
        ])

    class TorchDataset(Dataset):
        def __init__(self, hf_dataset, transform):
            self.hf_dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            item = self.hf_dataset[idx]
            image = Image.open(io.BytesIO(item["image"])).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = int(item["label"])
            return {"pixel_values": image, "labels": label}

    train_ds = TorchDataset(train_dataset, transform)
    val_ds = TorchDataset(val_dataset, transform)
    return train_ds, val_ds


def main(num_train_images=100, proportion_per_transform=0.2, resolution=224, batch_size=256, num_epochs=3, eval_steps=10, learning_rate=1e-4):
    wandb_config = {
        "num_train_images": num_train_images,
        "proportion_per_transform": proportion_per_transform,
        "resolution": resolution,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "eval_steps": eval_steps,
        "weight_decay": 0.01,
        "learning_rate": learning_rate,
        "gpu_available": GPU_AVAILABLE,
    }

    models = [
        {"name": "vit", "model_id": "google/vit-base-patch16-224", "type": "vit", "config": {
            "image_size": resolution,
            "num_labels": NUM_FILTERED_CLASSES,
            "ignore_mismatched_sizes": True
        }},
    ]

    dataset = load_dataset(
        "MKZuziak/ISIC_2019_224",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        split="train",
    )

    dataset = balance_dataset(dataset, num_train_images, FILTERED_CLASSES)

    for model_info in models:
        model, preprocessor = initialize_model_and_preprocessor(model_info, resolution)

        train_ds, val_ds = prepare_datasets(dataset, preprocessor, resolution, proportion_per_transform, apply_transforms=True)

        results = train_model(
            model, train_ds, val_ds, model_info["name"], model_info["type"],
            resolution, batch_size, num_epochs, learning_rate, eval_steps, wandb_config
        )

        print(f"Results for {model_info['name']}: {results}")


if __name__ == "__main__":
    main()