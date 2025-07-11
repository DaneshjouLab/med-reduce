# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
This script is a baseline for comparing different image classification models
at three different image compression levels, in comparison to the original.
It supports fine-tuning, linear probing, and optional degradation transforms.

Purpose: General pipeline for comparing image classification models (ViT, DINOv2, SimCLR) with
optional image degradation transforms.
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
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ViTFeatureExtractor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import wandb
import timm

from src.models.utils.constants import (
    SSL_MODEL, SIMCLR_BACKBONE, FILTERED_CLASSES, NUM_FILTERED_CLASSES
)
from src.models.utils.transforms import (
    JPEGCompressionTransform, GaussianBlurTransform,
)
from src.models.utils.utils_classes import SimCLRForClassification, LossLoggerCallback
from src.models.utils.utils_methods import get_gpu_memory, GPU_AVAILABLE, freeze_backbone
from src.models.utils.transforms import ColorQuantizationTransform

# Compatibility for LANCZOS resampling
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS # pylint: disable=no-member

# GPU Memory Monitoring (optional)
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("pynvml not installed, GPU memory monitoring disabled.")

class WandbCallback:
    """Callback for logging to Weights & Biases."""
    def __init__(self, model_name, phase):
        self.model_name = model_name
        self.phase = phase
        self.best_accuracy = 0.0

    def on_log(self, _args, _state, _control, logs=None, **_kwargs):
        """
        Log metrics to Weights & Biases."""
        if logs is not None:
            logs["model"] = self.model_name
            logs["phase"] = self.phase
            if GPU_AVAILABLE:
                logs["gpu_memory_mb"] = get_gpu_memory()
            wandb.log(logs)

    def on_evaluate(self, _args, _state, _control, metrics=None, **_kwargs):
        """Log evaluation metrics to Weights & Biases."""
        if metrics is not None:
            if "eval_accuracy" in metrics:
                self.best_accuracy = max(self.best_accuracy, metrics["eval_accuracy"])
                metrics["best_accuracy"] = self.best_accuracy
            wandb.log(metrics)


def initialize_model_and_preprocessor(model_info, resolution):
    """
    Initialize model and preprocessor.
    """
    _, model_id, typ, _ = (
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
            resample=LANCZOS,
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
            resample=LANCZOS,
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
    """
    Balance the dataset by sampling equal images per class.
    """
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


def prepare_datasets(dataset, _preprocessor, resolution, apply_transforms=False):
    """
    Prepare train and validation datasets.
    """
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
        """
        Custom dataset class for Hugging Face datasets.
        """
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


def train_model(
    model,
    train_ds,
    val_ds,
    config
):
    """
    Train the model using Hugging Face Trainer.

    Args:
        model: The model to train.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        config (dict): Configuration dictionary with keys:
            - model_name
            - model_type
            - resolution
            - batch_size
            - num_epochs
            - learning_rate
            - eval_steps
            - wandb_config

    Returns:
        dict: Evaluation results.
    """
    wandb.init(
        project="Model Comparison Baseline",
        name=f"{config['model_name']}_{config['resolution']}_{config['num_epochs']}_epochs",
        config=config['wandb_config'],
        tags=["baseline", config['model_name'], f"res_{config['resolution']}"],
        reinit=True,
    )

    train_args = TrainingArguments(
        output_dir=f"./outputs/{config['model_name']}",
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=f"./logs/{config['model_name']}",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=config['eval_steps'],
        save_strategy="steps",
        save_steps=config['eval_steps'],
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    wandb.finish()
    return eval_results


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


def main(config=None):
    """
    Main pipeline for model comparison.
    """
    if config is None:
        config = {
            "num_train_images": 100,
            "proportion_per_transform": 0.2,
            "resolution": 224,
            "batch_size": 256,
            "num_epochs": 3,
            "eval_steps": 10,
            "learning_rate": 1e-4,
            "gpu_available": GPU_AVAILABLE,
        }

    wandb_config = config.copy()
    wandb_config["weight_decay"] = 0.01

    models = [
        {"name": "vit", "model_id": "google/vit-base-patch16-224", "type": "vit", "config": {
            "image_size": config["resolution"],
            "num_labels": NUM_FILTERED_CLASSES,
            "ignore_mismatched_sizes": True
        }},
    ]

    dataset = load_dataset(
        "MKZuziak/ISIC_2019_224",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        split="train",
    )

    dataset = balance_dataset(dataset, config["num_train_images"], FILTERED_CLASSES)

    for model_info in models:
        model, preprocessor = initialize_model_and_preprocessor(model_info, config["resolution"])
        train_ds, val_ds = prepare_datasets(
            dataset, preprocessor, config["resolution"], apply_transforms=True
        )

        train_config = {
            "model_name": model_info["name"],
            "model_type": model_info["type"],
            "resolution": config["resolution"],
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
            "learning_rate": config["learning_rate"],
            "eval_steps": config["eval_steps"],
            "wandb_config": wandb_config,
        }

        results = train_model(
            model, train_ds, val_ds, train_config
        )
        print(f"Results for {model_info['name']}: {results}")

        metrics = {
            "learning_rate": config["learning_rate"],
            "model_name": model_info["name"],
            "model_type": model_info["type"],
            "peak_memory_mb": get_gpu_memory(),
            "flops_giga": None,  # Placeholder for FLOPs, calculate if needed
            "train_time_seconds": None,  # Placeholder for training time, calculate if needed
            "eval_time_seconds": None,  # Placeholder for evaluation time, calculate if needed
            "eval_metrics": results,
        }
        wandb.log({"metrics": metrics})

if __name__ == "__main__":
    main()
