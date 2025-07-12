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
import argparse
from PIL import Image
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


from src.compressed_perception.models.training.constants import (
    SSL_MODEL, SIMCLR_BACKBONE, FILTERED_CLASSES, NUM_FILTERED_CLASSES
)
from src.compressed_perception.models.training.utils_classes import SimCLRForClassification
from src.compressed_perception.models.training.utils_methods import get_gpu_memory, GPU_AVAILABLE, freeze_backbone
from src.compressed_perception.modules.data_preparation.preparation import (
    filter_and_cast_dataset,
    balance_dataset,
    split_dataset,
    get_default_transforms,
    prepare_datasets,
)

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



def main(config=None, dataset=None):
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

    MODEL_CONFIG_KEYS = {
        "name": "name",
        "model_id": "model_id",
        "type": "type",
        "config": "config",
    }

    models = [
        {
            MODEL_CONFIG_KEYS["name"]: "vit",
            MODEL_CONFIG_KEYS["model_id"]: "google/vit-base-patch16-224",
            MODEL_CONFIG_KEYS["type"]: "vit",
            MODEL_CONFIG_KEYS["config"]: {
                "image_size": config["resolution"],
                "num_labels": NUM_FILTERED_CLASSES,
                "ignore_mismatched_sizes": True
            }
        },
    ]

    if dataset is None:
        raise ValueError("Dataset must be provided via `dataset` argument or CLI.")

    # Use preparation.py functions for filtering, balancing, and splitting
    dataset = filter_and_cast_dataset(dataset, FILTERED_CLASSES, NUM_FILTERED_CLASSES)
    dataset = balance_dataset(dataset, config["num_train_images"], FILTERED_CLASSES)
    splits = split_dataset(dataset, test_size=0.2, stratify_by_column="label", seed=42)

    # Get transforms
    transform = get_default_transforms(config["resolution"], apply_transforms=True)

    # Prepare PyTorch datasets
    train_ds, val_ds = prepare_datasets(splits["train"], transform, split_ratio=1.0)
    val_ds, _ = prepare_datasets(splits["test"], transform, split_ratio=1.0)

    for model_info in models:
        model, _preprocessor = initialize_model_and_preprocessor(model_info, config["resolution"])

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

        METRIC_KEYS = {
            "learning_rate": "learning_rate",
            "model_name": "model_name",
            "model_type": "model_type",
            "peak_memory_mb": "peak_memory_mb",
            "flops_giga": "flops_giga",
            "train_time_seconds": "train_time_seconds",
            "eval_time_seconds": "eval_time_seconds",
            "eval_metrics": "eval_metrics",
        }

        metrics = {
            METRIC_KEYS["learning_rate"]: config["learning_rate"],
            METRIC_KEYS["model_name"]: model_info[MODEL_CONFIG_KEYS["name"]],
            METRIC_KEYS["model_type"]: model_info[MODEL_CONFIG_KEYS["type"]],
            METRIC_KEYS["peak_memory_mb"]: get_gpu_memory(),
            METRIC_KEYS["flops_giga"]: None,
            METRIC_KEYS["train_time_seconds"]: None,
            METRIC_KEYS["eval_time_seconds"]: None,
            METRIC_KEYS["eval_metrics"]: results,
        }
        wandb.log({"metrics": metrics})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline model comparison for image classification.")
    parser.add_argument(
        "--path_to_dataset",
        type=str,
        default=None,
        help="Path to local dataset directory. If not provided, loads from Hugging Face.",
    )
    args = parser.parse_args()

    dataset = None
    if args.path_to_dataset:
        dataset = load_dataset("imagefolder", data_dir=args.path_to_dataset, split="train")
    else:
        try:
            dataset = load_dataset(
                "MKZuziak/ISIC_2019_224",
                cache_dir=os.environ["HF_DATASETS_CACHE"],
                split="train",
            )
        except Exception as e:
            raise ValueError("No dataset provided and Hugging Face dataset failed to load.") from e

    main(dataset=dataset)