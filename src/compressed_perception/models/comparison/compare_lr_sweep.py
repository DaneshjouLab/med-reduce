# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# pylint: skip-file

"""
This script is a baseline for comparing different image classification models
at three different image compression levels, in comparison to the original.
It has a set number of augmentation transforms and does NOT combine them.
This does NOT experiment on JPEG compression levels


Purpose: Specialized for learning rate experiments with image classification models.
Features:
- Focuses on learning rate sweeps for a single model (currently DINOv2).
- Uses command-line argument parsing for hyperparameters.
- Applies a set number of augmentation transforms (does not combine them).
- Does not experiment with JPEG compression levels.
- Logs results for each learning rate and saves them to a JSON file.
- More flexible for hyperparameter tuning and ablation studies.
"""

# Environment Setup
import os

# Standard Library
import json
import time
import argparse
import shutil

# Scientific & Visualization Libraries
import numpy as np
from PIL import Image

# PyTorch & Torchvision
import torch
from torch.utils.data import Subset, ConcatDataset
from torchvision import transforms

# Hugging Face Transformers & Datasets
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTForImageClassification,
)
from datasets import load_dataset

# Weights & Biases
import wandb

# Model Profiling & Vision Backbones
import timm


# Local Application Imports
from src.compressed_perception.models.training.constants import (
    HF_MODELS, SSL_MODEL, SIMCLR_BACKBONE, FILTERED_CLASSES, NUM_FILTERED_CLASSES
)
from src.compressed_perception.modules.data_transformation.image_transformation import (
    JPEGCompressionTransform,
    GaussianBlurTransform,
    ColorQuantizationTransform,
)
from src.compressed_perception.models.training.utils_classes import (
    ISICDataset,
    SimCLRForClassification,
    WandbCallback,
    LossLoggerCallback,
    get_trainer_callbacks
)
from src.compressed_perception.models.training.utils_methods import (
    env_path,
    compute_metrics,
    cleanup_model_dirs,
    get_gpu_memory,
    freeze_backbone,
    GPU_AVAILABLE,
    get_flops,
    check_disk_space,
    save_model_and_preprocessor,
)

from src.compressed_perception.modules import (
    filter_and_cast_dataset, balance_dataset, split_dataset, get_default_transforms, prepare_datasets
)
# Set memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Cache paths
os.environ["TRANSFORMERS_CACHE"] = os.getenv(
    "TRANSFORMERS_CACHE", "~/.cache/huggingface/transformers"
)
os.environ["HF_DATASETS_CACHE"] = os.getenv(
    "HF_DATASETS_CACHE", "~/.cache/huggingface/datasets"
)
os.environ["HF_HOME"] = os.getenv("HF_HOME", "~/.cache/huggingface")


def create_preprocessors(model_config, config):
    """
    Create preprocessors for each model type.
    """
    preprocessors = {}
    for model_info in [model_config]:
        _name, model_id, typ, _config = (
            model_info["name"],
            model_info["model_id"],
            model_info["type"],
            model_info["config"],
        )
        if typ == "vit":
            preprocessors[typ] = ViTFeatureExtractor.from_pretrained(
                model_id,
                size=config["resolution"],
                do_resize=True,
                resample=Image.LANCZOS, # pylint: disable=no-member
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225]
            )
        elif typ == "dinov2":
            preprocessors[typ] = AutoImageProcessor.from_pretrained(
                model_id,
                size=config["resolution"],
                do_resize=True,
                resample=Image.LANCZOS, # pylint: disable=no-member
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225]
            )
        else:
            preprocessors[typ] = None

    return preprocessors

def train_for_learning_rate(    # pylint: disable=too-many-locals
    learning_rate, model_config, train_dataset, val_dataset, config
):
    """
    Train and evaluate the model for a given learning rate.
    """
    preprocessors = config["preprocessors"]
    name, model_id, typ, _config = (
        model_config["name"],
        model_config["model_id"],
        model_config["type"],
        model_config["config"],
    )

    train_ds = get_transformed_datasets(train_dataset, preprocessors, config, typ)
    val_ds = ISICDataset( # pylint: disable=too-many-function-args
        val_dataset,
        preprocessors[typ],
        config["resolution"],
        typ,
    )

    model = get_model(typ, model_id, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    flops = get_flops(model, config["resolution"])
    check_disk_space(min_gb=1)
    cleanup_model_dirs(name, learning_rate)

    wandb.init(
        entity="ericcui-use-stanford-university",
        project="CS231N Test",
        name=f"{name}_{config['resolution']}_lr_{learning_rate}",
        config={
            "model_name": name,
            "resolution": config["resolution"],
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
            "eval_steps": config["eval_steps"],
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "gpu_available": GPU_AVAILABLE,
        },
        tags=[
            "learning_rate_experiment",
            f"lr_{learning_rate}",
            f"resolution_{config['resolution']}"],
    )

    train_args = get_training_args(name, learning_rate, config)
    callbacks = get_trainer_callbacks(name)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda pred: compute_metrics(pred, name),
        callbacks=callbacks,
    )

    start_time = time.time()
    peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1

    if typ in HF_MODELS:
        wandb.watch(model, log="all", log_freq=100)
    elif typ == SSL_MODEL:
        wandb.watch(model.backbone, log="all", log_freq=100)

    trainer.train()

    current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
    peak_memory = max(peak_memory, current_memory)

    eval_start_time = time.time()
    eval_results = trainer.evaluate()
    eval_time = time.time() - eval_start_time
    train_time = time.time() - start_time - eval_time

    metrics = {
        "learning_rate": learning_rate,
        "model_name": name,
        "model_type": typ,
        "peak_memory_mb": peak_memory,
        "flops_giga": flops,
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
        "eval_metrics": eval_results,
    }
    wandb.log(metrics)

    model_dir = save_model_and_preprocessor(model, preprocessors, typ, name, learning_rate)
    log_wandb_artifact(model_dir, name, learning_rate)

    print(f"[Finetune] Learning Rate {learning_rate}: {metrics}")

    wandb.finish()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics

def get_model(typ, model_id, config):
    """
    Returns the initialized model based on type.
    """
    if typ == "vit":
        return ViTForImageClassification.from_pretrained(
            model_id,
            num_labels=NUM_FILTERED_CLASSES,
            ignore_mismatched_sizes=True,
            image_size=config["resolution"],
        )
    if typ == "dinov2":
        return AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=NUM_FILTERED_CLASSES,
            ignore_mismatched_sizes=True,
            image_size=config["resolution"]
        )
    if typ == SSL_MODEL:
        backbone = timm.create_model( # pylint: disable=too-many-function-args
            SIMCLR_BACKBONE,
            pretrained=True,
            num_classes=0
        )
        model = SimCLRForClassification(backbone, NUM_FILTERED_CLASSES)
        freeze_backbone(model, SSL_MODEL)
        return model
    raise ValueError(f"Unknown model type: {typ}")

def get_transformed_datasets(train_dataset, preprocessors, config, typ):
    """
    Returns a concatenated dataset with optional degradation transforms applied.
    """
    num_images = len(train_dataset)
    images_per_transform = int(num_images * config["proportion_per_transform"])
    indices = np.random.permutation(num_images)

    transforms_list = [
        JPEGCompressionTransform(),
        GaussianBlurTransform(),
        ColorQuantizationTransform(),
    ]

    def make_subset(indices_subset, transform=None):
        subset = Subset(train_dataset, indices_subset)
        transform_compose = transforms.Compose([transform]) if transform else None
        return ISICDataset( # pylint: disable=too-many-function-args
            subset,
            preprocessors[typ],
            config["resolution"],
            transform_compose,
            typ
            )

    datasets = []
    used_indices = set()

    for i, transform in enumerate(transforms_list):
        idx = indices[i * images_per_transform : (i + 1) * images_per_transform]
        used_indices.update(idx)
        datasets.append(make_subset(idx, transform))

    remaining = np.setdiff1d(indices, list(used_indices))
    if len(remaining) > 0:
        datasets.append(make_subset(remaining))

    return ConcatDataset(datasets)

def log_wandb_artifact(model_dir, name, learning_rate):
    """
    Logs the saved model directory as a wandb artifact.
    """
    artifact = wandb.Artifact(
        name=f"{name}_lr_{learning_rate}_model",
        type="model",
        description=f"Trained {name} model with {learning_rate} learning rate"
    )
    artifact.add_dir(model_dir)
    wandb.log_artifact(artifact)

def get_training_args(name, learning_rate, config):
    """
    Returns a TrainingArguments object for Hugging Face Trainer.
    """
    return TrainingArguments(
        output_dir=os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), f"{name}_lr_{learning_rate}"),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=os.path.join(env_path("LOG_DIR", "."), f"{name}_lr_{learning_rate}"),
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["eval_steps"],
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        save_safetensors=False,
        hub_model_id=None,
        hub_strategy="end",
        push_to_hub=False,
        save_only_model=True,
    )

def main(config=None):
    """
    Main function for running learning rate sweep experiments on image classification models.

    Args:
        config (dict): Configuration dictionary.
    """
    if config is None:
        config = {
            "num_train_images": 25000,
            "proportion_per_transform": 0.2,
            "resolution": 224,
            "batch_size": 256,
            "num_epochs": 3,
            "eval_steps": 10,
            "learning_rate": 1e-4,
        }

    model_config = {
        "name": "dinov2",
        "model_id": "facebook/dinov2-base",
        "type": "dinov2",
        "config": {
            "image_size": config["resolution"],
            "num_labels": NUM_FILTERED_CLASSES,
            "ignore_mismatched_sizes": True
        }
    }

    learning_rates = [config["learning_rate"]]
    results = {}

    dataset = load_dataset(
        "MKZuziak/ISIC_2019_224",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        split="train",
    )

    filtered_dataset = filter_and_cast_dataset(dataset, FILTERED_CLASSES, NUM_FILTERED_CLASSES)
    balanced_dataset = balance_dataset(filtered_dataset, config["num_train_images"], FILTERED_CLASSES)
    splits = split_dataset(balanced_dataset)
    transform = get_default_transforms(resolution=224, apply_transforms=True)
    train_dataset, val_dataset = prepare_datasets(splits["train"], transform)

    preprocessors = create_preprocessors(model_config, config)
    config["preprocessors"] = preprocessors

    for lr in learning_rates:
        results[str(lr)] = train_for_learning_rate(
            lr, model_config, train_dataset, val_dataset, config
        )

    # Save results
    with open(
        os.path.join(
            env_path("TRAIN_OUTPUT_DIR", "."), "results_metrics_lr_experiment.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Entry point for running the learning rate sweep experiment from the command line.
    # Parses command-line arguments and calls main().
    parser = argparse.ArgumentParser(
        description="Learning rate experiment for image classification."
    )
    parser.add_argument(
        '--resolution', type=int, default=224,
        help='Input image resolution (default: 224)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size for training and evaluation (default: 128)'
    )
    parser.add_argument(
        '--num_train_images', type=int, default=500,
        help='Number of training images to use per class (default: 500)'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--eval_steps', type=int, default=100,
        help='Number of steps between evaluations (default: 100)'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    args = parser.parse_args()
    main({
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "num_train_images": args.num_train_images,
        "num_epochs": args.num_epochs,
        "eval_steps": args.eval_steps,
        "learning_rate": args.learning_rate,
    })
