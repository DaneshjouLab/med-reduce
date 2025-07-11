'''
This script is a baseline for comparing different image classification models
at three different image compression levels, in comparison to the original.
It has a set number of augmentation transforms and does NOT combine them.
This does NOT experiment on JPEG compression levels
'''

# Environment Setup
import os

# Set memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Standard Libraries
import io
import json
import random
import time
import argparse

# Scientific & Visualization Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# PyTorch & Torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, ConcatDataset
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
from datasets import load_dataset, ClassLabel

# Weights & Biases
import wandb

# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# Model Profiling & Vision Backbones
import timm
from thop import profile

# Local Application Imports
from utils.constants import HF_MODELS, SSL_MODEL, SIMCLR_BACKBONE, NUM_CLASSES, FILTERED_CLASSES, NUM_FILTERED_CLASSES
from utils.transforms_test import (
    JPEGCompressionTransform,
    GaussianBlurTransform,
    ColorQuantizationTransform,
)
from utils.util_classes_test import (
    ISICDataset,
    SimCLRForClassification,
    LossLoggerCallback,
)
from utils.util_methods import (
    env_path,
    compute_metrics,
    get_gpu_memory,
    freeze_backbone,
)

# GPU Memory Monitoring (optional)
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("pynvml not installed, GPU memory monitoring disabled.")

# Cache paths
os.environ["TRANSFORMERS_CACHE"] = os.getenv(
    "TRANSFORMERS_CACHE", "~/.cache/huggingface/transformers"
)
os.environ["HF_DATASETS_CACHE"] = os.getenv(
    "HF_DATASETS_CACHE", "~/.cache/huggingface/datasets"
)
os.environ["HF_HOME"] = os.getenv("HF_HOME", "~/.cache/huggingface")

class WandbCallback(TrainerCallback):
    def __init__(self, model_name, phase):
        self.model_name = model_name
        self.phase = phase
        self.best_accuracy = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Add model name and phase to logs
            logs["model"] = self.model_name
            logs["phase"] = self.phase
            
            # Track GPU memory if available
            if GPU_AVAILABLE:
                logs["gpu_memory_mb"] = get_gpu_memory()
            
            # Log to wandb
            wandb.log(logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # Track best accuracy
            if "eval_accuracy" in metrics:
                self.best_accuracy = max(self.best_accuracy, metrics["eval_accuracy"])
                metrics["best_accuracy"] = self.best_accuracy
            
            # Log evaluation metrics
            wandb.log(metrics)

def main(num_train_images=100, proportion_per_transform=0.2, resolution=224, batch_size=256, num_epochs=3, eval_steps=10, learning_rate=1e-4):
    
    # Initialize wandb config
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
        # {"name": "dinov2", "model_id": "facebook/dinov2-base", "type": "dinov2", "config": {
        #     "image_size": resolution,
        #     "num_labels": NUM_FILTERED_CLASSES,
        #     "ignore_mismatched_sizes": True
        # }},
        # {"name": "simclr", "model_id": "resnet50", "type": "simclr", "config": {
        #     "img_size": resolution
        # }},
    ]

    results = {m["name"]: {} for m in models}
    results_linear_probe = {m["name"]: {} for m in models}

    dataset = load_dataset(
        "MKZuziak/ISIC_2019_224",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        split="train",
    )

    print(f"Initial dataset size: {len(dataset)} images")

    # Get indices of images with desired labels
    filtered_indices = [
        i for i, label in enumerate(dataset["label"])
        if str(label) in FILTERED_CLASSES  # Convert to string for comparison
    ]
    
    # Select only those indices
    dataset = dataset.select(filtered_indices)
    print(f"Number of images after filtering for classes {FILTERED_CLASSES}: {len(dataset)}")
    dataset = dataset.cast_column("label", ClassLabel(num_classes=NUM_FILTERED_CLASSES))

    # Get class counts and balance dataset - optimized version
    print("Balancing dataset...")
    # Get counts for each class
    class_counts = {label: 0 for label in FILTERED_CLASSES}
    for label in dataset["label"]:
        class_counts[str(label)] += 1  # Convert to string for dictionary key
    
    print(f"Class counts: {class_counts}")  # Debug print to verify counts
    
    # Calculate how many images to use per class
    min_class_size = min(class_counts.values())
    images_per_class = min(num_train_images // 2, min_class_size)
    
    # Sample indices for each class
    np.random.seed(42)
    balanced_indices = []
    for label in FILTERED_CLASSES:
        class_indices = [i for i, l in enumerate(dataset["label"]) if str(l) == label]  # Convert to string for comparison
        print(f"Found {len(class_indices)} images for class {label}")  # Debug print
        sampled_indices = np.random.choice(class_indices, images_per_class, replace=False)
        balanced_indices.extend(sampled_indices)
    
    np.random.shuffle(balanced_indices)
    balanced_dataset = dataset.select(balanced_indices)

    # Split into train and validation
    full_dataset = balanced_dataset.train_test_split(
        test_size=0.2, stratify_by_column="label", seed=42
    )

    train_dataset, val_dataset = full_dataset["train"], full_dataset["test"]
    
    degradation_transforms = [
        JPEGCompressionTransform(),
        GaussianBlurTransform(),
        ColorQuantizationTransform(),
    ]

    num_transforms = len(degradation_transforms)
    num_images = len(train_dataset)
    images_per_transform = int(num_images * proportion_per_transform)

    # Create a single preprocessor for each model type
    preprocessors = {}
    for model_info in models:
        name, model_id, typ, config = (
            model_info["name"],
            model_info["model_id"],
            model_info["type"],
            model_info["config"],
        )
        if typ == "vit":
            preprocessors[typ] = ViTFeatureExtractor.from_pretrained(
                model_id,
                size=resolution,
                do_resize=True,
                resample=Image.LANCZOS,
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225]
            )
        elif typ == "dinov2":
            preprocessors[typ] = AutoImageProcessor.from_pretrained(
                model_id,
                size=resolution,
                do_resize=True,
                resample=Image.LANCZOS,
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225]
            )
        else:
            preprocessors[typ] = None

    # Process each model type separately
    for model_info in models:
        name, model_id, typ, config = (
            model_info["name"],
            model_info["model_id"],
            model_info["type"],
            model_info["config"],
        )

        # Initialize wandb for this specific model run
        wandb.init(
            entity="ericcui-use-stanford-university",
            project="CS231N Test",
            name=f"{name}_{resolution}_{num_epochs}_epochs_finetune",
            config={**wandb_config, "model_config": config},
            tags=["baseline", "model-comparison", "finetune", name, f"res_{resolution}"],
            reinit=True
        )

        transformed_datasets = []
        indices = np.arange(num_images)
        np.random.shuffle(indices)

        used_indices = []
        for i, transform in enumerate(degradation_transforms):
            subset_indices = indices[i * images_per_transform:(i + 1) * images_per_transform]
            used_indices.extend(subset_indices)
            subset = Subset(train_dataset, subset_indices)
            transform_compose = transforms.Compose([transform])
            
            transformed_ds = ISICDataset(
                subset, 
                preprocessors[typ], 
                resolution, 
                transform_compose, 
                typ
            )
            transformed_datasets.append(transformed_ds)

        remaining_indices = np.setdiff1d(indices, used_indices)

        if len(remaining_indices) > 0:
            remaining_subset = Subset(train_dataset, remaining_indices)
            untransformed_ds = ISICDataset(
                remaining_subset, 
                preprocessors[typ], 
                resolution, 
                None, 
                typ
            )
            transformed_datasets.append(untransformed_ds)

        train_ds = ConcatDataset(transformed_datasets)
        val_ds = ISICDataset(
            val_dataset,
            preprocessors[typ],
            resolution,
            model_type=typ,
        )

        if typ == "vit":
            model = ViTForImageClassification.from_pretrained(
                model_id,
                num_labels=NUM_FILTERED_CLASSES,
                ignore_mismatched_sizes=True,
                image_size=resolution,
            )
        elif typ == "dinov2":
            model = AutoModelForImageClassification.from_pretrained(
                model_id,
                num_labels=NUM_FILTERED_CLASSES,
                ignore_mismatched_sizes=True,
                image_size=resolution
            )
        elif typ == SSL_MODEL:
            backbone = timm.create_model(
                SIMCLR_BACKBONE,
                pretrained=True,
                num_classes=0  # Remove classification head
            )
            model = SimCLRForClassification(backbone, NUM_FILTERED_CLASSES)
            freeze_backbone(model, SSL_MODEL)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        try:
            dummy_input = torch.randn(1, 3, resolution, resolution).to(device)
            model.to(device)
            flops, _ = profile(model, inputs=(dummy_input,))
            flops /= 1e9
        except Exception as e:
            print(f"FLOP profiling failed: {e}")
            flops = -1

        train_args = TrainingArguments(
            output_dir=os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), f"{name}"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            logging_dir=os.path.join(env_path("LOG_DIR", "."), f"{name}"),
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            save_safetensors=False,
            hub_model_id=None,
            hub_strategy="end",
            push_to_hub=False,
            save_only_model=True,
        )
        
        # Clean up old model directories before training
        model_dirs = [
            os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), f"{name}"),
            os.path.join(env_path("MODEL_DIR", "."), f"{name}"),
            os.path.join(env_path("LOG_DIR", "."), f"{name}"),
        ]
        
        for dir_path in model_dirs:
            if os.path.exists(dir_path):
                print(f"Cleaning up directory: {dir_path}")
                import shutil
                shutil.rmtree(dir_path)
                os.makedirs(dir_path, exist_ok=True)

        # Monitor disk space before training
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"Disk space before training {name}:")
        print(f"Total: {total // (2**30)} GB")
        print(f"Used: {used // (2**30)} GB")
        print(f"Free: {free // (2**30)} GB")
        
        # If less than 1GB free, raise an error
        if free < 1 * (2**30):  # 1GB in bytes
            raise RuntimeError("Not enough disk space to train model. Please free up at least 1GB of space.")
        
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=lambda pred: compute_metrics(pred, name),
            callbacks=[
                LossLoggerCallback(
                    log_dir=env_path("LOG_DIR", "./logs"),
                    phase="finetune",
                    model_name=name,
                ),
                WandbCallback(name, "finetune"),
            ],
        )

        # ---- TRAINING PHASE ----
        start_time = time.time()
        peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1

        # Log model architecture
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

        # Log model-specific metrics
        model_metrics = {
            "model_name": name,
            "model_type": typ,
            "peak_memory_mb": peak_memory,
            "flops_giga": flops,
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time,
            "eval_metrics": eval_results,
        }
        wandb.log(model_metrics)

        # Create parameterized directory path
        dir_name = f"{name}_{typ}_lr{learning_rate}_bs{batch_size}"
        model_dir = os.path.join(env_path("MODEL_DIR", "."), dir_name)
        os.makedirs(model_dir, exist_ok=True)

        if typ in HF_MODELS:
            model.save_pretrained(model_dir)
            preprocessors[typ].save_pretrained(model_dir)
        elif typ == SSL_MODEL:
            torch.save(
                model.state_dict(), os.path.join(model_dir, "pytorch_model.bin")
            )
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "model_type": SSL_MODEL,
                        "backbone": "resnet50",
                        "num_classes": NUM_FILTERED_CLASSES,
                    },
                    f,
                )

        # Save model as wandb artifact
        artifact = wandb.Artifact(
            name=f"{dir_name}_model",
            type="model",
            description=f"Trained {name} model with {typ} architecture"
        )
        artifact.add_dir(model_dir)
        wandb.log_artifact(artifact)

        results[name] = model_metrics

        print(f"[Finetune] {name}: {results[name]}")
        
        # Close the wandb run for this model
        wandb.finish()

        # Clear GPU memory after model is done
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Remove the final wandb.finish() since we're now closing each run individually
    with open(
        os.path.join(
            env_path("TRAIN_OUTPUT_DIR", "."), f"results_metrics_finetune_{dir_name}.json"
        ),
        "w",
    ) as f:
        json.dump(results, f, indent=4)

    with open(
        os.path.join(
            env_path("TRAIN_OUTPUT_DIR", "."), f"results_metrics_linear_probe_{dir_name}.json"
        ),
        "w",
    ) as f:
        json.dump(results_linear_probe, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison script for image classification.")
    parser.add_argument('--resolution', type=int, default=224, help='Input image resolution (default: 224)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation (default: 128)')
    parser.add_argument('--num_train_images', type=int, default=500, help='Number of training images to use per class (default: 500)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs (default: 3)')
    parser.add_argument('--eval_steps', type=int, default=100, help='Number of steps between evaluations (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    args = parser.parse_args()
    main(
        resolution=args.resolution, 
        batch_size=args.batch_size, 
        num_train_images=args.num_train_images,
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate
    )

