'''
This script is a baseline for comparing different image classification models
at three different image compression levels, in comparison to the original.
It has a set number of augmentation transforms and does NOT combine them.
This does NOT experiment on JPEG compression levels
'''

# Environment Setup
import os

# Standard Libraries
import io
import json
import random
import time

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
from utils.transforms import (
    JPEGCompressionTransform,
    GaussianBlurTransform,
    ColorQuantizationTransform,
)
from utils.util_classes import (
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

wandb.login(key=os.getenv("WANDB_API_KEY", ""))

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

def main(num_train_images=5000, proportion_per_transform=0.2, resolution=224):
    
    # Simple batch size configuration
    batch_size = 256
    
    # Initialize wandb
    wandb_config = {
        "num_train_images": num_train_images,
        "proportion_per_transform": proportion_per_transform,
        "resolution": resolution,
        "batch_size": batch_size,
        "num_epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "gpu_available": GPU_AVAILABLE,
    }
    
    wandb.init(
        entity="ericcui-use-stanford-university",
        project="CS231N Test",
        config=wandb_config,
        tags=["baseline", "model-comparison"]
    )

    models = [
        {"name": "vit", "model_id": "google/vit-base-patch16-224", "type": "vit"},
        {"name": "dinov2", "model_id": "facebook/dinov2-base", "type": "dinov2"},
        {"name": "simclr", "model_id": "resnet50", "type": "simclr"},
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

    transformed_datasets = []
    indices = np.arange(num_images)
    np.random.shuffle(indices)

    used_indices = []
    for i, transform in enumerate(degradation_transforms):
        subset_indices = indices[i * images_per_transform:(i + 1) * images_per_transform]
        used_indices.extend(subset_indices)
        subset = Subset(train_dataset, subset_indices)
        transform_compose = transforms.Compose([transform])
        
        for model_info in models:
            name, model_id, typ = (
                model_info["name"],
                model_info["model_id"],
                model_info["type"],
            )
            if typ == "vit":
                preprocessor = ViTFeatureExtractor.from_pretrained(model_id, size=resolution)
            elif typ == "dinov2":
                preprocessor = AutoImageProcessor.from_pretrained(model_id, size=resolution)
            else:
                preprocessor = None

            transformed_ds = ISICDataset(subset, preprocessor, resolution, transform_compose, typ)
            transformed_datasets.append(transformed_ds)

    remaining_indices = np.setdiff1d(indices, used_indices)

    if len(remaining_indices) > 0:
        remaining_subset = Subset(train_dataset, remaining_indices)
        for model_info in models:
            name, model_id, typ = (
                model_info["name"],
                model_info["model_id"],
                model_info["type"],
            )
            if typ == "vit":
                preprocessor = ViTFeatureExtractor.from_pretrained(model_id, size=resolution)
            elif typ == "dinov2":
                preprocessor = AutoImageProcessor.from_pretrained(model_id, size=resolution)
            else:
                preprocessor = None

            # No transform applied to remaining indices
            untransformed_ds = ISICDataset(remaining_subset, preprocessor, resolution, None, typ)
            transformed_datasets.append(untransformed_ds)

    train_ds = ConcatDataset(transformed_datasets)

    val_ds = ISICDataset(
        val_dataset,
        preprocessor,
        resolution,
        model_type=typ,
    )

    for model_info in models:
        name, model_id, typ = (
            model_info["name"],
            model_info["model_id"],
            model_info["type"],
        )
        if typ == "vit":
            preprocessor = ViTFeatureExtractor.from_pretrained(
                model_id, size=resolution
            )
        elif typ == "dinov2":
            preprocessor = AutoImageProcessor.from_pretrained(model_id, size=resolution)
        else:
            preprocessor = None

        if typ == "vit":
            model = ViTForImageClassification.from_pretrained(
                model_id, num_labels=NUM_FILTERED_CLASSES, ignore_mismatched_sizes=True
            )
        elif typ == "dinov2":
            model = AutoModelForImageClassification.from_pretrained(
                model_id, num_labels=NUM_FILTERED_CLASSES, ignore_mismatched_sizes=True
            )
        elif typ == SSL_MODEL:
            # Load pretrained ResNet50 backbone
            backbone = timm.create_model(
                SIMCLR_BACKBONE,
                pretrained=True,
                num_classes=0  # Remove classification head
            )
            # Create SimCLR model with the backbone
            model = SimCLRForClassification(backbone, NUM_FILTERED_CLASSES)
            # Freeze the backbone initially
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
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(env_path("LOG_DIR", "."), f"{name}"),
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,  # Only keep the best model
        )
        
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

        model_dir = os.path.join(
            env_path("MODEL_DIR", "."), f"{name}"
        )
        os.makedirs(model_dir, exist_ok=True)

        if typ in HF_MODELS:
            model.save_pretrained(model_dir)
            preprocessor.save_pretrained(model_dir)
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
            name=f"{name}_model",
            type="model",
            description=f"Trained {name} model with {typ} architecture"
        )
        artifact.add_dir(model_dir)
        wandb.log_artifact(artifact)

        results[name] = model_metrics

        print(
            f"[Finetune] {name}: {results[name]}"
        )

        # ---- LINEAR PROBE PHASE ----
        # Create a new wandb run for linear probe
        wandb.init(
            project="model-comparison-baseline",
            config=wandb_config,
            tags=["baseline", "model-comparison", "linear-probe"],
            name=f"{name}_linear_probe"
        )

        if typ == "vit":
            model = ViTForImageClassification.from_pretrained(
                model_id, num_labels=NUM_FILTERED_CLASSES, ignore_mismatched_sizes=True
            )
        elif typ == "dinov2":
            model = AutoModelForImageClassification.from_pretrained(
                model_id, num_labels=NUM_FILTERED_CLASSES, ignore_mismatched_sizes=True
            )
        elif typ == SSL_MODEL:
            backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
            model = SimCLRForClassification(backbone, NUM_FILTERED_CLASSES)

        model.to(device)
        freeze_backbone(model, typ)

        # Log model architecture for linear probe
        if typ in HF_MODELS:
            wandb.watch(model, log="all", log_freq=100)
        elif typ == SSL_MODEL:
            wandb.watch(model.backbone, log="all", log_freq=100)

        linear_args = TrainingArguments(
            output_dir=os.path.join(
                env_path("TRAIN_OUTPUT_DIR", "."),
                f"{name}_linear_probe",
            ),
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(
                env_path("LOG_DIR", "."), f"{name}_linear_probe"
            ),
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,  # Only keep the best model
        )

        trainer = Trainer(
            model=model,
            args=linear_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=lambda pred: compute_metrics(pred, name),
            callbacks=[
                LossLoggerCallback(
                    log_dir=env_path("LOG_DIR", "./logs"),
                    phase="linear_probe",
                    model_name=name,
                ),
                WandbCallback(name, "linear_probe"),
            ],
        )

        start_time = time.time()
        peak_memory = get_gpu_memory() if GPU_AVAILABLE else -1
        trainer.train()
        current_memory = get_gpu_memory() if GPU_AVAILABLE else -1
        peak_memory = max(peak_memory, current_memory)

        eval_start_time = time.time()
        eval_results = trainer.evaluate()
        eval_time = time.time() - eval_start_time
        train_time = time.time() - start_time - eval_time

        # Log linear probe metrics
        linear_probe_metrics = {
            "model_name": name,
            "model_type": typ,
            "phase": "linear_probe",
            "peak_memory_mb": peak_memory,
            "flops_giga": flops,
            "train_time_seconds": train_time,
            "eval_time_seconds": eval_time,
            "eval_metrics": eval_results,
        }
        wandb.log(linear_probe_metrics)

        model_dir = os.path.join(
            env_path("MODEL_DIR", "."), f"{name}_linear_probe"
        )
        os.makedirs(model_dir, exist_ok=True)

        if typ in HF_MODELS:
            model.save_pretrained(model_dir)
            preprocessor.save_pretrained(model_dir)
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

        # Save linear probe model as wandb artifact
        artifact = wandb.Artifact(
            name=f"{name}_linear_probe_model",
            type="model",
            description=f"Linear probe {name} model with {typ} architecture"
        )
        artifact.add_dir(model_dir)
        wandb.log_artifact(artifact)

        results_linear_probe[name] = linear_probe_metrics

        print(
            f"[LinearProbe] {name}: {results_linear_probe[name]}"
        )

        # Close the wandb run for linear probe
        wandb.finish()

    # Close the main wandb run
    wandb.finish()

    with open(
        os.path.join(
            env_path("TRAIN_OUTPUT_DIR", "."), "results_metrics_finetune.json"
        ),
        "w",
    ) as f:
        json.dump(results, f, indent=4)

    with open(
        os.path.join(
            env_path("TRAIN_OUTPUT_DIR", "."), "results_metrics_linear_probe.json"
        ),
        "w",
    ) as f:
        json.dump(results_linear_probe, f, indent=4)


if __name__ == "__main__":
    main()
