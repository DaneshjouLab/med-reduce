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

# Scientific & Visualization Libraries
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

# Local Application Imports
from utils.constants import SSL_MODEL, SIMCLR_BACKBONE, FILTERED_CLASSES, NUM_FILTERED_CLASSES
from utils.util_classes import (
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

def initialize_model_and_preprocessor(model_info, resolution):
    """
    Initialize the model and preprocessor based on the model type.

    Args:
        model_info (dict): Dictionary containing model details (name, model_id, type, config).
        resolution (int): Image resolution.

    Returns:
        model (torch.nn.Module): Initialized model.
        preprocessor (transformers.PreTrainedTokenizer or None): Preprocessor for the model.
    """
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
            num_classes=0,  # Remove classification head
        )
        model = SimCLRForClassification(backbone, NUM_FILTERED_CLASSES)
        freeze_backbone(model, SSL_MODEL)
        preprocessor = None
    else:
        raise ValueError(f"Unsupported model type: {typ}")

    return model, preprocessor

def balance_dataset(dataset, num_train_images, filtered_classes):
    """
    Balance the dataset by sampling an equal number of images per class.

    Args:
        dataset (Dataset): The dataset to balance.
        num_train_images (int): Total number of training images to use.
        filtered_classes (list): List of class labels to filter.

    Returns:
        balanced_dataset (Dataset): Balanced dataset with equal images per class.
    """
    print("Balancing dataset...")
    class_counts = {label: 0 for label in filtered_classes}
    for label in dataset["label"]:
        class_counts[str(label)] += 1

    print(f"Class counts: {class_counts}")  # Debug print to verify counts

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

def train_model(model, train_ds, val_ds, name, typ, resolution, batch_size, num_epochs, learning_rate, eval_steps, wandb_config):
    """
    Train the model using the Hugging Face Trainer.

    Args:
        model (torch.nn.Module): The model to train.
        train_ds (Dataset): Training dataset.
        val_ds (Dataset): Validation dataset.
        name (str): Model name.
        typ (str): Model type.
        resolution (int): Image resolution.
        batch_size (int): Batch size.
        num_epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        eval_steps (int): Evaluation steps.
        wandb_config (dict): Configuration for wandb logging.

    Returns:
        dict: Training results and metrics.
    """
    wandb.init(
        entity="ericcui-use-stanford-university",
        project="CS231N Test",
        name=f"{name}_{resolution}_{num_epochs}_epochs_finetune",
        config={**wandb_config, "model_name": name, "model_type": typ},
        tags=["baseline", "model-comparison", "finetune", name, f"res_{resolution}"],
        reinit=True,
    )

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

    trainer.train()
    eval_results = trainer.evaluate()

    wandb.finish()
    return eval_results

def prepare_datasets(dataset, preprocessor, resolution, proportion_per_transform):
            """
            Prepare training and validation datasets with optional preprocessing.
        
            Args:
                dataset: The balanced HuggingFace dataset.
                preprocessor: Preprocessing function or None.
                resolution: Image resolution.
                proportion_per_transform: Proportion for each transform.
        
            Returns:
                train_ds, val_ds: Torch-compatible datasets for training and validation.
            """
            # Split dataset into train and validation (80/20 split)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset = dataset.select(range(train_size))
            val_dataset = dataset.select(range(train_size, train_size + val_size))
        
            # Define basic transform
            transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

        # Prepare datasets
        train_ds, val_ds = prepare_datasets(dataset, preprocessor, resolution, proportion_per_transform)

        # Train the model
        results = train_model(
            model, train_ds, val_ds, model_info["name"], model_info["type"],
            resolution, batch_size, num_epochs, learning_rate, eval_steps, wandb_config
        )

        print(f"Results for {model_info['name']}: {results}")
