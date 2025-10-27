"""Main training script."""
import os
import time
import argparse
import numpy as np
import torch
from PIL import Image
import wandb
from datasets import load_dataset, ClassLabel
from transformers import Trainer, TrainingArguments
from typing import Dict, Any
from torch.utils.data import DataLoader
from umap_viz import create_umap_callback

from src.config import (
    TrainingConfig, MODEL_REGISTRY, FILTERED_CLASSES, 
    NUM_FILTERED_CLASSES, HF_MODELS
)
from src.utils import (
    setup_environment, env_path, get_gpu_memory, 
    check_disk_space, save_results
)
from src.models import (
    create_model, create_preprocessor, freeze_backbone, save_model
)
from src.data_utils import (
    ISICDataset, create_transformed_datasets, balance_dataset,
    create_multi_validation_datasets, evaluate_all_datasets
)
from src.transforms import (
    get_degradation_transforms
)

def train_model(
    model_info: dict,
    train_dataset,
    val_dataset,
    config: TrainingConfig,
    degradation_transforms: list,
    training_mode: str = "finetune"  # "finetune" or "linear_probe"
) -> dict:
    """
    Train a single model with specified training mode.
    
    Args:
        model_info: Model configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        degradation_transforms: List of data augmentations
        training_mode: "finetune" or "linear_probe"
        
    Returns:
        Dictionary of training results
    """
    name = model_info["name"]
    model_type = model_info["type"]
    
    print(f"\n{'='*50}")
    print(f"Training {name} ({model_type}) - Mode: {training_mode}")
    print(f"{'='*50}")
 
    # Create preprocessor
    preprocessor = create_preprocessor(model_info, config.resolution)
    
    # Create multiple validation datasets
    val_datasets = create_multi_validation_datasets(
        val_dataset,
        preprocessor,
        config.resolution,
        model_type
    )

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Save images in each validation dataset
    for val_name, val_dataset in val_datasets.items():
        output_dir = os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), f"{name}_{training_mode}", val_name)
        os.makedirs(output_dir, exist_ok=True)

        for i, data in enumerate(val_dataset):
            # 1. Get the tensor and ensure it's on CPU
            pixel_values_tensor = data['pixel_values'].cpu() 
            
            # 2. De-normalize the tensor (reverse the standard normalization)
            # Apply multiplication by STD
            denorm_tensor = pixel_values_tensor * IMAGENET_STD 
            # Apply addition of MEAN
            denorm_tensor = denorm_tensor + IMAGENET_MEAN
            
            # 3. Scale to 0-255 and convert to unsigned 8-bit integers (uint8)
            # Clamp ensures values are within [0, 1] before scaling to [0, 255]
            denorm_tensor = torch.clamp(denorm_tensor, 0, 1) * 255
            
            # 4. Convert (C, H, W) to (H, W, C) using .permute() and to a NumPy array
            img_np = denorm_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

            # 5. Convert NumPy array to PIL Image
            img_pil = Image.fromarray(img_np)
            
            # 6. Save the image using the PIL Image's .save() method
            img_path = os.path.join(output_dir, f"{i:03d}_label_{data['labels'].item()}.png")
            img_pil.save(img_path)

def main(config: TrainingConfig):
    """Main training loop with both fine-tuning and linear probing."""
    # Setup environment
    setup_environment()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "MKZuziak/ISIC_2019_224",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        split="train",
    )
    print(f"Initial dataset size: {len(dataset)} images")

    # Filter for desired classes
    filtered_indices = [
        i for i, label in enumerate(dataset["label"])
        if str(label) in FILTERED_CLASSES
    ]

    dataset = dataset.select(filtered_indices)
    print(f"After filtering: {len(dataset)} images")
    
    # Cast labels to correct number of classes
    dataset = dataset.cast_column("label", ClassLabel(num_classes=NUM_FILTERED_CLASSES))
    
    # Balance dataset
    balanced_dataset = balance_dataset(dataset, FILTERED_CLASSES, config.num_train_images)
    print(f"Balanced dataset size: {len(balanced_dataset)} images")

    # Split into train and validation
    split_dataset = balanced_dataset.train_test_split(
        test_size=0.8,
        stratify_by_column="label",
        seed=42
    )
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Get degradation transforms
    degradation_transforms = get_degradation_transforms()
    
    # Select models to train
    models = [m for m in MODEL_REGISTRY if m["name"] in ["dinov2"]]  # Modify as needed
    
    # Store all results
    all_results = {
        "finetune": {},
        "linear_probe": {}
    }
    
    # Train each model with both strategies
    for model_info in models:
        model_name = model_info["name"]
        
        # Fine-tuning
        try:
            print(f"\n{'='*60}")
            print(f"FINE-TUNING: {model_name}")
            print(f"{'='*60}")
            
            train_model(
                model_info,
                train_dataset,
                val_dataset,
                config,
                degradation_transforms,
                training_mode="finetune"
            )
            
        except Exception as e:
            print(f"Error fine-tuning {model_name}: {e}")
            all_results["finetune"][model_name] = {"error": str(e)}
    
    print("\n" + "="*60)
    print("Downloading complete!")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison with fine-tuning and linear probing")
    parser.add_argument('--resolution', type=int, default=224,
                      help='Input image resolution (default: 224)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training (default: 128)')
    parser.add_argument('--num_train_images', type=int, default=10000,
                      help='Number of training images (default: 10000)')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs (default: 3)')
    parser.add_argument('--eval_steps', type=int, default=100,
                      help='Steps between evaluations (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate (default: 1e-4)')
    parser.add_argument('--mode', type=str, default='both',
                      choices=['finetune', 'linear_probe', 'both'],
                      help='Training mode (default: both)')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        num_train_images=50,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
    )
    
    main(config)