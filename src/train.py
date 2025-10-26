"""Main training script."""
import os
import time
import argparse
import numpy as np
import torch
import wandb
from datasets import load_dataset, ClassLabel
from transformers import Trainer, TrainingArguments
from typing import Dict, Any
from torch.utils.data import DataLoader
from umap_viz import create_umap_callback
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

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
    ISICDataCollator, create_transformed_datasets, balance_dataset,
    create_multi_validation_datasets, evaluate_all_datasets
)
from src.transforms import (
    get_degradation_transforms
)
from src.training import (
    LossLoggerCallback, WandbCallback, profile_model,
    create_compute_metrics_fn
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
    
    # Initialize wandb
    wandb.init(
        entity="sonnet-xu-stanford-university",
        project="Compressed Perception Test",
        name=f"{name}_{config.resolution}_{config.num_epochs}_epochs_{training_mode}",
        config={
            **config.to_wandb_config(),
            "model_config": model_info["config"],
            "training_mode": training_mode
        },
        tags=["baseline", "model-comparison", training_mode, name, f"res_{config.resolution}"],
        reinit=True
    )
    
    # Create model and preprocessor
    model = create_model(model_info, config.resolution)
    preprocessor = create_preprocessor(model_info, config.resolution)
    
    # Freeze backbone for linear probing
    if training_mode == "linear_probe":
        print(f"Freezing backbone for linear probing...")
        freeze_backbone(model, model_type)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create training dataset with transformations
    train_ds, _ = create_transformed_datasets(
        train_dataset,
        val_dataset,  # Not used but required by function signature
        degradation_transforms,
        config.proportion_per_transform,
        preprocessor,
        config.resolution,
        model_type
    )
    
    # Create multiple validation datasets
    val_datasets = create_multi_validation_datasets(
        val_dataset,
        preprocessor,
        config.resolution,
        model_type
    )
    
    # Profile model
    flops = profile_model(model, config.resolution)
    
    # Setup training arguments
    output_dir = os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), f"{name}_{training_mode}")
    log_dir = env_path("LOG_DIR", "./logs")
    
    # Adjust learning rate for linear probing (typically higher)
    learning_rate = config.learning_rate
    if training_mode == "linear_probe":
        learning_rate = config.learning_rate * 10  # Often need higher LR for linear probe
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=config.weight_decay,
        logging_dir=os.path.join(log_dir, f"{name}_{training_mode}"),
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        load_best_model_at_end=True,  # Load best model for final evaluation
        metric_for_best_model="eval_accuracy",  # Use eval accuracy for model selection
        greater_is_better=True,
        save_total_limit=1,
        save_safetensors=False,
        push_to_hub=False,
    )
    
    # Check disk space
    # check_disk_space(required_gb=1.0)
    
    # Create trainer with clean validation set for checkpointing
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_datasets['clean'],  # Use clean for model selection
        data_collator=ISICDataCollator(),
        compute_metrics=create_compute_metrics_fn(name),
        callbacks=[
            LossLoggerCallback(log_dir, training_mode, name),
            WandbCallback(name, training_mode),
        ],
    )
    
    # Log model to wandb
    if model_type in HF_MODELS:
        wandb.watch(model, log="all", log_freq=100)
    
    # Train
    start_time = time.time()
    peak_memory = get_gpu_memory()
    
    umap_callbacks = create_umap_callback(
        val_dataloader=DataLoader(val_datasets['clean'], batch_size=32),
        device=device,
        output_dir=os.path.join(log_dir, "umap_plots"),
        model_name=f"{name}_{training_mode}",
        max_samples=2000,
    )
    umap_callbacks["capture_before"](model)

    trainer.train()

    umap_callbacks["capture_after"](trainer.model)
    
    # Evaluate on all validation datasets
    eval_start_time = time.time()
    multi_eval_results = evaluate_all_datasets(trainer, val_datasets, name)
    eval_time = time.time() - eval_start_time
    train_time = time.time() - start_time - eval_time
    
    # Track peak memory
    current_memory = get_gpu_memory()
    peak_memory = max(peak_memory, current_memory) if peak_memory > 0 else current_memory
    
    # Prepare comprehensive results
    results = {
        "model_name": name,
        "model_type": model_type,
        "training_mode": training_mode,
        "peak_memory_mb": peak_memory,
        "flops_giga": flops,
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
        "eval_results_by_degradation": multi_eval_results,
        # Summary statistics
        "clean_accuracy": multi_eval_results['clean']['accuracy'],
        "avg_degraded_accuracy": np.mean([
            res['accuracy'] for key, res in multi_eval_results.items() 
            if key != 'clean'
        ]),
        "robustness_gap": multi_eval_results['clean']['accuracy'] - multi_eval_results['jpeg_20']['accuracy'],
    }
    
    # Log summary to wandb
    wandb.log({
        "summary/clean_accuracy": results["clean_accuracy"],
        "summary/avg_degraded_accuracy": results["avg_degraded_accuracy"],
        "summary/robustness_gap": results["robustness_gap"],
    })
    
    # Save model
    model_dir = os.path.join(
        env_path("MODEL_DIR", "."),
        f"{name}_{model_type}_{training_mode}_lr{learning_rate}_bs{config.batch_size}"
    )
    save_model(model, model_info, model_dir, preprocessor)
    
    # Save as wandb artifact
    artifact = wandb.Artifact(
        name=f"{name}_{training_mode}_model",
        type="model",
        description=f"Trained {name} model with {model_type} architecture in {training_mode} mode"
    )
    artifact.add_dir(model_dir)
    wandb.log_artifact(artifact)
    
    # Finish wandb run
    wandb.finish()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

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
        test_size=0.2,
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
    models = [m for m in MODEL_REGISTRY if m["name"] in ["dinov3"]]  # Modify as needed
    
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
            
            finetune_results = train_model(
                model_info,
                train_dataset,
                val_dataset,
                config,
                degradation_transforms,
                training_mode="finetune"
            )
            all_results["finetune"][model_name] = finetune_results
            
        except Exception as e:
            print(f"Error fine-tuning {model_name}: {e}")
            all_results["finetune"][model_name] = {"error": str(e)}
        
        # Linear probing
        try:
            print(f"\n{'='*60}")
            print(f"LINEAR PROBING: {model_name}")
            print(f"{'='*60}")
            
            probe_results = train_model(
                model_info,
                train_dataset,
                val_dataset,
                config,
                degradation_transforms,
                training_mode="linear_probe"
            )
            all_results["linear_probe"][model_name] = probe_results
            
        except Exception as e:
            print(f"Error linear probing {model_name}: {e}")
            all_results["linear_probe"][model_name] = {"error": str(e)}
    
    # Save comprehensive results
    output_filename = (
        f"results_lr{config.learning_rate}_{config.resolution}_"
        f"bs{config.batch_size}_ep{config.num_epochs}.json"
    )
    save_results(
        all_results,
        os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), output_filename)
    )
    
    # Print summary comparison
    print_results_summary(all_results)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

def print_results_summary(results: Dict[str, Any]):
    """Print a formatted summary of results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Create comparison table
    print("\nClean Accuracy Comparison:")
    print("-" * 40)
    print(f"{'Model':<15} {'Fine-tune':<12} {'Linear Probe':<12}")
    print("-" * 40)
    
    for model_name in results["finetune"].keys():
        ft_acc = results["finetune"][model_name].get("clean_accuracy", 0)
        lp_acc = results["linear_probe"][model_name].get("clean_accuracy", 0)
        print(f"{model_name:<15} {ft_acc:<12.3f} {lp_acc:<12.3f}")
    
    print("\nRobustness (Clean - JPEG20 Accuracy):")
    print("-" * 40)
    print(f"{'Model':<15} {'Fine-tune':<12} {'Linear Probe':<12}")
    print("-" * 40)
    
    for model_name in results["finetune"].keys():
        ft_rob = results["finetune"][model_name].get("robustness_gap", 0)
        lp_rob = results["linear_probe"][model_name].get("robustness_gap", 0)
        print(f"{model_name:<15} {ft_rob:<12.3f} {lp_rob:<12.3f}")

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
        num_train_images=args.num_train_images,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
    )
    
    main(config)