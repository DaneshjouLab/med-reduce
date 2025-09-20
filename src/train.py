"""Main training script."""
import os
import time
import argparse
import torch
import wandb
from dotenv import load_dotenv

from datasets import load_dataset, ClassLabel
from transformers import Trainer, TrainingArguments

# Load environment variables from .env file
load_dotenv()

from src.config import (
    TrainingConfig, MODEL_REGISTRY, FILTERED_CLASSES, 
    NUM_FILTERED_CLASSES, HF_MODELS, SSL_MODEL
)
from src.utils import (
    setup_environment, env_path, get_gpu_memory, 
    check_disk_space, save_results
)
from src.models import (
    create_model, create_preprocessor, save_model
)
from src.datasets import (
    create_transformed_datasets, balance_dataset
)
from src.transforms import get_degradation_transforms
from src.training import (
    LossLoggerCallback, WandbCallback, profile_model,
    create_compute_metrics_fn
)

def train_model(
    model_info: dict,
    train_dataset,
    val_dataset, 
    config: TrainingConfig,
    degradation_transforms: list
) -> dict:
    """
    Train a single model.
    
    Args:
        model_info: Model configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        degradation_transforms: List of data augmentations
        
    Returns:
        Dictionary of training results
    """
    name = model_info["name"]
    model_type = model_info["type"]
    
    print(f"\n{'='*50}")
    print(f"Training {name} ({model_type})")
    print(f"{'='*50}")
    
    # Initialize wandb
    wandb.init(
        entity=os.getenv("WANDB_ENTITY", "default-entity"),
        project=os.getenv("WANDB_PROJECT", "default-project"),
        name=f"{name}_{config.resolution}_{config.num_epochs}_epochs_finetune",
        config={**config.to_wandb_config(), "model_config": model_info["config"]},
        tags=["baseline", "model-comparison", "finetune", name, f"res_{config.resolution}"],
        reinit=True
    )
    
    # Create model and preprocessor
    model = create_model(model_info, config.resolution)
    preprocessor = create_preprocessor(model_info, config.resolution)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create datasets with transformations
    train_ds, val_ds = create_transformed_datasets(
        train_dataset,
        val_dataset,
        degradation_transforms,
        config.proportion_per_transform,
        preprocessor,
        config.resolution,
        model_type
    )
    
    # Profile model
    flops = profile_model(model, config.resolution)
    
    # Setup training arguments
    output_dir = os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), name)
    log_dir = env_path("LOG_DIR", "./logs")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=config.weight_decay,
        logging_dir=os.path.join(log_dir, name),
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        save_safetensors=False,
        push_to_hub=False,
    )
    
    # Check disk space
    check_disk_space(required_gb=1.0)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=create_compute_metrics_fn(name),
        callbacks=[
            LossLoggerCallback(log_dir, "finetune", name),
            WandbCallback(name, "finetune"),
        ],
    )
    
    # Log model to wandb
    if model_type in HF_MODELS:
        wandb.watch(model, log="all", log_freq=100)
    elif model_type == SSL_MODEL:
        wandb.watch(model.backbone, log="all", log_freq=100)
    
    # Train
    start_time = time.time()
    peak_memory = get_gpu_memory()
    
    trainer.train()
    
    # Evaluate
    eval_start_time = time.time()
    eval_results = trainer.evaluate()
    eval_time = time.time() - eval_start_time
    train_time = time.time() - start_time - eval_time
    
    # Track peak memory
    current_memory = get_gpu_memory()
    peak_memory = max(peak_memory, current_memory) if peak_memory > 0 else current_memory
    
    # Prepare results
    results = {
        "model_name": name,
        "model_type": model_type,
        "peak_memory_mb": peak_memory,
        "flops_giga": flops,
        "train_time_seconds": train_time,
        "eval_time_seconds": eval_time,
        "eval_metrics": eval_results,
    }
    
    # Log to wandb
    wandb.log(results)
    
    # Save model
    model_dir = os.path.join(
        env_path("MODEL_DIR", "."),
        f"{name}_{model_type}_lr{config.learning_rate}_bs{config.batch_size}"
    )
    save_model(model, model_info, model_dir, preprocessor)
    
    # Save as wandb artifact
    artifact = wandb.Artifact(
        name=f"{name}_model",
        type="model",
        description=f"Trained {name} model with {model_type} architecture"
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
    """Main training loop."""
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
    models = [m for m in MODEL_REGISTRY if m["name"] in ["vit"]]  # Modify as needed
    
    # Train each model
    results = {}
    for model_info in models:
        try:
            model_results = train_model(
                model_info,
                train_dataset,
                val_dataset,
                config,
                degradation_transforms
            )
            results[model_info["name"]] = model_results
            print(f"\n[Finetune] {model_info['name']}: {model_results}")
            
        except Exception as e:
            print(f"Error training {model_info['name']}: {e}")
            results[model_info['name']] = {"error": str(e)}
    
    # Save all results
    output_filename = (
        f"results_finetune_lr{config.learning_rate}_"
        f"bs{config.batch_size}_ep{config.num_epochs}.json"
    )
    save_results(
        results,
        os.path.join(env_path("TRAIN_OUTPUT_DIR", "."), output_filename)
    )
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison training script")
    parser.add_argument('--resolution', type=int, default=224,
                      help='Input image resolution (default: 224)')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training (default: 128)')
    parser.add_argument('--num_train_images', type=int, default=500,
                      help='Number of training images (default: 500)')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs (default: 3)')
    parser.add_argument('--eval_steps', type=int, default=100,
                      help='Steps between evaluations (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate (default: 1e-4)')
    
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