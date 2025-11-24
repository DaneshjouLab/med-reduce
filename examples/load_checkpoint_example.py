"""
Example: Loading and using saved model checkpoints.

This script demonstrates how to:
1. Load a single checkpoint
2. Find the best checkpoint from a directory
3. Load all fold models for ensemble prediction
"""
import torch
from pathlib import Path

# pylint: disable=import-error
from src.utils.checkpoint_utils import (
    load_checkpoint,
    load_model_from_checkpoint,
    find_best_checkpoint,
    load_all_fold_models,
    ensemble_predict,
)


def example_load_single_checkpoint():
    """Load a specific checkpoint and inspect it."""
    checkpoint_path = "outputs/2024-01-01/12-00-00/checkpoints/cv_run_fold1_metric0.8567.pt"

    # Load checkpoint dict
    checkpoint = load_checkpoint(checkpoint_path)
    print(f"Fold: {checkpoint['fold']}")
    print(f"Best metric: {checkpoint['metric']:.4f}")
    print(f"Model config: {checkpoint['model_config']}")

    # Load model directly
    model = load_model_from_checkpoint(checkpoint_path)
    print(f"Model loaded: {type(model).__name__}")
    return model


def example_find_best_checkpoint():
    """Find and load the best checkpoint from a directory."""
    checkpoint_dir = "outputs/2024-01-01/12-00-00/checkpoints"

    # Find best checkpoint (highest val_acc)
    best_checkpoint = find_best_checkpoint(checkpoint_dir, metric_key="val_acc")
    print(f"Best checkpoint: {best_checkpoint}")

    # Load the best model
    model = load_model_from_checkpoint(best_checkpoint)
    return model


def example_ensemble_prediction():
    """Load all fold models and make ensemble predictions."""
    checkpoint_dir = "outputs/2024-01-01/12-00-00/checkpoints"

    # Load all fold models
    models = load_all_fold_models(checkpoint_dir)
    print(f"Loaded {len(models)} fold models")

    # Create dummy input (batch_size=4, channels=3, height=224, width=224)
    dummy_input = torch.randn(4, 3, 224, 224)

    # Get ensemble predictions
    ensemble_output = ensemble_predict(models, dummy_input, average_logits=True)
    print(f"Ensemble output shape: {ensemble_output.shape}")

    # Get predicted classes
    predictions = ensemble_output.argmax(dim=-1)
    print(f"Predicted classes: {predictions}")

    return models, ensemble_output


def example_inference_on_new_data():
    """Load a checkpoint and run inference on new data."""
    checkpoint_path = "outputs/2024-01-01/12-00-00/checkpoints/cv_run_fold1_metric0.8567.pt"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(checkpoint_path, device=device)

    # Example: Load your actual data here
    # For demonstration, using dummy data
    batch = torch.randn(8, 3, 224, 224).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(batch)

        # Handle different output formats
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output

        # Get predictions
        predictions = logits.argmax(dim=-1)
        probabilities = torch.softmax(logits, dim=-1)

    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")

    return predictions, probabilities


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Load single checkpoint")
    print("=" * 60)
    # example_load_single_checkpoint()

    print("\n" + "=" * 60)
    print("Example 2: Find and load best checkpoint")
    print("=" * 60)
    # example_find_best_checkpoint()

    print("\n" + "=" * 60)
    print("Example 3: Ensemble prediction from all folds")
    print("=" * 60)
    # example_ensemble_prediction()

    print("\n" + "=" * 60)
    print("Example 4: Inference on new data")
    print("=" * 60)
    # example_inference_on_new_data()

    print("\n✅ Uncomment the examples above to run them with your actual checkpoint paths")
