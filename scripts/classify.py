"""Train and evaluate linear classifier on WSI embeddings."""
import os
import sys
import numpy as np
import torch

from src.config import (
    ClassificationConfig,
    DataSplitConfig,
    OUTPUTS_DIR,
)
from src.data import TCGAPrediction
from src.data.datamodule import split_dataset_by_patient
from src.engines import train_logistic_regression
from src.evaluation import evaluate_model, plot_roc_curve
from src.utils import save_pickle

if __name__ == "__main__":
    # Parse command line arguments
    encoder_name = sys.argv[1]  # "univ2"
    var = sys.argv[2]  # "subtype"
    level = int(sys.argv[3])  # 0
    datasets = sys.argv[4:]  # ["luad", "lusc"]
    
    # Configuration
    class_config = ClassificationConfig(
        num_epochs=20,
        batch_size=16,
        num_workers=16,
        lr_range=(1e-6, 1e2),
        num_lr_steps=33
    )
    
    split_config = DataSplitConfig(
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=42
    )
    
    print(
        f'\n🎯 Predicting {var} using {encoder_name.upper()} '
        f'on TCGA {"-".join(datasets).upper()} level {level}...'
    )
    
    # Compile dataset
    print("\n⏳ Compiling dataset...")
    prediction_dataset = TCGAPrediction(encoder_name, level, datasets, var)
    
    # Split dataset
    print("\n⏳ Splitting dataset...")
    (train_loader, val_loader, test_loader,
     train_indices, val_indices, test_indices) = split_dataset_by_patient(
        prediction_dataset,
        split_config,
        class_config.batch_size,
        class_config.num_workers
    )
    
    # Train model
    print("\n⏳ Training model...")
    lr_list = np.logspace(
        np.log10(class_config.lr_range[0]),
        np.log10(class_config.lr_range[1]),
        num=class_config.num_lr_steps
    )
    model, losses = train_logistic_regression(train_loader, val_loader, lr_list)
    
    # Evaluate model
    print("\n⌛️ Evaluating model...")
    metrics = evaluate_model(model, test_loader)
    
    # Save results
    filename = f'{encoder_name}_{var}_{level}_{"-".join(datasets)}'
    
    # Create output directories
    models_dir = os.path.join(OUTPUTS_DIR, "models")
    output_dir = os.path.join(OUTPUTS_DIR, "output")
    plots_dir = os.path.join(OUTPUTS_DIR, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(models_dir, f"{filename}.pth"))
    
    # Save output
    output = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "lr_list": lr_list.tolist(),
        "losses": losses,
        "metrics": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
    }
    save_pickle(output, os.path.join(output_dir, f"{filename}.pkl"))
    
    # Save ROC curve
    plot_roc_curve(
        metrics["fpr"],
        metrics["tpr"],
        metrics["auc"],
        save_path=os.path.join(plots_dir, f"{filename}_roc.png"),
        show=False
    )
    
    print(f"\n✅ Model saved as {filename}")
    print(f"📊 Results saved to {output_dir}")
    print(f"📈 Plots saved to {plots_dir}")