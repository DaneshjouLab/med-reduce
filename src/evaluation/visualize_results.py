# src/evaluation/visualize_results.py
"""Visualize robustness results across models and degradations."""
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def create_robustness_heatmap(results_path: str):
    """Create heatmap showing model performance across degradations."""
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Prepare data for heatmap
    models = []
    degradations = []
    accuracies = []
    
    for training_mode in ['finetune', 'linear_probe']:
        for model_name, model_results in results[training_mode].items():
            if 'error' not in model_results:
                eval_results = model_results['eval_results_by_degradation']
                
                for degradation, metrics in eval_results.items():
                    models.append(f"{model_name}_{training_mode[:2]}")
                    degradations.append(degradation)
                    accuracies.append(metrics['accuracy'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Degradation': degradations,
        'Accuracy': accuracies
    })
    
    # Pivot for heatmap
    pivot_df = df.pivot(index='Model', columns='Degradation', values='Accuracy')
    
    # Reorder columns logically
    column_order = ['clean', 
                   'jpeg_90', 'jpeg_50', 'jpeg_20',
                   'blur_1.0', 'blur_3.0', 'blur_5.0',
                   'color_64', 'color_16', 'color_4']
    pivot_df = pivot_df[column_order]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn',
                vmin=0.5, 
                vmax=1.0,
                cbar_kws={'label': 'Accuracy'})
    
    plt.title('Model Robustness Across Different Degradations')
    plt.xlabel('Degradation Type')
    plt.ylabel('Model (ft=finetune, lp=linear probe)')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(results_path).parent / 'robustness_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return pivot_df

def create_degradation_curves(results_path: str):
    """Create line plots showing accuracy degradation."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # JPEG degradation
    jpeg_qualities = [90, 50, 20]
    ax = axes[0]
    
    for training_mode in ['finetune', 'linear_probe']:
        for model_name, model_results in results[training_mode].items():
            if 'error' not in model_results:
                eval_results = model_results['eval_results_by_degradation']
                
                clean_acc = eval_results['clean']['accuracy']
                jpeg_accs = [eval_results[f'jpeg_{q}']['accuracy'] for q in jpeg_qualities]
                
                label = f"{model_name} ({training_mode})"
                ax.plot([100] + jpeg_qualities, [clean_acc] + jpeg_accs, 
                       marker='o', label=label)
    
    ax.set_xlabel('JPEG Quality')
    ax.set_ylabel('Accuracy')
    ax.set_title('JPEG Compression Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Blur degradation
    blur_radii = [0, 1.0, 3.0, 5.0]
    ax = axes[1]
    
    for training_mode in ['finetune', 'linear_probe']:
        for model_name, model_results in results[training_mode].items():
            if 'error' not in model_results:
                eval_results = model_results['eval_results_by_degradation']
                
                blur_accs = []
                blur_accs.append(eval_results['clean']['accuracy'])
                for r in blur_radii[1:]:
                    blur_accs.append(eval_results[f'blur_{r:.1f}']['accuracy'])
                
                label = f"{model_name} ({training_mode})"
                ax.plot(blur_radii, blur_accs, marker='o', label=label)
    
    ax.set_xlabel('Blur Radius')
    ax.set_ylabel('Accuracy')
    ax.set_title('Gaussian Blur Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Color quantization
    color_levels = [256, 64, 16, 4]
    ax = axes[2]
    
    for training_mode in ['finetune', 'linear_probe']:
        for model_name, model_results in results[training_mode].items():
            if 'error' not in model_results:
                eval_results = model_results['eval_results_by_degradation']
                
                color_accs = []
                color_accs.append(eval_results['clean']['accuracy'])
                for c in color_levels[1:]:
                    color_accs.append(eval_results[f'color_{c}']['accuracy'])
                
                label = f"{model_name} ({training_mode})"
                ax.plot(color_levels, color_accs, marker='o', label=label)
    
    ax.set_xlabel('Number of Colors')
    ax.set_ylabel('Accuracy')
    ax.set_title('Color Quantization Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.invert_xaxis()
    
    plt.suptitle('Model Robustness to Different Degradation Types')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(results_path).parent / 'degradation_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize robustness results across models and degradations"
    )
    parser.add_argument(
        "results_path",
        type=str,
        help="Path to results JSON file (e.g., results/results_comprehensive.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output figures. If not specified, uses parent directory of results_path"
    )

    args = parser.parse_args()

    results_path = args.results_path

    # Validate that results file exists
    if not Path(results_path).exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Create visualizations
    pivot_df = create_robustness_heatmap(results_path)
    create_degradation_curves(results_path)

    # Print summary statistics
    print("\nModel Rankings by Robustness:")
    print("-" * 40)
    robustness_scores = pivot_df.mean(axis=1).sort_values(ascending=False)
    for model, score in robustness_scores.items():
        print(f"{model}: {score:.3f}")