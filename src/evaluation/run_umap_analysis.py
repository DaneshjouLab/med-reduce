"""
Standalone script to perform UMAP analysis on saved model embeddings.
"""
import torch
import numpy as np
import os
from typing import List, Optional
import argparse
import sys

# --- Re-import necessary UMAP functions from your existing file ---
# NOTE: Assuming your existing UMAP functions are now saved in a file 
# like 'src/evaluation/umap_viz.py'
try:
    from src.evaluation.run_umap_analysis import plot_umap, plot_umap_comparison
except ImportError:
    # Fallback if running standalone
    print("Warning: Could not import plot_umap and plot_umap_comparison. Please ensure the functions are accessible.")
    sys.exit(1)


def analyze_saved_embeddings(
    embedding_dir: str,
    plot_dir: str,
    run_name: str,
    class_names: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """
    Loads saved embeddings and performs UMAP projection and plotting.

    Args:
        embedding_dir: Directory containing saved embedding files (*.pt).
        plot_dir: Directory to save the final UMAP plots.
        run_name: Name of the training run (used for plot titles/filenames).
        class_names: Optional list of class names for the plot legend.
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Discover all saved embedding files
    embedding_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith('.pt')])
    
    if not embedding_files:
        print(f"Error: No .pt embedding files found in {embedding_dir}")
        return

    print(f"Found {len(embedding_files)} embedding files for analysis.")
    
    embeddings_before = None
    labels = None
    
    # 2. Find the "Before Training" file
    before_file = next((f for f in embedding_files if 'e000' in f or 'before' in f.lower()), embedding_files[0])
    
    # Load the first (or designated 'before') file
    try:
        data_before = torch.load(os.path.join(embedding_dir, before_file))
        embeddings_before = data_before['embeddings']
        labels = data_before['label']
        print(f"Loaded 'Before Training' embeddings from: {before_file}")
    except Exception as e:
        print(f"Error loading initial embeddings from {before_file}: {e}")
        return

    # 3. Plot all individual epochs
    for filename in embedding_files:
        epoch_num = filename.split('_e')[-1].split('.')[0]
        
        # Load data for this epoch
        data_epoch = torch.load(os.path.join(embedding_dir, filename))
        embeddings = data_epoch['embeddings']
        
        # Note: Your training code saves NumPy arrays, but we load them via 
        # torch.load() which converts to a Tensor. We must convert back to NumPy.
        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy() # Assumes labels are the same
        
        # Call the existing plot_umap function
        plot_path = os.path.join(plot_dir, f"{run_name}_umap_epoch_{epoch_num}.png")
        plot_umap(
            embeddings=embeddings_np,
            labels=labels_np,
            title=f"{run_name} - Epoch {int(epoch_num)}",
            save_path=plot_path,
            class_names=class_names,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

    # 4. Create Before/After Comparison
    if len(embedding_files) > 1:
        # Get the last (final) embedding file
        last_file = embedding_files[-1]
        data_after = torch.load(os.path.join(embedding_dir, last_file))
        embeddings_after_np = data_after['embeddings'].cpu().numpy()
        
        print(f"Creating comparison plot using final embeddings from: {last_file}")

        comparison_path = os.path.join(plot_dir, f"{run_name}_umap_final_comparison.png")
        plot_umap_comparison(
            embeddings_before=embeddings_before.cpu().numpy(),
            embeddings_after=embeddings_after_np,
            labels=labels_np,
            save_path=comparison_path,
            class_names=class_names,
            title_prefix=f"{run_name} - ",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP Analysis Script")
    parser.add_argument("--embedding_dir", required=True, help="Directory where embeddings (.pt files) are saved.")
    parser.add_argument("--plot_dir", default="umap_plots_analysis", help="Directory to save UMAP plots.")
    parser.add_argument("--run_name", default="Finetune_Run", help="Name of the run for titles.")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist parameter.")
    parser.add_argument("--class_names", nargs='+', type=str, default=None, help="List of class names for the legend.")

    args = parser.parse_args()

    analyze_saved_embeddings(
        embedding_dir=args.embedding_dir,
        plot_dir=args.plot_dir,
        run_name=args.run_name,
        class_names=args.class_names,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )