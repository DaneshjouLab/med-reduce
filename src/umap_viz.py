"""
UMAP visualization functions for model embeddings.
Integrates with your existing training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import umap
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb


# ============================================================================
# Core Embedding Extraction
# ============================================================================

def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from model's backbone (before classifier).
    
    Args:
        model: Trained model
        dataloader: DataLoader with images
        device: Device to run on
        max_samples: Maximum number of samples to extract (None = all)
    
    Returns:
        embeddings: (N, embedding_dim) array
        labels: (N,) array
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            # Handle different batch formats
            if isinstance(batch, dict):
                pixel_values = batch['pixel_values'].to(device)
                batch_labels = batch['labels'].to(device)
            else:
                pixel_values, batch_labels = batch[0].to(device), batch[1].to(device)
            
            # Get embeddings before classifier
            if hasattr(model, 'backbone'):
                # DINOv3 wrapper
                outputs = model.backbone(pixel_values=pixel_values)
                emb = outputs.pooler_output
            elif hasattr(model, 'vit'):
                # ViT models
                outputs = model.vit(pixel_values=pixel_values)
                emb = outputs.last_hidden_state[:, 0]  # CLS token
            elif hasattr(model, 'dinov2'):
                # DINOv2 models
                outputs = model.dinov2(pixel_values=pixel_values)
                emb = outputs.last_hidden_state[:, 0]
            else:
                # Fallback: use model's forward but extract features
                outputs = model(pixel_values=pixel_values, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    emb = outputs.hidden_states[-1][:, 0]
                else:
                    raise ValueError("Cannot extract embeddings from this model")
            
            embeddings.append(emb.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
            
            # Early stop if max_samples reached
            if max_samples and len(embeddings) * emb.shape[0] >= max_samples:
                break
    
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    if max_samples:
        embeddings = embeddings[:max_samples]
        labels = labels[:max_samples]
    
    return embeddings, labels


# ============================================================================
# UMAP Plotting
# ============================================================================

def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Create UMAP visualization and save to file.
    
    Args:
        embeddings: (N, D) embedding array
        labels: (N,) label array
        title: Plot title
        save_path: Path to save figure
        class_names: Optional class names for legend
        figsize: Figure size
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed
    
    Returns:
        embedding_2d: (N, 2) UMAP coordinates
    """
    # Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=random_state
    )
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Use appropriate colormap
    if n_classes == 2:
        cmap = 'bwr'
    elif n_classes <= 10:
        cmap = 'tab10'
    else:
        cmap = 'tab20'
    
    # Plot scatter
    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap=cmap,
        s=10,
        alpha=0.7,
        edgecolors='none'
    )
    
    # Add legend if class names provided
    if class_names and len(class_names) == n_classes:
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=scatter.cmap(scatter.norm(i)), 
                             markersize=8, label=class_names[i])
                  for i in unique_labels]
        ax.legend(handles=handles, loc='best', framealpha=0.9)
    else:
        plt.colorbar(scatter, ax=ax, label='Class')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP plot saved to: {save_path}")
    
    return embedding_2d


# ============================================================================
# Before/After Comparison
# ============================================================================

def plot_umap_comparison(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    title_prefix: str = "",
) -> None:
    """
    Create side-by-side UMAP comparison (before and after training).
    
    Args:
        embeddings_before: (N, D) embeddings before training
        embeddings_after: (N, D) embeddings after training
        labels: (N,) label array
        save_path: Path to save figure
        class_names: Optional class names
        title_prefix: Prefix for titles (e.g., model name)
    """
    # Fit UMAP for both
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    
    emb_2d_before = reducer.fit_transform(embeddings_before)
    emb_2d_after = reducer.fit_transform(embeddings_after)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    cmap = 'bwr' if n_classes == 2 else ('tab10' if n_classes <= 10 else 'tab20')
    
    # Plot before
    scatter_before = axes[0].scatter(
        emb_2d_before[:, 0], emb_2d_before[:, 1],
        c=labels, cmap=cmap, s=10, alpha=0.7, edgecolors='none'
    )
    axes[0].set_title(f"{title_prefix}Before Training", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("UMAP-1", fontsize=12)
    axes[0].set_ylabel("UMAP-2", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot after
    scatter_after = axes[1].scatter(
        emb_2d_after[:, 0], emb_2d_after[:, 1],
        c=labels, cmap=cmap, s=10, alpha=0.7, edgecolors='none'
    )
    axes[1].set_title(f"{title_prefix}After Training", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("UMAP-1", fontsize=12)
    axes[1].set_ylabel("UMAP-2", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add shared legend
    if class_names and len(class_names) == n_classes:
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=scatter_after.cmap(scatter_after.norm(i)),
                             markersize=8, label=class_names[i])
                  for i in unique_labels]
        fig.legend(handles=handles, loc='center right', framealpha=0.9, bbox_to_anchor=(1.12, 0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {save_path}")


# ============================================================================
# Integration with Training Pipeline
# ============================================================================

def create_umap_callback(
    val_dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    model_name: str,
    class_names: Optional[List[str]] = None,
    max_samples: int = 2000,
):
    """
    Create a callback for UMAP visualization during training.
    Call this before training to capture initial embeddings.
    
    Args:
        val_dataloader: Validation dataloader
        device: Device
        output_dir: Directory to save plots
        model_name: Model name for filenames
        class_names: Class names for legend
        max_samples: Max samples to visualize
    
    Returns:
        Dictionary with callback functions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for before embeddings
    embeddings_before = None
    labels_before = None
    
    def capture_before(model: nn.Module):
        """Capture embeddings before training"""
        nonlocal embeddings_before, labels_before
        
        print("Capturing embeddings before training...")
        embeddings_before, labels_before = extract_embeddings(
            model, val_dataloader, device, max_samples=max_samples
        )
        
        # Plot before
        plot_path = os.path.join(output_dir, f"{model_name}_umap_before.png")
        plot_umap(
            embeddings_before,
            labels_before,
            title=f"{model_name} - Before Training",
            save_path=plot_path,
            class_names=class_names,
        )
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"umap/before_training": wandb.Image(plot_path)})
    
    def capture_after(model: nn.Module):
        """Capture embeddings after training and create comparison"""
        nonlocal embeddings_before, labels_before
        
        print("Capturing embeddings after training...")
        embeddings_after, labels_after = extract_embeddings(
            model, val_dataloader, device, max_samples=max_samples
        )
        
        # Plot after
        plot_path = os.path.join(output_dir, f"{model_name}_umap_after.png")
        plot_umap(
            embeddings_after,
            labels_after,
            title=f"{model_name} - After Training",
            save_path=plot_path,
            class_names=class_names,
        )
        
        # Create comparison if we have before embeddings
        if embeddings_before is not None:
            comparison_path = os.path.join(output_dir, f"{model_name}_umap_comparison.png")
            plot_umap_comparison(
                embeddings_before,
                embeddings_after,
                labels_after,
                save_path=comparison_path,
                class_names=class_names,
                title_prefix=f"{model_name} - ",
            )
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "umap/after_training": wandb.Image(plot_path),
                    "umap/comparison": wandb.Image(comparison_path),
                })
    
    return {
        "capture_before": capture_before,
        "capture_after": capture_after,
    }