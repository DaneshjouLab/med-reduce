"""Patch encoding engine for extracting features from WSI patches."""
import os
import torch
from typing import List, Callable

from src.config import EncodingConfig, get_tissue_info_dir, get_features_dir
from src.models import get_patch_encoder
from src.utils import load_pickle, save_pickle
from src.data import get_tissue_patches_loader


def encode_slides(
    model_name: str,
    slides_loader,
    config: EncodingConfig,
    patch_transforms: List[Callable] = None,
    verbose: bool = True
) -> None:
    """
    Encode slide patches using a pre-trained model.
    
    Args:
        model_name: Name of encoder model ('univ2' or 'dinov3')
        slides_loader: DataLoader for slides
        config: Encoding configuration
        patch_transforms: Additional patch transformations
        verbose: Whether to print progress
    """
    if patch_transforms is None:
        patch_transforms = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load encoder
    patch_encoder, encoder_transforms = get_patch_encoder(model_name)
    patch_encoder = patch_encoder.eval().to(device)

    num_slides = len(slides_loader)
    
    for idx, (slide_path, label, pt_id, d) in enumerate(slides_loader, 1):
        tissue_info_dir = get_tissue_info_dir(d, config.level)
        slide_name = slide_path.split("/")[-1][:-4]
        
        features_dir = get_features_dir(model_name, d, config.level)
        os.makedirs(features_dir, exist_ok=True)
        features_path = os.path.join(features_dir, slide_name + ".pkl")

        # Skip if already processed
        if os.path.exists(features_path):
            if verbose:
                print(f"⏭️  Slide {idx}/{num_slides}: Features already exist, skipping...")
            continue
        
        # Load tissue info
        tissue_info_path = os.path.join(tissue_info_dir, slide_name + ".pkl")
        if not os.path.exists(tissue_info_path):
            print(f"⚠️  Tissue info not found for {slide_name}, skipping...")
            continue
            
        pkl = load_pickle(tissue_info_path)
        
        # Create tissue patches loader
        tissue_patches_loader = get_tissue_patches_loader(
            slide_path, pkl, config.level,
            encoder_transforms, patch_transforms,
            config.threshold, config.patch_len,
            config.batch_size, config.num_workers
        )
        
        if verbose:
            num_patches = len(tissue_patches_loader)
            print(
                f"-> Slide {idx}/{num_slides}: "
                f"Processing {num_patches} patches from {slide_path} ..."
            )
        
        # Extract features
        coords_to_features = {}
        for patch_batch, coords_list in tissue_patches_loader:
            with torch.no_grad():
                if model_name == "dinov3":
                    pixel_values = torch.stack(
                        patch_batch["pixel_values"], dim=0
                    ).squeeze(0).to(device)
                    features = patch_encoder(pixel_values=pixel_values)
                    features = features.last_hidden_state.mean(dim=1)
                else:
                    features = patch_encoder(patch_batch.to(device))
                    
            for i in range(features.shape[0]):
                coords = (coords_list[0][i].item(), coords_list[1][i].item())
                coords_to_features[coords] = features[i].detach().cpu().numpy()
        
        save_pickle(coords_to_features, features_path)
        
        if verbose:
            print(f"✅ Slide {idx}/{num_slides}: Features saved to {features_path}")

