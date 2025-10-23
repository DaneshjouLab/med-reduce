"""Tissue segmentation engine for whole slide images."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from huggingface_hub import hf_hub_download
from typing import Optional

from src.config import (
    IMAGE_NORMALIZATION,
    SegmentationConfig,
    get_tissue_info_dir,
)
from src.utils import save_pickle
from src.data import get_patches_loader


class HESTSegmenter(nn.Module):
    """
    HEST tissue segmentation model using DeepLabV3.
    
    Args:
        confidence_thresh: Confidence threshold for tissue detection
        device: Device to run model on (None = auto-detect)
    """
    
    def __init__(
        self,
        confidence_thresh: float = 0.5,
        device: Optional[str] = None
    ):
        super().__init__()
        self.confidence_thresh = confidence_thresh

        # Pick GPU if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(f"Using {self.device}...")

        # Download checkpoint
        ckpt_path = hf_hub_download(
            repo_id="MahmoodLab/hest-tissue-seg",
            filename="deeplabv3_seg_v4.ckpt"
        )

        # Build backbone
        self.model = deeplabv3_resnet50(weights=None)
        self.model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

        # Load weights
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = {
            k.replace("model.", ""): v
            for k, v in ckpt.get("state_dict", ckpt).items()
            if "aux" not in k
        }
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Send model to device
        self.model.to(self.device)

        # Normalization
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_NORMALIZATION["mean"],
                std=IMAGE_NORMALIZATION["std"]
            )
        ])

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for tissue segmentation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Binary tissue mask of shape (B, H, W)
        """
        # Ensure input is on the same device as the model
        x = x.to(self.device)

        # Optionally accelerate with mixed precision
        with torch.amp.autocast(
            device_type="cuda",
            enabled=self.device.type == "cuda"
        ):
            logits = self.model(x)["out"]            # [B, 2, H, W]
            probs = F.softmax(logits, dim=1)[:, 1]   # take tissue channel
            return (probs > self.confidence_thresh).to(torch.uint8)  # [B, H, W]


def segment_slides(
    slides_loader,
    segmenter: HESTSegmenter,
    config: SegmentationConfig,
    verbose: bool = True
) -> None:
    """
    Segment tissue in whole slide images.
    
    Args:
        slides_loader: DataLoader for slides
        segmenter: HEST segmentation model
        config: Segmentation configuration
        verbose: Whether to print progress
    """
    num_slides = len(slides_loader)
    idx = 1
    
    for slide_path, label, pt_id, d in slides_loader:
        patches_loader = get_patches_loader(
            slide_path,
            config.level,
            segmenter.transforms,
            config.patch_len,
            config.batch_size,
            config.num_workers
        )
        
        if verbose:
            num_patches = len(patches_loader)
            print(
                f"-> Slide {idx}/{num_slides}: "
                f"Processing {num_patches} patches from {slide_path} ..."
            )
        
        tissue_frac_dir = get_tissue_info_dir(d, config.level)
        os.makedirs(tissue_frac_dir, exist_ok=True)
        
        slide_name = slide_path.split("/")[-1][:-4]
        patch_info_path = os.path.join(tissue_frac_dir, slide_name + ".pkl")
        
        if not os.path.exists(patch_info_path):
            coords_to_tissue_fraction = {}
            
            for patch_batch, coords_list in patches_loader:
                masks = segmenter(patch_batch)
                
                for i in range(masks.shape[0]):
                    mask = masks[i].cpu().numpy()
                    tissue_fraction = mask.sum() / (config.patch_len ** 2)
                    coords = (coords_list[0][i].item(), coords_list[1][i].item())
                    coords_to_tissue_fraction[coords] = tissue_fraction
            
            save_pickle(coords_to_tissue_fraction, patch_info_path)
            
            if verbose:
                print(
                    f"✅ Slide {idx}/{num_slides}: "
                    f"Patch info saved to {patch_info_path}"
                )
        
        idx += 1

