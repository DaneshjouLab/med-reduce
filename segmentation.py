import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from huggingface_hub import hf_hub_download

from tcgapipeline.utils import save_pickle
from tcgapipeline.wsi import get_patches_loader

root = "/oak/stanford/groups/roxanad/"

def segment_slides(slides_loader, segmenter, level, batch_size, num_workers, patch_len=512, verbose=True):
    num_slides = slides_loader.__len__()
    idx = 1
    for slide_path, label, pt_id, d in slides_loader:
        patches_loader = get_patches_loader(slide_path, level, segmenter.transforms, patch_len, batch_size, num_workers)
        if verbose:
            num_patches = patches_loader.__len__()
            print(f"-> Slide {idx}/{num_slides}: Processing {num_patches} patches from {slide_path} ...")
        
        tissue_frac_dir = root + f"rpark23/outputs/hest/tcga/{d}/level_{level}/"
        os.makedirs(tissue_frac_dir, exist_ok=True)
        patch_info_path = tissue_frac_dir + slide_path.split("/")[-1][:-4] + ".pkl"
        if not os.path.exists(patch_info_path):
            coords_to_tissue_fraction = {}
            for patch_batch, coords_list in patches_loader:
                masks = segmenter(patch_batch)
                for i in range(masks.shape[0]):
                    mask = masks[i].cpu().numpy()
                    tissue_fraction = np.sum(mask) / patch_len**2
                    coords_to_tissue_fraction[(coords_list[0][i].item(), coords_list[1][i].item())] = tissue_fraction
        
            save_pickle(coords_to_tissue_fraction, patch_info_path)
            if verbose:
                print(f"✅ Slide {idx}/{num_slides}: Patch info saved to {patch_info_path}")
        
        idx += 1

class HESTSegmenter(nn.Module):
    def __init__(self, confidence_thresh: float = 0.5, device=None):
        super().__init__()
        self.confidence_thresh = confidence_thresh

        # Pick GPU if available
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"Using {device}...")

        # Download only the checkpoint file
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

        # Send model to GPU
        self.model.to(self.device)

        # Normalization
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the same device as the model
        x = x.to(self.device)

        # Optionally accelerate with mixed precision
        with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            logits = self.model(x)["out"]            # [B, 2, H, W]
            probs = F.softmax(logits, dim=1)[:, 1]   # take tissue channel
            return (probs > self.confidence_thresh).to(torch.uint8)  # [B, H, W]