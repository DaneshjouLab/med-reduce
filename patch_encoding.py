import os
import timm
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from tcgapipeline.utils import load_pickle, save_pickle
from tcgapipeline.wsi import get_tissue_patches_loader, get_slide_by_path, view_whole_slide

root = "/oak/stanford/groups/roxanad/"

univ2_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def get_univ2_path():
    return root + "rpark23/cache/hub/models--MahmoodLab--UNI2-h/snapshots/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin"

def load_univ2():
    weights_path = get_univ2_path()
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    model = timm.create_model(model_name='vit_giant_patch14_224', pretrained=False, **timm_kwargs)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return model, univ2_transforms

def load_dinov3():
    dinov3_transforms = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    return model, dinov3_transforms

def get_patch_encoder(model_name):
    if model_name == "univ2":
        return load_univ2()
    elif model_name == "dinov3":
        return load_dinov3()
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from ['univ2', 'dinov3'].")


def patch_slides(model_name, slides_loader, level, patch_transforms, THRESHOLD, patch_len, batch_size, num_workers, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    patch_encoder, encoder_transforms = get_patch_encoder(model_name)
    patch_encoder = patch_encoder.eval().to(device)

    num_slides = len(slides_loader)
    idx = 0
    for slide_path, label, pt_id, d in slides_loader:
        idx += 1
        tissue_info_dir = root + f"rpark23/outputs/hest/tcga/{d}/level_{level}/"
        slide_name = slide_path.split("/")[-1][:-4]
        # features_dir = root + f"rpark23/outputs/{model_name}/tcga/{d}/level_{level}/"
        features_dir = root + f"rpark23/datasets/tcga-resized/1/{d}/"
        os.makedirs(features_dir, exist_ok=True)
        features_path = features_dir + slide_name + ".pkl"

        if os.path.exists(features_path):
            continue
        
        pkl = load_pickle(tissue_info_dir + slide_name + ".pkl")
        
        tissue_patches_loader = get_tissue_patches_loader(slide_path, pkl, level, encoder_transforms, patch_transforms, THRESHOLD, patch_len, batch_size, num_workers)
        if verbose:
            num_patches = len(tissue_patches_loader)
            print(f"-> Slide {idx}/{num_slides}: Processing {num_patches} patches from {slide_path} ...")
        
        coords_to_features = {}
        for patch_batch, coords_list in tissue_patches_loader:
            with torch.no_grad():
                if model_name == "dinov3":
                    pixel_values = torch.stack(patch_batch["pixel_values"], dim=0).squeeze(0).to(device)
                    features = patch_encoder(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
                else:
                    features = patch_encoder(patch_batch.to(device))
            for i in range(features.shape[0]):
                coords_to_features[(coords_list[0][i].item(), coords_list[1][i].item())] = features[i].detach().cpu().numpy()
        save_pickle(coords_to_features, features_path)
        if verbose:
            print(f"✅ Slide {idx}/{num_slides}: Features saved to {features_path}")
        
        