import os
import numpy as np
import math
import openslide
import torch
from torch.utils.data import Dataset, DataLoader

from tcgapipeline.utils import get_root_dir, get_outputs_dir, get_clinical_dir, load_json

root = get_root_dir()
outputs_dir = get_outputs_dir()
clinical_dir = get_clinical_dir()

def get_slide_dir(d):
    return root + f"wsi-datasets/tcga/{d}/svs/"

def is_svs(path):
    return os.path.exists(path) and path[-4:] == ".svs"

def get_patient_id(filename):
    f = filename.split("/")[-1]
    return "-".join(f.split("-")[:3])

def is_dx(filename):
    f = filename.split("/")[-1]
    return f.split("-")[5][:2] == "DX"

class TCGASlides(Dataset):
    def __init__(self, datasets, var=None, dx_only=True):
        slide_paths = []
        pt_ids = []
        dataset_list = []
        labels = []
        for d in datasets:
            d_dir = get_slide_dir(d)
            for f in os.listdir(d_dir):
                if dx_only and not is_dx(f):
                    continue
                slide_path = os.path.join(d_dir, f)
                if is_svs(slide_path):
                    pt_id = get_patient_id(f)
                    slide_paths.append(slide_path)
                    pt_ids.append(pt_id)
                    dataset_list.append(d)
                    if var:
                        label = 0 ## TODO
                        labels.append(label) 
                    else:
                        labels.append(0)
                else:
                    print(f'"{slide_path}" is not a valid .svs file')
                
                # label = clinical_df.loc[pt_id][var]
                # if label in metadata_var_dict[var]:
                #     paths.append(os.path.join(features_dir, d, "mean", f))
                #     labels.append(metadata_var_dict[var][label])
                #     pt_ids.append(pt_id)
                #     dataset_list.append(d)
        self.slide_paths = slide_paths
        self.pt_ids = pt_ids
        self.dataset_list = dataset_list
        self.labels = labels
        

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        slide_path = self.slide_paths[idx]
        label = self.labels[idx]
        pt_id = self.pt_ids[idx]
        d = self.dataset_list[idx]
        
        return slide_path, label, pt_id, d

def single_sample_collate(batch):
    return batch[0]

def get_slides_loader(datasets, var=None, verbose=True):
    slides_loader = DataLoader(TCGASlides(datasets, var=var), collate_fn=single_sample_collate)
    if verbose:
        num_slides = slides_loader.__len__()
        print(f"Processing {num_slides} {(', ').join([d.upper() for d in datasets])} slides ...")
    return slides_loader

def get_slide_by_path(slide_path):    
    if os.path.exists(slide_path):
        slide = openslide.OpenSlide(slide_path)
        return slide
    raise FileNotFoundError(f"{slide_path} could not be found")

def view_whole_slide(slide):
    max_level = slide.level_count - 1
    whole_slide = slide.read_region((0, 0), max_level, slide.level_dimensions[max_level])
    return whole_slide

def get_slide_path(slide_dir, short_name):
    slide_path = slide_dir + [f for f in os.listdir(slide_dir) if f.startswith(short_name)][0]
    return slide_path

class TCGAPatches(Dataset):
    def __init__(self, slide_path, level, transforms, patch_len=512):
        slide = get_slide_by_path(slide_path)
        (width, height) = slide.level_dimensions[level]
        rows = math.ceil(height / patch_len)
        cols = math.ceil(width / patch_len)
        
        self.slide = slide
        self.level = level
        self.patch_len = patch_len
        self.rows = rows
        self.cols = cols
        self.transforms = transforms

    def __len__(self):
        return self.rows * self.cols

    def __getitem__(self, idx):
        i = idx // self.rows * self.patch_len * 4**self.level
        j = idx % self.rows * self.patch_len * 4**self.level
        level = self.level
        patch_dims = (self.patch_len, self.patch_len)

        patch = self.slide.read_region((i, j), level, patch_dims).convert("RGB")
        x = self.transforms(patch)
        return x, (i, j)

def get_patches_loader(slide_path, level, transforms, patch_len, batch_size, num_workers, verbose=True):
    patches_loader = DataLoader(TCGAPatches(slide_path, level, transforms, patch_len), batch_size=batch_size, num_workers=num_workers)
    return patches_loader

class TCGATissuePatches(Dataset):
    def __init__(self, slide_path, pkl, level, encoder_transforms, patch_transforms, THRESHOLD, patch_len):
        coords_list = []
        for coords in pkl:
            if pkl[coords] > THRESHOLD:
                coords_list.append(coords)

        slide = get_slide_by_path(slide_path)
        
        self.slide = slide
        self.coords_list = coords_list
        self.level = level
        self.patch_len = patch_len
        self.encoder_transforms = encoder_transforms
        self.patch_transforms = patch_transforms

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        coords = self.coords_list[idx]
        patch = self.slide.read_region(coords, self.level, (self.patch_len, self.patch_len)).convert("RGB")
        for f in self.patch_transforms:
            patch = f(patch)
        x = self.encoder_transforms(patch)
        return x, coords

def get_tissue_patches_loader(slide_path, pkl, level, encoder_transforms, patch_transforms, THRESHOLD, patch_len, batch_size, num_workers, verbose=True):
    tissue_patches_loader = DataLoader(TCGATissuePatches(slide_path, pkl, level, encoder_transforms, patch_transforms, THRESHOLD, patch_len), batch_size=batch_size, num_workers=num_workers)
    return tissue_patches_loader

####

# GENES = {
#     "luad"
# }

# def get_genetic_label():

def get_label(var, label_dict, genes, pt_id, d):
    if var == "subtype":
        return label_dict[d]
    if var in genes and pt_id in genes[var]:
        return genes[var][pt_id]
    return None

class TCGAPrediction(Dataset):
    def __init__(self, encoder_name, level, datasets, var):
        embeddings_paths = []
        pt_ids = []
        dataset_list = []
        labels = []
        for d in datasets:
            genetic_dir = os.path.join(clinical_dir, f"tcga-maf-summaries/{d.upper()}/")
            genes = load_json(genetic_dir + 'top_gene_statuses.json')
            
            label_dict = {}
            if var == "subtype":
                label_dict = {v: i for i, v in enumerate(datasets)}
            elif var not in genes:
                print(f"{var} not found")
                return
            
            d_dir = os.path.join(outputs_dir, f"{encoder_name}/tcga/{d}/level_{level}_mean/")
            for f in os.listdir(d_dir):
                embedding_path = os.path.join(d_dir, f)
                pt_id = get_patient_id(f)
                label = get_label(var, label_dict, genes, pt_id, d)
                if label is not None and os.path.exists(embedding_path):
                    embeddings_paths.append(embedding_path)
                    labels.append(label)
                    pt_ids.append(pt_id)
                    dataset_list.append(d)
                else:
                    print(f"No genetic status found for {pt_id}")
                
        self.embeddings_paths = embeddings_paths
        self.pt_ids = pt_ids
        self.dataset_list = dataset_list
        self.labels = labels

    def __len__(self):
        return len(self.embeddings_paths)

    def __getitem__(self, idx):
        embedding = np.load(self.embeddings_paths[idx])
        embedding = torch.from_numpy(embedding).float()
        label = torch.tensor(self.labels[idx]).float()

        # label = self.labels[idx]
        pt_id = self.pt_ids[idx]
        d = self.dataset_list[idx]
        
        return embedding, label, pt_id, d

# def get_prediction_loader(encoder_name, level, datasets, var, batch_size, num_workers):
#     prediction_loader = DataLoader(TCGAPrediction(encoder_name, level, datasets, var), batch_size=batch_size, num_workers=num_workers)
#     return prediction_loader