"""Dataset classes for TCGA whole slide images."""
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Tuple, List, Optional, Dict, Callable

from src.config import get_slide_dir, OUTPUTS_DIR, CLINICAL_DIR
from src.utils import load_json
from .data_utils import get_slide_by_path, get_patient_id, is_svs, is_dx


class TCGASlides(Dataset):
    """
    Dataset for TCGA slide paths and metadata.
    
    Args:
        datasets: List of dataset names (e.g., ['luad', 'lusc'])
        var: Optional variable for labeling
        dx_only: Whether to use only DX slides
    """
    
    def __init__(
        self,
        datasets: List[str],
        var: Optional[str] = None,
        dx_only: bool = True
    ):
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
                        label = 0  # TODO: Implement proper labeling logic
                        labels.append(label)
                    else:
                        labels.append(0)
                else:
                    print(f'"{slide_path}" is not a valid .svs file')
        
        self.slide_paths = slide_paths
        self.pt_ids = pt_ids
        self.dataset_list = dataset_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.slide_paths)

    def __getitem__(self, idx: int) -> Tuple[str, int, str, str]:
        """
        Get slide information.
        
        Returns:
            Tuple of (slide_path, label, patient_id, dataset)
        """
        slide_path = self.slide_paths[idx]
        label = self.labels[idx]
        pt_id = self.pt_ids[idx]
        d = self.dataset_list[idx]
        
        return slide_path, label, pt_id, d


class TCGAPatches(Dataset):
    """
    Dataset for extracting patches from a whole slide image.
    
    Args:
        slide_path: Path to the slide file
        level: Pyramid level to extract patches from
        transforms: Transformation pipeline
        patch_len: Size of patches (square)
    """
    
    def __init__(
        self,
        slide_path: str,
        level: int,
        transforms: Any,
        patch_len: int = 512
    ):
        slide = get_slide_by_path(slide_path)
        width, height = slide.level_dimensions[level]
        rows = math.ceil(height / patch_len)
        cols = math.ceil(width / patch_len)
        
        self.slide = slide
        self.level = level
        self.patch_len = patch_len
        self.rows = rows
        self.cols = cols
        self.transforms = transforms

    def __len__(self) -> int:
        return self.rows * self.cols

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Get patch and its coordinates.
        
        Returns:
            Tuple of (transformed_patch, (x_coord, y_coord))
        """
        i = idx // self.rows * self.patch_len * 4**self.level
        j = idx % self.rows * self.patch_len * 4**self.level
        patch_dims = (self.patch_len, self.patch_len)

        patch = self.slide.read_region((i, j), self.level, patch_dims).convert("RGB")
        x = self.transforms(patch)
        return x, (i, j)


class TCGATissuePatches(Dataset):
    """
    Dataset for extracting tissue patches above a threshold.
    
    Args:
        slide_path: Path to the slide file
        pkl: Dictionary mapping coordinates to tissue fractions
        level: Pyramid level
        encoder_transforms: Encoder-specific transforms
        patch_transforms: Additional patch transforms
        threshold: Minimum tissue fraction threshold
        patch_len: Size of patches (square)
    """
    
    def __init__(
        self,
        slide_path: str,
        pkl: Dict[Tuple[int, int], float],
        level: int,
        encoder_transforms: Any,
        patch_transforms: List[Callable],
        threshold: float,
        patch_len: int
    ):
        coords_list = []
        for coords in pkl:
            if pkl[coords] > threshold:
                coords_list.append(coords)

        slide = get_slide_by_path(slide_path)
        
        self.slide = slide
        self.coords_list = coords_list
        self.level = level
        self.patch_len = patch_len
        self.encoder_transforms = encoder_transforms
        self.patch_transforms = patch_transforms

    def __len__(self) -> int:
        return len(self.coords_list)

    def __getitem__(self, idx: int) -> Tuple[Any, Tuple[int, int]]:
        """
        Get tissue patch and its coordinates.
        
        Returns:
            Tuple of (transformed_patch, coordinates)
        """
        coords = self.coords_list[idx]
        patch = self.slide.read_region(
            coords, self.level, (self.patch_len, self.patch_len)
        ).convert("RGB")
        
        for f in self.patch_transforms:
            patch = f(patch)
            
        x = self.encoder_transforms(patch)
        return x, coords


class TCGAPrediction(Dataset):
    """
    Dataset for loading pre-computed embeddings for prediction tasks.
    
    Args:
        encoder_name: Name of the encoder used
        level: Pyramid level
        datasets: List of dataset names
        var: Variable to predict ('subtype' or gene name)
    """
    
    def __init__(
        self,
        encoder_name: str,
        level: int,
        datasets: List[str],
        var: str
    ):
        embeddings_paths = []
        pt_ids = []
        dataset_list = []
        labels = []
        
        for d in datasets:
            genetic_dir = os.path.join(CLINICAL_DIR, f"tcga-maf-summaries/{d.upper()}/")
            genes = load_json(genetic_dir + 'top_gene_statuses.json')
            
            label_dict = {}
            if var == "subtype":
                label_dict = {v: i for i, v in enumerate(datasets)}
            elif var not in genes:
                print(f"{var} not found")
                return
            
            d_dir = os.path.join(OUTPUTS_DIR, f"{encoder_name}/tcga/{d}/level_{level}_mean/")
            for f in os.listdir(d_dir):
                embedding_path = os.path.join(d_dir, f)
                pt_id = get_patient_id(f)
                label = self._get_label(var, label_dict, genes, pt_id, d)
                
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

    def __len__(self) -> int:
        return len(self.embeddings_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Get embedding, label, patient ID, and dataset.
        
        Returns:
            Tuple of (embedding, label, patient_id, dataset)
        """
        embedding = np.load(self.embeddings_paths[idx])
        embedding = torch.from_numpy(embedding).float()
        label = torch.tensor(self.labels[idx]).float()
        pt_id = self.pt_ids[idx]
        d = self.dataset_list[idx]
        
        return embedding, label, pt_id, d

    @staticmethod
    def _get_label(
        var: str,
        label_dict: Dict[str, int],
        genes: Dict[str, Any],
        pt_id: str,
        d: str
    ) -> Optional[int]:
        """Get label for a patient based on variable."""
        if var == "subtype":
            return label_dict[d]
        if var in genes and pt_id in genes[var]:
            return genes[var][pt_id]
        return None

