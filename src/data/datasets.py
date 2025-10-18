# src/data/datasets.py
from src.data.isic_dataset import ISICDataset
from src.data.tcga_dataset import TCGADataset
from src.data.merlin_dataset import MerlinDataset

def get_dataset(dataset_name, data_dir, split, cfg):
    if dataset_name.lower() == "isic":
        return ISICDataset(data_dir=data_dir, split=split, cfg=cfg)
    elif dataset_name.lower() == "tcga":
        return TCGADataset(data_dir=data_dir, split=split, cfg=cfg)
    elif dataset_name.lower() == "merlin":
        return MerlinDataset(data_dir=data_dir, split=split, cfg=cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
