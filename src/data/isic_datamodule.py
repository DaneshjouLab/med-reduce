# isic_datamodule.py

from typing import Optional, Any, List
from torch.utils.data import DataLoader, random_split, Subset
import torch
from torchvision import transforms

from src.data.datamodule import BaseDataModule 
from src.data.dataset_factory import balance_dataset
from src.data.isic_loader import ISICHFRawSplit 

# --- Constants
VAL_SPLIT_RATIO = 0.1
DEFAULT_REPO_ID = "MKZuziak/ISIC_2019_224"

class ISICDataModule(BaseDataModule):
    """
    A specific DataModule for the Hugging Face-backed ISIC dataset, 
    using ISICHFRawSplit for loading.
    
    It overrides the 'setup' method to directly instantiate the dataset, 
    bypassing the generic dataset_factory.
    """

    def __init__(
        self,
        dataset_name: str, 
        data_dir: str,
        *,
        num_workers: int = 8,
        batch_size: int = 32,
        pin_memory: bool = True,
        drop_last: bool = False,
        split_seed: int = 42,
        transform: Optional[Any] = None, 
        filter_fn: Optional[Any] = None,
        keep_indices: Optional[Any] = None,
        full_cfg: Optional[Any] = None,
        filtered_classes: Optional[List[str]] = None, 
        balance_data: bool = True,
        num_train_images: Optional[int] = 1000,
        **kwargs # Catch-all for any other parameters from BaseDataModule
    ):
        self.full_cfg = full_cfg

        super().__init__(
            cfg=full_cfg,
            dataset_name=dataset_name,
            data_dir=data_dir,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            drop_last=drop_last,
            split_seed=split_seed,
            transform=transform,
            **kwargs
        )
        
        self.filter_fn = filter_fn
        self.keep_indices = keep_indices
        self.repo_id = dataset_name 

        self.filtered_classes = filtered_classes or ["0", "1"] 
        self.balance_data = balance_data
        self.num_train_images = num_train_images

    def _load_split(
        self, 
        split: str, 
        transform: Optional[Any] = None,
        balance: bool = False
    ):
        """Load a dataset split with optional balancing."""
        
        # Load the raw split
        dataset = ISICHFRawSplit(
            repo_id=self.repo_id,
            split=split,
            cache_dir=self.data_dir,
            transform=transform,
            filter_fn=self.filter_fn,
            keep_indices=self.keep_indices,
        )
        
        if balance and self.balance_data and self.num_train_images:
            print(f"  Balancing {split} set to {self.num_train_images} samples...")
            dataset = balance_dataset(
                dataset=dataset,
                filtered_classes=self.filtered_classes,
                num_train_images=self.num_train_images,
                seed=self.split_seed
            )
            print(f"  → Balanced to {len(dataset)} samples")
        elif balance and self.balance_data:
            print(f"  Filtering {split} set to classes {self.filtered_classes}...")
            dataset = balance_dataset(
                dataset=dataset,
                filtered_classes=self.filtered_classes,
                num_train_images=len(dataset), 
                seed=self.split_seed
            )
            print(f"  → Filtered to {len(dataset)} samples")
        
        return dataset

    def setup(self, _stage: Optional[str] = None):
        """Initialize datasets using ISICHFRawSplit with balancing."""
        print(f"\n{'='*60}")
        print(f"Setting up ISIC DataModule")
        print(f"  Dataset: {self.repo_id}")
        if self.balance_data:
            print(f"  Filtered classes: {self.filtered_classes}")
            print(f"  Balance data: {self.balance_data}")
            if self.num_train_images:
                print(f"  Num train images: {self.num_train_images}")
        print(f"{'='*60}\n")
        
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_transform = self.transform if self.transform is not None else val_test_transform
        
        print("Loading training set...")
        self.train_set = self._load_split(
            split="train",
            transform=train_transform,
            balance=True  
        )
        print(f"  ✓ {len(self.train_set)} training samples\n")
        
        print("Loading validation set...")
        try:
            self.val_set = self._load_split(
                split="validation",
                transform=val_test_transform,
                balance=self.balance_data
            )
            print(f"  ✓ {len(self.val_set)} validation samples\n")
        except Exception as e:
            print(f"  No validation split found, splitting from training...")
            
            val_base = self._load_split(
                split="train",
                transform=val_test_transform,
                balance=True 
            )
            
            n_total = len(self.train_set)
            n_val = max(1, int(VAL_SPLIT_RATIO * n_total))
            n_train = n_total - n_val
            
            g = torch.Generator().manual_seed(self.split_seed)
            train_subset, val_subset = random_split(
                range(n_total), [n_train, n_val], generator=g
            )
            
            self.train_set = Subset(self.train_set, train_subset.indices)
            self.val_set = Subset(val_base, val_subset.indices)
            
            print(f"  ✓ Split: {n_train} train, {n_val} val\n")
        
        print("Loading test set...")
        try:
            self.test_set = self._load_split(
                split="test",
                transform=val_test_transform,
                balance=self.balance_data  
            )
            print(f"  ✓ {len(self.test_set)} test samples\n")
        except Exception:
            print(f"  No test split found\n")
            self.test_set = None
        
        print(f"{'='*60}")
        print(f"DataModule setup complete!")
        print(f"  Train: {len(self.train_set)} samples")
        print(f"  Val: {len(self.val_set)} samples")
        if self.test_set:
            print(f"  Test: {len(self.test_set)} samples")
        print(f"{'='*60}\n")
