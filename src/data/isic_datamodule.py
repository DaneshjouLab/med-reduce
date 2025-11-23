# isic_datamodule.py

from typing import Optional, Any, List
import os
from torch.utils.data import DataLoader, random_split, Subset
import torch
from torchvision import transforms
from torchvision import set_image_backend
set_image_backend("accimage")

from src.data.datamodule import BaseDataModule 
from src.data.dataset_factory import balance_dataset
from src.data.isic_loader import ISICHFRawSplit, ISICHFRawSplitLocal
from src.transformations.transforms import SegmentationTransform

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
        dataset_name: Optional[str] = DEFAULT_REPO_ID, 
        data_dir: Optional[str] = None,
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
        local_label_file: str = "labels.csv", 
        local_image_column: str = "file_name",
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
        
        if dataset_name and not os.path.isdir(str(dataset_name)):
            self.data_source = "remote_hf"
            self.source_id = dataset_name
        elif data_dir and os.path.isdir(str(data_dir)):
            self.data_source = "local"
            self.source_id = data_dir
        else:
            raise ValueError(
                "Must provide valid `repo_id` (HF Hub) or `data_dir` (local directory)."
            )
        
        self.local_label_file = local_label_file
        self.local_image_column = local_image_column
    def _load_split(
        self, 
        split: str, 
        transform: Optional[Any] = None,
        balance: bool = False
    ):
        """Load a dataset split with optional balancing."""
        
        print(f"Loading split from: {self.data_source}")

        if self.data_source == "remote_hf":
            wrapper = ISICHFRawSplit(
                repo_id=self.source_id,
                split=split,
                cache_dir=self.data_dir, 
                transform=transform,
                filter_fn=self.filter_fn,
                keep_indices=self.keep_indices,
            )
        elif self.data_source == "local":
            wrapper = ISICHFRawSplitLocal(
                data_dir=self.source_id,
                split=split,
                transform=transform,
                filter_fn=self.filter_fn,
                keep_indices=self.keep_indices,
                label_file=self.local_label_file, 
                image_column=self.local_image_column,
            )
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        dataset = wrapper.ds

        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset features/columns: {dataset.column_names if hasattr(dataset, 'column_names') else 'N/A'}")
        if hasattr(dataset, 'features'):
            print(f"Features: {dataset.features}")

        if balance and self.balance_data:
            print(f"  Balancing {split}...")
            dataset = balance_dataset(
                dataset=dataset,
                filtered_classes=self.filtered_classes,
                num_train_images=self.num_train_images or len(dataset),
                seed=self.split_seed,
            )
            print(f"  → Balanced to {len(dataset)} samples")

        wrapper.ds = dataset
        return wrapper


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
            
            total = len(self.train_set)
            n_val = max(1, int(total * VAL_SPLIT_RATIO))
            n_train = total - n_val

            g = torch.Generator().manual_seed(self.split_seed)
            idx_train, idx_val = random_split(
                range(total), [n_train, n_val], generator=g
            )

            self.val_set = Subset(self.train_set, idx_val.indices)
            self.train_set = Subset(self.train_set, idx_train.indices)
            
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

class ISICSegDataModule(BaseDataModule):
    def __init__(
        self,
        repo_id: str, 
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
        **kwargs
    ):
        self.full_cfg = full_cfg

        super().__init__(
            cfg=full_cfg,
            dataset_name=repo_id,
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
        self.repo_id = repo_id 

        self.filtered_classes = filtered_classes or ["0", "1"] 
        self.balance_data = balance_data
        self.num_train_images = num_train_images
    
    def setup(self, _stage: Optional[str] = None):
        """Initialize datasets using ISICSegRawSplit for segmentation."""
        print(f"\n{'='*60}")
        print(f"Setting up ISIC Segmentation DataModule")
        print(f" Dataset: {self.repo_id or self.data_dir}")
        print(f"{'='*60}\n")

        val_test_transform = SegmentationTransform() 
        train_transform = self.transform if self.transform is not None else val_test_transform
        
        print("Loading training set...")
        self.train_set = self._load_split(
            split="train",
            transform=train_transform,
            balance=False 
        )
        print(f"  ✓ {len(self.train_set)} training samples\n")
        
        print("Loading validation set...")
        try:
            self.val_set = self._load_split(
                split="validation",
                transform=val_test_transform,
                balance=False
            )
            print(f"  ✓ {len(self.val_set)} validation samples\n")
        except Exception:
            print(f"  No validation split found. Consider defining a 'validation' split in your data source.")
            self.val_set = None # Or implement the Subset logic similar to the classification DM
            
        print("Loading test set...")
        try:
            self.test_set = self._load_split(
                split="test",
                transform=val_test_transform,
                balance=False
            )
            print(f" ✓ {len(self.test_set)} test samples\n")
        except Exception:
            print(f" No test split found\n")
            self.test_set = None
        
        print(f"{'='*60}")
        print(f"DataModule setup complete!")
        print(f" Train: {len(self.train_set)} samples")
        print(f" Val: {len(self.val_set) if self.val_set else '0'} samples")
        if self.test_set:
            print(f" Test: {len(self.test_set)} samples")
        print(f"{'='*60}\n")