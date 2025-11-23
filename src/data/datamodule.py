# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
DataModule for managing dataset loading, splitting, and dataloader creation.
Provides a unified interface for working with different datasets.
"""

# Standard library imports
from typing import Optional, Any

# Third-party imports
import torch    # pylint: disable=import-error
from torch.utils.data import DataLoader, random_split, Subset # pylint: disable=import-error

# Local imports
# pylint: disable=import-error
from src.data.dataset_factory import get_dataset

class BaseDataModule:
    """
    Dataset-agnostic data module.

    - If get_dataset(dataset_name, split=...) exists, we use provided splits.
    - Otherwise we create train/val via random_split from a single 'train' split.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        cfg: Any,
        dataset_name: str,
        data_dir: str,
        *,
        num_workers: int = 8,
        batch_size: int = 32,
        pin_memory: bool = True,
        drop_last: bool = False,
        split_seed: int = 42,
        preprocessor: Any = None,
        resolution: int = 224,
        transform: Any = None,
        model_type: str = "vit",
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        **kwargs, 
    ): 
        """
        Initialize the DataModule.
        """
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.split_seed = split_seed

        # Plumb-through for dataset construction
        self.preprocessor = preprocessor
        self.resolution = resolution
        self.transform = transform
        self.model_type = model_type

        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
        if kwargs:
            print(f"Warning: Unused config keys passed to DataModule: {list(kwargs.keys())}")

        self.train_set = None
        self.val_set = None
        self.test_set = None

    # ------------------------------------------------------------------
    def setup(self, _stage: Optional[str] = None):
        """
        Initialize datasets. _stage: 'fit' | 'validate' | 'test' | None
        Tries explicit splits first; falls back to random_split from a 'train' split.

        Args:
            _stage: Current stage of training pipeline (unused but kept for API compatibility)
        """
        # Try to fetch explicit train/val
        ds_train = get_dataset(
            self.dataset_name, self.data_dir, split="train", cfg=self.cfg,
            preprocessor=self.preprocessor, resolution=self.resolution,
            transform=self.transform, model_type=self.model_type,
        )

        try:
            ds_val = get_dataset(
                self.dataset_name, self.data_dir, split="val", cfg=self.cfg,
                preprocessor=self.preprocessor, resolution=self.resolution,
                transform=None, model_type=self.model_type,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught, unused-variable
            # Exception is broad to handle any dataset-specific errors
            ds_val = None

        try:
            ds_test = get_dataset(
                self.dataset_name, self.data_dir, split="test", cfg=self.cfg,
                preprocessor=self.preprocessor, resolution=self.resolution,
                transform=None, model_type=self.model_type,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught, unused-variable
            # Exception is broad to handle any dataset-specific errors
            ds_test = None

        if ds_val is None:
            # Fallback: split train into train/val
            n_total = len(ds_train)
            n_val = max(1, int(0.1 * n_total))
            n_train = n_total - n_val
            g = torch.Generator().manual_seed(self.split_seed)
            self.train_set, self.val_set = random_split(ds_train, [n_train, n_val], generator=g)
        else:
            self.train_set, self.val_set = ds_train, ds_val

        self.test_set = ds_test

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers, 
            prefetch_factor=self.prefetch_factor        
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor        
        )

    def test_dataloader(self):
        if self.test_set is None:
            return DataLoader(Subset(self.val_set, []), batch_size=self.batch_size)
            
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, 
            prefetch_factor=self.prefetch_factor       
        )