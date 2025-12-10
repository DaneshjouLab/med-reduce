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
from torch.utils.data import DataLoader, Subset # pylint: disable=import-error

class BaseDataModule:
    """
    Base class for data modules providing common dataloader interface.

    Subclasses must implement the setup() method to load their specific datasets.
    Examples: ISICDataModule, ISICSegDataModule, etc.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        cfg: Any = None,  # Made optional to support Hydra instantiation
        dataset_name: str = None,
        data_dir: str = None,
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
        Initialize datasets. Must be overridden by subclasses.

        Args:
            _stage: Current stage of training pipeline (unused but kept for API compatibility)

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement setup() method. "
            "Use a specific datamodule like ISICDataModule instead of BaseDataModule directly."
        )

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