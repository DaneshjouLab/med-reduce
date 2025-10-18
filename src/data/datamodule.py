# src/data/datamodule.py
# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, random_split
from typing import Optional
from .datasets import get_dataset


class BaseDataModule:
    """
    Dataset-agnostic data module.

    Responsibilities:
    - Instantiate train/val/test datasets using `get_dataset()`.
    - Build and cache dataloaders.
    - Apply consistent transforms and batching options across datasets.
    """

    def __init__(
        self,
        cfg,
        dataset_name: str,
        data_dir: str,
        num_workers: int = 8,
        batch_size: int = 32,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train_set = None
        self.val_set = None
        self.test_set = None

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        """
        Called once to initialize datasets.
        stage: 'fit' | 'validate' | 'test' | None
        """
        ds_full = get_dataset(self.dataset_name, self.data_dir, split="train", cfg=self.cfg)
        n_total = len(ds_full)
        n_val = int(0.1 * n_total)
        n_train = n_total - n_val
        self.train_set, self.val_set = random_split(ds_full, [n_train, n_val])

        # test set
        self.test_set = get_dataset(self.dataset_name, self.data_dir, split="test", cfg=self.cfg)

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
