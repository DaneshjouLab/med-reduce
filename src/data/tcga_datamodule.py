# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
TCGA classification data module for two-stage linear probing.

Provides TCGASlideDataset (plain PyTorch Dataset) and TCGADataModule
(subclasses BaseDataModule) for slide-level classification tasks using
thumbnails extracted by the TCGA ETL pipeline.

Supported tasks:
  - luad_vs_lusc: Lung adenocarcinoma vs squamous cell carcinoma
  - lgg_vs_gbm:   Low-grade glioma vs glioblastoma
  - kras, tp53, egfr, idh: Gene mutation binary classification
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from src.data.datamodule import BaseDataModule
from src.utils.split_manager import SplitManager
from src.utils.logging_core import get_logger

log = get_logger(__name__)

VAL_SPLIT_RATIO = 0.1

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
TASK_CONFIGS = {
    "luad_vs_lusc": {
        "filter_col": "project_id",
        "filter_values": ["TCGA-LUAD", "TCGA-LUSC"],
        "label_source": "project_id",
        "label_map": {"TCGA-LUAD": 0, "TCGA-LUSC": 1},
        "num_classes": 2,
        "require_maf": False,
    },
    "lgg_vs_gbm": {
        "filter_col": "project_id",
        "filter_values": ["TCGA-LGG", "TCGA-GBM"],
        "label_source": "project_id",
        "label_map": {"TCGA-LGG": 0, "TCGA-GBM": 1},
        "num_classes": 2,
        "require_maf": False,
    },
    "kras": {
        "label_source": "KRAS",
        "num_classes": 2,
        "require_maf": True,
    },
    "tp53": {
        "label_source": "TP53",
        "num_classes": 2,
        "require_maf": True,
    },
    "egfr": {
        "label_source": "EGFR",
        "num_classes": 2,
        "require_maf": True,
    },
    "idh": {
        "label_source": "IDH",
        "num_classes": 2,
        "require_maf": True,
    },
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TCGASlideDataset(Dataset):
    """PyTorch Dataset backed by a pandas DataFrame of TCGA slide metadata."""

    def __init__(self, df: pd.DataFrame, thumbnails_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.transform = transform
        self.labels = self.df["label"].values  # numpy array for fast stratification

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.thumbnails_dir / f"{row['slide_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "label": int(row["label"]), "image_id": str(row["slide_id"])}


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class TCGADataModule(BaseDataModule):
    """
    TCGA slide classification data module with persistent splits.

    Subclasses BaseDataModule directly. Uses SplitManager for reproducible
    stratified train/test splits. Compatible with ProbeTwoStageWrapper.
    """

    def __init__(
        self,
        task: str,
        dataset_csv: str,
        data_dir: str,
        split_dir: str = "./splits",
        split_seed: int = 42,
        use_val_split: bool = False,
        train_ratio: float = 0.8,
        force_recompute: bool = False,
        *,
        full_cfg: Any = None,
        num_workers: int = 8,
        batch_size: int = 64,
        pin_memory: bool = True,
        **kwargs,
    ):
        if task not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS.keys())}"
            )

        self.task = task
        self.task_cfg = TASK_CONFIGS[task]
        self.thumbnails_dir = Path(data_dir)
        self.use_val_split = use_val_split
        self.train_ratio = train_ratio
        self.force_recompute = force_recompute
        self.full_cfg = full_cfg

        # --- Read and filter CSV -------------------------------------------
        log.info(f"Reading dataset CSV: {dataset_csv}")
        df = pd.read_csv(dataset_csv)
        log.info(f"  Raw rows: {len(df)}")

        # Require jpg_path present
        df = df[df["jpg_path"].notna()].copy()
        log.info(f"  After jpg_path filter: {len(df)}")

        # Task-specific filter
        if self.task_cfg.get("require_maf"):
            df = df[df["has_maf"] == True].copy()  # noqa: E712
            log.info(f"  After has_maf filter: {len(df)}")

        if "filter_col" in self.task_cfg:
            col = self.task_cfg["filter_col"]
            vals = self.task_cfg["filter_values"]
            df = df[df[col].isin(vals)].copy()
            log.info(f"  After {col} filter ({vals}): {len(df)}")

        # Create integer label column
        if "label_map" in self.task_cfg:
            df["label"] = df[self.task_cfg["label_source"]].map(self.task_cfg["label_map"])
        else:
            df["label"] = df[self.task_cfg["label_source"]].astype(int)

        self.df = df.reset_index(drop=True)
        log.info(f"  Final dataset size: {len(self.df)}")
        log.info(f"  Label distribution: {self.df['label'].value_counts().to_dict()}")

        # --- Call parent init -----------------------------------------------
        super().__init__(
            cfg=full_cfg,
            data_dir=data_dir,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            split_seed=split_seed,
            **kwargs,
        )

        # --- SplitManager + identifier -------------------------------------
        self.dataset_identifier = f"tcga_{task}"
        self.split_manager = SplitManager(
            split_dir=split_dir,
            dataset_name=self.dataset_identifier,
            seed=split_seed,
        )

    # -----------------------------------------------------------------------
    # Properties expected by the wrapper / BaseDataModule
    # -----------------------------------------------------------------------
    @property
    def image_size(self) -> int:
        """Dynamically read image_size from cfg.data.image_size."""
        if (
            self.full_cfg is not None
            and hasattr(self.full_cfg, "data")
            and hasattr(self.full_cfg.data, "image_size")
        ):
            return int(self.full_cfg.data.image_size)
        return self._fallback_resolution

    @property
    def native_resolution(self) -> int | None:
        """Read native_resolution from cfg.data.native_resolution.

        When set, images are downsampled to image_size then upsampled back to
        native_resolution before being fed to the encoder.
        """
        if (
            self.full_cfg
            and hasattr(self.full_cfg, "data")
            and hasattr(self.full_cfg.data, "native_resolution")
        ):
            return int(self.full_cfg.data.native_resolution)
        return None

    # -----------------------------------------------------------------------
    # Stratification helper
    # -----------------------------------------------------------------------
    def get_labels_for_stratification(self, dataset) -> np.ndarray | None:
        """Return numpy array of integer labels for stratified splitting."""
        if hasattr(dataset, "labels"):
            return dataset.labels
        return None

    # -----------------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------------
    def setup(self, _stage: Optional[str] = None):
        """Initialize datasets with persistent splits."""
        log.info(f"\n{'='*60}")
        log.info(f"Setting up TCGADataModule")
        log.info(f"  Task: {self.task}")
        log.info(f"  Dataset size: {len(self.df)}")
        log.info(f"  Image size: {self.image_size}")
        if self.native_resolution and self.native_resolution != self.image_size:
            log.info(f"  Native resolution: {self.native_resolution} (downsample→upsample degradation)")
        log.info(f"  Split dir: {self.split_manager.dataset_dir}")
        log.info(f"{'='*60}\n")

        # --- Transforms -----------------------------------------------------
        resize_steps = [transforms.Resize((self.image_size, self.image_size))]
        if self.native_resolution and self.native_resolution != self.image_size:
            resize_steps.append(transforms.Resize((self.native_resolution, self.native_resolution)))

        base_transform = transforms.Compose([
            *resize_steps,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        # For linear probing we use the same transform for train/val/test
        train_transform = base_transform

        # --- Full dataset ---------------------------------------------------
        full_dataset = TCGASlideDataset(
            self.df, self.thumbnails_dir, transform=train_transform
        )
        self.full_dataset = full_dataset

        dataset_size = len(full_dataset)
        stratify_labels = self.get_labels_for_stratification(full_dataset)

        # --- Clear splits if forced -----------------------------------------
        if self.force_recompute and self.split_manager.exists():
            log.info("force_recompute=True: Clearing existing splits...")
            self.split_manager.clear_splits()
            self.split_manager.dataset_dir.mkdir(parents=True, exist_ok=True)

        # --- Create or load splits ------------------------------------------
        if not self.split_manager.exists():
            log.info("Creating new persistent splits...")
            splits = self.split_manager.create_splits(
                dataset_size=dataset_size,
                use_val_split=self.use_val_split,
                train_ratio=self.train_ratio,
                stratify_labels=stratify_labels,
            )
        else:
            log.info("Loading existing persistent splits...")
            splits = self.split_manager.load_splits()

        # --- Assign subsets --------------------------------------------------
        self.train_set = Subset(full_dataset, splits["train"])
        self.test_set = Subset(full_dataset, splits["test"]) if "test" in splits else None

        if "val" in splits:
            self.val_set = Subset(full_dataset, splits["val"])
        else:
            # Carve validation from training set
            log.info("Creating validation split from training set...")
            train_indices = splits["train"]
            n_val = max(1, int(len(train_indices) * VAL_SPLIT_RATIO))

            np.random.seed(self.split_seed)
            val_indices = np.random.choice(train_indices, n_val, replace=False)
            train_remaining = np.setdiff1d(train_indices, val_indices)

            self.val_set = Subset(full_dataset, val_indices)
            self.train_set = Subset(full_dataset, train_remaining)

            log.info(f"  Split: {len(train_remaining)} train, {n_val} val")

        log.info(f"\n{'='*60}")
        log.info(f"TCGADataModule setup complete!")
        log.info(f"  Train: {len(self.train_set)} samples")
        log.info(f"  Val:   {len(self.val_set)} samples")
        if self.test_set:
            log.info(f"  Test:  {len(self.test_set)} samples")
        log.info(f"{'='*60}\n")
