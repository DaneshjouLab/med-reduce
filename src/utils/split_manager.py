# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""
Split management for consistent train/val/test splits across experiments.

Directory structure:
  splits/
    {dataset_name}/
      train_indices.npy
      val_indices.npy    
      test_indices.npy
      cv_folds.json      (5-fold CV indices for hyperparameter tuning)
      metadata.json
"""
from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from src.utils.logging_core import get_logger

log = get_logger(__name__)


class SplitManager:
    """
    Manages dataset splits and ensures consistency across experiments.

    Supports two modes:
    1. Train/Test split (80/20)
    2. Train/Val/Test split (70/10/20)

    For hyperparameter tuning, stores 5-fold CV indices on the training set.
    """

    def __init__(
        self,
        split_dir: str,
        dataset_name: str,
        seed: int = 42,
    ):
        self.split_dir = Path(split_dir)
        self.dataset_name = dataset_name
        self.seed = seed

        self.dataset_dir = self.split_dir / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def _get_split_path(self, split: str) -> Path:
        return self.dataset_dir / f"{split}_indices.npy"

    def _get_cv_path(self) -> Path:
        return self.dataset_dir / "cv_folds.json"

    def _get_metadata_path(self) -> Path:
        return self.dataset_dir / "metadata.json"

    def exists(self) -> bool:
        train_path = self._get_split_path("train")
        test_path = self._get_split_path("test")
        metadata_path = self._get_metadata_path()

        return train_path.exists() and test_path.exists() and metadata_path.exists()

    def create_splits(
        self,
        dataset_size: int,
        use_val_split: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        stratify_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            dataset_size: Total number of samples in dataset
            use_val_split: If True, create train/val/test. Otherwise train/test.
            train_ratio: Proportion for training (default 0.8 for 80/20 split)
            val_ratio: Proportion for validation (only used if use_val_split=True)
            stratify_labels: Optional labels for stratified splitting

        Returns:
            Dict with 'train', 'test', and optionally 'val' indices
        """
        if self.exists():
            log.warning(f"⚠️  Splits already exist for {self.dataset_name}. Loading existing splits.")
            return self.load_splits()

        log.info(f"📊 Creating splits for {self.dataset_name} ({dataset_size} samples)")

        indices = np.arange(dataset_size)

        if use_val_split:
            test_ratio = 1.0 - train_ratio - val_ratio

            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=test_ratio,
                random_state=self.seed,
                stratify=stratify_labels if stratify_labels is not None else None,
            )

            val_size = val_ratio / (train_ratio + val_ratio)
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_size,
                random_state=self.seed,
                stratify=stratify_labels[train_val_indices] if stratify_labels is not None else None,
            )

            splits = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }

            log.info(f"  Train: {len(train_indices)} ({len(train_indices)/dataset_size*100:.1f}%)")
            log.info(f"  Val:   {len(val_indices)} ({len(val_indices)/dataset_size*100:.1f}%)")
            log.info(f"  Test:  {len(test_indices)} ({len(test_indices)/dataset_size*100:.1f}%)")

        else:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=(1.0 - train_ratio),
                random_state=self.seed,
                stratify=stratify_labels if stratify_labels is not None else None,
            )

            splits = {
                "train": train_indices,
                "test": test_indices,
            }

            log.info(f"  Train: {len(train_indices)} ({len(train_indices)/dataset_size*100:.1f}%)")
            log.info(f"  Test:  {len(test_indices)} ({len(test_indices)/dataset_size*100:.1f}%)")

        for split_name, split_indices in splits.items():
            split_path = self._get_split_path(split_name)
            np.save(split_path, split_indices)
            log.info(f"💾 Saved {split_name} indices to {split_path}")

        metadata = {
            "dataset_name": self.dataset_name,
            "dataset_size": dataset_size,
            "seed": self.seed,
            "use_val_split": use_val_split,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio if use_val_split else None,
            "stratified": stratify_labels is not None,
            "split_sizes": {k: len(v) for k, v in splits.items()},
        }

        metadata_path = self._get_metadata_path()
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"✓ Split creation complete for {self.dataset_name}")

        return splits

    def load_splits(self) -> Dict[str, np.ndarray]:
        if not self.exists():
            raise FileNotFoundError(
                f"Splits not found for {self.dataset_name}. "
                f"Run create_splits() first."
            )

        splits = {}

        splits["train"] = np.load(self._get_split_path("train"))
        splits["test"] = np.load(self._get_split_path("test"))

        val_path = self._get_split_path("val")
        if val_path.exists():
            splits["val"] = np.load(val_path)

        log.info(f"📥 Loaded splits for {self.dataset_name}")
        for split_name, split_indices in splits.items():
            log.info(f"  {split_name}: {len(split_indices)} samples")

        return splits

    def create_cv_folds(
        self,
        train_indices: np.ndarray,
        n_folds: int = 5,
        stratify_labels: Optional[np.ndarray] = None,
        force_recompute: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create and save 5-fold CV splits for hyperparameter tuning.

        Args:
            train_indices: Indices of training samples
            n_folds: Number of folds (default 5)
            stratify_labels: Optional labels for stratified folding
            force_recompute: If True, regenerate folds even if cached

        Returns:
            List of (train_fold_indices, val_fold_indices) tuples
        """
        cv_path = self._get_cv_path()

        if cv_path.exists() and not force_recompute:
            log.info(f"📥 Loading existing CV folds from {cv_path}")
            return self.load_cv_folds()

        log.info(f"📊 Creating {n_folds}-fold CV splits on training set ({len(train_indices)} samples)")

        if stratify_labels is not None:
            from sklearn.model_selection import StratifiedKFold
            train_labels = stratify_labels[train_indices]
            unique, counts = np.unique(train_labels, return_counts=True)
            log.info(f"  Using StratifiedKFold with label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            splits = list(kf.split(train_indices, train_labels))
        else:
            log.warning("  ⚠️ stratify_labels is None - using regular KFold (not stratified!)")
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            splits = list(kf.split(train_indices))

        folds = []
        for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(splits):
            train_fold = train_indices[train_fold_idx]
            val_fold = train_indices[val_fold_idx]
            folds.append((train_fold, val_fold))

            # Log class distribution in each fold's validation set
            if stratify_labels is not None:
                val_labels = stratify_labels[val_fold]
                unique, counts = np.unique(val_labels, return_counts=True)
                log.info(f"  Fold {fold_idx + 1} val set: {len(val_fold)} samples, classes: {dict(zip(unique.tolist(), counts.tolist()))}")

        cv_data = {
            "n_folds": n_folds,
            "seed": self.seed,
            "folds": [
                {
                    "fold": i + 1,
                    "train": train_fold.tolist(),
                    "val": val_fold.tolist(),
                }
                for i, (train_fold, val_fold) in enumerate(folds)
            ],
        }

        with open(cv_path, "w") as f:
            json.dump(cv_data, f, indent=2)

        log.info(f"💾 Saved {n_folds}-fold CV splits to {cv_path}")

        return folds

    def load_cv_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns:
            List of (train_fold_indices, val_fold_indices) tuples
        """
        cv_path = self._get_cv_path()

        if not cv_path.exists():
            raise FileNotFoundError(
                f"CV folds not found at {cv_path}. "
                f"Run create_cv_folds() first."
            )

        with open(cv_path, "r") as f:
            cv_data = json.load(f)

        folds = []
        for fold_data in cv_data["folds"]:
            train_fold = np.array(fold_data["train"])
            val_fold = np.array(fold_data["val"])
            folds.append((train_fold, val_fold))

        log.info(f"📥 Loaded {len(folds)}-fold CV splits")

        return folds

    def get_metadata(self) -> Dict[str, Any]:
        """Load split metadata."""
        metadata_path = self._get_metadata_path()

        if not metadata_path.exists():
            return {}

        with open(metadata_path, "r") as f:
            return json.load(f)

    def clear_splits(self):
        """Clear all splits for this dataset."""
        if self.dataset_dir.exists():
            import shutil
            shutil.rmtree(self.dataset_dir)
            log.info(f"🗑️  Cleared all splits for {self.dataset_name}")
