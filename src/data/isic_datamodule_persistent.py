# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""
ISIC DataModule with persistent splits.

train/val/test splits are:
1. Created once and stored to disk
2. Reused across all experiments and resolutions
3. Consistent across different models

Splits stored in ./splits/isic/
"""

from typing import Optional, Any, List
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.datamodule import BaseDataModule
from src.data.dataset_factory import balance_dataset
from src.data.isic_loader import ISICHFRawSplit, ISICHFRawSplitLocal
from src.transformations.transforms import SegmentationTransform, FeatureDetectionTransform
from src.utils.split_manager import SplitManager
from src.utils.logging_core import get_logger

log = get_logger(__name__)

# --- Constants
VAL_SPLIT_RATIO = 0.1
DEFAULT_REPO_ID = "MKZuziak/ISIC_2019_224"


class ISICDataModulePersistent(BaseDataModule):
    """
    ISIC DataModule with persistent split management.
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
        image_size: int = 224,
        local_label_file: Optional[str] = None,
        local_image_id_column: str = "image_id",
        local_label_column: Any = "label",
        local_image_extension: str = ".jpg",
        split_dir: str = "./splits",
        use_val_split: bool = False,  # If True, create train/val/test. Otherwise train/test.
        train_ratio: float = 0.8,
        force_recompute_embeddings: bool = False,  # Force recomputation of embeddings even if cached
        **kwargs
    ):
        self.full_cfg = full_cfg
        # Store fallback but prefer cfg.data.image_size at runtime
        self._fallback_image_size = image_size

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

        self.use_val_split = use_val_split
        self.train_ratio = train_ratio
        self.force_recompute_embeddings = force_recompute_embeddings

        if dataset_name and not os.path.isdir(str(dataset_name)):
            self.data_source = "remote_hf"
            self.source_id = dataset_name
            self.dataset_identifier = dataset_name.replace("/", "_")  # For split folder naming
        elif data_dir and os.path.isdir(str(data_dir)):
            self.data_source = "local"
            self.source_id = data_dir
            self.dataset_identifier = os.path.basename(data_dir)
            if local_label_file is None:
                raise ValueError(
                    "When using local data (data_dir is a directory), "
                    "you must specify 'local_label_file' parameter."
                )
        else:
            raise ValueError(
                "Must provide valid `dataset_name` (HF Hub) or `data_dir` (local directory)."
            )

        self.local_label_file = local_label_file
        self.local_image_id_column = local_image_id_column
        self.local_label_column = local_label_column
        self.local_image_extension = local_image_extension

        # Initialize split manager
        self.split_manager = SplitManager(
            split_dir=split_dir,
            dataset_name=self.dataset_identifier,
            seed=split_seed,
        )

    def _load_full_dataset(
        self,
        split: str,
        transform: Optional[Any] = None,
    ):
        """Load the full dataset (before applying split indices)."""
        log.info(f"Loading dataset from: {self.data_source}")

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
                label_file=self.local_label_file,
                image_id_column=self.local_image_id_column,
                label_column=self.local_label_column,
                image_extension=self.local_image_extension,
                transform=transform,
                filter_fn=self.filter_fn,
                keep_indices=self.keep_indices,
            )
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        return wrapper

    def get_labels_for_stratification(self, dataset) -> Optional[np.ndarray]:
        """Extract labels from dataset for stratified splitting."""
        try:
            if hasattr(dataset, 'ds'):
                # Wrapped dataset
                inner_ds = dataset.ds
                if hasattr(inner_ds, 'targets'):
                    return np.array(inner_ds.targets)
                elif hasattr(inner_ds, 'labels'):
                    return np.array(inner_ds.labels)
                elif hasattr(inner_ds, '__getitem__'):
                    # Try to extract labels by sampling
                    labels = []
                    for i in range(len(inner_ds)):
                        try:
                            item = inner_ds[i]
                            if isinstance(item, (tuple, list)) and len(item) >= 2:
                                labels.append(item[1])
                            elif isinstance(item, dict) and 'label' in item:
                                labels.append(item['label'])
                        except:
                            break
                    if labels:
                        return np.array(labels)
            elif hasattr(dataset, 'targets'):
                return np.array(dataset.targets)
            elif hasattr(dataset, 'labels'):
                return np.array(dataset.labels)
        except Exception as e:
            log.warning(f"Could not extract labels for stratification: {e}")

        return None

    @property
    def image_size(self) -> int:
        """
        Dynamically read image_size from cfg.data.image_size.

        This ensures consistency - all image sizes come from cfg.data.image_size.

        Returns:
            Current image_size from cfg.data.image_size, or fallback value
        """
        if self.full_cfg is not None and hasattr(self.full_cfg, 'data') and hasattr(self.full_cfg.data, 'image_size'):
            return int(self.full_cfg.data.image_size)
        return self._fallback_image_size

    def setup(self, _stage: Optional[str] = None):
        """Initialize datasets with persistent splits."""
        log.info(f"\n{'='*60}")
        log.info(f"Setting up ISIC DataModule (Persistent Splits)")
        log.info(f"  Dataset: {self.repo_id}")
        log.info(f"  Split directory: {self.split_manager.dataset_dir}")
        if self.balance_data:
            log.info(f"  Filtered classes: {self.filtered_classes}")
            log.info(f"  Balance data: {self.balance_data}")
            if self.num_train_images:
                log.info(f"  Num train images: {self.num_train_images}")
        log.info(f"  Image size: {self.image_size}")
        if self.force_recompute_embeddings:
            log.info(f"  Force recompute: {self.force_recompute_embeddings} (will regenerate splits & embeddings)")
        log.info(f"{'='*60}\n")

        val_test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        if self.transform is not None:
            train_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                self.transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = val_test_transform

        log.info("Loading full dataset...")
        full_dataset = self._load_full_dataset(
            split="train",
            transform=train_transform,
        )

        if self.balance_data:
            log.info(f"  Balancing dataset...")
            full_dataset.ds = balance_dataset(
                dataset=full_dataset.ds,
                filtered_classes=self.filtered_classes,
                num_train_images=self.num_train_images or len(full_dataset.ds),
                seed=self.split_seed,
            )
            log.info(f"  → Balanced to {len(full_dataset.ds)} samples")

        dataset_size = len(full_dataset)
        log.info(f"  ✓ Full dataset: {dataset_size} samples\n")

        self.full_dataset = full_dataset

        stratify_labels = self.get_labels_for_stratification(full_dataset)

        # Clear existing splits if force_recompute_embeddings is enabled
        if self.force_recompute_embeddings and self.split_manager.exists():
            log.info("🔄 force_recompute_embeddings=True: Clearing existing splits...")
            self.split_manager.clear_splits()
            self.split_manager.dataset_dir.mkdir(parents=True, exist_ok=True)

        if not self.split_manager.exists():
            log.info("📊 Creating new persistent splits...")
            splits = self.split_manager.create_splits(
                dataset_size=dataset_size,
                use_val_split=self.use_val_split,
                train_ratio=self.train_ratio,
                stratify_labels=stratify_labels,
            )
        else:
            log.info("📥 Loading existing persistent splits...")
            splits = self.split_manager.load_splits()

            total_split_samples = sum(len(indices) for indices in splits.values())
            if total_split_samples != dataset_size:
                log.warning(
                    f"⚠️  Split size mismatch! "
                    f"Expected {dataset_size} samples, but splits contain {total_split_samples}. "
                    f"This may be due to different balancing settings. "
                    f"Set force_recompute_embeddings=true to regenerate splits."
                )

        self.train_set = Subset(full_dataset, splits["train"])
        self.test_set = Subset(full_dataset, splits["test"]) if "test" in splits else None

        if "val" in splits:
            self.val_set = Subset(full_dataset, splits["val"])
        else:
            log.info("Creating validation split from training set...")
            train_indices = splits["train"]
            n_val = max(1, int(len(train_indices) * VAL_SPLIT_RATIO))

            np.random.seed(self.split_seed)
            val_indices_from_train = np.random.choice(train_indices, n_val, replace=False)
            train_indices_remaining = np.setdiff1d(train_indices, val_indices_from_train)

            self.val_set = Subset(full_dataset, val_indices_from_train)
            self.train_set = Subset(full_dataset, train_indices_remaining)

            log.info(f"  ✓ Split: {len(train_indices_remaining)} train, {n_val} val\n")

        log.info(f"{'='*60}")
        log.info(f"DataModule setup complete!")
        log.info(f"  Train: {len(self.train_set)} samples")
        log.info(f"  Val: {len(self.val_set)} samples")
        if self.test_set:
            log.info(f"  Test: {len(self.test_set)} samples")
        log.info(f"{'='*60}\n")


class ISICSegDataModulePersistent(BaseDataModule):
    """
    ISIC Segmentation DataModule with persistent splits.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
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
        image_size: int = 256,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        image_extension: str = ".jpg",
        mask_extension: str = ".png",
        mask_suffix: str = "_segmentation",
        split_dir: str = "./splits",
        use_val_split: bool = False,
        train_ratio: float = 0.8,
        force_recompute_embeddings: bool = False,  # Force recomputation of embeddings even if cached
        **kwargs
    ):
        self.full_cfg = full_cfg
        # Store fallback but prefer cfg.data.image_size at runtime
        self._fallback_image_size = image_size

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

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.mask_suffix = mask_suffix

        self.use_val_split = use_val_split
        self.train_ratio = train_ratio
        self.force_recompute_embeddings = force_recompute_embeddings

        if image_dir and mask_dir:
            self.data_source = "local_dirs"
            self.dataset_identifier = f"isic_seg_{os.path.basename(image_dir)}"
            if not os.path.isdir(image_dir):
                raise ValueError(f"image_dir must be a valid directory: {image_dir}")
            if not os.path.isdir(mask_dir):
                raise ValueError(f"mask_dir must be a valid directory: {mask_dir}")
        elif dataset_name:
            self.data_source = "remote_hf"
            self.dataset_identifier = dataset_name.replace("/", "_")
        else:
            raise ValueError(
                "Must provide either:\n"
                "  1. image_dir + mask_dir (for directory-based loading), or\n"
                "  2. data_dir + label_file (for CSV-based loading), or\n"
                "  3. dataset_name (for HuggingFace Hub)"
            )

        self.split_manager = SplitManager(
            split_dir=split_dir,
            dataset_name=self.dataset_identifier,
            seed=split_seed,
        )

    def _load_full_dataset(self, transform: Optional[Any] = None):
        """Load full segmentation dataset."""
        from src.data.isic_loader import ISICSegRawSplitLocal

        if self.data_source == "local_dirs":
            wrapper = ISICSegRawSplitLocal(
                image_dir=self.image_dir,
                mask_dir=self.mask_dir,
                image_extension=self.image_extension,
                mask_extension=self.mask_extension,
                mask_suffix=self.mask_suffix,
                transform=transform,
                filter_fn=self.filter_fn,
                keep_indices=self.keep_indices,
            )
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        return wrapper

    @property
    def image_size(self) -> int:
        """
        Dynamically read image_size from cfg.data.image_size.

        This ensures consistency - all image sizes come from cfg.data.image_size.

        Returns:
            Current image_size from cfg.data.image_size, or fallback value
        """
        if self.full_cfg is not None and hasattr(self.full_cfg, 'data') and hasattr(self.full_cfg.data, 'image_size'):
            return int(self.full_cfg.data.image_size)
        return self._fallback_image_size

    def setup(self, _stage: Optional[str] = None):
        """Initialize segmentation datasets with persistent splits."""
        log.info(f"\n{'='*60}")
        log.info(f"Setting up ISIC Segmentation DataModule (Persistent Splits)")
        log.info(f"  Data source: {self.data_source}")
        log.info(f"  Split directory: {self.split_manager.dataset_dir}")
        log.info(f"  Image size: {self.image_size}")
        log.info(f"{'='*60}\n")

        val_test_transform = SegmentationTransform(target_size=self.image_size)
        train_transform = self.transform if self.transform is not None else val_test_transform

        log.info("Loading full dataset...")
        full_dataset = self._load_full_dataset(transform=train_transform)
        dataset_size = len(full_dataset)
        log.info(f"  ✓ Full dataset: {dataset_size} samples\n")

        # Clear existing splits if force_recompute_embeddings is enabled
        if self.force_recompute_embeddings and self.split_manager.exists():
            log.info("🔄 force_recompute_embeddings=True: Clearing existing splits...")
            self.split_manager.clear_splits()
            self.split_manager.dataset_dir.mkdir(parents=True, exist_ok=True)

        if not self.split_manager.exists():
            log.info("📊 Creating new persistent splits...")
            splits = self.split_manager.create_splits(
                dataset_size=dataset_size,
                use_val_split=self.use_val_split,
                train_ratio=self.train_ratio,
                stratify_labels=None,
            )
        else:
            log.info("📥 Loading existing persistent splits...")
            splits = self.split_manager.load_splits()

        self.train_set = Subset(full_dataset, splits["train"])
        self.test_set = Subset(full_dataset, splits["test"]) if "test" in splits else None

        if "val" in splits:
            self.val_set = Subset(full_dataset, splits["val"])
        else:
            train_indices = splits["train"]
            n_val = max(1, int(len(train_indices) * VAL_SPLIT_RATIO))

            np.random.seed(self.split_seed)
            val_indices_from_train = np.random.choice(train_indices, n_val, replace=False)
            train_indices_remaining = np.setdiff1d(train_indices, val_indices_from_train)

            self.val_set = Subset(full_dataset, val_indices_from_train)
            self.train_set = Subset(full_dataset, train_indices_remaining)

        log.info(f"{'='*60}")
        log.info(f"DataModule setup complete!")
        log.info(f"  Train: {len(self.train_set)} samples")
        log.info(f"  Val: {len(self.val_set)} samples")
        if self.test_set:
            log.info(f"  Test: {len(self.test_set)} samples")
        log.info(f"{'='*60}\n")
