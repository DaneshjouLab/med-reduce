# isic_datamodule.py

from typing import Optional, Any, List
import os
from torch.utils.data import DataLoader, random_split, Subset
import torch
from torchvision import transforms

from src.data.datamodule import BaseDataModule
from src.data.dataset_factory import balance_dataset
from src.data.isic_loader import ISICHFRawSplit, ISICHFRawSplitLocal
from src.transformations.transforms import SegmentationTransform, FeatureDetectionTransform

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
        image_size: int = 224,  # Target image size for the model
        # Local dataset parameters (only used when data_dir points to local files)
        local_label_file: Optional[str] = None,
        local_image_id_column: str = "image_id",
        local_label_column: Any = "label",  # Can be str or list of str
        local_image_extension: str = ".jpg",
        **kwargs # Catch-all for any other parameters from BaseDataModule
    ):
        self.full_cfg = full_cfg
        self.image_size = image_size

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

        # Determine data source
        if dataset_name and not os.path.isdir(str(dataset_name)):
            self.data_source = "remote_hf"
            self.source_id = dataset_name
        elif data_dir and os.path.isdir(str(data_dir)):
            self.data_source = "local"
            self.source_id = data_dir
            # For local data, label_file is required
            if local_label_file is None:
                raise ValueError(
                    "When using local data (data_dir is a directory), "
                    "you must specify 'local_label_file' parameter."
                )
        else:
            raise ValueError(
                "Must provide valid `dataset_name` (HF Hub) or `data_dir` (local directory)."
            )

        # Store local dataset parameters
        self.local_label_file = local_label_file
        self.local_image_id_column = local_image_id_column
        self.local_label_column = local_label_column
        self.local_image_extension = local_image_extension

        if self.data_source == "remote_hf":
            self.dataset_identifier = dataset_name.replace("/", "_")
        else:
            self.dataset_identifier = os.path.basename(data_dir)

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
        print(f"  Image size: {self.image_size}")
        print(f"{'='*60}\n")

        # Create transforms with proper resizing
        val_test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # If custom transform provided, add resize + normalize after it
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
    """
    DataModule for ISIC segmentation tasks.

    Supports both:
    1. HF dataset with CSV (repo_id + label_file)
    2. Local directories (image_dir + mask_dir)
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
        # For CSV-based loading
        label_file: Optional[str] = None,
        image_column: str = "image_path",
        mask_column: str = "mask_path",
        # For directory-based loading
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        image_extension: str = ".jpg",
        mask_extension: str = ".png",
        mask_suffix: str = "_segmentation",
        **kwargs
    ):
        self.full_cfg = full_cfg
        self.image_size = image_size

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

        # Store segmentation-specific parameters
        self.label_file = label_file
        self.image_column = image_column
        self.mask_column = mask_column
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.mask_suffix = mask_suffix

        # Determine data source
        if image_dir and mask_dir:
            # Directory-based loading
            self.data_source = "local_dirs"
            if not os.path.isdir(image_dir):
                raise ValueError(f"image_dir must be a valid directory: {image_dir}")
            if not os.path.isdir(mask_dir):
                raise ValueError(f"mask_dir must be a valid directory: {mask_dir}")
        elif data_dir and label_file:
            # CSV-based loading
            self.data_source = "csv"
            if not os.path.isdir(data_dir):
                raise ValueError(f"data_dir must be a valid directory: {data_dir}")
        elif dataset_name:
            # Remote HF dataset
            self.data_source = "remote_hf"
        else:
            raise ValueError(
                "Must provide either:\n"
                "  1. image_dir + mask_dir (for directory-based loading), or\n"
                "  2. data_dir + label_file (for CSV-based loading), or\n"
                "  3. dataset_name (for HuggingFace Hub)"
            )

    def _load_split(
        self,
        split: str = "train",
        transform: Optional[Any] = None,
        balance: bool = False
    ):
        """Load a segmentation dataset split."""
        from src.data.isic_loader import ISICSegRawSplit, ISICSegRawSplitLocal

        print(f"Loading {split} split from: {self.data_source}")

        if self.data_source == "local_dirs":
            # Directory-based loading
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
        elif self.data_source == "csv":
            # CSV-based loading
            wrapper = ISICSegRawSplit(
                data_dir=self.data_dir,
                label_file=self.label_file,
                image_column=self.image_column,
                mask_column=self.mask_column,
                transform=transform,
            )
        elif self.data_source == "remote_hf":
            # Remote HF dataset (would need ISICSegRawSplit to support HF datasets)
            raise NotImplementedError(
                "Remote HuggingFace segmentation datasets not yet supported. "
                "Use local_dirs or csv mode instead."
            )
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        dataset = wrapper

        print(f"Dataset loaded: {len(dataset)} samples")
        return dataset

    def setup(self, _stage: Optional[str] = None):
        """Initialize datasets using ISICSegRawSplit for segmentation."""
        print(f"\n{'='*60}")
        print(f"Setting up ISIC Segmentation DataModule")
        print(f"  Data source: {self.data_source}")
        if self.data_source == "local_dirs":
            print(f"  Image dir: {self.image_dir}")
            print(f"  Mask dir: {self.mask_dir}")
        elif self.data_source == "csv":
            print(f"  Data dir: {self.data_dir}")
            print(f"  Label file: {self.label_file}")
        print(f"  Image size: {self.image_size}")
        print(f"{'='*60}\n")

        # Create segmentation transforms
        val_test_transform = SegmentationTransform(target_size=self.image_size)
        train_transform = self.transform if self.transform is not None else val_test_transform

        print("Loading training set...")
        self.train_set = self._load_split(
            split="train",
            transform=train_transform
        )
        print(f"  ✓ {len(self.train_set)} training samples\n")

        # For directory-based loading, we only have one dataset
        # Split it into train/val if needed
        if self.data_source == "local_dirs":
            print("Splitting dataset into train/val...")
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
            self.test_set = None
        else:
            # For CSV or HF datasets, try to load separate splits
            print("Loading validation set...")
            try:
                self.val_set = self._load_split(
                    split="validation",
                    transform=val_test_transform
                )
                print(f"  ✓ {len(self.val_set)} validation samples\n")
            except Exception:
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
                    transform=val_test_transform
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


class ISICFeatureDataModule(BaseDataModule):
    """
    DataModule for ISIC dermoscopic feature detection tasks.

    This task involves multi-label classification of superpixel regions.
    Each image has:
    - An RGB image
    - A superpixel mask (encoded as RGB PNG)
    - JSON annotations with 4 features per superpixel
    """

    def __init__(
        self,
        image_dir: str,
        superpixel_dir: str,
        annotation_dir: str,
        *,
        num_workers: int = 8,
        batch_size: int = 16,
        pin_memory: bool = True,
        drop_last: bool = False,
        split_seed: int = 42,
        transform: Optional[Any] = None,
        filter_fn: Optional[Any] = None,
        keep_indices: Optional[Any] = None,
        full_cfg: Optional[Any] = None,
        image_size: int = 256,
        image_extension: str = ".jpg",
        superpixel_extension: str = ".png",
        annotation_extension: str = ".json",
        superpixel_suffix: str = "_superpixels",
        **kwargs
    ):
        self.full_cfg = full_cfg
        self.image_size = image_size

        super().__init__(
            cfg=full_cfg,
            dataset_name=None,
            data_dir=None,
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

        # Store feature detection-specific parameters
        self.image_dir = image_dir
        self.superpixel_dir = superpixel_dir
        self.annotation_dir = annotation_dir
        self.image_extension = image_extension
        self.superpixel_extension = superpixel_extension
        self.annotation_extension = annotation_extension
        self.superpixel_suffix = superpixel_suffix

        # Validate directories
        if not os.path.isdir(image_dir):
            raise ValueError(f"image_dir must be a valid directory: {image_dir}")
        if not os.path.isdir(superpixel_dir):
            raise ValueError(f"superpixel_dir must be a valid directory: {superpixel_dir}")
        if not os.path.isdir(annotation_dir):
            raise ValueError(f"annotation_dir must be a valid directory: {annotation_dir}")

    def _load_split(
        self,
        split: str = "train",
        transform: Optional[Any] = None
    ):
        """Load a feature detection dataset split."""
        from src.data.isic_feature_loader import ISICFeatureDetectionDataset

        print(f"Loading {split} split from local directories")

        dataset = ISICFeatureDetectionDataset(
            image_dir=self.image_dir,
            superpixel_dir=self.superpixel_dir,
            annotation_dir=self.annotation_dir,
            image_extension=self.image_extension,
            superpixel_extension=self.superpixel_extension,
            annotation_extension=self.annotation_extension,
            superpixel_suffix=self.superpixel_suffix,
            transform=transform,
            filter_fn=self.filter_fn,
            keep_indices=self.keep_indices,
        )

        print(f"Dataset loaded: {len(dataset)} samples")
        return dataset

    def setup(self, _stage: Optional[str] = None):
        """Initialize datasets for feature detection."""
        print(f"Setting up ISIC Feature Detection DataModule")
        print(f"  Image dir: {self.image_dir}")
        print(f"  Superpixel dir: {self.superpixel_dir}")
        print(f"  Annotation dir: {self.annotation_dir}")
        print(f"  Image size: {self.image_size}")

        # Create feature detection transforms
        val_test_transform = FeatureDetectionTransform(target_size=self.image_size)
        train_transform = self.transform if self.transform is not None else val_test_transform

        print("Loading dataset...")
        full_dataset = self._load_split(
            split="train",
            transform=train_transform
        )
        print(f"  ✓ {len(full_dataset)} total samples\n")

        # Split into train/val
        print("Splitting dataset into train/val...")
        total = len(full_dataset)
        n_val = max(1, int(total * VAL_SPLIT_RATIO))
        n_train = total - n_val

        g = torch.Generator().manual_seed(self.split_seed)
        idx_train, idx_val = random_split(
            range(total), [n_train, n_val], generator=g
        )

        self.train_set = Subset(full_dataset, idx_train.indices)
        self.val_set = Subset(full_dataset, idx_val.indices)

        print(f"  ✓ Split: {n_train} train, {n_val} val\n")

        # No separate test set for feature detection (typically)
        self.test_set = None

        print(f"DataModule setup complete!")
        print(f"  Train: {len(self.train_set)} samples")
        print(f"  Val: {len(self.val_set)} samples")

