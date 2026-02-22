# src/wrappers/distillation_wrapper.py
# -*- coding: utf-8 -*-
"""
Distillation pipeline wrapper.

Pipeline:
  Stage 1: Cache teacher (DINOv3) embeddings at full resolution
  Stage 2: Train student (TinyViT / ResNet18) end-to-end on degraded images
           to match teacher embeddings
  Stage 3: (Optional) Freeze student, run LP evaluation at each resolution
"""
from __future__ import annotations
from typing import Any, Dict

import os
import json
import torch
import torch.nn as nn
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler

from src.engines.distillation_engine import train_distillation
from src.losses.distillation import embedding_distillation_loss
from src.models.factory import create_model, get_embedding_dim, extract_embeddings
from src.utils.teacher_cache import TeacherEmbeddingCache, TeacherEmbeddingLookup, MultiResolutionTeacherLookup
from src.utils.embedding_cache import EmbeddingCache
from src.utils.split_manager import SplitManager
from src.utils.optim import make_optimizer_and_scheduler
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.transformations.transforms import ResolutionReductionTransform

log = get_logger(__name__)


class MultiViewDataset(Dataset):
    """Wraps a base dataset to produce multiple augmented views per image.

    Each original sample is returned ``n_views`` times with independently
    sampled augmentations.  The ``image_id`` is preserved so that the
    teacher-embedding lookup still works.
    """

    def __init__(self, base_dataset: Dataset, view_transforms: list, n_views: int = 2,
                 view_resolutions: list[int] | None = None):
        """
        Args:
            base_dataset: Underlying dataset returning dicts with
                ``pixel_values``, ``label``, and optionally ``image_id``.
            view_transforms: List of ``torchvision.transforms.Compose``
                pipelines, one per view.  Cycled if ``n_views`` exceeds the
                list length.
            n_views: Number of augmented copies per image (>= 1).
            view_resolutions: Target teacher resolution for each view.
                Used by the training loop to look up the matching teacher
                embedding.  If None, all views target the max resolution.
        """
        self.base = base_dataset
        self.view_transforms = view_transforms
        self.n_views = max(1, n_views)
        self.view_resolutions = view_resolutions or [512] * n_views

    def __len__(self):
        return len(self.base) * self.n_views

    def __getitem__(self, idx):
        base_idx = idx // self.n_views
        view_idx = idx % self.n_views

        # Access the raw PIL image directly (bypassing the base dataset's
        # transform) so we can apply our own view-specific transform.
        raw = self._get_raw_item(base_idx)
        image = raw["image"]
        tfm = self.view_transforms[view_idx % len(self.view_transforms)]
        image = tfm(image)

        result = {
            "pixel_values": image,
            "label": raw["label"],
            "target_resolution": self.view_resolutions[view_idx % len(self.view_resolutions)],
        }
        if "image_id" in raw:
            result["image_id"] = raw["image_id"]
        return result

    # ------------------------------------------------------------------

    def _get_raw_item(self, base_idx):
        """Return the raw PIL image + metadata without the base transform."""
        ds = self.base
        # Unwrap torch Subset
        while hasattr(ds, "dataset"):
            if hasattr(ds, "indices"):
                base_idx = ds.indices[base_idx]
            ds = ds.dataset

        # ds is now the leaf dataset (ISICHFRawSplitLocal / TCGASlideDataset / …)
        from PIL import Image as PILImage

        if hasattr(ds, "ds"):
            # HuggingFace-backed (ISICHFRawSplitLocal)
            row = ds.ds[base_idx]
            img = row.get(ds.image_column, row.get("image"))
            if not isinstance(img, PILImage.Image):
                img = PILImage.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            label = row.get(ds.label_column if hasattr(ds, "label_column") else "label", 0)
            if isinstance(label, (list, tuple)):
                import torch
                label = torch.tensor(label, dtype=torch.float32)
            else:
                label = int(label)
            image_id = row.get("image_id")
        elif hasattr(ds, "df"):
            # TCGASlideDataset
            row = ds.df.iloc[base_idx]
            img_path = ds.thumbnails_dir / f"{row['slide_id']}.jpg"
            img = PILImage.open(img_path).convert("RGB")
            label = int(row["label"])
            image_id = str(row["slide_id"])
        else:
            raise TypeError(f"Unsupported base dataset type: {type(ds)}")

        out = {"image": img, "label": label}
        if image_id is not None:
            out["image_id"] = str(image_id)
        return out


class GroupedViewBatchSampler(Sampler):
    """Batch sampler that keeps all views of the same image in the same batch.

    Given a ``MultiViewDataset`` with *N* base images and *V* views each,
    this sampler shuffles the *N* image indices and yields batches where each
    batch contains ``batch_size_images * V`` samples (all views of
    ``batch_size_images`` images grouped together).
    """

    def __init__(self, n_images: int, n_views: int, batch_size_images: int, shuffle: bool = True, seed: int | None = None):
        self.n_images = n_images
        self.n_views = n_views
        self.batch_size_images = batch_size_images
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic but different shuffling each epoch."""
        self._epoch = epoch

    def __iter__(self):
        import numpy as np
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self._epoch if self.seed is not None else None)
            order = rng.permutation(self.n_images)
        else:
            order = np.arange(self.n_images)
        for start in range(0, self.n_images, self.batch_size_images):
            image_indices = order[start : start + self.batch_size_images]
            # Expand each image index into its n_views consecutive dataset indices
            batch = []
            for img_idx in image_indices:
                batch.extend(range(img_idx * self.n_views, (img_idx + 1) * self.n_views))
            yield batch

    def __len__(self):
        return (self.n_images + self.batch_size_images - 1) // self.batch_size_images


class _ResolutionTaggedDataset(Dataset):
    """Thin wrapper that injects a fixed ``target_resolution`` into every sample."""

    def __init__(self, base_dataset: Dataset, target_resolution: int):
        self.base = base_dataset
        self.target_resolution = target_resolution

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if isinstance(sample, dict):
            sample["target_resolution"] = self.target_resolution
        return sample


class DistillationWrapper:
    """
    Orchestrates the distillation pipeline:
      1. Setup data + splits (reuses SplitManager for consistency with LP baseline)
      2. Cache / load teacher embeddings at full resolution
      3. Train student end-to-end on degraded images
      4. Save best student checkpoint
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        setup_logging()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.domain = getattr(cfg, "domain", "dermatology")
        self.seed = int(getattr(cfg.train, "seed", 42))
        self.smoke_test = bool(getattr(cfg, "smoke_test", False))

        # --- Data module (same splits as LP baseline) ---
        self.dm = hydra.utils.instantiate(cfg.datamodule, full_cfg=cfg)
        self.dm.setup("fit")

        self.dataset_name = self.dm.dataset_identifier
        self.task_name = getattr(cfg.datamodule, "task", None)

        # --- Split manager (reuse the datamodule's split manager for consistency) ---
        self.split_manager = self.dm.split_manager

        # --- Model configs ---
        self.teacher_info = dict(cfg.teacher)
        self.student_info = dict(cfg.student)

        # --- Distillation config ---
        distill_cfg = cfg.distillation
        self.alpha = float(getattr(distill_cfg, "alpha", 0.5))
        self.teacher_resolution = int(getattr(distill_cfg, "teacher_resolution", 512))
        self.teacher_cache_dir = str(getattr(distill_cfg, "teacher_cache_dir", "./cache/teacher_embeddings"))
        self.lp_embedding_cache_dir = str(getattr(distill_cfg, "lp_embedding_cache_dir", ""))
        self.target_resolutions = list(getattr(distill_cfg, "target_resolutions", [512, 256, 128, 64]))

        # --- Run dir ---
        base_run_dir = getattr(cfg.runtime, "run_dir", "./runs/distillation")
        self.run_dir = os.path.join(base_run_dir, f"seed_{self.seed}")
        os.makedirs(self.run_dir, exist_ok=True)

        # --- WandB ---
        self.wandb = WandbLogger(
            project=getattr(cfg.logging, "project", "reduced-perception"),
            run_name=getattr(cfg.logging, "run_name", "distillation"),
            config=cfg,
            enabled=bool(getattr(cfg.logging, "wandb_enabled", True)),
            tags=["distillation", self.student_info.get("name", "student"), self.domain],
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full distillation pipeline."""
        log.info("=" * 60)
        log.info("DISTILLATION PIPELINE")
        if self.smoke_test:
            log.info("  *** SMOKE TEST MODE — 10 samples, 2 epochs ***")
        log.info("=" * 60)

        # Stage 1: Cache teacher embeddings
        teacher_lookup = self._cache_and_load_teacher_embeddings()

        # Stage 2: Train student
        result = self._train_student(teacher_lookup)

        # Save results JSON
        self._save_results(result)

        log.info("=" * 60)
        log.info("DISTILLATION COMPLETE")
        log.info(f"  Best val loss: {result['best_val_loss']:.6f}")
        student_name = self.student_info.get("name", "student")
        suffix = f"_{self.task_name}" if self.task_name else ""
        log.info(f"  Checkpoint: {self.run_dir}/distilled_{student_name}{suffix}.pt")
        log.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Stage 1: Teacher embedding caching
    # ------------------------------------------------------------------

    def _cache_and_load_teacher_embeddings(self) -> MultiResolutionTeacherLookup:
        """Load teacher embeddings at multiple resolutions.

        Strategy:
          1. Try to load from LP baseline's EmbeddingCache (already computed at
             512/256/128/64px during Pipeline A).
          2. Fall back to TeacherEmbeddingCache (extracts from scratch) if LP
             cache is not available.

        Returns a MultiResolutionTeacherLookup with per-resolution access.
        """
        log.info("\n--- Stage 1: Teacher Embedding Loading (multi-resolution) ---")

        # Try LP baseline cache first
        lookup = self._try_load_from_lp_cache()
        if lookup is not None:
            return lookup

        # Fall back to dedicated teacher cache (single resolution only)
        log.info("LP cache not available — extracting teacher embeddings from scratch")
        cache = TeacherEmbeddingCache(
            cache_dir=self.teacher_cache_dir,
            teacher_model_info=self.teacher_info,
            full_resolution=self.teacher_resolution,
            device=self.device,
        )

        full_loader = self._make_full_dataset_dataloader()

        cache.cache_embeddings(
            dataloader=full_loader,
            dataset_name=self.dataset_name,
            split="full",
            force_recompute=False,
        )

        data = cache.load_embeddings(self.dataset_name, "full", device="cpu")
        self.teacher_embedding_dim = data["embeddings"].shape[1]

        # Wrap single-resolution data in MultiResolutionTeacherLookup
        lookup = MultiResolutionTeacherLookup()
        lookup.add_resolution(self.teacher_resolution, data)
        # Use the same embeddings for all target resolutions (fallback)
        for res in self.target_resolutions:
            if res != self.teacher_resolution:
                lookup.add_resolution(res, data)
        log.warning(
            "LP cache not available — using teacher@%dpx for all resolutions. "
            "For best results, run LP baseline first to cache multi-resolution embeddings.",
            self.teacher_resolution,
        )

        log.info(f"  Teacher embedding dim: {self.teacher_embedding_dim}")
        log.info(f"  Total cached samples: {len(lookup)}")

        return lookup

    def _try_load_from_lp_cache(self) -> MultiResolutionTeacherLookup | None:
        """Try to load teacher embeddings at multiple resolutions from LP cache.

        The LP pipeline caches DINOv3 embeddings at 512/256/128/64px per seed.
        We load all available resolutions and combine them into a
        MultiResolutionTeacherLookup.

        Returns:
            MultiResolutionTeacherLookup if at least one resolution found, else None.
        """
        if not self.lp_embedding_cache_dir:
            return None

        teacher_name = self.teacher_info.get("name", "dinov3")
        lp_cache = EmbeddingCache(
            cache_dir=self.lp_embedding_cache_dir,
            dataset_name=self.dataset_name,
            model_name=teacher_name,
            seed=self.seed,
        )

        # Check which resolutions are available
        available = [
            res for res in self.target_resolutions
            if lp_cache.exists(res, "train") and lp_cache.exists(res, "test")
        ]
        if not available:
            log.info(
                f"LP cache not found for {teacher_name} at any target resolution "
                f"(seed={self.seed}) in {self.lp_embedding_cache_dir}"
            )
            return None

        log.info(f"Found LP baseline cache — loading {teacher_name} at {available}")

        # Load split indices
        if not self.split_manager.exists():
            log.warning("Splits not found — cannot reuse LP cache without split indices")
            return None
        splits = self.split_manager.load_splits()
        train_indices = splits["train"]
        test_indices = splits["test"]

        # Reconstruct image_ids once
        image_ids = self._get_image_ids_from_dataset()
        all_indices = list(train_indices) + list(test_indices)
        full_image_ids = [str(image_ids[idx]) for idx in all_indices]

        lookup = MultiResolutionTeacherLookup()

        for res in available:
            train_emb, train_labels = lp_cache.load(res, "train")
            test_emb, test_labels = lp_cache.load(res, "test")

            if len(train_indices) != train_emb.shape[0] or len(test_indices) != test_emb.shape[0]:
                log.warning(f"Index/embedding count mismatch at {res}px — skipping")
                continue

            full_embeddings = torch.cat([train_emb, test_emb], dim=0)
            full_labels = torch.cat([train_labels, test_labels], dim=0)
            del train_emb, test_emb, train_labels, test_labels

            data = {
                "embeddings": full_embeddings,
                "labels": full_labels,
                "image_ids": full_image_ids,
            }
            lookup.add_resolution(res, data)

        if not lookup.resolutions:
            return None

        self.teacher_embedding_dim = lookup.embedding_dim
        log.info(f"  Teacher embedding dim: {self.teacher_embedding_dim}")
        log.info(f"  Resolutions loaded: {lookup.resolutions}")

        return lookup

    def _get_image_ids_from_dataset(self) -> list:
        """Extract image_ids from the full dataset without loading pixel data.

        Tries to read image_ids directly from the underlying dataset's metadata
        (CSV / manifest) to avoid loading and decoding every image. Falls back to
        iterating the dataset with a minimal transform if metadata is unavailable.

        Returns a list where result[i] is the image_id for dataset index i.
        """
        # Use pre-built full_dataset when available (e.g. TCGADataModule sets
        # self.full_dataset in setup()), otherwise load via the TabularDataModule API.
        if hasattr(self.dm, "full_dataset") and not hasattr(self.dm, "_load_full_dataset"):
            full_dataset = self.dm.full_dataset
        else:
            full_dataset = self.dm._load_full_dataset(split="train", transform=None)

            if getattr(self.dm, "balance_data", False):
                from src.data.dataset_factory import balance_dataset
                full_dataset.ds = balance_dataset(
                    dataset=full_dataset.ds,
                    filtered_classes=self.dm.filtered_classes,
                    num_train_images=self.dm.num_train_images or len(full_dataset.ds),
                    seed=self.dm.split_seed,
                )

        # Fast path: read image_ids from the underlying dataset's metadata
        # without loading any images.
        inner = full_dataset.ds if hasattr(full_dataset, "ds") else full_dataset
        if hasattr(inner, "image_ids"):
            # Some datasets expose a pre-computed list of image_ids
            image_ids = [str(x) for x in inner.image_ids]
            log.info(f"  Read {len(image_ids)} image_ids from dataset.image_ids attribute")
            return image_ids

        if hasattr(inner, "df"):
            # Tabular datasets backed by a DataFrame — try image_id or slide_id
            for col in ("image_id", "slide_id"):
                if col in getattr(inner.df, "columns", []):
                    image_ids = [str(x) for x in inner.df[col].tolist()]
                    log.info(f"  Read {len(image_ids)} image_ids from dataset.df['{col}']")
                    return image_ids

        # Slow fallback: iterate the dataset. Use a tiny transform to minimise
        # decode cost, and only access the needed indices.
        log.info("  Falling back to iterating dataset for image_ids (slow path)")
        if hasattr(self.dm, "_load_full_dataset"):
            light_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            full_dataset = self.dm._load_full_dataset(split="train", transform=light_transform)
            if getattr(self.dm, "balance_data", False):
                from src.data.dataset_factory import balance_dataset
                full_dataset.ds = balance_dataset(
                    dataset=full_dataset.ds,
                    filtered_classes=self.dm.filtered_classes,
                    num_train_images=self.dm.num_train_images or len(full_dataset.ds),
                    seed=self.dm.split_seed,
                )

        image_ids = []
        for i in range(len(full_dataset)):
            sample = full_dataset[i]
            if isinstance(sample, dict):
                img_id = sample.get("image_id", f"sample_{i}")
            else:
                img_id = f"sample_{i}"
            image_ids.append(str(img_id))

        log.info(f"  Extracted {len(image_ids)} image_ids from dataset (slow path)")
        return image_ids

    # ------------------------------------------------------------------
    # Stage 2: Student distillation training
    # ------------------------------------------------------------------

    def _train_student(self, teacher_lookup: TeacherEmbeddingLookup) -> Dict[str, Any]:
        """Create student model and train via distillation."""
        log.info("\n--- Stage 2: Student Distillation Training ---")

        image_size = int(getattr(self.cfg.data, "image_size", 512))

        # Create student model
        student = create_model(self.student_info, resolution=image_size).to(self.device)
        student_type = self.student_info["type"]

        student_emb_dim = get_embedding_dim(student, student_type)
        log.info(f"  Student: {self.student_info.get('name')} (emb_dim={student_emb_dim})")

        # MLP projection head: maps student embeddings into teacher space.
        # Discarded after training — LP evaluates backbone embeddings directly.
        # The hidden layer lets the backbone learn its own feature arrangement
        # while the head handles translation to the teacher's geometry.
        proj_hidden = int(getattr(self.cfg.distillation, "projection_hidden_dim", student_emb_dim))
        projection = nn.Sequential(
            nn.Linear(student_emb_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, self.teacher_embedding_dim),
        ).to(self.device)
        log.info(f"  Projection head: {student_emb_dim} -> {proj_hidden} -> {self.teacher_embedding_dim}")

        # Loss
        loss_fn = embedding_distillation_loss(alpha=self.alpha)
        log.info(f"  Loss: alpha={self.alpha} (MSE={self.alpha}, cosine={1-self.alpha})")

        # Collect trainable parameters
        params = list(student.parameters())
        if projection is not None:
            params += list(projection.parameters())

        # Optimizer and scheduler
        optimizer, scheduler = make_optimizer_and_scheduler(self.cfg, params)

        # Create dataloaders with degradation transforms
        train_loader = self._make_degraded_dataloader(split="train", shuffle=True)
        val_loader = self._make_degraded_dataloader(split="val", shuffle=False)

        loaders = {"train": train_loader, "val": val_loader}

        epochs = int(getattr(self.cfg.train, "epochs", 100))
        if self.smoke_test:
            epochs = min(epochs, 2)
        grad_clip = getattr(self.cfg.train, "grad_clip", None)
        mixed_precision = bool(getattr(self.cfg.train, "mixed_precision", True))
        log_interval = int(getattr(self.cfg.train, "log_interval", 50))

        log.info(f"  Epochs: {epochs}")
        log.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        result = train_distillation(
            student=student,
            student_model_type=student_type,
            teacher_lookup=teacher_lookup,
            loaders=loaders,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            projection=projection,
            device=self.device,
            epochs=epochs,
            grad_clip=float(grad_clip) if grad_clip else None,
            mixed_precision=mixed_precision,
            log_interval=log_interval,
            wandb_logger=self.wandb,
        )

        # Save best student checkpoint
        self._save_checkpoint(result.get("best_state"), student, projection)

        return result

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

    def _save_checkpoint(self, best_state, student, projection):
        """Save distilled student checkpoint."""
        checkpoint = {
            "student_model_info": self.student_info,
            "teacher_model_info": self.teacher_info,
            "alpha": self.alpha,
            "teacher_resolution": self.teacher_resolution,
        }

        if best_state is not None:
            checkpoint["student_state_dict"] = best_state["student"]
            if "projection" in best_state:
                checkpoint["projection_state_dict"] = best_state["projection"]
        else:
            checkpoint["student_state_dict"] = student.state_dict()
            if projection is not None:
                checkpoint["projection_state_dict"] = projection.state_dict()

        student_name = self.student_info.get("name", "student")
        suffix = f"_{self.task_name}" if self.task_name else ""
        ckpt_path = os.path.join(self.run_dir, f"distilled_{student_name}{suffix}.pt")

        # Backup existing checkpoint before overwriting
        if os.path.exists(ckpt_path):
            from datetime import datetime
            backup_path = os.path.join(
                self.run_dir,
                f"distilled_{student_name}{suffix}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
            )
            import shutil
            shutil.copy2(ckpt_path, backup_path)
            log.warning(f"Checkpoint exists. Backed up to: {backup_path}")

        torch.save(checkpoint, ckpt_path)
        log.info(f"  Saved checkpoint to {ckpt_path}")

    def _save_results(self, training_result: Dict[str, Any]):
        """Save distillation results to JSON (mirrors LP baseline format).

        If the results file already exists, creates a timestamped backup first.
        """
        from datetime import datetime

        history = training_result.get("history", {})
        best_val_loss = training_result.get("best_val_loss", None)

        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [])
        lrs = history.get("lr", [])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        student_name = self.student_info.get("name", "student")
        suffix = f"_{self.task_name}" if self.task_name else ""
        ckpt_path_ref = os.path.join(self.run_dir, f"distilled_{student_name}{suffix}.pt")

        results = {
            "experiment_info": {
                "pipeline": "distillation",
                "domain": self.domain,
                "dataset": self.dataset_name,
                "seed": self.seed,
                "teacher_model": self.teacher_info.get("name", "dinov3"),
                "teacher_model_id": self.teacher_info.get("model_id", ""),
                "student_model": self.student_info.get("name", "resnet18"),
                "student_model_id": self.student_info.get("model_id", ""),
                "teacher_resolution": self.teacher_resolution,
                "timestamp": timestamp,
            },
            "distillation_metrics": {
                "best_val_loss": best_val_loss,
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
            },
            "hyperparameters": {
                "alpha": self.alpha,
                "lr": float(self.cfg.train.optimizer.lr),
                "weight_decay": float(self.cfg.train.optimizer.weight_decay),
                "batch_size": int(self.cfg.data.batch_size),
                "epochs": int(self.cfg.train.epochs),
                "image_size": int(getattr(self.cfg.data, "image_size", 512)),
                "mixed_precision": bool(getattr(self.cfg.train, "mixed_precision", True)),
                "grad_clip": getattr(self.cfg.train, "grad_clip", None),
            },
            "training_history": {
                "num_epochs": len(train_losses),
                "train_loss": train_losses,
                "val_loss": val_losses,
                "lr": lrs,
            },
            "checkpoint": ckpt_path_ref,
        }

        results_path = os.path.join(self.run_dir, f"results_distillation_{student_name}{suffix}.json")

        if os.path.exists(results_path):
            import shutil
            backup_path = os.path.join(
                self.run_dir,
                f"results_distillation_{student_name}{suffix}_backup_{timestamp}.json",
            )
            shutil.copy2(results_path, backup_path)
            log.warning(f"Results file exists. Backed up to: {backup_path}")

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        log.info(f"Results saved to: {os.path.abspath(results_path)}")

        if self.wandb:
            self.wandb.log({
                "summary/best_val_loss": best_val_loss,
                "summary/final_train_loss": train_losses[-1] if train_losses else None,
                "summary/seed": self.seed,
            })

    # ------------------------------------------------------------------
    # Dataloader helpers
    # ------------------------------------------------------------------

    def _make_full_dataset_dataloader(self) -> DataLoader:
        """Create a clean dataloader for the FULL dataset (no split filtering).

        Used for teacher embedding caching so the cache is seed-independent.
        """
        clean_transform = transforms.Compose([
            transforms.Resize((self.teacher_resolution, self.teacher_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Load full dataset WITHOUT subsetting to any split
        if hasattr(self.dm, "_load_full_dataset"):
            full_dataset = self.dm._load_full_dataset(split="train", transform=clean_transform)

            # Apply same balancing as the datamodule if needed
            if getattr(self.dm, "balance_data", False):
                from src.data.dataset_factory import balance_dataset
                full_dataset.ds = balance_dataset(
                    dataset=full_dataset.ds,
                    filtered_classes=self.dm.filtered_classes,
                    num_train_images=self.dm.num_train_images or len(full_dataset.ds),
                    seed=self.dm.split_seed,
                )
        else:
            full_dataset = self.dm.full_dataset
            full_dataset.transform = clean_transform

        batch_size = int(getattr(self.cfg.data, "batch_size", 256))

        return DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(getattr(self.cfg.datamodule, "num_workers", 8)),
            pin_memory=True,
            drop_last=False,
        )

    def _make_degraded_dataloader(self, split: str, shuffle: bool = True) -> DataLoader:
        """Create a dataloader with multi-view resolution-targeted transforms.

        Each training image produces one view per target resolution.  Every
        view is matched against the teacher embedding extracted at that same
        resolution, so the student learns resolution-appropriate features.

          - View 0 (512px): clean resize (no degradation)
          - View 1 (256px): downsample to 256px then upsample back to 512px
          - View 2 (128px): downsample to 128px then upsample back to 512px
          - View 3 ( 64px): downsample to  64px then upsample back to 512px

        Validation uses a single deterministic degraded view (256px equivalent).
        """
        image_size = int(getattr(self.cfg.data, "image_size", 512))

        _norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
        _to_tensor = [transforms.ToTensor(), _norm]

        # Build one view per target resolution
        view_transforms = []
        view_resolutions = []

        for res in sorted(self.target_resolutions, reverse=True):
            factor = res / image_size  # e.g. 256/512 = 0.5
            if factor >= 1.0:
                # Clean view (no degradation)
                tfm = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    *_to_tensor,
                ])
            else:
                tfm = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    ResolutionReductionTransform(reduction_factor=factor),
                    *_to_tensor,
                ])
            view_transforms.append(tfm)
            view_resolutions.append(res)

        n_views = len(view_transforms)

        # Base dataset with a dummy transform (MultiViewDataset bypasses it
        # and applies its own transforms to the raw PIL image).
        base_dataset = self._get_split_dataset(split, transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            _norm,
        ]))

        if split == "train":
            dataset = MultiViewDataset(
                base_dataset, view_transforms, n_views=n_views,
                view_resolutions=view_resolutions,
            )
            res_str = ", ".join(f"{r}px" for r in view_resolutions)
            log.info(
                f"  Multi-resolution training: {n_views} views/image "
                f"({len(base_dataset)} base -> {len(dataset)} samples): [{res_str}]"
            )
        else:
            # Validation: single deterministic degraded view (256px equivalent)
            val_resolution = 256
            val_factor = val_resolution / image_size
            val_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                ResolutionReductionTransform(reduction_factor=val_factor),
                transforms.ToTensor(),
                _norm,
            ])
            base_val = self._get_split_dataset(split, transform=val_transform)
            dataset = _ResolutionTaggedDataset(base_val, target_resolution=val_resolution)

        batch_size = int(getattr(self.cfg.data, "batch_size", 64))

        if split == "train":
            # Use grouped batch sampler so all views of the same image
            # always appear together in the same batch.
            n_base = len(base_dataset)
            batch_sampler = GroupedViewBatchSampler(
                n_images=n_base,
                n_views=n_views,
                batch_size_images=batch_size,  # images per batch (actual samples = batch_size * n_views)
                shuffle=shuffle,
                seed=self.seed,
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=int(getattr(self.cfg.datamodule, "num_workers", 8)),
                pin_memory=True,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(getattr(self.cfg.datamodule, "num_workers", 8)),
                pin_memory=True,
                drop_last=False,
            )

    def _get_split_dataset(self, split: str, transform):
        """
        Get a dataset for a specific split with the given transform.

        Re-creates the dataset from the datamodule's source with the provided
        transform, then subsets it using the same persistent split indices.
        """
        from torch.utils.data import Subset

        # Load full dataset with the desired transform
        if hasattr(self.dm, "_load_full_dataset"):
            full_dataset = self.dm._load_full_dataset(split="train", transform=transform)

            # Apply same balancing as the datamodule if needed
            if getattr(self.dm, "balance_data", False):
                from src.data.dataset_factory import balance_dataset
                full_dataset.ds = balance_dataset(
                    dataset=full_dataset.ds,
                    filtered_classes=self.dm.filtered_classes,
                    num_train_images=self.dm.num_train_images or len(full_dataset.ds),
                    seed=self.dm.split_seed,
                )
        else:
            # TCGADataModule (and similar): reuse the pre-built dataset, swap transform
            full_dataset = self.dm.full_dataset
            full_dataset.transform = transform

        # Load the persistent split indices
        splits = self.split_manager.load_splits()

        if split == "train":
            if "val" in splits:
                indices = splits["train"]
            else:
                # Exclude val samples from train when val is carved from train
                import numpy as np
                val_indices = self._get_or_create_val_indices(splits["train"])
                indices = np.setdiff1d(splits["train"], val_indices)
        elif split == "val":
            if "val" in splits:
                indices = splits["val"]
            else:
                indices = self._get_or_create_val_indices(splits["train"])
        elif split == "test":
            indices = splits["test"]
        else:
            raise ValueError(f"Unknown split: {split}")

        if self.smoke_test:
            SMOKE_N = 10
            indices = indices[:SMOKE_N]
            log.info(f"  [SMOKE TEST] Truncated {split} split to {len(indices)} samples")

        return Subset(full_dataset, indices)

    def _get_or_create_val_indices(self, train_indices) -> "np.ndarray":
        """Carve a val split from train indices and persist to disk.

        On first call, creates val_indices.npy in the split directory so that
        every subsequent run (same seed) gets the exact same val set.
        """
        import numpy as np

        val_path = self.split_manager.dataset_dir / "val_indices.npy"

        if val_path.exists():
            indices = np.load(val_path)
            log.info(f"  Loaded persisted val split ({len(indices)} samples) from {val_path}")
            return indices

        # First time: carve out 10% of train as val (deterministic via split seed)
        n_val = max(1, int(len(train_indices) * 0.1))
        rng = np.random.RandomState(self.dm.split_seed)
        indices = rng.choice(train_indices, n_val, replace=False)

        np.save(val_path, indices)
        log.info(f"  Created and saved val split ({len(indices)} samples) to {val_path}")

        return indices


def run(cfg: Any) -> Dict[str, Any]:
    """Entry point."""
    wrapper = DistillationWrapper(cfg)
    return wrapper.run()
