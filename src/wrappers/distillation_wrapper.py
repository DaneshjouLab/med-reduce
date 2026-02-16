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
from torch.utils.data import DataLoader

from src.engines.distillation_engine import train_distillation
from src.losses.distillation import embedding_distillation_loss
from src.models.factory import create_model, get_embedding_dim, extract_embeddings
from src.utils.teacher_cache import TeacherEmbeddingCache, TeacherEmbeddingLookup
from src.utils.embedding_cache import EmbeddingCache
from src.utils.split_manager import SplitManager
from src.utils.optim import make_optimizer_and_scheduler
from src.utils.logging_core import setup_logging, get_logger, WandbLogger
from src.transformations.transforms import ResolutionReductionTransform

log = get_logger(__name__)


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

        # --- Data module (same splits as LP baseline) ---
        self.dm = hydra.utils.instantiate(cfg.datamodule, full_cfg=cfg)
        self.dm.setup("fit")

        self.dataset_name = self.dm.dataset_identifier
        self.task_name = getattr(cfg.datamodule, "task", None)

        # --- Split manager (same as LP baseline) ---
        split_dir = getattr(cfg, "split_dir", "./splits")
        self.split_manager = SplitManager(
            split_dir=split_dir,
            dataset_name=self.dataset_name,
            seed=self.seed,
        )

        # --- Model configs ---
        self.teacher_info = dict(cfg.teacher)
        self.student_info = dict(cfg.student)

        # --- Distillation config ---
        distill_cfg = cfg.distillation
        self.alpha = float(getattr(distill_cfg, "alpha", 0.5))
        self.teacher_resolution = int(getattr(distill_cfg, "teacher_resolution", 512))
        self.teacher_cache_dir = str(getattr(distill_cfg, "teacher_cache_dir", "./cache/teacher_embeddings"))
        self.lp_embedding_cache_dir = str(getattr(distill_cfg, "lp_embedding_cache_dir", ""))

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

    def _cache_and_load_teacher_embeddings(self) -> TeacherEmbeddingLookup:
        """Load teacher embeddings, reusing LP baseline cache when available.

        Strategy:
          1. Try to load from LP baseline's EmbeddingCache (already computed at 512px
             during Pipeline A). The LP cache stores {embeddings, labels} per split
             but lacks image_ids, so we reconstruct them from the dataset.
          2. Fall back to TeacherEmbeddingCache (extracts from scratch) if LP cache
             is not available.

        In both cases, the result is a TeacherEmbeddingLookup with image_id-keyed
        access for the distillation training loop.
        """
        log.info("\n--- Stage 1: Teacher Embedding Loading ---")

        # Try LP baseline cache first
        lookup = self._try_load_from_lp_cache()
        if lookup is not None:
            return lookup

        # Fall back to dedicated teacher cache
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
        lookup = TeacherEmbeddingLookup(data)
        self.teacher_embedding_dim = data["embeddings"].shape[1]

        log.info(f"  Teacher embedding dim: {self.teacher_embedding_dim}")
        log.info(f"  Total cached samples: {len(lookup)}")

        return lookup

    def _try_load_from_lp_cache(self) -> TeacherEmbeddingLookup | None:
        """Try to load teacher embeddings from LP baseline's EmbeddingCache.

        The LP pipeline caches DINOv3 embeddings at 512px per seed/split.
        We load train + test embeddings, reconstruct image_ids from the dataset
        ordering, and combine them into a full-dataset lookup.

        Returns:
            TeacherEmbeddingLookup if LP cache found and loaded, else None.
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

        resolution = self.teacher_resolution

        # Check if LP cache has both train and test at teacher resolution
        if not (lp_cache.exists(resolution, "train") and lp_cache.exists(resolution, "test")):
            log.info(
                f"LP cache not found for {teacher_name} at {resolution}px "
                f"(seed={self.seed}) in {self.lp_embedding_cache_dir}"
            )
            return None

        log.info(f"Found LP baseline cache — reusing {teacher_name} embeddings at {resolution}px")

        # Load embeddings from LP cache (ordered by split indices, no shuffle)
        train_emb, train_labels = lp_cache.load(resolution, "train")
        test_emb, test_labels = lp_cache.load(resolution, "test")

        # Load split indices to know which dataset positions each embedding maps to
        splits = self.split_manager.load_splits()
        train_indices = splits["train"]
        test_indices = splits["test"]

        if len(train_indices) != train_emb.shape[0]:
            log.warning(
                f"Train index count ({len(train_indices)}) != train embeddings ({train_emb.shape[0]}). "
                "Skipping LP cache reuse."
            )
            return None
        if len(test_indices) != test_emb.shape[0]:
            log.warning(
                f"Test index count ({len(test_indices)}) != test embeddings ({test_emb.shape[0]}). "
                "Skipping LP cache reuse."
            )
            return None

        # Reconstruct image_ids from the dataset metadata (no image loading).
        image_ids = self._get_image_ids_from_dataset()

        # Combine train + test embeddings via concatenation (vectorised)
        full_embeddings = torch.cat([train_emb, test_emb], dim=0)
        full_labels = torch.cat([train_labels, test_labels], dim=0)

        # Free the per-split tensors now that they're concatenated
        del train_emb, test_emb, train_labels, test_labels

        # Build image_id list in the same order: train indices then test indices
        all_indices = list(train_indices) + list(test_indices)
        full_image_ids = [str(image_ids[idx]) for idx in all_indices]

        self.teacher_embedding_dim = emb_dim
        log.info(f"  Reused LP cache: {total} embeddings (train={len(train_indices)}, test={len(test_indices)})")
        log.info(f"  Teacher embedding dim: {emb_dim}")

        data = {
            "embeddings": full_embeddings,
            "labels": full_labels,
            "image_ids": full_image_ids,
        }
        return TeacherEmbeddingLookup(data)

    def _get_image_ids_from_dataset(self) -> list:
        """Extract image_ids from the full dataset without loading pixel data.

        Tries to read image_ids directly from the underlying dataset's metadata
        (CSV / manifest) to avoid loading and decoding every image. Falls back to
        iterating the dataset with a minimal transform if metadata is unavailable.

        Returns a list where result[i] is the image_id for dataset index i.
        """
        full_dataset = self.dm._load_full_dataset(split="train", transform=None)

        if self.dm.balance_data:
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

        if hasattr(inner, "df") and "image_id" in getattr(inner.df, "columns", []):
            # Tabular datasets backed by a DataFrame with image_id column
            image_ids = [str(x) for x in inner.df["image_id"].tolist()]
            log.info(f"  Read {len(image_ids)} image_ids from dataset.df['image_id']")
            return image_ids

        # Slow fallback: iterate the dataset. Use a tiny transform to minimise
        # decode cost, and only access the needed indices.
        log.info("  Falling back to iterating dataset for image_ids (slow path)")
        light_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        full_dataset = self.dm._load_full_dataset(split="train", transform=light_transform)
        if self.dm.balance_data:
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

        # Projection layer if dimension mismatch
        projection = None
        if student_emb_dim != self.teacher_embedding_dim:
            log.info(f"  Adding projection: {student_emb_dim} -> {self.teacher_embedding_dim}")
            projection = nn.Linear(student_emb_dim, self.teacher_embedding_dim).to(self.device)

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
        full_dataset = self.dm._load_full_dataset(split="train", transform=clean_transform)

        # Apply same balancing as the datamodule if needed
        if self.dm.balance_data:
            from src.data.dataset_factory import balance_dataset
            full_dataset.ds = balance_dataset(
                dataset=full_dataset.ds,
                filtered_classes=self.dm.filtered_classes,
                num_train_images=self.dm.num_train_images or len(full_dataset.ds),
                seed=self.dm.split_seed,
            )

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
        """Create a dataloader with degradation transforms for student training."""
        image_size = int(getattr(self.cfg.data, "image_size", 512))

        degraded_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ResolutionReductionTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        dataset = self._get_split_dataset(split, transform=degraded_transform)
        batch_size = int(getattr(self.cfg.data, "batch_size", 64))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
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
        full_dataset = self.dm._load_full_dataset(split="train", transform=transform)

        # Apply same balancing as the datamodule if needed
        if self.dm.balance_data:
            from src.data.dataset_factory import balance_dataset
            full_dataset.ds = balance_dataset(
                dataset=full_dataset.ds,
                filtered_classes=self.dm.filtered_classes,
                num_train_images=self.dm.num_train_images or len(full_dataset.ds),
                seed=self.dm.split_seed,
            )

        # Load the persistent split indices
        splits = self.split_manager.load_splits()

        if split == "train":
            indices = splits["train"]
        elif split == "val":
            if "val" in splits:
                indices = splits["val"]
            else:
                # Reproduce the same val carve-out logic as the datamodule
                import numpy as np
                train_indices = splits["train"]
                n_val = max(1, int(len(train_indices) * 0.1))
                np.random.seed(self.dm.split_seed)
                indices = np.random.choice(train_indices, n_val, replace=False)
        elif split == "test":
            indices = splits["test"]
        else:
            raise ValueError(f"Unknown split: {split}")

        return Subset(full_dataset, indices)


def run(cfg: Any) -> Dict[str, Any]:
    """Entry point."""
    wrapper = DistillationWrapper(cfg)
    return wrapper.run()
