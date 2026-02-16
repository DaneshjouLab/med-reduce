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

        log.info("=" * 60)
        log.info("DISTILLATION COMPLETE")
        log.info(f"  Best val loss: {result['best_val_loss']:.6f}")
        log.info(f"  Checkpoint: {self.run_dir}/distilled_student.pt")
        log.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Stage 1: Teacher embedding caching
    # ------------------------------------------------------------------

    def _cache_and_load_teacher_embeddings(self) -> TeacherEmbeddingLookup:
        """Cache teacher embeddings at full resolution, then build lookup table."""
        log.info("\n--- Stage 1: Teacher Embedding Caching ---")

        cache = TeacherEmbeddingCache(
            cache_dir=self.teacher_cache_dir,
            teacher_model_info=self.teacher_info,
            full_resolution=self.teacher_resolution,
            device=self.device,
        )

        # Build a clean-image dataloader from the datamodule's full dataset
        # using only the train split indices (same splits as LP baseline).
        train_loader = self._make_clean_dataloader(split="train")

        cache.cache_embeddings(
            dataloader=train_loader,
            dataset_name=self.dataset_name,
            split="train",
            force_recompute=False,
        )

        # Also cache val split for validation during distillation
        val_loader = self._make_clean_dataloader(split="val")
        cache.cache_embeddings(
            dataloader=val_loader,
            dataset_name=self.dataset_name,
            split="val",
            force_recompute=False,
        )

        # Load cached embeddings and build lookup for train + val
        train_data = cache.load_embeddings(self.dataset_name, "train", device="cpu")
        val_data = cache.load_embeddings(self.dataset_name, "val", device="cpu")

        # Merge into a single lookup (train + val embeddings)
        merged = {
            "embeddings": torch.cat([train_data["embeddings"], val_data["embeddings"]], dim=0),
            "labels": torch.cat([train_data["labels"], val_data["labels"]], dim=0),
            "image_ids": train_data["image_ids"] + val_data["image_ids"],
        }
        lookup = TeacherEmbeddingLookup(merged)
        self.teacher_embedding_dim = train_data["embeddings"].shape[1]

        log.info(f"  Teacher embedding dim: {self.teacher_embedding_dim}")
        log.info(f"  Total cached samples: {len(lookup)}")

        return lookup

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

        ckpt_path = os.path.join(self.run_dir, "distilled_student.pt")
        torch.save(checkpoint, ckpt_path)
        log.info(f"  Saved checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------
    # Dataloader helpers
    # ------------------------------------------------------------------

    def _make_clean_dataloader(self, split: str) -> DataLoader:
        """Create a clean (no degradation) dataloader for teacher embedding extraction."""
        clean_transform = transforms.Compose([
            transforms.Resize((self.teacher_resolution, self.teacher_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        dataset = self._get_split_dataset(split, transform=clean_transform)
        batch_size = int(getattr(self.cfg.data, "batch_size", 256))

        return DataLoader(
            dataset,
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
