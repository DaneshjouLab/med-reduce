#!/usr/bin/env python3
# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""
Script to pre-cache teacher embeddings for distillation.

This script extracts and caches embeddings from a teacher model at full resolution
for all training images. These cached embeddings can then be used during student
training without re-running the teacher model.

Usage:
    python -m src.cli.cache_teacher_embeddings --config configs/config.yaml

The script will:
1. Load the teacher model specified in the config
2. Extract embeddings at full resolution for all training images
3. Cache the embeddings to disk for later use during distillation
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.teacher_cache import TeacherEmbeddingCache, create_clean_image_dataloader
from src.utils.logging_core import setup_logging, get_logger

log = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cache teacher embeddings for knowledge distillation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/teacher_embeddings",
        help="Directory to store cached embeddings"
    )
    parser.add_argument(
        "--full-resolution",
        type=int,
        default=224,
        help="Full resolution for teacher embeddings (default: 224)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding extraction (default: 256)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache exists"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to cache (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to cache (for testing)"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Override teacher model ID from config"
    )

    return parser.parse_args()


def cache_teacher_embeddings_for_split(
    cache: TeacherEmbeddingCache,
    cfg: DictConfig,
    split: str,
    batch_size: int,
    num_workers: int,
    force: bool = False,
    max_samples: int = None,
):
    """Cache teacher embeddings for a single dataset split."""
    log.info(f"\n{'='*60}")
    log.info(f"Caching teacher embeddings for split: {split}")
    log.info(f"{'='*60}\n")

    # Get dataset configuration
    dataset_name = cfg.datamodule.dataset_name
    data_dir = getattr(cfg.datamodule, 'data_dir', None)

    # Create dataloader for clean full-resolution images
    log.info("Creating dataloader for clean images...")
    dataloader = create_clean_image_dataloader(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=cache.full_resolution,
    )
    log.info(f"  ✓ Loaded {len(dataloader.dataset)} images")

    # Cache embeddings
    cache_path = cache.cache_embeddings(
        dataloader=dataloader,
        dataset_name=dataset_name,
        split=split,
        force_recompute=force,
        max_samples=max_samples,
    )

    return cache_path


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging()

    log.info("="*60)
    log.info("Teacher Embedding Cache Generator")
    log.info("="*60)

    # Load config
    log.info(f"Loading config from {args.config}")
    cfg = OmegaConf.load(args.config)

    # Override teacher model if specified
    teacher_model_info = cfg.model
    if args.teacher_model:
        log.info(f"Overriding teacher model: {args.teacher_model}")
        teacher_model_info.model_id = args.teacher_model

    # Display configuration
    log.info(f"\nConfiguration:")
    log.info(f"  Dataset: {cfg.datamodule.dataset_name}")
    log.info(f"  Teacher model: {teacher_model_info.model_id}")
    log.info(f"  Full resolution: {args.full_resolution}px")
    log.info(f"  Cache directory: {args.cache_dir}")
    log.info(f"  Splits to cache: {args.splits}")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  Force recompute: {args.force}")
    if args.max_samples:
        log.info(f"  Max samples: {args.max_samples}")

    # Initialize cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\nUsing device: {device}")

    cache = TeacherEmbeddingCache(
        cache_dir=args.cache_dir,
        teacher_model_info=teacher_model_info,
        full_resolution=args.full_resolution,
        device=device,
    )

    # Cache embeddings for each split
    cache_paths = {}
    for split in args.splits:
        try:
            cache_path = cache_teacher_embeddings_for_split(
                cache=cache,
                cfg=cfg,
                split=split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                force=args.force,
                max_samples=args.max_samples,
            )
            cache_paths[split] = cache_path
            log.info(f"✓ Successfully cached {split} split to {cache_path}\n")
        except Exception as e:
            log.error(f"✗ Failed to cache {split} split: {e}\n")

    # Summary
    log.info("\n" + "="*60)
    log.info("Caching Complete!")
    log.info("="*60)
    for split, path in cache_paths.items():
        log.info(f"  {split}: {path}")

    log.info(f"\nYou can now use these cached embeddings for distillation by setting:")
    log.info(f"  distillation.use_cached_embeddings: true")
    log.info(f"  distillation.teacher_cache_dir: {args.cache_dir}")
    log.info(f"  distillation.teacher_model: {teacher_model_info.model_id}")


if __name__ == "__main__":
    main()
