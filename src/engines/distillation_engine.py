# src/engines/distillation_engine.py
# -*- coding: utf-8 -*-
"""
Training engine for embedding distillation.

Student models (TinyViT, ResNet18) learn to match frozen teacher (DINOv3)
embeddings while being trained on degraded inputs.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import math
import torch
from torch import nn

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from src.utils.logging_core import get_logger
from src.utils.teacher_cache import TeacherEmbeddingLookup
from src.models.factory import extract_embeddings
from src.engines.training_core import (
    _maybe_scheduler_step,
    _create_grad_scaler,
    _update_history_and_log,
)

log = get_logger(__name__)


def train_distillation(
    *,
    student: nn.Module,
    student_model_type: str,
    teacher_lookup: TeacherEmbeddingLookup,
    loaders: Dict[str, Any],
    loss_fn,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Tuple[Any, Dict[str, Any]]] = None,
    projection: Optional[nn.Module] = None,
    device: torch.device,
    epochs: int,
    grad_clip: Optional[float] = None,
    mixed_precision: bool = True,
    log_interval: int = 50,
    wandb_logger=None,
) -> Dict[str, Any]:
    """
    End-to-end distillation training loop.

    Per batch:
      1. Load degraded images + image_ids from dataloader
      2. Forward through student → extract pre-classifier embeddings
      3. Look up teacher embeddings via TeacherEmbeddingLookup
      4. Optionally project student embeddings if dim mismatch
      5. Compute distillation loss
      6. Backprop through entire student model

    Args:
        student: Student model (trained end-to-end).
        student_model_type: Model type string ('timm', 'vit', etc.).
        teacher_lookup: Pre-built lookup for teacher embeddings.
        loaders: Dict with 'train' and 'val' DataLoaders.
        loss_fn: Distillation loss function (student_emb, teacher_emb) -> Tensor.
        optimizer: Optimizer for student + optional projection parameters.
        scheduler: Optional (scheduler, metadata) tuple.
        projection: Optional linear projection layer if dim mismatch.
        device: Device to train on.
        epochs: Number of training epochs.
        grad_clip: Optional gradient clipping value.
        mixed_precision: Whether to use mixed precision.
        log_interval: Steps between log messages.
        wandb_logger: Optional WandB logger.

    Returns:
        Dict with best_val_loss, history, best_state_dict.
    """
    scaler = _create_grad_scaler(mixed_precision)
    sched, sched_meta = scheduler or (None, {})

    best_val_loss = math.inf
    best_state = None

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        # --- Training ---
        student.train()
        if projection is not None:
            projection.train()

        running_loss, n_seen = 0.0, 0

        for step, batch in enumerate(loaders["train"], start=1):
            pixel_values, image_ids = _unpack_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=mixed_precision):
                student_emb = extract_embeddings(student, pixel_values, student_model_type)

                if projection is not None:
                    student_emb = projection(student_emb)

                teacher_emb = teacher_lookup.get_embeddings(image_ids).to(device).detach()
                loss = loss_fn(student_emb, teacher_emb)

            if mixed_precision:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        _trainable_params(student, projection), grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        _trainable_params(student, projection), grad_clip
                    )
                optimizer.step()

            if sched is not None:
                _maybe_scheduler_step(sched_meta, sched, on="batch")

            running_loss += float(loss.item()) * pixel_values.size(0)
            n_seen += pixel_values.size(0)

            if step % log_interval == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                if wandb_logger:
                    wandb_logger.log({"distill/train_loss": float(loss.item()), "lr": cur_lr})

        train_loss = running_loss / max(n_seen, 1)

        # --- Validation ---
        val_loss = _run_distillation_validation(
            student=student,
            student_model_type=student_model_type,
            teacher_lookup=teacher_lookup,
            loader=loaders["val"],
            loss_fn=loss_fn,
            projection=projection,
            device=device,
            mixed_precision=mixed_precision,
        )

        # Scheduler step
        if sched is not None:
            _maybe_scheduler_step(sched_meta, sched, on="epoch", metric=val_loss)

        cur_lr = optimizer.param_groups[0]["lr"]
        _update_history_and_log(
            history=history,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            cur_lr=cur_lr,
            wandb_logger=wandb_logger,
            log=log,
        )

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = _snapshot_state(student, projection)
            log.info(f"  New best val_loss: {best_val_loss:.6f}")

    # Restore best model
    if best_state is not None:
        student.load_state_dict(best_state["student"])
        if projection is not None and "projection" in best_state:
            projection.load_state_dict(best_state["projection"])

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "best_state": best_state,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpack_batch(batch, device) -> Tuple[torch.Tensor, list]:
    """Extract pixel_values and image_ids from a batch."""
    if isinstance(batch, dict):
        pixel_values = batch["pixel_values"].to(device)
        image_ids = batch.get("image_id", [])
    else:
        pixel_values = batch[0].to(device)
        image_ids = batch[2] if len(batch) > 2 else []

    # Normalise image_ids to a list of strings
    if isinstance(image_ids, torch.Tensor):
        image_ids = image_ids.tolist()
    image_ids = [str(x) for x in image_ids]
    return pixel_values, image_ids


def _trainable_params(student, projection):
    """Collect all trainable parameters."""
    params = list(student.parameters())
    if projection is not None:
        params += list(projection.parameters())
    return params


def _snapshot_state(student, projection):
    """CPU snapshot of model state dicts."""
    state = {"student": {k: v.cpu().clone() for k, v in student.state_dict().items()}}
    if projection is not None:
        state["projection"] = {k: v.cpu().clone() for k, v in projection.state_dict().items()}
    return state


def _run_distillation_validation(
    *,
    student: nn.Module,
    student_model_type: str,
    teacher_lookup: TeacherEmbeddingLookup,
    loader,
    loss_fn,
    projection: Optional[nn.Module],
    device: torch.device,
    mixed_precision: bool,
) -> float:
    """Compute average distillation loss on the validation set."""
    student.eval()
    if projection is not None:
        projection.eval()

    running_loss, n_seen = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            pixel_values, image_ids = _unpack_batch(batch, device)

            with autocast(device_type=device.type, enabled=mixed_precision):
                student_emb = extract_embeddings(student, pixel_values, student_model_type)
                if projection is not None:
                    student_emb = projection(student_emb)
                teacher_emb = teacher_lookup.get_embeddings(image_ids).to(device).detach()
                loss = loss_fn(student_emb, teacher_emb)

            running_loss += float(loss.item()) * pixel_values.size(0)
            n_seen += pixel_values.size(0)

    return running_loss / max(n_seen, 1)
