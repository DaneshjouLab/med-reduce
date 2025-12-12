# src/engines/linear_probe_engine.py
# -*- coding: utf-8 -*-
"""Linear probing engine for training classification heads on frozen backbones."""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import math
import torch
from torch import nn
from torch.utils.data import DataLoader

# pylint: disable=import-error
from src.utils.logging import get_logger, MetricAverager, WandbLogger
from src.utils.optim import step_scheduler

log = get_logger(__name__)


def _unpack_loaders(loaders: Any) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Accepts either:
      - {"train": ..., "val": ..., "test": ...}
      - (train_loader, val_loader)
    Returns (train, val, test)
    """
    if isinstance(loaders, dict):
        return loaders.get("train"), loaders.get("val"), loaders.get("test")
    if isinstance(loaders, (tuple, list)) and len(loaders) >= 2:
        return loaders[0], loaders[1], loaders[2] if len(loaders) > 2 else None
    raise ValueError(
        "`loaders` must be a dict with keys train/val[/test] or a (train,val) tuple."
    )


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    # if the model exposes a criterion attribute, prefer the external loss
    # passed in train loop anyway.
    ce = nn.CrossEntropyLoss()

    for batch in loader:
        # support (x,y) or dict
        if isinstance(batch, dict):
            x = (batch.get("pixel_values") or
                 batch.get("images") or
                 batch.get("x"))
            y = batch.get("labels") or batch.get("y")
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        loss = ce(logits, y)
        running_loss += float(loss.item()) * x.size(0)

        preds = logits.argmax(dim=-1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    acc = correct / max(total, 1)
    avg_loss = running_loss / max(total, 1)
    return {"val_loss": avg_loss, "val_acc": acc}


def train_probe(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
    model: nn.Module,
    loaders: Any,
    loss_fn,                                 # callable: (logits, targets) -> loss tensor
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,                   # either a scheduler or (scheduler, meta)
    device: torch.device = torch.device("cpu"),
    epochs: int = 10,
    grad_clip: Optional[float] = None,
    mixed_precision: bool = True,
    log_interval: int = 50,
    wandb_logger: Optional[WandbLogger] = None,
    metric_key: str = "val_acc",
) -> Dict[str, Any]:
    """
    Linear probing engine.
    Expects the backbone to be frozen already; only the head should have requires_grad=True.
    """
    train_loader, val_loader, _ = _unpack_loaders(loaders)
    assert train_loader is not None, "train_loader is required"
    assert val_loader is not None, "val_loader is required"

    scaler = torch.amp.GradScaler(enabled=(mixed_precision and device.type == "cuda"))

    # unwrap (scheduler, meta) if it came from our utils
    sched, sched_meta = None, {"by": "epoch"}
    if scheduler is not None:
        if isinstance(scheduler, tuple) and len(scheduler) == 2:
            sched, sched_meta = scheduler[0], scheduler[1]
        else:
            sched = scheduler

    # sanity: log trainable parameter ratio
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[probe] trainable params: {trainable_params:,} / {total_params:,} "
             f"({100.0 * trainable_params / max(total_params,1):.2f}%)")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_metric = -math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        ma = MetricAverager()

        for it, batch in enumerate(train_loader, start=1):
            if isinstance(batch, dict):
                x = (batch.get("pixel_values") or
                     batch.get("images") or
                     batch.get("x"))
                y = batch.get("labels") or batch.get("y")
            else:
                x, y = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.amp.autocast(device_type=device.type):
                    logits = model(x)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        max_norm=grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = loss_fn(logits, y)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        max_norm=grad_clip
                    )
                optimizer.step()

            # running metrics
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()
            ma.update(loss=float(loss.item()), acc=acc)

            if it % log_interval == 0:
                avgs = ma.averages()
                log.info(f"[probe] epoch {epoch:03d} iter {it:05d} "
                         f"| loss {avgs['loss']:.4f} | acc {avgs['acc']:.4f}")
                if wandb_logger:
                    wandb_logger.log({
                        "train/loss": avgs["loss"],
                        "train/acc": avgs["acc"],
                        "epoch": epoch,
                        "iter": it,
                    })

        # end epoch → validation
        val_metrics = _evaluate(model, val_loader, device=device)
        history["train_loss"].append(ma.averages()["loss"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])

        log.info(f"[probe] epoch {epoch:03d} | val_loss {val_metrics['val_loss']:.4f} "
                 f"| val_acc {val_metrics['val_acc']:.4f}")

        if wandb_logger:
            wandb_logger.log({
                "val/loss": val_metrics["val_loss"],
                "val/acc": val_metrics["val_acc"],
                "epoch": epoch,
            })

        # scheduler step
        if sched is not None:
            # if ReduceLROnPlateau, step on val metric; else per-epoch
            if hasattr(sched, "step") and sched_meta.get("by") == "val_metric":
                # for plateau: lower is better typically; if you want higher-better, pass negative
                step_scheduler(sched, sched_meta, epoch=epoch, val_metric=val_metrics["val_loss"])
            else:
                step_scheduler(sched, sched_meta, epoch=epoch)

        # track best
        current = val_metrics.get(metric_key, val_metrics["val_acc"])
        if current > best_metric:
            best_metric = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best weights (optional but standard)
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_metric": float(best_metric),
        "best_metric_name": metric_key,
        "final_val_acc": float(history["val_acc"][-1]) if history["val_acc"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
