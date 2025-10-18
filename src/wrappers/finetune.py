# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

from typing import Dict, Any
import torch

from src.engines.finetune_engine import train_finetune
from src.data.datamodule import build_datamodule
from src.models.factory import build_classifier
from src.losses.classification import cross_entropy_loss
from src.utils.optim import make_optimizer_and_scheduler
from src.utils.logging import get_logger # TODO

log = get_logger(__name__)

def run(cfg) -> Dict[str, Any]:
    device = torch.device(cfg.runtime["device"])
    dm = build_datamodule(cfg)
    model = build_classifier(cfg).to(device)         # backbone + classification head

    # Unfreeze all params for full finetuning
    if hasattr(model, "unfreeze_all"):
        model.unfreeze_all()

    loss_fn = cross_entropy_loss()

    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model.parameters())

    log.info("Starting end-to-end finetuning...")
    metrics = train_finetune(
        model=model,
        loaders=dm.loaders(),
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg.train.epochs,
        grad_clip=getattr(cfg.train, "grad_clip_norm", None),
        mixed_precision=getattr(cfg.train, "amp", True),
    )
    return metrics
