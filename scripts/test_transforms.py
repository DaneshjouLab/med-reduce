#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal check: dataset loads and a degradation transform is applied to 5 images.
- Uses your BaseDataModule and get_degradation_transforms()
- No training, no W&B.
- Saves clean and degraded PNGs.
"""

from types import SimpleNamespace
from pathlib import Path
import argparse

from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image

from src.data.datamodule import BaseDataModule
from src.transformation.transforms import get_degradation_transforms
from src.utils import setup_environment


def main():
    ap = argparse.ArgumentParser(description="Quick dataset+transform smoke test (5 images).")
    ap.add_argument("--dataset", type=str, required=True, choices=["isic", "tcga", "merlin"])
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/test_transforms")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    setup_environment()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- minimal cfg your DataModule/get_dataset can read ----
    cfg = SimpleNamespace(
        resolution=args.resolution,
        batch_size=5,           # just enough to grab 5 samples
    )

    # ---- build the DataModule exactly as your code expects ----
    dm = BaseDataModule(
        cfg=cfg,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        batch_size=cfg.batch_size,
        pin_memory=True,
        drop_last=False,
    )
    dm.setup(stage="fit")

    # ---- pull one small batch (5 images) from val ----
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))

    if isinstance(batch, dict):
        x = batch.get("pixel_values") or batch.get("images") or batch.get("x")
        y = batch.get("labels") or batch.get("y")
    else:
        x, y = batch

    # keep exactly 5
    x = x[:5]

    for i in range(x.size(0)):
        save_image(x[i], out_root / f"clean_{i}.png")

    degradations = get_degradation_transforms()
    if not degradations:
        print("No degradations returned by get_degradation_transforms(); saved only clean images.")
        return

    degr = degradations[0]
    tag = degr.__class__.__name__.lower()
    print(f"Applying degradation: {tag}")

    for i in range(x.size(0)):
        pil_img = TF.to_pil_image(x[i].cpu())
        pil_out = degr(pil_img)
        t_out = TF.to_tensor(pil_out)
        save_image(t_out, out_root / f"{tag}_{i}.png")

    print(f"✅ Done. Wrote images to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
