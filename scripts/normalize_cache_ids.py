#!/usr/bin/env python3
"""Strip image extensions from image_ids in existing LP embedding caches.

The distillation ImageOnlyDataset keys teacher embeddings by filename STEM
(basename, no extension). CheXpert's CSV image_id keeps a ".jpg"
(e.g. patient11938_study1_view1_frontal.jpg), so reuse-by-id misses. This rewrites
the cached image_ids to stems in place. No-op for ISIC/TCGA (ids have no
extension), and idempotent (running twice changes nothing).

Usage:
  python scripts/normalize_cache_ids.py <cache_dir> [<cache_dir> ...]
e.g.
  python scripts/normalize_cache_ids.py \
    /scratch/users/$USER/med-reduce-dinov3/radiology/cache/embeddings
"""
import glob
import os
import sys

import torch


def _stem(i) -> str:
    return os.path.splitext(os.path.basename(str(i)))[0]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: normalize_cache_ids.py <cache_dir> [<cache_dir> ...]")

    files = []
    for root in sys.argv[1:]:
        files += glob.glob(os.path.join(root, "**", "*_embeddings.pt"), recursive=True)
    if not files:
        sys.exit(f"No *_embeddings.pt found under: {sys.argv[1:]}")

    changed = 0
    for p in sorted(files):
        obj = torch.load(p, map_location="cpu", weights_only=False)
        ids = obj.get("image_ids")
        if ids is None:
            print(f"[skip] no image_ids: {p}")
            continue
        new = [_stem(i) for i in ids]
        if new != [str(i) for i in ids]:
            obj["image_ids"] = new
            torch.save(obj, p)
            changed += 1
            print(f"[fixed] {p} ({len(new)} ids)")
        else:
            print(f"[ok]    already stems: {p}")
    print(f"Done: {changed}/{len(files)} files updated.")


if __name__ == "__main__":
    main()
