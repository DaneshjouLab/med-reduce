# src/data/isic_loader.py
from typing import Any, Dict, Union
from torch.utils.data import Dataset, Subset

class ISICBaseDataset(Dataset):
    """
    Minimal, transformation-free wrapper for ISIC (or ISIC-like) datasets.

    Expects the backing dataset (or Subset) to yield items with:
        item["image"] : PIL.Image (or array/tensor if your source uses that)
        item["label"] : int-like

    Returns each sample unchanged:
        {"image": <original image>, "label": <int>}
    """

    def __init__(self, dataset: Union[Dataset, Subset]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Handle Subset wrapping transparently
        base = self.dataset
        if hasattr(base, "dataset") and hasattr(base, "indices"):
            # Subset case
            item = base.dataset[int(base.indices[idx])]
        else:
            item = base[idx]

        # Do NOT touch the image (no resize, no cast, no transforms)
        image = item["image"]
        label = item["label"]

        # Make sure label is int-like, but don't coerce image
        try:
            label = int(label)
        except Exception:
            raise TypeError("Label must be convertible to int.")

        return {"image": image, "label": label}
