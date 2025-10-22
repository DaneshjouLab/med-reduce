# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""ISIC dataset loader implementation for dermatology image datasets."""
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
        # Handle Subset wrapping
        base = self.dataset
        if hasattr(base, "dataset") and hasattr(base, "indices"):
            # Subset case
            item = base.dataset[int(base.indices[idx])]
        else:
            item = base[idx]

        # no resize, no cast, no transforms
        image = item["image"]
        label = item["label"]

        try:
            label = int(label)
        except Exception as exc:
            raise TypeError("Label must be convertible to int.") from exc

        return {"image": image, "label": label}
