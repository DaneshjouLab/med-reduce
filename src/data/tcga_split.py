# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University
# SPDX-License-Identifier: MIT

"""Global, leakage-safe case-level train/test split for TCGA.

All pathology tasks (LUAD/LUSC, LGG/GBM, KRAS, TP53, EGFR, IDH) and the
distillation share a single patient/case-level partition: every slide of a given
``case_id`` is assigned wholly to train or wholly to test, and the assignment is
computed over the *full* case universe (independent of which task/subset consumes
it). This guarantees:

* no patient leakage across train/test, and
* the *same* partition across all tasks — a case in "test" for one task is in
  "test" for every task and for the distillation.

The assignment is deterministic (seeded shuffle of the sorted unique case ids), so
it is reproducible and identical across runs, models, and codebases that call it
with the same case universe + seed.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def global_train_case_ids(
    all_case_ids: Iterable, seed: int, train_ratio: float = 0.8
) -> set:
    """Case ids assigned to TRAIN, computed over the full case universe.

    Args:
        all_case_ids: every case id in the dataset (the full universe; NOT a
            task-filtered subset), duplicates allowed.
        seed: split seed.
        train_ratio: fraction of *cases* (not slides) in train.

    Returns:
        Set of case ids (as ``str``) in the train partition. The complement is
        the test partition.
    """
    uniq = sorted({str(c) for c in all_case_ids})
    if not uniq:
        return set()
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(len(uniq))
    n_train = int(round(len(uniq) * float(train_ratio)))
    return {uniq[i] for i in perm[:n_train]}


def positional_split(
    row_case_ids: Iterable, train_cases: set
) -> Tuple[np.ndarray, np.ndarray]:
    """Positional train/test indices for a dataset whose rows carry ``row_case_ids``.

    A row goes to train iff its case id is in ``train_cases``. Leakage-safe by
    construction; asserts train/test case sets are disjoint.
    """
    rows = [str(c) for c in row_case_ids]
    train_idx = np.array([i for i, c in enumerate(rows) if c in train_cases], dtype=np.int64)
    test_idx = np.array([i for i, c in enumerate(rows) if c not in train_cases], dtype=np.int64)

    train_case_set = {rows[i] for i in train_idx}
    test_case_set = {rows[i] for i in test_idx}
    assert train_case_set.isdisjoint(test_case_set), (
        "Case-level leakage detected: a case appears in both train and test."
    )
    return train_idx, test_idx
