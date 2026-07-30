"""Tests for the global case-level TCGA split (src.data.tcga_split)."""
import numpy as np

from src.data.tcga_split import global_train_case_ids, positional_split


def _universe(n_cases=50, slides_per_case=3):
    """Full universe: n_cases patients, each with several slides. Returns the
    per-row (slide) case-id list."""
    return [f"case_{c:03d}" for c in range(n_cases) for _ in range(slides_per_case)]


class TestGlobalTrainCaseIds:
    def test_deterministic(self):
        u = _universe()
        assert global_train_case_ids(u, seed=42) == global_train_case_ids(u, seed=42)

    def test_ratio_on_cases(self):
        u = _universe(n_cases=100, slides_per_case=2)
        train = global_train_case_ids(u, seed=42, train_ratio=0.8)
        assert abs(len(train) - 80) <= 1          # ~80 of 100 cases
        assert train != global_train_case_ids(u, seed=123, train_ratio=0.8)  # seed matters

    def test_empty(self):
        assert global_train_case_ids([], seed=42) == set()


class TestPositionalSplit:
    def test_leakage_safe_and_partition(self):
        u = _universe(n_cases=40, slides_per_case=4)
        train_cases = global_train_case_ids(u, seed=42)
        tr, te = positional_split(u, train_cases)
        assert len(tr) + len(te) == len(u)
        assert set(tr).isdisjoint(set(te))
        # every slide of a case lands on one side only
        tr_cases = {u[i] for i in tr}; te_cases = {u[i] for i in te}
        assert tr_cases.isdisjoint(te_cases)

    def test_same_partition_across_task_subsets(self):
        """A case's train/test membership is identical whether the consumer sees
        the full universe or only a task-specific subset of it."""
        u = _universe(n_cases=60, slides_per_case=3)
        train_cases = global_train_case_ids(u, seed=42)   # computed over FULL universe

        # Task A: a subset of cases (e.g. lung cohort)
        task_a = [c for c in u if int(c.split("_")[1]) < 30]
        # Task B: a different, overlapping subset
        task_b = [c for c in u if int(c.split("_")[1]) >= 20]

        a_tr, _ = positional_split(task_a, train_cases)
        b_tr, _ = positional_split(task_b, train_cases)
        a_train_cases = {task_a[i] for i in a_tr}
        b_train_cases = {task_b[i] for i in b_tr}
        # overlap cases (20..29) must have the SAME train/test side in both tasks
        overlap = {c for c in task_a if c in task_b}
        for c in overlap:
            in_a_train = c in a_train_cases
            in_b_train = c in b_train_cases
            assert in_a_train == in_b_train, f"{c} inconsistent across tasks"
