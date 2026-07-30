"""Tests for src.engines.classification_metrics.compute_auroc_and_f1."""
import numpy as np

from src.engines.classification_metrics import compute_auroc_and_f1


class TestBinarySingleLabel:
    def test_perfect_separation(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        auroc, f1, per_class = compute_auroc_and_f1(labels, probs, multi_label=False)
        assert auroc == 1.0
        assert per_class is None  # binary has no per-class breakdown
        assert 0.0 <= f1 <= 1.0

    def test_single_class_returns_nan(self):
        labels = np.array([1, 1, 1])
        probs = np.array([[0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])
        auroc, _f1, _ = compute_auroc_and_f1(labels, probs, multi_label=False)
        assert np.isnan(auroc)


class TestMultiClass:
    def test_per_class_dict_populated(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.eye(3)[labels] * 0.7 + 0.1  # confident-ish correct predictions
        auroc, f1, per_class = compute_auroc_and_f1(
            labels, probs, multi_label=False, label_names=["a", "b", "c"]
        )
        assert per_class is not None
        assert set(per_class.keys()) == {"a", "b", "c"}
        assert not np.isnan(auroc)


class TestMultiLabel:
    def test_masks_uncertain_and_averages(self):
        # 2 labels; label 1 has an uncertain (-1) entry that must be ignored.
        labels = np.array([[1, 0], [0, 1], [1, -1], [0, 0]])
        probs = np.array([[0.9, 0.2], [0.1, 0.8], [0.85, 0.5], [0.2, 0.1]])
        auroc, f1, per_class = compute_auroc_and_f1(
            labels, probs, multi_label=True, label_names=["x", "y"]
        )
        assert per_class is not None
        assert "x" in per_class
        assert not np.isnan(auroc)

    def test_all_uncertain_label_skipped(self):
        labels = np.array([[1, -1], [0, -1], [1, -1]])
        probs = np.array([[0.8, 0.5], [0.2, 0.5], [0.7, 0.5]])
        auroc, _f1, per_class = compute_auroc_and_f1(labels, probs, multi_label=True)
        # Only label 0 is scorable; label 1 fully masked.
        assert "0" in per_class
        assert "1" not in per_class
