"""Tests for src.losses — classification losses."""
import pytest
import torch

from src.losses.classification import cross_entropy_loss, bce_with_logits_loss


class TestCrossEntropyLoss:
    def test_basic_ce(self):
        loss_fn = cross_entropy_loss()
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_label_smoothing(self):
        loss_no_smooth = cross_entropy_loss(label_smoothing=0.0)
        loss_smooth = cross_entropy_loss(label_smoothing=0.1)
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        l1 = loss_no_smooth(logits, targets)
        l2 = loss_smooth(logits, targets)
        # Smoothed loss is typically different (and often smaller gap to perfect)
        assert l1.item() != pytest.approx(l2.item(), abs=1e-4)

    def test_class_weights(self):
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = cross_entropy_loss(class_weight=weights)
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0


class TestBCEWithLogitsLoss:
    def test_basic_bce(self):
        loss_fn = bce_with_logits_loss()
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 2, (4, 5)).float()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_pos_weight(self):
        pw = torch.ones(5) * 2.0
        loss_fn = bce_with_logits_loss(pos_weight=pw)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 2, (4, 5)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_ignores_negative_one_labels(self):
        """Labels of -1 (uncertain) are masked out and don't produce NaN."""
        loss_fn = bce_with_logits_loss()
        logits = torch.randn(4, 5)
        targets = torch.zeros(4, 5)
        targets[0, 0] = -1.0  # uncertain
        targets[1, 2] = -1.0
        targets[2, :] = -1.0  # entire row uncertain
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss), "Loss should not be NaN with -1 labels"
        assert loss.item() > 0

    def test_all_uncertain_returns_zero(self):
        """All -1 labels should give zero loss (no valid entries)."""
        loss_fn = bce_with_logits_loss()
        logits = torch.randn(4, 5)
        targets = torch.full((4, 5), -1.0)
        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0

    def test_reduction_sum(self):
        loss_fn = bce_with_logits_loss(reduction="sum")
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 2, (4, 5)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_reduction_none(self):
        loss_fn = bce_with_logits_loss(reduction="none")
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 2, (4, 5)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == (4, 5)
