"""Tests for src.losses — distillation and classification losses."""
import pytest
import torch

from src.losses.distillation import embedding_distillation_loss
from src.losses.classification import cross_entropy_loss, bce_with_logits_loss


class TestEmbeddingDistillationLoss:
    def test_identical_embeddings_give_zero(self):
        """Loss is zero when student == teacher."""
        loss_fn = embedding_distillation_loss(alpha=0.5)
        emb = torch.randn(4, 128)
        loss = loss_fn(emb, emb.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_is_positive_for_different_embeddings(self):
        loss_fn = embedding_distillation_loss(alpha=0.5)
        loss = loss_fn(torch.randn(4, 128), torch.randn(4, 128))
        assert loss.item() > 0

    def test_alpha_one_is_pure_mse(self):
        """alpha=1 means only MSE, no cosine."""
        loss_fn = embedding_distillation_loss(alpha=1.0)
        s, t = torch.randn(4, 64), torch.randn(4, 64)
        loss = loss_fn(s, t)
        expected = torch.nn.functional.mse_loss(s, t)
        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_alpha_zero_is_pure_cosine(self):
        """alpha=0 means only cosine loss."""
        loss_fn = embedding_distillation_loss(alpha=0.0)
        s, t = torch.randn(4, 64), torch.randn(4, 64)
        loss = loss_fn(s, t)
        expected = (1 - torch.nn.functional.cosine_similarity(s, t)).mean()
        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_output_is_scalar(self):
        loss_fn = embedding_distillation_loss(alpha=0.7)
        loss = loss_fn(torch.randn(8, 256), torch.randn(8, 256))
        assert loss.dim() == 0


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
