"""Tests for src.utils.optim — parameter grouping, optimizer/scheduler building, warmup cosine."""
import math
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.utils.optim import (
    _param_groups_decay,
    _warmup_cosine_lambda_fn,
    _build_optimizer,
    _build_scheduler,
    step_scheduler,
    make_optimizer_and_scheduler,
)


class TestParamGroupsDecay:
    def test_splits_decay_and_no_decay(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.Linear(10, 3),
        )
        groups = _param_groups_decay(model, weight_decay=0.01)
        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0
        # bias and norm params go to no_decay
        total_decay = sum(p.numel() for p in groups[0]["params"])
        total_no_decay = sum(p.numel() for p in groups[1]["params"])
        assert total_decay > 0
        assert total_no_decay > 0

    def test_frozen_params_excluded(self):
        model = nn.Linear(10, 3)
        model.weight.requires_grad = False
        groups = _param_groups_decay(model, weight_decay=0.01)
        all_params = sum(len(g["params"]) for g in groups)
        assert all_params == 1  # only bias


class TestWarmupCosine:
    def test_warmup_ramp(self):
        """During warmup, LR increases linearly."""
        fn = _warmup_cosine_lambda_fn(epochs=100, warmup_epochs=10, min_lr_ratio=0.01)
        assert fn(0) == pytest.approx(1 / 10)
        assert fn(4) == pytest.approx(5 / 10)
        assert fn(9) == pytest.approx(10 / 10)

    def test_post_warmup_starts_at_one(self):
        fn = _warmup_cosine_lambda_fn(epochs=100, warmup_epochs=10, min_lr_ratio=0.0)
        assert fn(10) == pytest.approx(1.0)

    def test_cosine_ends_at_min_lr_ratio(self):
        fn = _warmup_cosine_lambda_fn(epochs=100, warmup_epochs=0, min_lr_ratio=0.01)
        # At end of training, should approach min_lr_ratio
        val = fn(99)
        assert val == pytest.approx(0.01, abs=0.01)

    def test_no_warmup(self):
        fn = _warmup_cosine_lambda_fn(epochs=50, warmup_epochs=0, min_lr_ratio=0.0)
        assert fn(0) == pytest.approx(1.0)


class TestBuildOptimizer:
    def _make_cfg(self, name="adamw", lr=1e-3, wd=0.01, **kwargs):
        opt = {"name": name, "lr": lr, "weight_decay": wd}
        opt.update(kwargs)
        return SimpleNamespace(train={"optimizer": opt})

    def test_adamw(self):
        cfg = self._make_cfg("adamw")
        model = nn.Linear(10, 3)
        opt = _build_optimizer(cfg, model)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd(self):
        cfg = self._make_cfg("sgd")
        model = nn.Linear(10, 3)
        opt = _build_optimizer(cfg, model)
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self):
        cfg = self._make_cfg("fake_opt")
        model = nn.Linear(10, 3)
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            _build_optimizer(cfg, model)


class TestBuildScheduler:
    def _make_cfg(self, sched_name="cosine", epochs=100, warmup=10, lr=1e-3):
        return SimpleNamespace(train={
            "scheduler": {"name": sched_name, "epochs": epochs, "warmup_epochs": warmup, "min_lr": 1e-6},
            "optimizer": {"lr": lr},
            "epochs": epochs,
        })

    def test_cosine(self):
        cfg = self._make_cfg("cosine")
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        sched, meta = _build_scheduler(cfg, opt)
        assert sched is not None
        assert meta["by"] == "epoch"

    def test_step(self):
        cfg = self._make_cfg("step")
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        sched, meta = _build_scheduler(cfg, opt)
        assert meta["by"] == "epoch"

    def test_plateau(self):
        cfg = self._make_cfg("plateau")
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        sched, meta = _build_scheduler(cfg, opt)
        assert meta["by"] == "val_metric"

    def test_none(self):
        cfg = self._make_cfg("none")
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        sched, meta = _build_scheduler(cfg, opt)
        assert sched is None

    def test_unknown_raises(self):
        cfg = self._make_cfg("fake_sched")
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            _build_scheduler(cfg, opt)


class TestMakeOptimizerAndScheduler:
    def test_returns_tuple(self):
        cfg = SimpleNamespace(train={
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01},
            "scheduler": {"name": "cosine", "epochs": 50, "warmup_epochs": 5, "min_lr": 1e-6},
            "epochs": 50,
        })
        model = nn.Linear(10, 3)
        opt, (sched, meta) = make_optimizer_and_scheduler(cfg, model)
        assert isinstance(opt, torch.optim.AdamW)
        assert sched is not None


class TestStepScheduler:
    def test_none_scheduler(self):
        """Calling step_scheduler with None does nothing."""
        step_scheduler(None, {"by": "epoch"}, epoch=1)

    def test_epoch_scheduler(self):
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        step_scheduler(sched, {"by": "epoch"}, epoch=1)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_val_metric_scheduler(self):
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=0.1)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1, factor=0.5)
        step_scheduler(sched, {"by": "val_metric"}, epoch=1, val_metric=0.5)
        step_scheduler(sched, {"by": "val_metric"}, epoch=2, val_metric=0.5)
        step_scheduler(sched, {"by": "val_metric"}, epoch=3, val_metric=0.5)
        # After patience=1 with no improvement, LR should decrease
        assert opt.param_groups[0]["lr"] < 0.1
