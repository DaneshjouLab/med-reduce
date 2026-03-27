"""Tests for src.utils.reproducibility."""
import json
import random

import numpy as np
import torch
import pytest

from src.utils.reproducibility import (
    seed_everything,
    get_worker_init_fn,
    SeedTracker,
    verify_reproducibility_settings,
)


class TestSeedEverything:
    def test_deterministic_random(self):
        """Same seed produces same random numbers."""
        seed_everything(42)
        a = random.random()
        seed_everything(42)
        b = random.random()
        assert a == b

    def test_deterministic_numpy(self):
        seed_everything(42)
        a = np.random.rand(5)
        seed_everything(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_torch(self):
        seed_everything(42)
        a = torch.randn(5)
        seed_everything(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        seed_everything(42)
        a = torch.randn(5)
        seed_everything(123)
        b = torch.randn(5)
        assert not torch.equal(a, b)


class TestWorkerInitFn:
    def test_returns_callable(self):
        fn = get_worker_init_fn(42)
        assert callable(fn)

    def test_different_workers_different_seeds(self):
        fn = get_worker_init_fn(42)
        fn(0)
        a = np.random.rand(3)
        fn(1)
        b = np.random.rand(3)
        assert not np.array_equal(a, b)

    def test_same_worker_id_reproducible(self):
        fn = get_worker_init_fn(42)
        fn(0)
        a = np.random.rand(3)
        fn(0)
        b = np.random.rand(3)
        np.testing.assert_array_equal(a, b)


class TestSeedTracker:
    def test_log_and_summary(self):
        tracker = SeedTracker(42)
        tracker.log_seed("dataloader", 42, {"workers": 8})
        tracker.log_seed("split_manager", 42)
        summary = tracker.get_summary()
        assert "Base Seed: 42" in summary
        assert "dataloader" in summary
        assert "split_manager" in summary

    def test_save_json(self, tmp_path):
        tracker = SeedTracker(123)
        tracker.log_seed("model", 123)
        path = tmp_path / "seeds.json"
        tracker.save(path)
        data = json.loads(path.read_text())
        assert data["base_seed"] == 123
        assert "model" in data["components"]


class TestVerifyReproducibility:
    def test_returns_dict(self):
        settings = verify_reproducibility_settings()
        assert isinstance(settings, dict)
        assert "torch_version" in settings
        assert "numpy_version" in settings


class TestLogReproducibilityInfo:
    def test_returns_settings(self):
        from src.utils.reproducibility import log_reproducibility_info
        result = log_reproducibility_info()
        assert isinstance(result, dict)
        assert "torch_version" in result

    def test_saves_to_file(self, tmp_path):
        from src.utils.reproducibility import log_reproducibility_info
        result = log_reproducibility_info(output_dir=tmp_path)
        settings_file = tmp_path / "reproducibility_settings.json"
        assert settings_file.exists()
        data = json.loads(settings_file.read_text())
        assert "torch_version" in data
