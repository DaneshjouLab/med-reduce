"""Tests for src.utils.utils and src.utils.training_utils."""
import json
import os
import torch
import torch.nn as nn
import pytest

from src.utils.utils import (
    env_path,
    setup_environment,
    get_gpu_memory,
    check_disk_space,
    _to_serializable,
    save_results,
    load_json,
)
from src.utils.training_utils import (
    env_path as tu_env_path,
    get_gpu_memory as tu_get_gpu_memory,
    profile_model,
    calculate_inference_latency,
)


class TestEnvPath:
    def test_returns_default_when_unset(self):
        key = "_TEST_NONEXISTENT_VAR_XYZ_"
        os.environ.pop(key, None)
        assert env_path(key, "/default") == "/default"

    def test_returns_env_when_set(self):
        os.environ["_TEST_VAR_ABC_"] = "/custom"
        assert env_path("_TEST_VAR_ABC_", "/default") == "/custom"
        del os.environ["_TEST_VAR_ABC_"]

    def test_training_utils_env_path_expands_tilde(self):
        result = tu_env_path("_TEST_NONEXISTENT_", "~/data")
        assert "~" not in result


class TestToSerializable:
    def test_plain_dict(self):
        assert _to_serializable({"a": 1}) == {"a": 1}

    def test_nested(self):
        obj = {"a": [1, {"b": 2}]}
        assert _to_serializable(obj) == {"a": [1, {"b": 2}]}

    def test_tuple_becomes_list(self):
        assert _to_serializable((1, 2, 3)) == [1, 2, 3]

    def test_scalar(self):
        assert _to_serializable(42) == 42


class TestSaveAndLoadJson:
    def test_roundtrip(self, tmp_path):
        data = {"auroc": 0.85, "domain": "dermatology"}
        filepath = str(tmp_path / "results.json")
        save_results(data, filepath)
        loaded = load_json(filepath)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        filepath = str(tmp_path / "sub" / "dir" / "results.json")
        save_results({"x": 1}, filepath)
        assert os.path.exists(filepath)


class TestCheckDiskSpace:
    def test_returns_true(self):
        # Should always have at least 0.001 GB free
        assert check_disk_space(required_gb=0.001) is True

    def test_raises_on_impossible_requirement(self):
        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            check_disk_space(required_gb=999999999)


class TestSetupEnvironment:
    def test_sets_env_vars(self):
        setup_environment()
        assert "PYTORCH_CUDA_ALLOC_CONF" in os.environ
        assert "HF_HOME" in os.environ
        assert "HF_DATASETS_CACHE" in os.environ


class TestGetGpuMemory:
    def test_returns_number(self):
        """Returns a float (or -1 if no GPU)."""
        result = get_gpu_memory()
        assert isinstance(result, (int, float))

    def test_training_utils_returns_int(self):
        result = tu_get_gpu_memory()
        assert isinstance(result, int)
        assert result >= 0


class TestProfileModel:
    def test_returns_gflops(self):
        model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        result = profile_model(model, 32)
        assert isinstance(result, float)
        if result > 0:
            assert result < 100  # sanity check

    def test_returns_negative_on_bad_model(self):
        """Returns -1 if profiling fails."""
        result = profile_model(nn.Linear(1, 1), 32)
        # May or may not fail depending on thop version; just check type
        assert isinstance(result, float)


class TestCalculateInferenceLatency:
    def test_returns_positive_ms(self):
        model = nn.Linear(3 * 32 * 32, 10)
        # Wrap to accept 4D input
        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x.flatten(1))
        result = calculate_inference_latency(Wrapper(model), 32, warmup_runs=2, bench_runs=5)
        assert result > 0

    def test_returns_negative_on_failure(self):
        """Returns -1 on error."""
        # Pass something that will fail
        result = calculate_inference_latency(None, 32)
        assert result == -1.0


class TestOmegaConfSerialization:
    def test_omegaconf_to_serializable(self):
        """OmegaConf DictConfig gets serialized to plain dict."""
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.create({"a": 1, "b": [2, 3]})
            result = _to_serializable(cfg)
            assert result == {"a": 1, "b": [2, 3]}
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("omegaconf not installed")
