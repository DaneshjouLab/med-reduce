"""Tests for src.utils.checkpoint_utils — extract_state_dict, find_best_checkpoint, load_checkpoint, load_model_from_checkpoint, ensemble_predict."""
import pytest
import torch
import torch.nn as nn
import timm
from collections import OrderedDict
from pathlib import Path

from src.utils.checkpoint_utils import extract_state_dict, find_best_checkpoint, load_checkpoint, load_model_from_checkpoint, ensemble_predict


# ---------------------------------------------------------------------------
# extract_state_dict
# ---------------------------------------------------------------------------

class TestExtractStateDict:
    """Tests for the extract_state_dict function covering all checkpoint formats."""

    def test_raw_state_dict(self):
        """Raw OrderedDict of tensors is returned as-is."""
        sd = OrderedDict({"conv1.weight": torch.randn(3, 3), "bn1.weight": torch.randn(3)})
        result = extract_state_dict(sd)
        assert result is sd

    def test_lightning_no_prefix(self):
        """Lightning checkpoint with no prefix on state_dict keys."""
        ckpt = {
            "epoch": 10,
            "state_dict": {"conv1.weight": torch.randn(3, 3), "bn1.weight": torch.randn(3)},
        }
        result = extract_state_dict(ckpt)
        assert "conv1.weight" in result
        assert "bn1.weight" in result

    def test_lightning_model_prefix(self):
        """Lightning checkpoint with 'model.' prefix gets stripped."""
        ckpt = {
            "epoch": 5,
            "state_dict": {
                "model.conv1.weight": torch.randn(3, 3),
                "model.bn1.weight": torch.randn(3),
            },
        }
        result = extract_state_dict(ckpt)
        assert "conv1.weight" in result
        assert "bn1.weight" in result
        assert not any(k.startswith("model.") for k in result)

    def test_lightning_backbone_model_prefix(self):
        """Lightning checkpoint with 'backbone.model.' prefix gets stripped."""
        ckpt = {
            "epoch": 5,
            "state_dict": {
                "backbone.model.conv1.weight": torch.randn(3, 3),
                "backbone.model.bn1.weight": torch.randn(3),
            },
        }
        result = extract_state_dict(ckpt)
        assert "conv1.weight" in result
        assert "bn1.weight" in result

    def test_lightning_model_backbone_model_prefix(self):
        """Lightning checkpoint with 'model.backbone.model.' prefix (3 levels) gets stripped."""
        ckpt = {
            "epoch": 5,
            "state_dict": {
                "model.backbone.model.conv1.weight": torch.randn(3, 3),
                "model.backbone.model.bn1.weight": torch.randn(3),
            },
        }
        result = extract_state_dict(ckpt)
        assert "conv1.weight" in result
        assert "bn1.weight" in result
        assert not any("backbone" in k for k in result)

    def test_lightning_backbone_prefix(self):
        """Lightning checkpoint with 'backbone.' prefix gets stripped."""
        ckpt = {
            "epoch": 5,
            "state_dict": {
                "backbone.conv1.weight": torch.randn(3, 3),
                "backbone.bn1.weight": torch.randn(3),
            },
        }
        result = extract_state_dict(ckpt)
        assert "conv1.weight" in result

    def test_student_state_dict(self):
        """Distillation checkpoint with 'student_state_dict' key."""
        student_sd = {"conv1.weight": torch.randn(3, 3), "bn1.weight": torch.randn(3)}
        ckpt = {
            "student_state_dict": student_sd,
            "projection_state_dict": {"linear.weight": torch.randn(384, 512)},
            "alpha": 0.7,
        }
        result = extract_state_dict(ckpt)
        assert result is student_sd

    def test_model_state_dict(self):
        """Probe/CV checkpoint with 'model_state_dict' key."""
        model_sd = {"fc.weight": torch.randn(3, 512), "fc.bias": torch.randn(3)}
        ckpt = {"model_state_dict": model_sd, "fold": 0, "metric": 0.85}
        result = extract_state_dict(ckpt)
        assert result is model_sd

    def test_fallback_returns_input(self):
        """Non-dict input is returned as-is (fallback)."""
        obj = "unexpected"
        assert extract_state_dict(obj) == obj

    def test_prefix_priority_longest_first(self):
        """The longest matching prefix is stripped, not a shorter one."""
        ckpt = {
            "epoch": 1,
            "state_dict": {
                "model.backbone.model.layer1.weight": torch.randn(2, 2),
            },
        }
        result = extract_state_dict(ckpt)
        assert "layer1.weight" in result

    def test_mixed_keys_with_and_without_prefix(self):
        """When some keys have a prefix and some don't, only prefixed keys get stripped."""
        ckpt = {
            "epoch": 1,
            "state_dict": {
                "model.conv1.weight": torch.randn(3, 3),
                "extra_param": torch.randn(1),  # no 'model.' prefix
            },
        }
        result = extract_state_dict(ckpt)
        # 'model.' is stripped from matching keys; non-matching keys keep their name
        assert "conv1.weight" in result
        assert "extra_param" in result


# ---------------------------------------------------------------------------
# find_best_checkpoint
# ---------------------------------------------------------------------------

class TestFindBestCheckpoint:
    """Tests for find_best_checkpoint."""

    def test_finds_best_accuracy(self, tmp_path):
        """Selects checkpoint with highest metric for accuracy-like keys."""
        (tmp_path / "model_metric0.80.pt").touch()
        (tmp_path / "model_metric0.92.pt").touch()
        (tmp_path / "model_metric0.85.pt").touch()
        best = find_best_checkpoint(tmp_path, metric_key="val_acc")
        assert best.name == "model_metric0.92.pt"

    def test_finds_best_loss(self, tmp_path):
        """Selects checkpoint with lowest metric for loss keys."""
        (tmp_path / "model_metric0.50.pt").touch()
        (tmp_path / "model_metric0.30.pt").touch()
        (tmp_path / "model_metric0.45.pt").touch()
        best = find_best_checkpoint(tmp_path, metric_key="val_loss")
        assert best.name == "model_metric0.30.pt"

    def test_empty_dir_raises(self, tmp_path):
        """Raises FileNotFoundError when no checkpoints exist."""
        with pytest.raises(FileNotFoundError):
            find_best_checkpoint(tmp_path)

    def test_no_parseable_metric_raises(self, tmp_path):
        """Raises ValueError when filenames don't contain metrics."""
        (tmp_path / "model_best.pt").touch()
        with pytest.raises(ValueError):
            find_best_checkpoint(tmp_path)

    def test_handles_ckpt_extension(self, tmp_path):
        """Works with .ckpt files too."""
        (tmp_path / "epoch10_metric0.75.ckpt").touch()
        (tmp_path / "epoch20_metric0.88.ckpt").touch()
        best = find_best_checkpoint(tmp_path, metric_key="val_auroc")
        assert best.name == "epoch20_metric0.88.ckpt"


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    def test_loads_pt_file(self, tmp_path):
        """Loads a .pt checkpoint file."""
        ckpt = {"model_state_dict": {"w": torch.randn(3, 3)}, "epoch": 5}
        path = tmp_path / "model.pt"
        torch.save(ckpt, path)
        loaded = load_checkpoint(path)
        assert loaded["epoch"] == 5
        assert "model_state_dict" in loaded

    def test_map_location(self, tmp_path):
        """Respects map_location argument."""
        ckpt = {"w": torch.randn(3)}
        path = tmp_path / "model.pt"
        torch.save(ckpt, path)
        loaded = load_checkpoint(path, map_location="cpu")
        assert loaded["w"].device == torch.device("cpu")

    def test_device_arg(self, tmp_path):
        """device arg is converted to map_location."""
        ckpt = {"w": torch.randn(3)}
        path = tmp_path / "model.pt"
        torch.save(ckpt, path)
        loaded = load_checkpoint(path, device=torch.device("cpu"))
        assert loaded["w"].device == torch.device("cpu")


# ---------------------------------------------------------------------------
# ensemble_predict
# ---------------------------------------------------------------------------

class TestEnsemblePredict:
    def test_averages_logits(self):
        """Ensemble averages logits from multiple models."""
        model1 = nn.Linear(10, 3)
        model2 = nn.Linear(10, 3)
        inputs = torch.randn(4, 10)
        result = ensemble_predict([model1, model2], inputs, device=torch.device("cpu"))
        assert result.shape == (4, 3)

    def test_averages_probabilities(self):
        """With average_logits=False, averages softmax probabilities."""
        model1 = nn.Linear(10, 3)
        model2 = nn.Linear(10, 3)
        inputs = torch.randn(4, 10)
        result = ensemble_predict([model1, model2], inputs, device=torch.device("cpu"), average_logits=False)
        assert result.shape == (4, 3)
        # Probabilities should sum to ~1
        assert result.sum(dim=1).allclose(torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# load_model_from_checkpoint
# ---------------------------------------------------------------------------

class TestLoadModelFromCheckpoint:
    def test_loads_timm_checkpoint(self, tmp_path):
        """Saves and reloads a timm model checkpoint."""
        model = timm.create_model("resnet18", pretrained=False, num_classes=3)
        sd = model.state_dict()
        ckpt = {
            "model_config": {
                "type": "timm",
                "model_id": "resnet18",
                "config": {"num_labels": 3, "pretrained": False},
            },
            "model_state_dict": sd,
        }
        path = tmp_path / "model.pt"
        torch.save(ckpt, path)
        loaded = load_model_from_checkpoint(path, device=torch.device("cpu"))
        assert isinstance(loaded, nn.Module)
        # Verify weights match
        for k in sd:
            assert torch.equal(loaded.state_dict()[k], sd[k])

    def test_loads_with_cfg_image_size(self, tmp_path):
        """Respects cfg.data.image_size for resolution."""
        from types import SimpleNamespace
        model = timm.create_model("resnet18", pretrained=False, num_classes=5)
        sd = model.state_dict()
        ckpt = {
            "model_config": {
                "type": "timm",
                "model_id": "resnet18",
                "config": {"num_labels": 5, "pretrained": False},
            },
            "model_state_dict": sd,
            "cfg": SimpleNamespace(data=SimpleNamespace(image_size=128)),
        }
        path = tmp_path / "model.pt"
        torch.save(ckpt, path)
        loaded = load_model_from_checkpoint(path, device=torch.device("cpu"))
        assert loaded.fc.out_features == 5
