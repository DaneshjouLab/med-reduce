"""Tests for src.models.factory — create_model, extract_embeddings, get_embedding_dim, freeze_backbone, save_model."""
import json
import pytest
import torch
import torch.nn as nn
import timm

from src.models.factory import create_model, create_preprocessor, extract_embeddings, get_embedding_dim, freeze_backbone, save_model


# ---------------------------------------------------------------------------
# extract_embeddings (timm branch)
# ---------------------------------------------------------------------------

class TestExtractEmbeddingsTimm:
    """Tests for extract_embeddings with model_type='timm'."""

    def test_resnet18_returns_512_dim(self):
        """ResNet18 produces 512-dim embeddings."""
        model = timm.create_model("resnet18", pretrained=False, num_classes=3)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        emb = extract_embeddings(model, x, "timm")
        assert emb.shape == (2, 512)

    def test_resnet50_returns_2048_dim(self):
        """ResNet50 produces 2048-dim embeddings."""
        model = timm.create_model("resnet50", pretrained=False, num_classes=3)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        emb = extract_embeddings(model, x, "timm")
        assert emb.shape == (2, 2048)

    def test_resnet50_not_num_classes(self):
        """Embedding dim is independent of num_classes."""
        model = timm.create_model("resnet50", pretrained=False, num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        emb = extract_embeddings(model, x, "timm")
        assert emb.shape[1] == 2048

    def test_unknown_model_type_raises(self):
        """Unknown model_type raises ValueError."""
        model = nn.Linear(10, 10)
        x = torch.randn(1, 10)
        with pytest.raises(ValueError, match="Unknown model_type"):
            extract_embeddings(model, x, "unknown")


# ---------------------------------------------------------------------------
# get_embedding_dim (timm branch)
# ---------------------------------------------------------------------------

class TestGetEmbeddingDimTimm:
    """Tests for get_embedding_dim with timm models."""

    def test_resnet18_dim(self):
        model = timm.create_model("resnet18", pretrained=False)
        assert get_embedding_dim(model, "timm") == 512

    def test_resnet50_dim(self):
        model = timm.create_model("resnet50", pretrained=False)
        assert get_embedding_dim(model, "timm") == 2048


# ---------------------------------------------------------------------------
# freeze_backbone
# ---------------------------------------------------------------------------

class TestFreezeBackbone:
    """Tests for freeze_backbone."""

    def test_timm_freezes_backbone_keeps_fc(self):
        """For timm, backbone is frozen but fc/head/classifier remain trainable."""
        model = timm.create_model("resnet18", pretrained=False, num_classes=3)
        freeze_backbone(model, "timm")

        for name, param in model.named_parameters():
            if "fc" in name or "head" in name or "classifier" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_unknown_type_raises(self):
        """Unknown model_type raises ValueError."""
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Unsupported model_type"):
            freeze_backbone(model, "unknown")


# ---------------------------------------------------------------------------
# create_model (timm branch)
# ---------------------------------------------------------------------------

class TestCreateModelTimm:
    """Tests for create_model with model_type='timm'."""

    def test_creates_resnet18(self):
        model_info = {"type": "timm", "model_id": "resnet18", "config": {"num_labels": 3, "pretrained": False}}
        model = create_model(model_info)
        assert isinstance(model, nn.Module)
        # Verify classifier head has correct num_classes
        assert model.fc.out_features == 3

    def test_creates_resnet50(self):
        model_info = {"type": "timm", "model_id": "resnet50", "config": {"num_labels": 5, "pretrained": False}}
        model = create_model(model_info)
        assert model.fc.out_features == 5

    def test_default_num_classes(self):
        model_info = {"type": "timm", "model_id": "resnet18", "config": {"pretrained": False}}
        model = create_model(model_info)
        assert model.fc.out_features == 1000

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model({"type": "fake", "model_id": "x"})


# ---------------------------------------------------------------------------
# create_preprocessor
# ---------------------------------------------------------------------------

class TestCreatePreprocessor:
    def test_timm_returns_none(self):
        """timm models use torchvision transforms, so preprocessor is None."""
        result = create_preprocessor({"type": "timm", "model_id": "resnet18"})
        assert result is None

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_preprocessor({"type": "fake", "model_id": "x"})


# ---------------------------------------------------------------------------
# save_model (timm branch)
# ---------------------------------------------------------------------------

class TestSaveModelTimm:
    def test_save_timm_model(self, tmp_path):
        model_info = {"type": "timm", "model_id": "resnet18", "config": {"num_labels": 3, "pretrained": False}}
        model = create_model(model_info)
        save_dir = str(tmp_path / "saved")
        save_model(model, model_info, save_dir)
        assert (tmp_path / "saved" / "pytorch_model.bin").exists()
        assert (tmp_path / "saved" / "config.json").exists()
        config = json.loads((tmp_path / "saved" / "config.json").read_text())
        assert config["model_type"] == "timm"
        assert config["model_id"] == "resnet18"

    def test_save_unknown_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported model_type"):
            save_model(nn.Linear(1, 1), {"type": "fake"}, str(tmp_path))


# ---------------------------------------------------------------------------
# get_embedding_dim — additional branches
# ---------------------------------------------------------------------------

class TestGetEmbeddingDimExtra:
    def test_dinov3_with_config(self):
        """DINOv3 model with config.hidden_size."""
        from types import SimpleNamespace
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=384))
        assert get_embedding_dim(model, "dinov3") == 384

    def test_dinov3_default(self):
        """DINOv3 model without config falls back to 384."""
        model = nn.Linear(10, 10)
        assert get_embedding_dim(model, "dinov3") == 384

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_embedding_dim(nn.Linear(10, 10), "fake")


# ---------------------------------------------------------------------------
# freeze_backbone — HF model branch
# ---------------------------------------------------------------------------

class TestFreezeBackboneHF:
    def test_hf_model_freezes_non_classifier(self):
        """For HF model types, non-classifier params are frozen."""
        model = nn.Module()
        model.backbone_conv = nn.Linear(10, 10)
        model.classifier = nn.Linear(10, 3)
        model.head_norm = nn.LayerNorm(10)
        freeze_backbone(model, "dinov3")
        assert not model.backbone_conv.weight.requires_grad
        assert model.classifier.weight.requires_grad

    def test_hf_vit_type(self):
        """Works with 'vit' type too."""
        model = nn.Module()
        model.encoder = nn.Linear(10, 10)
        model.classifier = nn.Linear(10, 3)
        freeze_backbone(model, "vit")
        assert not model.encoder.weight.requires_grad
        assert model.classifier.weight.requires_grad


# ---------------------------------------------------------------------------
# extract_embeddings — timm transformer (3D) branch
# ---------------------------------------------------------------------------

class TestExtractEmbeddingsTimm3D:
    def test_transformer_timm_avg_pool(self):
        """Timm transformer returning 3D -> global average pool."""
        class MockTimm3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Linear(1, 1)
            def forward_features(self, x):
                return torch.randn(x.shape[0], 49, 576)
        model = MockTimm3D()
        emb = extract_embeddings(model, torch.randn(2, 3, 224, 224), "timm")
        assert emb.shape == (2, 576)


class TestGetEmbeddingDimFallback:
    def test_timm_no_num_features_probes_3d(self):
        """When num_features is absent and forward_features returns 3D, uses shape[-1]."""
        class MockNoNumFeatures(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(1))
            def forward_features(self, x):
                return torch.randn(x.shape[0], 49, 576)
        model = MockNoNumFeatures()
        assert get_embedding_dim(model, "timm") == 576

    def test_timm_no_num_features_probes_4d(self):
        """When num_features is absent and forward_features returns 4D, uses shape[1]."""
        class MockNoNumFeatures4D(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(1))
            def forward_features(self, x):
                return torch.randn(x.shape[0], 2048, 7, 7)
        model = MockNoNumFeatures4D()
        assert get_embedding_dim(model, "timm") == 2048


# ---------------------------------------------------------------------------
# create_model / create_preprocessor — HF branches (mocked)
# ---------------------------------------------------------------------------

class TestCreateModelHFBranches:
    def test_timm_not_available_raises(self, monkeypatch):
        """Raises RuntimeError when timm is not available."""
        import src.models.factory as f
        monkeypatch.setattr(f, "_TIMM_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="timm is not installed"):
            create_model({"type": "timm", "model_id": "resnet18", "config": {}})

    def test_vit_type_calls_hf(self, monkeypatch):
        """'vit' type attempts HF download — mock to verify it's called."""
        called = {}
        def mock_from_pretrained(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs
            return nn.Linear(10, 3)  # dummy

        import src.models.factory as f
        monkeypatch.setattr(f, "ViTForImageClassification", type("Mock", (), {"from_pretrained": staticmethod(mock_from_pretrained)}))
        create_model({"type": "vit", "model_id": "google/vit-base", "config": {"num_labels": 3}})
        assert called["args"][0] == "google/vit-base"

    def test_dinov2_type_calls_hf(self, monkeypatch):
        """'dinov2' type attempts HF download."""
        called = {}
        def mock_from_pretrained(*args, **kwargs):
            called["args"] = args
            return nn.Linear(10, 3)

        import src.models.factory as f
        monkeypatch.setattr(f, "AutoModelForImageClassification", type("Mock", (), {"from_pretrained": staticmethod(mock_from_pretrained)}))
        create_model({"type": "dinov2", "model_id": "facebook/dinov2-small", "config": {"num_labels": 3}})
        assert called["args"][0] == "facebook/dinov2-small"

    def test_dinov3_type_creates_model(self, monkeypatch):
        """'dinov3' type creates DINOv3ForImageClassification."""
        called = {}
        def mock_init(self, config):
            called["config"] = config
            nn.Module.__init__(self)

        import src.models.factory as f
        monkeypatch.setattr(f.DINOv3ForImageClassification, "__init__", mock_init)
        model = create_model({
            "type": "dinov3",
            "model_id": "facebook/dinov3-vits16",
            "config": {"num_labels": 3, "hidden_size": 384},
        })
        assert called["config"].num_labels == 3


class TestCreatePreprocessorHF:
    def test_vit_calls_hf(self, monkeypatch):
        called = {}
        def mock_from_pretrained(*args, **kwargs):
            called["ok"] = True
            return "preprocessor"
        import src.models.factory as f
        monkeypatch.setattr(f, "ViTImageProcessor", type("Mock", (), {"from_pretrained": staticmethod(mock_from_pretrained)}))
        result = create_preprocessor({"type": "vit", "model_id": "google/vit-base"}, resolution=224)
        assert called["ok"]

    def test_dinov2_calls_hf(self, monkeypatch):
        called = {}
        def mock_from_pretrained(*args, **kwargs):
            called["ok"] = True
            return "preprocessor"
        import src.models.factory as f
        monkeypatch.setattr(f, "AutoImageProcessor", type("Mock", (), {"from_pretrained": staticmethod(mock_from_pretrained)}))
        result = create_preprocessor({"type": "dinov2", "model_id": "facebook/dinov2"}, resolution=224)
        assert called["ok"]

    def test_dinov3_calls_hf(self, monkeypatch):
        called = {}
        def mock_from_pretrained(*args, **kwargs):
            called["ok"] = True
            return "preprocessor"
        import src.models.factory as f
        monkeypatch.setattr(f, "AutoImageProcessor", type("Mock", (), {"from_pretrained": staticmethod(mock_from_pretrained)}))
        result = create_preprocessor({"type": "dinov3", "model_id": "facebook/dinov3"}, resolution=224)
        assert called["ok"]


class TestSaveModelHF:
    def test_save_hf_model(self, tmp_path, monkeypatch):
        """Saves HF model via save_pretrained."""
        saved = {}
        class MockHFModel(nn.Module):
            def save_pretrained(self, path):
                saved["model"] = path
        class MockProcessor:
            def save_pretrained(self, path):
                saved["proc"] = path

        model = MockHFModel()
        save_model(model, {"type": "vit"}, str(tmp_path), preprocessor=MockProcessor())
        assert saved["model"] == str(tmp_path)
        assert saved["proc"] == str(tmp_path)
