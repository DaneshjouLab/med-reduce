"""Tests for src.utils.embedding_cache — _extract_embeddings_from_model dispatch."""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from src.utils.embedding_cache import EmbeddingCache


# ---------------------------------------------------------------------------
# Helpers — lightweight mock models
# ---------------------------------------------------------------------------

class MockBackboneModel(nn.Module):
    """Mimics DINOv3: has model.backbone that returns pooler_output."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = self._Backbone(embed_dim)

    class _Backbone(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, pixel_values):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                pooler_output=torch.randn(B, self.embed_dim),
                last_hidden_state=torch.randn(B, 197, self.embed_dim),
            )


class MockCNNModel(nn.Module):
    """Mimics timm CNN (ResNet): forward_features returns [B, D, H, W]."""

    def __init__(self, embed_dim: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.dummy = nn.Linear(1, 1)  # so parameters() works

    def forward_features(self, x):
        B = x.shape[0]
        return torch.randn(B, self.embed_dim, 7, 7)

    def forward(self, x):
        # Full forward would return [B, num_classes] — this should NOT be called
        return torch.randn(x.shape[0], 3)


class MockTransformerTimmModel(nn.Module):
    """Mimics timm transformer (TinyViT): forward_features returns [B, tokens, D]."""

    def __init__(self, embed_dim: int = 576, num_tokens: int = 49):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.dummy = nn.Linear(1, 1)

    def forward_features(self, x):
        B = x.shape[0]
        return torch.randn(B, self.num_tokens, self.embed_dim)

    def forward(self, x):
        return torch.randn(x.shape[0], 3)


class MockViTModel(nn.Module):
    """Mimics HuggingFace ViT: has model.vit returning pooler_output."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self.vit = self._ViT(embed_dim)

    class _ViT(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, pixel_values, **kwargs):  # accepts interpolate_pos_encoding
            B = pixel_values.shape[0]
            return SimpleNamespace(
                pooler_output=torch.randn(B, self.embed_dim),
                last_hidden_state=torch.randn(B, 197, self.embed_dim),
            )


class MockDinoV2Model(nn.Module):
    """Mimics HuggingFace DINOv2: has model.dinov2."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.dinov2 = self._Dinov2(embed_dim)

    class _Dinov2(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, pixel_values):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                pooler_output=torch.randn(B, self.embed_dim),
                last_hidden_state=torch.randn(B, 197, self.embed_dim),
            )


class MockBackboneNullPooler(nn.Module):
    """Backbone model where pooler_output is None — falls back to last_hidden_state."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = self._Backbone(embed_dim)

    class _Backbone(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, pixel_values):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                pooler_output=None,
                last_hidden_state=torch.randn(B, 197, self.embed_dim),
            )


class MockBaseModelModule(nn.Module):
    """Model with base_model attribute."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_model = self._Base(embed_dim)

    class _Base(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, pixel_values):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                pooler_output=torch.randn(B, self.embed_dim),
                last_hidden_state=torch.randn(B, 197, self.embed_dim),
            )


class MockDictOutputModel(nn.Module):
    """Model whose forward() returns a dict with last_hidden_state."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        B = x.shape[0]
        return {"last_hidden_state": torch.randn(B, 50, self.embed_dim)}


class MockNamespaceOutputModel(nn.Module):
    """Model whose forward() returns namespace with last_hidden_state."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        B = x.shape[0]
        return SimpleNamespace(
            pooler_output=None,
            last_hidden_state=torch.randn(B, 50, self.embed_dim),
        )


class MockTensor3DOutputModel(nn.Module):
    """Model whose forward() returns a raw 3D tensor."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        B = x.shape[0]
        return torch.randn(B, 50, self.embed_dim)


class MockFallbackModel(nn.Module):
    """Model with no special attributes — falls through to forward()."""

    def __init__(self, out_dim: int = 10):
        super().__init__()
        self.fc = nn.Linear(3, out_dim)

    def forward(self, x):
        return torch.randn(x.shape[0], self.fc.out_features)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractEmbeddingsFromModel:
    """Tests for EmbeddingCache._extract_embeddings_from_model."""

    @pytest.fixture
    def cache(self, tmp_path):
        return EmbeddingCache(
            cache_dir=str(tmp_path),
            dataset_name="test_ds",
            model_name="test_model",
            seed=42,
        )

    def test_backbone_model_returns_pooler_output(self, cache):
        """DINOv3-style model with backbone returns pooler_output."""
        model = MockBackboneModel(embed_dim=384)
        images = torch.randn(4, 3, 224, 224)
        emb = cache._extract_embeddings_from_model(model, images)
        assert emb.shape == (4, 384)

    def test_cnn_timm_returns_pooled_features(self, cache):
        """ResNet-style timm model: forward_features [B,D,H,W] -> avg pool [B,D]."""
        model = MockCNNModel(embed_dim=2048)
        images = torch.randn(4, 3, 224, 224)
        emb = cache._extract_embeddings_from_model(model, images)
        assert emb.shape == (4, 2048)

    def test_transformer_timm_returns_pooled_features(self, cache):
        """TinyViT-style timm model: forward_features [B,T,D] -> avg pool [B,D]."""
        model = MockTransformerTimmModel(embed_dim=576, num_tokens=49)
        images = torch.randn(4, 3, 224, 224)
        emb = cache._extract_embeddings_from_model(model, images)
        assert emb.shape == (4, 576)

    def test_cnn_timm_does_not_call_forward(self, cache):
        """Ensure forward_features is used, not forward() (which returns logits)."""
        model = MockCNNModel(embed_dim=2048)
        images = torch.randn(2, 3, 224, 224)
        emb = cache._extract_embeddings_from_model(model, images)
        # forward() would return [B, 3]; forward_features returns [B, 2048]
        assert emb.shape[1] == 2048, "Should use forward_features, not forward()"

    def test_fallback_model_returns_tensor(self, cache):
        """Model with no special attrs falls through to forward()."""
        model = MockFallbackModel(out_dim=10)
        images = torch.randn(2, 3, 224, 224)
        emb = cache._extract_embeddings_from_model(model, images)
        assert emb.shape == (2, 10)

    def test_forward_features_priority_over_fallback(self, cache):
        """forward_features is preferred over forward() when both exist."""
        model = MockCNNModel(embed_dim=512)
        images = torch.randn(2, 3, 224, 224)
        emb = cache._extract_embeddings_from_model(model, images)
        # If fallback were used, shape would be [2, 3] (from forward())
        assert emb.shape[1] == 512

    def test_vit_model(self, cache):
        """HuggingFace ViT model with .vit attribute."""
        model = MockViTModel(embed_dim=768)
        emb = cache._extract_embeddings_from_model(model, torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 768)

    def test_dinov2_model(self, cache):
        """HuggingFace DINOv2 model with .dinov2 attribute."""
        model = MockDinoV2Model(embed_dim=384)
        emb = cache._extract_embeddings_from_model(model, torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 384)

    def test_backbone_null_pooler_fallback(self, cache):
        """Backbone with pooler_output=None falls back to last_hidden_state CLS."""
        model = MockBackboneNullPooler(embed_dim=384)
        emb = cache._extract_embeddings_from_model(model, torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 384)

    def test_base_model(self, cache):
        """Model with .base_model attribute."""
        model = MockBaseModelModule(embed_dim=512)
        emb = cache._extract_embeddings_from_model(model, torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 512)

    def test_dict_output_last_hidden_state(self, cache):
        """Model returning dict with last_hidden_state."""
        model = MockDictOutputModel(embed_dim=256)
        emb = cache._extract_embeddings_from_model(model, torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 256)

    def test_namespace_output_last_hidden_state(self, cache):
        """Model returning namespace with last_hidden_state (pooler=None)."""
        model = MockNamespaceOutputModel(embed_dim=128)
        emb = cache._extract_embeddings_from_model(model, torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 128)

    def test_tensor_3d_output(self, cache):
        """Model returning raw 3D tensor — takes first token."""
        model = MockTensor3DOutputModel(embed_dim=64)
        emb = cache._extract_embeddings_from_model(model, torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 64)

    def test_2d_forward_features(self, cache):
        """Model whose forward_features returns 2D tensor directly."""
        class Mock2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Linear(1, 1)
            def forward_features(self, x):
                return torch.randn(x.shape[0], 256)
        emb = cache._extract_embeddings_from_model(Mock2D(), torch.randn(2, 3, 224, 224))
        assert emb.shape == (2, 256)

    def test_raises_on_unextractable(self, cache):
        """Raises ValueError when no extraction method works."""
        class BadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Linear(1, 1)
            def forward(self, x):
                return 42  # not a tensor or dict
        with pytest.raises(ValueError, match="Could not extract"):
            cache._extract_embeddings_from_model(BadModel(), torch.randn(2, 3, 224, 224))


# ---------------------------------------------------------------------------
# Cache path helpers and persistence
# ---------------------------------------------------------------------------

class TestEmbeddingCachePaths:
    """Tests for cache directory structure and path resolution."""

    @pytest.fixture
    def cache(self, tmp_path):
        return EmbeddingCache(
            cache_dir=str(tmp_path),
            dataset_name="isic",
            model_name="resnet50_distilled",
            seed=42,
        )

    def test_dataset_dir_structure(self, cache, tmp_path):
        expected = tmp_path / "isic" / "resnet50_distilled" / "seed_42"
        assert cache.dataset_dir == expected
        assert expected.exists()

    def test_resolution_dir(self, cache):
        res_dir = cache._get_resolution_dir(512)
        assert str(res_dir).endswith("512px")

    def test_embedding_path(self, cache):
        path = cache._get_embedding_path(256, "train")
        assert path.name == "train_embeddings.pt"
        assert "256px" in str(path)

    def test_exists_false_initially(self, cache):
        assert cache.exists(512, "train") is False

    def test_get_cache_key_deterministic(self, cache):
        info = {"model_id": "resnet50", "type": "timm"}
        k1 = cache.get_cache_key(info, 512)
        k2 = cache.get_cache_key(info, 512)
        assert k1 == k2
        assert len(k1) == 8

    def test_get_cache_key_differs_by_resolution(self, cache):
        info = {"model_id": "resnet50", "type": "timm"}
        assert cache.get_cache_key(info, 512) != cache.get_cache_key(info, 256)

    def test_list_cached_resolutions_empty(self, cache):
        assert cache.list_cached_resolutions() == []

    def test_list_cached_resolutions_after_mkdir(self, cache):
        (cache.dataset_dir / "512px").mkdir(parents=True)
        (cache.dataset_dir / "256px").mkdir(parents=True)
        assert cache.list_cached_resolutions() == [256, 512]

    def test_get_metadata_empty(self, cache):
        assert cache.get_metadata(512) == {}

    def test_get_metadata_after_write(self, cache):
        import json
        res_dir = cache._get_resolution_dir(512)
        res_dir.mkdir(parents=True)
        meta_path = cache._get_metadata_path(512)
        meta_path.write_text(json.dumps({"resolution": 512, "num_samples": 100}))
        result = cache.get_metadata(512)
        assert result["resolution"] == 512
        assert result["num_samples"] == 100

    def test_load_missing_raises(self, cache):
        with pytest.raises(FileNotFoundError):
            cache.load(512, "train")

    def test_save_and_load_roundtrip(self, cache):
        """Manually save embeddings and verify load works."""
        import json
        res_dir = cache._get_resolution_dir(512)
        res_dir.mkdir(parents=True)
        emb = torch.randn(10, 2048)
        labels = torch.randint(0, 3, (10,))
        torch.save({"embeddings": emb, "labels": labels}, cache._get_embedding_path(512, "train"))
        cache._get_metadata_path(512).write_text(json.dumps({"resolution": 512}))
        assert cache.exists(512, "train")
        loaded_emb, loaded_labels = cache.load(512, "train")
        assert torch.equal(emb, loaded_emb)
        assert torch.equal(labels, loaded_labels)

    def test_clear_cache_single_resolution(self, cache):
        (cache.dataset_dir / "512px").mkdir(parents=True)
        (cache.dataset_dir / "256px").mkdir(parents=True)
        cache.clear_cache(resolution=512)
        assert not (cache.dataset_dir / "512px").exists()
        assert (cache.dataset_dir / "256px").exists()

    def test_clear_cache_all(self, cache):
        (cache.dataset_dir / "512px").mkdir(parents=True)
        cache.clear_cache()
        assert not cache.dataset_dir.exists()


# ---------------------------------------------------------------------------
# extract_and_cache integration
# ---------------------------------------------------------------------------

class TestExtractAndCache:
    """Tests for the full extract_and_cache pipeline."""

    @pytest.fixture
    def cache(self, tmp_path):
        return EmbeddingCache(
            cache_dir=str(tmp_path),
            dataset_name="test_ds",
            model_name="test_model",
            seed=42,
            device=torch.device("cpu"),
        )

    def _make_dataloader(self, n_samples=20, n_classes=3):
        from torch.utils.data import TensorDataset, DataLoader
        images = torch.randn(n_samples, 3, 32, 32)
        labels = torch.randint(0, n_classes, (n_samples,))
        ds = TensorDataset(images, labels)
        return DataLoader(ds, batch_size=8)

    def test_extract_and_cache_creates_files(self, cache):
        model = MockCNNModel(embed_dim=64)
        dl = self._make_dataloader()
        model_info = {"model_id": "test", "type": "timm"}
        emb, labels = cache.extract_and_cache(
            model, dl, resolution=32, split="train",
            model_info=model_info, mixed_precision=False,
        )
        assert emb.shape == (20, 64)
        assert labels.shape == (20,)
        assert cache.exists(32, "train")

    def test_extract_and_cache_loads_from_cache(self, cache):
        model = MockCNNModel(embed_dim=64)
        dl = self._make_dataloader()
        model_info = {"model_id": "test", "type": "timm"}
        emb1, _ = cache.extract_and_cache(
            model, dl, resolution=32, split="train",
            model_info=model_info, mixed_precision=False,
        )
        # Second call should load from cache
        emb2, _ = cache.extract_and_cache(
            model, dl, resolution=32, split="train",
            model_info=model_info, mixed_precision=False,
        )
        assert torch.equal(emb1, emb2)

    def test_force_recompute(self, cache):
        model = MockCNNModel(embed_dim=64)
        dl = self._make_dataloader()
        model_info = {"model_id": "test", "type": "timm"}
        cache.extract_and_cache(
            model, dl, resolution=32, split="train",
            model_info=model_info, mixed_precision=False,
        )
        # Force recompute — should succeed without error
        emb, _ = cache.extract_and_cache(
            model, dl, resolution=32, split="train",
            model_info=model_info, mixed_precision=False,
            force_recompute=True,
        )
        assert emb.shape == (20, 64)

    def test_metadata_written(self, cache):
        model = MockCNNModel(embed_dim=64)
        dl = self._make_dataloader(n_samples=10)
        model_info = {"model_id": "test", "type": "timm"}
        cache.extract_and_cache(
            model, dl, resolution=32, split="train",
            model_info=model_info, mixed_precision=False,
        )
        meta = cache.get_metadata(32)
        assert meta["num_samples"] == 10
        assert meta["embedding_dim"] == 64
        assert meta["split"] == "train"

    def test_multi_label(self, cache):
        """Multi-label targets (2D) are stored as float."""
        from torch.utils.data import TensorDataset, DataLoader
        images = torch.randn(10, 3, 32, 32)
        labels = torch.randint(0, 2, (10, 5)).float()  # multi-label
        dl = DataLoader(TensorDataset(images, labels), batch_size=5)
        model = MockCNNModel(embed_dim=64)
        model_info = {"model_id": "test", "type": "timm"}
        emb, lbl = cache.extract_and_cache(
            model, dl, resolution=32, split="train",
            model_info=model_info, mixed_precision=False,
        )
        assert lbl.shape == (10, 5)
        assert lbl.dtype == torch.float32
