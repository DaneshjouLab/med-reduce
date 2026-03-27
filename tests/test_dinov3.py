"""Tests for src.models.dinov3 — DINOv3Config."""
from src.models.dinov3 import DINOv3Config


class TestDINOv3Config:
    def test_defaults(self):
        cfg = DINOv3Config()
        assert cfg.backbone_model_id == "facebook/dinov3-vits16-pretrain-lvd1689m"
        assert cfg.num_labels == 2
        assert cfg.hidden_size == 384
        assert cfg.dropout_rate == 0.1
        assert cfg.use_quantization is False

    def test_custom_values(self):
        cfg = DINOv3Config(
            backbone_model_id="custom/model",
            num_labels=10,
            hidden_size=768,
            dropout_rate=0.3,
            use_quantization=True,
        )
        assert cfg.backbone_model_id == "custom/model"
        assert cfg.num_labels == 10
        assert cfg.hidden_size == 768
        assert cfg.dropout_rate == 0.3
        assert cfg.use_quantization is True

    def test_model_type(self):
        assert DINOv3Config.model_type == "dinov3_classifier"
