"""Tests for CLI helper functions: _resolve_model_name, _resolve_dataset_name."""
import pytest
from types import SimpleNamespace

from src.cli.run_multiresolution_probe import _resolve_model_name, _resolve_dataset_name


# ---------------------------------------------------------------------------
# _resolve_model_name
# ---------------------------------------------------------------------------

class TestResolveModelName:
    """Tests for _resolve_model_name."""

    def test_default_from_model_configs(self):
        """Without overrides, returns model.name from MODEL_CONFIGS."""
        assert _resolve_model_name("dinov3") == "dinov3"
        assert _resolve_model_name("resnet18") == "resnet18"
        assert _resolve_model_name("resnet50") == "resnet50"
        assert _resolve_model_name("tiny_vit_21m_224") == "tiny_vit"

    def test_override_via_extra_overrides(self):
        """model.name= in extra_overrides takes precedence."""
        result = _resolve_model_name("dinov3", ["model.name=resnet50_distilled"])
        assert result == "resnet50_distilled"

    def test_unrelated_override_ignored(self):
        """Overrides not matching model.name= are ignored."""
        result = _resolve_model_name("dinov3", ["model.type=timm", "data.batch_size=32"])
        assert result == "dinov3"

    def test_last_override_wins(self):
        """If multiple model.name= overrides, last one wins."""
        result = _resolve_model_name("dinov3", ["model.name=a", "model.name=b"])
        assert result == "b"


# ---------------------------------------------------------------------------
# _resolve_dataset_name
# ---------------------------------------------------------------------------

class TestResolveDatasetName:
    """Tests for _resolve_dataset_name."""

    def test_pathology_tcga(self):
        """TCGADataModule returns tcga_{task}."""
        config = SimpleNamespace(
            datamodule={"_target_": "src.data.tcga_datamodule.TCGADataModule", "task": "kras"}
        )
        assert _resolve_dataset_name(config) == "tcga_kras"

    def test_pathology_task_override(self):
        """extra_overrides can override the task."""
        config = SimpleNamespace(
            datamodule={"_target_": "src.data.tcga_datamodule.TCGADataModule", "task": "kras"}
        )
        result = _resolve_dataset_name(config, ["datamodule.task=tp53"])
        assert result == "tcga_tp53"

    def test_dermatology_data_dir(self):
        """TabularDataModulePersistent returns basename of data_dir."""
        config = SimpleNamespace(
            datamodule={
                "_target_": "src.data.tabular_datamodule_persistent.TabularDataModulePersistent",
                "data_dir": "/scratch/datasets/isic/images",
            }
        )
        assert _resolve_dataset_name(config) == "images"

    def test_data_dir_override(self):
        """extra_overrides can override data_dir."""
        config = SimpleNamespace(
            datamodule={
                "_target_": "src.data.tabular_datamodule_persistent.TabularDataModulePersistent",
                "data_dir": "/scratch/datasets/isic/images",
            }
        )
        result = _resolve_dataset_name(config, ["datamodule.data_dir=/data/chexpert/combined"])
        assert result == "combined"

    def test_dataset_name_fallback(self):
        """Falls back to dataset_name when data_dir is absent."""
        config = SimpleNamespace(
            datamodule={"_target_": "other", "dataset_name": "my/dataset"}
        )
        assert _resolve_dataset_name(config) == "my_dataset"

    def test_unknown_fallback(self):
        """Returns 'unknown' when nothing matches."""
        config = SimpleNamespace(datamodule={"_target_": "other"})
        assert _resolve_dataset_name(config) == "unknown"
