"""Configuration for TCGA data pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TCGAConfig:
    """Configuration for TCGA data pipeline.

    Usage:
        config = TCGAConfig(
            project_ids=["TCGA-LUAD", "TCGA-LUSC"],
            data_dir=Path("data/tcga"),
        )
        config.ensure_directories()
    """

    # Project selection - user must specify
    project_ids: List[str] = field(default_factory=list)

    # Base data directory
    data_dir: Path = field(default_factory=lambda: Path("data/tcga"))

    # ETL options
    include_demographics: bool = True
    include_diagnosis: bool = True
    include_maf: bool = True
    access: str = "open"  # "open" or "controlled"

    def __post_init__(self):
        """Convert paths and validate."""
        self.data_dir = Path(self.data_dir)
        if not self.project_ids:
            raise ValueError("project_ids cannot be empty")

    @property
    def slides_dir(self) -> Path:
        """Directory for slide image downloads."""
        return self.data_dir / "slides"

    @property
    def maf_dir(self) -> Path:
        """Directory for MAF file downloads."""
        return self.data_dir / "maf"

    @property
    def manifests_dir(self) -> Path:
        """Directory for manifest files."""
        return self.data_dir / "manifests"

    @property
    def tables_dir(self) -> Path:
        """Directory for output tables (CSV/parquet)."""
        return self.data_dir / "tables"

    @property
    def thumbnails_dir(self) -> Path:
        """Directory for slide thumbnail images."""
        return self.data_dir / "thumbnails"

    def ensure_directories(self):
        """Create all configured directories if they don't exist."""
        for d in [self.slides_dir, self.maf_dir, self.manifests_dir, self.tables_dir, self.thumbnails_dir]:
            d.mkdir(parents=True, exist_ok=True)
