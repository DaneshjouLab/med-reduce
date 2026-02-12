# src/data/tcga/__init__.py
"""
TCGA / GDC Data Module

Provides tools for querying and managing TCGA data from the Genomic Data Commons.

Example:
    from src.data.tcga import GDCClient

    client = GDCClient()
    projects = client.list_projects(program="TCGA")
    cases = client.get_cases(project_id="TCGA-BRCA", max_results=10)

ETL Example:
    from src.data.tcga import TCGASlideETL, TCGAConfig

    config = TCGAConfig(project_ids=["TCGA-LUAD"])
    etl = TCGASlideETL()
    df = etl.build_slide_table(
        project_ids=config.project_ids,
        include_demographics=True,
        include_diagnosis=True,
        include_maf=True,
    )
    df = etl.add_local_paths(df, config)

Manifest & Download Example:
    from src.data.tcga import ManifestGenerator, TCGADownloader

    manifest_gen = ManifestGenerator()
    manifest_gen.create_slide_manifest(df, config.manifests_dir / "slides.txt")

    downloader = TCGADownloader()
    result = downloader.download_from_manifest(manifest_path, config.slides_dir)
"""

from src.data.tcga.gdc_client import (
    # Main client
    GDCClient,

    # Filter building
    GDCFilterBuilder,
    FilterOp,

    # Data classes
    GDCProject,
    GDCCase,
    GDCFile,
    GDCAnnotation,

    # Field reference classes (for documentation, prefer discover_fields())
    CaseFields,
    FileFields,
    ProjectFields,
    AnnotationFields,
)

from src.data.tcga.hierarchy import (
    HierarchyBuilder,
    HierarchyNode,
)

from src.data.tcga.etl import (
    TCGASlideETL,
)

from src.data.tcga.config import (
    TCGAConfig,
)

from src.data.tcga.manifest import (
    ManifestGenerator,
)

from src.data.tcga.downloader import (
    TCGADownloader,
    DownloadStatus,
    DownloadResult,
)

from src.data.tcga.gene_matrix import (
    GeneMatrix,
)

__all__ = [
    # Client
    "GDCClient",
    "GDCFilterBuilder",
    "FilterOp",
    # Data classes
    "GDCProject",
    "GDCCase",
    "GDCFile",
    "GDCAnnotation",
    # Field references
    "CaseFields",
    "FileFields",
    "ProjectFields",
    "AnnotationFields",
    # Hierarchy
    "HierarchyBuilder",
    "HierarchyNode",
    # ETL
    "TCGASlideETL",
    # Config
    "TCGAConfig",
    # Manifest
    "ManifestGenerator",
    # Downloader
    "TCGADownloader",
    "DownloadStatus",
    "DownloadResult",
    # Gene Matrix
    "GeneMatrix",
]
