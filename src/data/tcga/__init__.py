# src/data/tcga/__init__.py
"""
TCGA / GDC Data Module

Provides tools for querying and managing TCGA data from the Genomic Data Commons.

Example:
    from src.data.tcga import GDCClient

    client = GDCClient()
    projects = client.list_projects(program="TCGA")
    cases = client.get_cases(project_id="TCGA-BRCA", max_results=10)
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

__all__ = [
    "GDCClient",
    "GDCFilterBuilder",
    "FilterOp",
    "GDCProject",
    "GDCCase",
    "GDCFile",
    "GDCAnnotation",
    "CaseFields",
    "FileFields",
    "ProjectFields",
    "AnnotationFields",
]
