"""
Hierarchy Builder for TCGA Case Data.

Builds index structures from case data to enable proper hierarchical joins:
    case → sample → portion → slide

This module is responsible for:
- Extracting the full path through the biospecimen hierarchy for each slide
- Building lookup indices for multi-level joins (e.g., MAF files at sample level)

Usage:
    from src.data.tcga import GDCClient, HierarchyBuilder

    client = GDCClient()
    cases = client._paginate("cases",
        filters={"op": "=", "content": {"field": "project.project_id", "value": "TCGA-LUAD"}},
        expand=["samples", "samples.portions", "samples.portions.slides"],
    )

    builder = HierarchyBuilder()
    index = builder.build_index(cases)
    # index[slide_id] → HierarchyNode with all parent info
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class HierarchyNode:
    """One path through case → sample → portion → slide."""

    # Case level
    case_id: str
    case_submitter_id: str

    # Sample level
    sample_id: str
    sample_submitter_id: str
    sample_type: str
    tissue_type: str

    # Portion level
    portion_id: str
    portion_submitter_id: str
    is_ffpe: Optional[bool]

    # Slide level
    slide_id: str
    slide_submitter_id: str
    percent_tumor_cells: Optional[float]
    percent_necrosis: Optional[float]


class HierarchyBuilder:
    """
    Builds hierarchy index from case data.

    Usage:
        builder = HierarchyBuilder()
        index = builder.build_index(cases_data)
        # index[slide_id] → HierarchyNode with all parent info
    """

    def build_index(self, cases: List[Dict[str, Any]]) -> Dict[str, HierarchyNode]:
        """
        Build index: slide_id → HierarchyNode.

        Args:
            cases: List of case dicts with expanded samples/portions/slides

        Returns:
            Dict mapping slide_id to its full hierarchy path
        """
        index = {}

        for case in cases:
            case_id = case.get("case_id")
            case_submitter_id = case.get("submitter_id")

            for sample in case.get("samples", []):
                sample_id = sample.get("sample_id")
                sample_submitter_id = sample.get("submitter_id")
                sample_type = sample.get("sample_type")
                tissue_type = sample.get("tissue_type")

                for portion in sample.get("portions", []):
                    portion_id = portion.get("portion_id")
                    portion_submitter_id = portion.get("submitter_id")
                    is_ffpe = portion.get("is_ffpe")

                    for slide in portion.get("slides", []):
                        slide_id = slide.get("slide_id")

                        index[slide_id] = HierarchyNode(
                            case_id=case_id,
                            case_submitter_id=case_submitter_id,
                            sample_id=sample_id,
                            sample_submitter_id=sample_submitter_id,
                            sample_type=sample_type,
                            tissue_type=tissue_type,
                            portion_id=portion_id,
                            portion_submitter_id=portion_submitter_id,
                            is_ffpe=is_ffpe,
                            slide_id=slide_id,
                            slide_submitter_id=slide.get("submitter_id"),
                            percent_tumor_cells=slide.get("percent_tumor_cells"),
                            percent_necrosis=slide.get("percent_necrosis"),
                        )

        return index

    def build_sample_index(self, cases: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Build index: sample_id → sample data.
        Used for joining MAF files at sample level.

        Args:
            cases: List of case dicts with expanded samples

        Returns:
            Dict mapping sample_id to sample data with case info
        """
        index = {}
        for case in cases:
            for sample in case.get("samples", []):
                index[sample.get("sample_id")] = {
                    **sample,
                    "case_id": case.get("case_id"),
                    "case_submitter_id": case.get("submitter_id"),
                }
        return index
