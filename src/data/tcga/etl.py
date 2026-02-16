"""
TCGA ETL for building flat slide tables.

Builds a flat table where each row is a slide image file, with proper hierarchy
broadcasting from parent levels (case → sample → portion → slide → file).

This module:
- Uses GDCClient to fetch data (no direct API calls)
- Uses HierarchyBuilder to index relationships
- Joins files to hierarchy via associated_entities.entity_id
- Supports multi-level joins (MAF at sample level, demographics at case level)

Usage:
    from src.data.tcga import TCGASlideETL

    etl = TCGASlideETL()
    df = etl.build_slide_table(
        project_ids=["TCGA-LUAD", "TCGA-LUSC"],
        include_demographics=True,
        include_diagnosis=True,
        include_maf=True,
    )

    # Each row is a slide image file with:
    # - File metadata (file_id, filename, file_size, md5sum)
    # - Slide data (slide_id, percent_tumor_cells, etc.)
    # - Portion data broadcast (portion_id, is_ffpe)
    # - Sample data broadcast (sample_id, sample_type, tissue_type)
    # - Case data broadcast (case_id, demographics, diagnosis)
    # - MAF data at sample level (all slides from same sample share MAF)
    # - Local path columns can be added with add_local_paths()
"""

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path
import pandas as pd
from src.data.tcga.gdc_client import GDCClient
from src.data.tcga.hierarchy import HierarchyBuilder, HierarchyNode

if TYPE_CHECKING:
    from src.data.tcga.config import TCGAConfig


class TCGASlideETL:
    """
    ETL for building flat slide tables with proper hierarchy broadcasting.

    Usage:
        etl = TCGASlideETL()
        df = etl.build_slide_table(
            project_ids=["TCGA-LUAD", "TCGA-LUSC"],
            include_demographics=True,
            include_diagnosis=True,
            include_maf=True,
        )
    """

    def __init__(self, client: Optional[GDCClient] = None):
        self.client = client or GDCClient()
        self.hierarchy_builder = HierarchyBuilder()

    def build_slide_table(
        self,
        project_ids: List[str],
        include_demographics: bool = True,
        include_diagnosis: bool = True,
        include_maf: bool = False,
        access: str = "open",
    ) -> pd.DataFrame:
        """
        Build flat table with one row per slide image file.

        Steps:
        1. Query cases with hierarchy (samples/portions/slides)[top down]
        2. Build hierarchy index (slide_id → parents)[bottom up, asymettric parent depths]
        3. Query slide files
        4. Join files to hierarchy via associated_entities.entity_id
        5. Optionally add demographics, diagnosis, MAF at appropriate levels

        Args:
            project_ids: List of TCGA project IDs (e.g., ["TCGA-LUAD", "TCGA-LUSC"])
            include_demographics: Include case-level demographics (gender, race, etc.)
            include_diagnosis: Include case-level diagnosis (tumor_stage, vital_status, etc.)
            include_maf: Include sample-level MAF file linkage
            access: Access level filter ("open" or "controlled")

        Returns:
            DataFrame with one row per slide image file
        """
        all_rows = []
        # TODO: please update to register new types of column filters relative to some expected fuzzy search across hiearchy.... example. I know i want details X_1,X_2,Xn and i know the parent that should have the details... supposedly, but i should allow a fuzzy search across entities and broadcast the informaiton to the level i envision. .. later challenge. 
        for project_id in project_ids:
            rows = self._process_project(
                project_id, include_demographics, include_diagnosis, include_maf, access
            )
            all_rows.extend(rows)

        return pd.DataFrame(all_rows)

    def _process_project(
        self,
        project_id: str,
        include_demographics: bool,
        include_diagnosis: bool,
        include_maf: bool,
        access: str,
    ) -> List[Dict]:
        """Process one project."""

        # Step 1: Get cases with full hierarchy
        expand = ["samples", "samples.portions", "samples.portions.slides"]
        if include_demographics:
            expand.append("demographic")
        if include_diagnosis:
            expand.append("diagnoses")

        cases = self.client._paginate(
            "cases",
            filters={"op": "=", "content": {"field": "project.project_id", "value": project_id}},
            expand=expand,
        )

        # Step 2: Build hierarchy index
        hierarchy_index = self.hierarchy_builder.build_index(cases)

        # Step 3: Build case-level data index (demographics, diagnosis)
        case_data = self._build_case_data_index(cases, include_demographics, include_diagnosis)

        # Step 4: Query slide files (include md5sum and state for manifest generation)
        files = self.client._paginate(
            "files",
            filters={
                "op": "and",
                "content": [
                    {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                    {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
                    {"op": "=", "content": {"field": "access", "value": access}},
                ]
            },
            expand=["associated_entities"],
            fields=["file_id", "file_name", "file_size", "md5sum", "state"],
        )

        # Step 5: Join MAF at sample level (optional)
        maf_by_sample = {}
        if include_maf:
            maf_by_sample = self._get_maf_by_sample(project_id)

        # Step 6: Build rows
        rows = []
        for file in files:
            # Get slide_id from associated_entities
            assoc = (file.get("associated_entities") or [{}])[0]
            slide_id = assoc.get("entity_id")

            # Lookup hierarchy
            hierarchy = hierarchy_index.get(slide_id)
            if not hierarchy:
                continue  # Skip if can't find in hierarchy

            # Get case-level data
            cdata = case_data.get(hierarchy.case_id, {})

            row = {
                # File level (includes md5sum for manifest generation)
                "file_id": file.get("file_id"),
                "filename": file.get("file_name"),
                "file_size": file.get("file_size"),
                "md5sum": file.get("md5sum"),
                "file_state": file.get("state"),
                "project_id": project_id,

                # Slide level
                "slide_id": hierarchy.slide_id,
                "slide_submitter_id": hierarchy.slide_submitter_id,
                "percent_tumor_cells": hierarchy.percent_tumor_cells,
                "percent_necrosis": hierarchy.percent_necrosis,

                # Portion level (broadcast)
                "portion_id": hierarchy.portion_id,
                "is_ffpe": hierarchy.is_ffpe,

                # Sample level (broadcast)
                "sample_id": hierarchy.sample_id,
                "sample_submitter_id": hierarchy.sample_submitter_id,
                "sample_type": hierarchy.sample_type,
                "tissue_type": hierarchy.tissue_type,

                # Case level (broadcast)
                "case_id": hierarchy.case_id,
                "case_submitter_id": hierarchy.case_submitter_id,

                # Demographics (case level broadcast)
                **cdata.get("demographics", {}),

                # Diagnosis (case level broadcast)
                **cdata.get("diagnosis", {}),
            }

            # MAF (sample level broadcast) - full metadata for manifest generation
            if include_maf:
                maf_data = maf_by_sample.get(hierarchy.sample_id, {})
                row["maf_file_id"] = maf_data.get("file_id")
                row["maf_filename"] = maf_data.get("filename")
                row["maf_file_size"] = maf_data.get("file_size")
                row["maf_md5sum"] = maf_data.get("md5sum")
                row["has_maf"] = bool(maf_data)

            rows.append(row)

        return rows

    def _build_case_data_index(
        self,
        cases: List[Dict],
        include_demographics: bool,
        include_diagnosis: bool,
    ) -> Dict[str, Dict]:
        """Build index: case_id → {demographics, diagnosis}."""
        index = {}

        for case in cases:
            case_id = case.get("case_id")
            data = {}

            if include_demographics:
                demo = case.get("demographic") or {}
                data["demographics"] = {
                    "gender": demo.get("gender"),
                    "race": demo.get("race"),
                    "ethnicity": demo.get("ethnicity"),
                    "year_of_birth": demo.get("year_of_birth"),
                }

            if include_diagnosis:
                # Find primary diagnosis
                diagnoses = case.get("diagnoses") or []
                primary_diag = {}
                for d in diagnoses:
                    if d.get("diagnosis_is_primary_disease") in ["Yes", True, "yes"]:
                        primary_diag = d
                        break
                if not primary_diag and diagnoses:
                    primary_diag = diagnoses[0]

                data["diagnosis"] = {
                    "primary_diagnosis": primary_diag.get("primary_diagnosis"),
                    "diagnosis_is_primary": primary_diag.get("diagnosis_is_primary_disease"),
                    "tumor_stage": primary_diag.get("tumor_stage"),
                    "tumor_grade": primary_diag.get("tumor_grade"),
                    "vital_status": primary_diag.get("vital_status"),
                    "days_to_death": primary_diag.get("days_to_death"),
                    "age_at_diagnosis": primary_diag.get("age_at_diagnosis"),
                }

            index[case_id] = data

        return index

    def _get_maf_by_sample(self, project_id: str) -> Dict[str, Dict[str, Any]]:
        """Get MAF file metadata indexed by sample_id.

        Returns full metadata dict (not just file_id) for manifest generation.
        """
        maf_files = self.client._paginate(
            "files",
            filters={
                "op": "and",
                "content": [
                    {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                    {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
                ]
            },
            expand=["associated_entities", "cases.samples"],
            fields=["file_id", "file_name", "file_size", "md5sum", "state"],
        )

        maf_by_sample = {}
        for maf in maf_files:
            maf_metadata = {
                "file_id": maf.get("file_id"),
                "filename": maf.get("file_name"),
                "file_size": maf.get("file_size"),
                "md5sum": maf.get("md5sum"),
                "state": maf.get("state"),
            }
            # Get sample from expanded cases.samples
            for case in maf.get("cases", []):
                for sample in case.get("samples", []):
                    sample_id = sample.get("sample_id")
                    if sample_id:
                        maf_by_sample[sample_id] = maf_metadata

        return maf_by_sample

    def add_local_paths(
        self,
        df: pd.DataFrame,
        config: "TCGAConfig",
    ) -> pd.DataFrame:
        """Add local file path columns based on config directories.

        gdc-client downloads files to: <dir>/<file_uuid>/<filename>

        Note: These are computed paths - files may not exist yet.
        Use validate_local_paths() to check which files exist.

        Args:
            df: DataFrame from build_slide_table()
            config: TCGAConfig with directory settings

        Returns:
            DataFrame with slide_local_path and maf_local_path columns added
        """
        df = df.copy()

        # Slide paths: slides_dir / file_id / filename
        df["slide_local_path"] = df.apply(
            lambda row: config.slides_dir / row["file_id"] / row["filename"],
            axis=1
        )

        # MAF paths (only for rows with MAF): maf_dir / maf_file_id / maf_filename
        def get_maf_path(row):
            if row.get("has_maf") and row.get("maf_file_id") and row.get("maf_filename"):
                return config.maf_dir / row["maf_file_id"] / row["maf_filename"]
            return None

        df["maf_local_path"] = df.apply(get_maf_path, axis=1)

        return df

    def validate_local_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate which local files actually exist.

        Adds boolean columns indicating whether files are present on disk.
        Run this after add_local_paths() and after downloading.

        Args:
            df: DataFrame with slide_local_path (and optionally maf_local_path)

        Returns:
            DataFrame with slide_exists and maf_exists columns added
        """
        df = df.copy()

        # Check slide files
        if "slide_local_path" in df.columns:
            df["slide_exists"] = df["slide_local_path"].apply(
                lambda p: Path(p).exists() if pd.notna(p) else False
            )

        # Check MAF files
        if "maf_local_path" in df.columns:
            df["maf_exists"] = df["maf_local_path"].apply(
                lambda p: Path(p).exists() if pd.notna(p) else False
            )

        return df
