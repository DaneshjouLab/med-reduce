"""
GDC API Wrapper for TCGA Data Management.

Purpose: Query and manage TCGA data from the Genomic Data Commons (GDC).



GDC Data Model (Biospecimen Hierarchy):
    Program (e.g., TCGA)
        └── Project (e.g., TCGA-BRCA)
             └── Case (patient/donor)
                  ├── demographic (gender, race, ethnicity, year_of_birth, ...)
                  ├── diagnoses (primary_diagnosis, tumor_stage, vital_status, ...)
                  │    └── treatments (therapeutic_agents, treatment_type, ...)
                  ├── exposures (bmi, alcohol_history, cigarettes_per_day, ...)
                  ├── family_histories
                  ├── follow_ups (molecular_tests, disease response, ...)
                  └── samples (sample_type, tissue_type, tumor_descriptor, ...)
                       └── portions
                            ├── analytes → aliquots
                            └── slides (percent_tumor_cells, percent_necrosis, ...)

    Files are linked to cases/samples through associated_entities.

Usage:
    client = GDCClient()

    # List all TCGA projects
    projects = client.list_projects()

    # Get cases for a project with clinical data
    cases = client.get_cases(project_id="TCGA-BRCA", fields=["demographic", "diagnoses"])

    # Get slide images
    slides = client.get_files(project_id="TCGA-BRCA", data_type="Slide Image", access="open")

    # Get clinical supplement files
    clinical = client.get_files(project_id="TCGA-BRCA", data_category="Clinical")

    # Generate download manifest
    manifest = client.create_manifest(files)
"""

from __future__ import annotations

import json
import time
import requests
from enum import Enum
from typing import Any, Dict, List, Optional, Literal, Union, Iterator
from dataclasses import dataclass, field, asdict
from pathlib import Path
from urllib.parse import urljoin

# =============================================================================
# Constants
# =============================================================================

GDC_API_BASE = "https://api.gdc.cancer.gov"
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 10000
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1.0


# =============================================================================
# Field Definitions - Reference lists (use discover_fields() for dynamic discovery)
# =============================================================================

# These are REFERENCE lists based on GDC documentation.
# For dynamic discovery, use client.discover_fields(endpoint) instead.
# When fields=None is passed to queries, GDC returns default fields.
# Use expand=True to get expanded nested fields.

class CaseFields:
    """
    Reference fields for the cases endpoint.

    NOTE: These are not exhaustive. Use client.discover_fields("cases")
    to get all available fields dynamically.
    """

    # Core identifiers
    BASIC = [
        "case_id",
        "submitter_id",
        "state",
        "created_datetime",
        "updated_datetime",
        "days_to_index",
    ]

    # ID collections
    IDS = [
        "aliquot_ids",
        "analyte_ids",
        "portion_ids",
        "sample_ids",
        "slide_ids",
        "submitter_aliquot_ids",
        "submitter_analyte_ids",
        "submitter_portion_ids",
        "submitter_sample_ids",
        "submitter_slide_ids",
    ]

    # Demographic data
    DEMOGRAPHIC = [
        "demographic.demographic_id",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "demographic.year_of_birth",
        "demographic.year_of_death",
        "demographic.state",
        "demographic.submitter_id",
    ]

    # Diagnosis information
    DIAGNOSES = [
        "diagnoses.diagnosis_id",
        "diagnoses.primary_diagnosis",
        "diagnoses.age_at_diagnosis",
        "diagnoses.classification_of_tumor",
        "diagnoses.days_to_birth",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.days_to_last_known_disease_status",
        "diagnoses.days_to_recurrence",
        "diagnoses.last_known_disease_status",
        "diagnoses.morphology",
        "diagnoses.prior_malignancy",
        "diagnoses.progression_or_recurrence",
        "diagnoses.site_of_resection_or_biopsy",
        "diagnoses.tissue_or_organ_of_origin",
        "diagnoses.tumor_grade",
        "diagnoses.tumor_stage",
        "diagnoses.vital_status",
        "diagnoses.submitter_id",
    ]

    # Treatment data (nested under diagnoses)
    TREATMENTS = [
        "diagnoses.treatments.treatment_id",
        "diagnoses.treatments.treatment_or_therapy",
        "diagnoses.treatments.treatment_intent_type",
        "diagnoses.treatments.therapeutic_agents",
        "diagnoses.treatments.days_to_treatment",
        "diagnoses.treatments.submitter_id",
    ]

    # Exposure information
    EXPOSURES = [
        "exposures.exposure_id",
        "exposures.alcohol_history",
        "exposures.alcohol_intensity",
        "exposures.bmi",
        "exposures.cigarettes_per_day",
        "exposures.height",
        "exposures.weight",
        "exposures.years_smoked",
        "exposures.submitter_id",
    ]

    # Family history
    FAMILY_HISTORIES = [
        "family_histories.family_history_id",
        "family_histories.relationship_type",
        "family_histories.relative_with_cancer_history",
        "family_histories.relationship_primary_diagnosis",
        "family_histories.relationship_gender",
        "family_histories.relationship_age_at_diagnosis",
        "family_histories.submitter_id",
    ]

    # Sample information
    SAMPLES = [
        "samples.sample_id",
        "samples.submitter_id",
        "samples.sample_type",
        "samples.sample_type_id",
        "samples.composition",
        "samples.is_ffpe",
        "samples.tissue_type",
        "samples.tumor_code",
        "samples.tumor_code_id",
        "samples.tumor_descriptor",
        "samples.current_weight",
        "samples.initial_weight",
        "samples.longest_dimension",
        "samples.intermediate_dimension",
        "samples.shortest_dimension",
        "samples.days_to_collection",
        "samples.days_to_sample_procurement",
        "samples.preservation_method",
        "samples.freezing_method",
    ]

    # Portion information (nested under samples)
    PORTIONS = [
        "samples.portions.portion_id",
        "samples.portions.submitter_id",
        "samples.portions.portion_number",
        "samples.portions.weight",
        "samples.portions.is_ffpe",
    ]

    # Slide information (nested under portions) - PATHOLOGY DATA
    SLIDES = [
        "samples.portions.slides.slide_id",
        "samples.portions.slides.submitter_id",
        "samples.portions.slides.section_location",
        "samples.portions.slides.percent_tumor_cells",
        "samples.portions.slides.percent_tumor_nuclei",
        "samples.portions.slides.percent_normal_cells",
        "samples.portions.slides.percent_stromal_cells",
        "samples.portions.slides.percent_necrosis",
        "samples.portions.slides.percent_inflam_infiltration",
        "samples.portions.slides.percent_lymphocyte_infiltration",
        "samples.portions.slides.percent_monocyte_infiltration",
        "samples.portions.slides.percent_granulocyte_infiltration",
        "samples.portions.slides.percent_neutrophil_infiltration",
        "samples.portions.slides.percent_eosinophil_infiltration",
        "samples.portions.slides.number_proliferating_cells",
    ]

    # Project association
    PROJECT = [
        "project.project_id",
        "project.name",
        "project.disease_type",
        "project.primary_site",
        "project.program.name",
        "project.program.program_id",
    ]

    # Tissue source site
    TISSUE_SOURCE_SITE = [
        "tissue_source_site.tissue_source_site_id",
        "tissue_source_site.bcr_id",
        "tissue_source_site.code",
        "tissue_source_site.name",
        "tissue_source_site.project",
    ]

    # Summary data
    SUMMARY = [
        "summary.file_count",
        "summary.file_size",
        "summary.data_categories.data_category",
        "summary.data_categories.file_count",
        "summary.experimental_strategies.experimental_strategy",
        "summary.experimental_strategies.file_count",
    ]

    @classmethod
    def all_clinical(cls) -> List[str]:
        """Get all clinical-relevant fields including sample/slide IDs."""
        return (
            cls.BASIC
            + cls.IDS  # Include sample_ids, slide_ids, etc.
            + cls.DEMOGRAPHIC
            + cls.DIAGNOSES
            + cls.TREATMENTS
            + cls.EXPOSURES
            + cls.FAMILY_HISTORIES
            + cls.PROJECT
        )

    @classmethod
    def all_biospecimen(cls) -> List[str]:
        """Get all biospecimen-relevant fields."""
        return cls.BASIC + cls.IDS + cls.SAMPLES + cls.PORTIONS + cls.SLIDES + cls.PROJECT

    @classmethod
    def all(cls) -> List[str]:
        """Get ALL available fields."""
        return (
            cls.BASIC
            + cls.IDS
            + cls.DEMOGRAPHIC
            + cls.DIAGNOSES
            + cls.TREATMENTS
            + cls.EXPOSURES
            + cls.FAMILY_HISTORIES
            + cls.SAMPLES
            + cls.PORTIONS
            + cls.SLIDES
            + cls.PROJECT
            + cls.TISSUE_SOURCE_SITE
            + cls.SUMMARY
        )

    @classmethod
    def minimal(cls) -> List[str]:
        """Get minimal fields for listing."""
        return cls.BASIC + cls.IDS + ["project.project_id"]


class FileFields:
    """Available fields for the files endpoint."""

    # Core file metadata
    BASIC = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "file_state",
        "state",
        "data_category",
        "data_type",
        "data_format",
        "experimental_strategy",
        "access",
        "platform",
        "created_datetime",
        "updated_datetime",
    ]

    # Analysis information
    ANALYSIS = [
        "analysis.analysis_id",
        "analysis.analysis_type",
        "analysis.workflow_type",
        "analysis.workflow_version",
    ]

    # Associated case information
    CASES = [
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.project.name",
        "cases.project.disease_type",
        "cases.project.primary_site",
    ]

    # Case demographic (for filtering)
    CASES_DEMOGRAPHIC = [
        "cases.demographic.gender",
        "cases.demographic.race",
        "cases.demographic.ethnicity",
    ]

    # Case diagnoses (for filtering)
    CASES_DIAGNOSES = [
        "cases.diagnoses.primary_diagnosis",
        "cases.diagnoses.tumor_stage",
        "cases.diagnoses.tumor_grade",
        "cases.diagnoses.vital_status",
    ]

    # Sample info through cases
    CASES_SAMPLES = [
        "cases.samples.sample_id",
        "cases.samples.sample_type",
        "cases.samples.tissue_type",
        "cases.samples.tumor_descriptor",
    ]

    # Associated entities
    ASSOCIATED_ENTITIES = [
        "associated_entities.entity_id",
        "associated_entities.entity_submitter_id",
        "associated_entities.entity_type",
        "associated_entities.case_id",
    ]

    # Index files
    INDEX_FILES = [
        "index_files.file_id",
        "index_files.file_name",
        "index_files.file_size",
        "index_files.data_format",
    ]

    @classmethod
    def for_slides(cls) -> List[str]:
        """Fields useful for slide image queries."""
        return cls.BASIC + cls.CASES + cls.ASSOCIATED_ENTITIES

    @classmethod
    def for_clinical(cls) -> List[str]:
        """Fields useful for clinical data queries."""
        return cls.BASIC + cls.CASES + cls.CASES_DIAGNOSES

    @classmethod
    def minimal(cls) -> List[str]:
        """Minimal fields for listing."""
        return [
            "file_id",
            "file_name",
            "file_size",
            "data_type",
            "data_category",
            "access",
            "cases.case_id",
            "cases.submitter_id",
            "cases.project.project_id",
        ]


class ProjectFields:
    """Available fields for the projects endpoint."""

    BASIC = [
        "project_id",
        "name",
        "disease_type",
        "primary_site",
        "dbgap_accession_number",
        "released",
        "state",
    ]

    PROGRAM = [
        "program.name",
        "program.program_id",
        "program.dbgap_accession_number",
    ]

    SUMMARY = [
        "summary.case_count",
        "summary.file_count",
        "summary.file_size",
        "summary.data_categories.data_category",
        "summary.data_categories.file_count",
        "summary.experimental_strategies.experimental_strategy",
        "summary.experimental_strategies.file_count",
    ]

    @classmethod
    def all(cls) -> List[str]:
        return cls.BASIC + cls.PROGRAM + cls.SUMMARY


class AnnotationFields:
    """Available fields for the annotations endpoint."""

    ALL = [
        "annotation_id",
        "case_id",
        "case_submitter_id",
        "entity_id",
        "entity_submitter_id",
        "entity_type",
        "category",
        "classification",
        "notes",
        "status",
        "creator",
        "state",
        "created_datetime",
        "updated_datetime",
    ]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GDCFile:
    """Represents a file from GDC."""

    file_id: str
    filename: str
    data_type: str
    data_category: str
    data_format: str
    file_size: int
    access: str  # "open" or "controlled"
    md5sum: Optional[str] = None
    experimental_strategy: Optional[str] = None
    platform: Optional[str] = None
    state: Optional[str] = None

    # Associated case info
    case_id: Optional[str] = None
    case_submitter_id: Optional[str] = None
    project_id: Optional[str] = None

    # Associated entities
    associated_entities: List[Dict[str, Any]] = field(default_factory=list)

    # Raw data for additional access
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_hit(cls, hit: Dict[str, Any]) -> "GDCFile":
        """Create GDCFile from API response hit."""
        # Extract case info if available
        cases = hit.get("cases", [])
        case_id = None
        case_submitter_id = None
        project_id = None

        if cases:
            case_id = cases[0].get("case_id")
            case_submitter_id = cases[0].get("submitter_id")
            project = cases[0].get("project", {})
            project_id = project.get("project_id") if isinstance(project, dict) else None

        return cls(
            file_id=hit.get("file_id", hit.get("id", "")),
            filename=hit.get("file_name", ""),
            data_type=hit.get("data_type", ""),
            data_category=hit.get("data_category", ""),
            data_format=hit.get("data_format", ""),
            file_size=hit.get("file_size", 0),
            access=hit.get("access", ""),
            md5sum=hit.get("md5sum"),
            experimental_strategy=hit.get("experimental_strategy"),
            platform=hit.get("platform"),
            state=hit.get("state"),
            case_id=case_id,
            case_submitter_id=case_submitter_id,
            project_id=project_id,
            associated_entities=hit.get("associated_entities", []),
            _raw=hit,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding _raw)."""
        d = asdict(self)
        d.pop("_raw", None)
        return d


@dataclass
class GDCCase:
    """Represents a case (patient) from GDC."""

    case_id: str
    submitter_id: str
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    disease_type: Optional[str] = None
    primary_site: Optional[str] = None
    state: Optional[str] = None

    # Demographic
    gender: Optional[str] = None
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    year_of_birth: Optional[int] = None
    year_of_death: Optional[int] = None

    # Diagnosis (first diagnosis if multiple)
    primary_diagnosis: Optional[str] = None
    tumor_stage: Optional[str] = None
    tumor_grade: Optional[str] = None
    vital_status: Optional[str] = None
    days_to_death: Optional[int] = None
    age_at_diagnosis: Optional[int] = None

    # Collections
    sample_ids: List[str] = field(default_factory=list)
    slide_ids: List[str] = field(default_factory=list)

    # Full nested data
    diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    exposures: List[Dict[str, Any]] = field(default_factory=list)

    # Raw data
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_hit(cls, hit: Dict[str, Any]) -> "GDCCase":
        """Create GDCCase from API response hit."""
        # Extract project info
        project = hit.get("project", {})
        project_id = project.get("project_id") if isinstance(project, dict) else None
        project_name = project.get("name") if isinstance(project, dict) else None
        disease_type = project.get("disease_type") if isinstance(project, dict) else None
        primary_site = project.get("primary_site") if isinstance(project, dict) else None

        # Handle disease_type and primary_site as lists
        if isinstance(disease_type, list):
            disease_type = disease_type[0] if disease_type else None
        if isinstance(primary_site, list):
            primary_site = primary_site[0] if primary_site else None

        # Extract demographic
        demo = hit.get("demographic", {}) or {}

        # Extract first diagnosis
        diagnoses = hit.get("diagnoses", []) or []
        first_diag = diagnoses[0] if diagnoses else {}

        return cls(
            case_id=hit.get("case_id", ""),
            submitter_id=hit.get("submitter_id", ""),
            project_id=project_id,
            project_name=project_name,
            disease_type=disease_type,
            primary_site=primary_site,
            state=hit.get("state"),
            # Demographic
            gender=demo.get("gender"),
            race=demo.get("race"),
            ethnicity=demo.get("ethnicity"),
            year_of_birth=demo.get("year_of_birth"),
            year_of_death=demo.get("year_of_death"),
            # Diagnosis
            primary_diagnosis=first_diag.get("primary_diagnosis"),
            tumor_stage=first_diag.get("tumor_stage"),
            tumor_grade=first_diag.get("tumor_grade"),
            vital_status=first_diag.get("vital_status"),
            days_to_death=first_diag.get("days_to_death"),
            age_at_diagnosis=first_diag.get("age_at_diagnosis"),
            # Collections
            sample_ids=hit.get("sample_ids", []) or [],
            slide_ids=hit.get("slide_ids", []) or [],
            # Full nested
            diagnoses=diagnoses,
            samples=hit.get("samples", []) or [],
            exposures=hit.get("exposures", []) or [],
            _raw=hit,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding _raw)."""
        d = asdict(self)
        d.pop("_raw", None)
        return d


@dataclass
class GDCProject:
    """Represents a GDC project."""

    project_id: str
    name: str
    program_name: str
    disease_type: List[str] = field(default_factory=list)
    primary_site: List[str] = field(default_factory=list)
    dbgap_accession_number: Optional[str] = None
    released: bool = False
    state: Optional[str] = None

    # Summary stats
    case_count: int = 0
    file_count: int = 0
    file_size: int = 0

    # Detailed summaries
    data_categories: List[Dict[str, Any]] = field(default_factory=list)
    experimental_strategies: List[Dict[str, Any]] = field(default_factory=list)

    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_hit(cls, hit: Dict[str, Any]) -> "GDCProject":
        """Create GDCProject from API response hit."""
        program = hit.get("program", {}) or {}
        summary = hit.get("summary", {}) or {}

        return cls(
            project_id=hit.get("project_id", ""),
            name=hit.get("name", ""),
            program_name=program.get("name", ""),
            disease_type=hit.get("disease_type", []) or [],
            primary_site=hit.get("primary_site", []) or [],
            dbgap_accession_number=hit.get("dbgap_accession_number"),
            released=hit.get("released", False),
            state=hit.get("state"),
            case_count=summary.get("case_count", 0),
            file_count=summary.get("file_count", 0),
            file_size=summary.get("file_size", 0),
            data_categories=summary.get("data_categories", []) or [],
            experimental_strategies=summary.get("experimental_strategies", []) or [],
            _raw=hit,
        )


@dataclass
class GDCAnnotation:
    """Represents an annotation from GDC."""

    annotation_id: str
    case_id: Optional[str] = None
    case_submitter_id: Optional[str] = None
    entity_id: Optional[str] = None
    entity_submitter_id: Optional[str] = None
    entity_type: Optional[str] = None
    category: Optional[str] = None
    classification: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None

    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_hit(cls, hit: Dict[str, Any]) -> "GDCAnnotation":
        return cls(
            annotation_id=hit.get("annotation_id", ""),
            case_id=hit.get("case_id"),
            case_submitter_id=hit.get("case_submitter_id"),
            entity_id=hit.get("entity_id"),
            entity_submitter_id=hit.get("entity_submitter_id"),
            entity_type=hit.get("entity_type"),
            category=hit.get("category"),
            classification=hit.get("classification"),
            notes=hit.get("notes"),
            status=hit.get("status"),
            _raw=hit,
        )


# =============================================================================
# Filter Builder
# =============================================================================

class FilterOp(str, Enum):
    """GDC filter operators."""

    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN = "in"
    EXCLUDE = "exclude"
    IS_MISSING = "is"
    NOT_MISSING = "not"
    AND = "and"
    OR = "or"


class GDCFilterBuilder:
    """
    Builder for constructing GDC API filters.

    Usage:
        filter = (
            GDCFilterBuilder()
            .add("cases.project.project_id", "TCGA-BRCA")
            .add("access", "open")
            .add("data_type", ["Slide Image", "Clinical Supplement"], op=FilterOp.IN)
            .build()
        )
    """

    def __init__(self):
        self._conditions: List[Dict[str, Any]] = []

    def add(
        self,
        field: str,
        value: Union[str, int, float, List[Any]],
        op: FilterOp = None,
    ) -> "GDCFilterBuilder":
        """
        Add a filter condition.

        Args:
            field: Field name (e.g., "cases.project.project_id")
            value: Value or list of values
            op: Operator (auto-detected if not provided)

        Returns:
            self for chaining
        """
        # Auto-detect operator
        if op is None:
            if isinstance(value, list):
                op = FilterOp.IN
            else:
                op = FilterOp.EQ

        # Ensure value is a list for "in" operator
        if op == FilterOp.IN and not isinstance(value, list):
            value = [value]

        self._conditions.append({
            "op": op.value if isinstance(op, FilterOp) else op,
            "content": {
                "field": field,
                "value": value,
            }
        })

        return self

    def add_exists(self, field: str, exists: bool = True) -> "GDCFilterBuilder":
        """Add a filter for field existence."""
        op = FilterOp.NOT_MISSING if exists else FilterOp.IS_MISSING
        self._conditions.append({
            "op": op.value,
            "content": {
                "field": field,
                "value": "MISSING",
            }
        })
        return self

    def build(self) -> Dict[str, Any]:
        """Build the filter dict."""
        if not self._conditions:
            return {}

        if len(self._conditions) == 1:
            return self._conditions[0]

        return {
            "op": "and",
            "content": self._conditions,
        }

    def clear(self) -> "GDCFilterBuilder":
        """Clear all conditions."""
        self._conditions = []
        return self


# =============================================================================
# Main Client
# =============================================================================

class GDCClient:
    """
    Client for querying the GDC API.

    Handles:
    - Project listing and filtering
    - Case queries with clinical data
    - File queries (slides, clinical data, etc.)
    - Annotation queries
    - Manifest generation for gdc-client downloads
    - Pagination for large result sets
    - Retry logic for transient failures

    Example:
        client = GDCClient()

        # Get all TCGA projects
        projects = client.list_projects()

        # Get cases with clinical data
        cases = client.get_cases(
            project_id="TCGA-BRCA",
            fields=CaseFields.all_clinical()
        )

        # Get slide images
        slides = client.get_files(
            project_id="TCGA-BRCA",
            data_type="Slide Image",
            access="open"
        )

        # Generate manifest for download
        manifest = client.create_manifest(slides, Path("manifest.txt"))
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = GDC_API_BASE,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize GDC client.

        Args:
            token: Authentication token for controlled-access data.
                   Get from: https://portal.gdc.cancer.gov/
            base_url: GDC API base URL
            timeout: Request timeout in seconds
            max_retries: Max retries for failed requests
        """
        self.base_url = base_url
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        if token:
            self.session.headers.update({"X-Auth-Token": token})

    # =========================================================================
    # Low-level API Methods
    # =========================================================================

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make request to GDC API with retry logic.

        Args:
            endpoint: API endpoint (e.g., "files", "cases")
            params: Query parameters
            method: HTTP method
            data: JSON body for POST requests

        Returns:
            Response JSON
        """
        url = urljoin(self.base_url + "/", endpoint)

        for attempt in range(self.max_retries):
            try:
                if method == "GET":
                    response = self.session.get(
                        url, params=params, timeout=self.timeout
                    )
                elif method == "POST":
                    response = self.session.post(
                        url, params=params, json=data, timeout=self.timeout
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise RuntimeError(f"GDC API request failed after {self.max_retries} attempts: {e}") from e

    def _paginate(
        self,
        endpoint: str,
        filters: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        size: int = DEFAULT_PAGE_SIZE,
        max_results: Optional[int] = None,
        sort: Optional[str] = None,
        expand: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Paginate through all results for a query.

        Args:
            endpoint: API endpoint
            filters: GDC filter dict
            fields: Fields to return (None = default fields from API)
            size: Page size
            max_results: Maximum total results (None for all)
            sort: Sort field (e.g., "file_size:desc")
            expand: List of fields to expand (e.g., ["cases", "cases.samples"])

        Returns:
            List of all hits
        """
        all_hits = []
        offset = 0

        while True:
            params = {
                "from": offset,
                "size": min(size, MAX_PAGE_SIZE),
            }

            if filters:
                params["filters"] = json.dumps(filters)

            if fields:
                params["fields"] = ",".join(fields)

            if expand:
                params["expand"] = ",".join(expand)

            if sort:
                params["sort"] = sort

            response = self._request(endpoint, params)
            data = response.get("data", {})
            hits = data.get("hits", [])

            if not hits:
                break

            all_hits.extend(hits)

            # Check limits
            if max_results and len(all_hits) >= max_results:
                all_hits = all_hits[:max_results]
                break

            pagination = data.get("pagination", {})
            total = pagination.get("total", 0)

            if offset + len(hits) >= total:
                break

            offset += size

        return all_hits

    def _paginate_iter(
        self,
        endpoint: str,
        filters: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        size: int = DEFAULT_PAGE_SIZE,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate through results without loading all into memory.

        Yields:
            Individual hits
        """
        offset = 0

        while True:
            params = {
                "from": offset,
                "size": min(size, MAX_PAGE_SIZE),
            }

            if filters:
                params["filters"] = json.dumps(filters)

            if fields:
                params["fields"] = ",".join(fields)

            response = self._request(endpoint, params)
            data = response.get("data", {})
            hits = data.get("hits", [])

            if not hits:
                break

            for hit in hits:
                yield hit

            pagination = data.get("pagination", {})
            total = pagination.get("total", 0)

            if offset + len(hits) >= total:
                break

            offset += size

    # =========================================================================
    # Field Discovery - Get available fields dynamically
    # =========================================================================

    def discover_fields(self, endpoint: str) -> List[str]:
        """
        Discover all available fields for an endpoint.

        Uses the _mapping endpoint to get field definitions.

        Args:
            endpoint: API endpoint ("cases", "files", "projects", "annotations")

        Returns:
            List of all available field names
        """
        try:
            response = self._request(f"{endpoint}/_mapping")
            fields = response.get("fields", [])

            # Also include any nested fields
            all_fields = []
            for field_name in fields:
                all_fields.append(field_name)

            return sorted(all_fields)

        except Exception as e:
            # Fallback: do a sample query and extract keys from response
            print(f"Warning: _mapping endpoint failed ({e}), using sample query fallback")
            return self._discover_fields_from_sample(endpoint)

    def _discover_fields_from_sample(self, endpoint: str) -> List[str]:
        """
        Fallback: discover fields by examining a sample response.

        Args:
            endpoint: API endpoint

        Returns:
            List of field names found in sample
        """
        # Get one record without specifying fields (gets defaults)
        response = self._request(endpoint, {"size": 1})
        hits = response.get("data", {}).get("hits", [])

        if not hits:
            return []

        # Recursively extract all keys
        def extract_keys(obj: Any, prefix: str = "") -> List[str]:
            keys = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    keys.append(full_key)
                    keys.extend(extract_keys(v, full_key))
            elif isinstance(obj, list) and obj:
                # Check first element
                keys.extend(extract_keys(obj[0], prefix))
            return keys

        return sorted(set(extract_keys(hits[0])))

    def get_default_fields(self, endpoint: str) -> List[str]:
        """
        Get the default fields returned by an endpoint when no fields are specified.

        Args:
            endpoint: API endpoint

        Returns:
            List of default field names
        """
        response = self._request(endpoint, {"size": 1})
        hits = response.get("data", {}).get("hits", [])

        if not hits:
            return []

        return list(hits[0].keys())

    def get_expandable_fields(self, endpoint: str) -> List[str]:
        """
        Get fields that can be expanded for an endpoint.

        These are typically nested objects like 'cases', 'samples', etc.

        Args:
            endpoint: API endpoint

        Returns:
            List of expandable field names
        """
        try:
            response = self._request(f"{endpoint}/_mapping")
            # Look for fields that have nested structure
            fields = response.get("fields", [])
            expandable = response.get("expand", [])
            return sorted(expandable) if expandable else []
        except Exception:
            # Common expandable fields by endpoint
            defaults = {
                "files": ["cases", "cases.samples", "cases.samples.portions",
                         "cases.demographic", "cases.diagnoses", "annotations"],
                "cases": ["samples", "samples.portions", "samples.portions.slides",
                         "demographic", "diagnoses", "diagnoses.treatments",
                         "exposures", "family_histories", "files", "annotations"],
                "projects": ["summary"],
            }
            return defaults.get(endpoint, [])

    # =========================================================================
    # Project Queries
    # =========================================================================

    def list_projects(
        self,
        program: Optional[str] = "TCGA",
        fields: Optional[List[str]] = None,
    ) -> List[GDCProject]:
        """
        List all projects, optionally filtered by program.

        Args:
            program: Program name (e.g., "TCGA", "TARGET"). None for all.
            fields: Fields to return (default: all)

        Returns:
            List of GDCProject objects
        """
        builder = GDCFilterBuilder()
        if program:
            builder.add("program.name", program)

        filters = builder.build()
        fields = fields or ProjectFields.all()

        hits = self._paginate("projects", filters, fields, size=100)
        return [GDCProject.from_hit(hit) for hit in hits]

    def get_project(self, project_id: str) -> Optional[GDCProject]:
        """Get a specific project by ID."""
        builder = GDCFilterBuilder().add("project_id", project_id)
        hits = self._paginate("projects", builder.build(), ProjectFields.all())
        return GDCProject.from_hit(hits[0]) if hits else None

    # =========================================================================
    # Case Queries
    # =========================================================================

    def get_cases(
        self,
        project_id: Optional[str] = None,
        case_ids: Optional[List[str]] = None,
        submitter_ids: Optional[List[str]] = None,
        primary_site: Optional[str] = None,
        disease_type: Optional[str] = None,
        gender: Optional[str] = None,
        vital_status: Optional[str] = None,
        fields: Optional[List[str]] = None,
        expand: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        custom_filter: Optional[Dict[str, Any]] = None,
    ) -> List[GDCCase]:
        """
        Query cases with various filters.

        Args:
            project_id: Filter by project
            case_ids: Filter by specific case UUIDs
            submitter_ids: Filter by submitter IDs (e.g., "TCGA-A1-A0SB")
            primary_site: Filter by primary site
            disease_type: Filter by disease type
            gender: Filter by gender
            vital_status: Filter by vital status
            fields: Fields to return (None = API defaults, use discover_fields() to see all)
            expand: Fields to expand (e.g., ["demographic", "diagnoses", "samples"])
            max_results: Maximum results
            custom_filter: Additional custom filter to merge

        Returns:
            List of GDCCase objects

        Example:
            # Get cases with expanded clinical data
            cases = client.get_cases(
                project_id="TCGA-BRCA",
                expand=["demographic", "diagnoses", "samples", "samples.portions"],
                max_results=10
            )
        """
        builder = GDCFilterBuilder()
        # For constructing the api call.. honestly should be in it's own class... the ops are pretty consistent. 
        if project_id:
            builder.add("project.project_id", project_id)
        if case_ids:
            builder.add("case_id", case_ids, FilterOp.IN)
        if submitter_ids:
            builder.add("submitter_id", submitter_ids, FilterOp.IN)
        if primary_site:
            builder.add("primary_site", primary_site)
        if disease_type:
            builder.add("disease_type", disease_type)
        if gender:
            builder.add("demographic.gender", gender)
        if vital_status:
            builder.add("diagnoses.vital_status", vital_status)

        filters = builder.build()

        # Merge custom filter if provided
        if custom_filter:
            if filters:
                filters = {
                    "op": "and",
                    "content": [filters, custom_filter]
                }
            else:
                filters = custom_filter

        # If no fields or expand specified, use sensible defaults for clinical data
        if fields is None and expand is None:
            expand = ["demographic", "diagnoses", "project", "samples"]

        hits = self._paginate("cases", filters, fields, max_results=max_results, expand=expand)
        return [GDCCase.from_hit(hit) for hit in hits]

    def get_case(self, case_id: str, fields: Optional[List[str]] = None) -> Optional[GDCCase]:
        """Get a specific case by UUID."""
        cases = self.get_cases(case_ids=[case_id], fields=fields)
        # TODO: THIS MAKES A HUGE ASSUMPTION>> SO FAR IT WORKS,BUT ACCESSING NUMBER ONE CASE CAN BE DANGEROUS 
        return cases[0] if cases else None

    def get_case_by_submitter_id(
        self, submitter_id: str, fields: Optional[List[str]] = None
    ) -> Optional[GDCCase]:
        """Get a case by submitter ID (e.g., 'TCGA-A1-A0SB')."""
        cases = self.get_cases(submitter_ids=[submitter_id], fields=fields)
        return cases[0] if cases else None

    # =========================================================================
    # File Queries
    # =========================================================================

    def get_files(
        self,
        project_id: Optional[str] = None,
        case_id: Optional[str] = None,
        case_submitter_id: Optional[str] = None,
        data_type: Optional[Union[str, List[str]]] = None,
        data_category: Optional[Union[str, List[str]]] = None,
        data_format: Optional[Union[str, List[str]]] = None,
        experimental_strategy: Optional[str] = None,
        access: Optional[Literal["open", "controlled"]] = None,
        fields: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        custom_filter: Optional[Dict[str, Any]] = None,
    ) -> List[GDCFile]:
        """
        Query files with various filters.

        Args:
            project_id: Filter by project
            case_id: Filter by case UUID
            case_submitter_id: Filter by case submitter ID
            data_type: Filter by data type(s) (e.g., "Slide Image")
            data_category: Filter by data category(s) (e.g., "Biospecimen")
            data_format: Filter by format(s) (e.g., "SVS")
            experimental_strategy: Filter by strategy (e.g., "Diagnostic Slide")
            access: Filter by access level
            fields: Fields to return
            max_results: Maximum results
            custom_filter: Additional custom filter

        Returns:
            List of GDCFile objects
        """
        builder = GDCFilterBuilder()
        # TODO: once again redundant building of api request package. this should be refactored.
        if project_id:
            builder.add("cases.project.project_id", project_id)
        if case_id:
            builder.add("cases.case_id", case_id)
        if case_submitter_id:
            builder.add("cases.submitter_id", case_submitter_id)
        if data_type:
            builder.add("data_type", data_type, FilterOp.IN if isinstance(data_type, list) else FilterOp.EQ)
        if data_category:
            builder.add("data_category", data_category, FilterOp.IN if isinstance(data_category, list) else FilterOp.EQ)
        if data_format:
            builder.add("data_format", data_format, FilterOp.IN if isinstance(data_format, list) else FilterOp.EQ)
        if experimental_strategy:
            builder.add("experimental_strategy", experimental_strategy)
        if access:
            builder.add("access", access)

        filters = builder.build()

        if custom_filter:
            if filters:
                filters = {"op": "and", "content": [filters, custom_filter]}
            else:
                filters = custom_filter

        fields = fields or FileFields.minimal()

        hits = self._paginate("files", filters, fields, max_results=max_results)
        return [GDCFile.from_hit(hit) for hit in hits]

    def get_slide_images(
        self,
        project_id: str,
        access: Literal["open", "controlled"] = "open",
        experimental_strategy: Optional[str] = "Diagnostic Slide",
        max_results: Optional[int] = None,
    ) -> List[GDCFile]:
        """
        Get slide image files for a project.

        Args:
            project_id: TCGA project ID (e.g., "TCGA-BRCA")
            access: Access level filter
            experimental_strategy: "Diagnostic Slide" or "Tissue Slide"
            max_results: Maximum results

        Returns:
            List of slide image files
        """
        return self.get_files(
            project_id=project_id,
            data_type="Slide Image",
            access=access,
            experimental_strategy=experimental_strategy,
            fields=FileFields.for_slides(),
            max_results=max_results,
        )

    def get_clinical_files(
        self,
        project_id: str,
        access: Literal["open", "controlled"] = "open",
        max_results: Optional[int] = None,
    ) -> List[GDCFile]:
        """Get clinical supplement files (XMLs) for a project."""
        return self.get_files(
            project_id=project_id,
            data_category="Clinical",
            access=access,
            fields=FileFields.for_clinical(),
            max_results=max_results,
        )

    def get_biospecimen_files(
        self,
        project_id: str,
        access: Literal["open", "controlled"] = "open", 
        max_results: Optional[int] = None,
    ) -> List[GDCFile]:
        """Get biospecimen supplement files for a project.
        
        """
        return self.get_files(
            project_id=project_id,
            data_category="Biospecimen",
            access=access,
            max_results=max_results,
        )

    # =========================================================================
    # Annotation Queries
    # =========================================================================

    def get_annotations(
        self,
        project_id: Optional[str] = None,
        case_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        category: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[GDCAnnotation]:
        """
        Query annotations.

        Args:
            project_id: Filter by project
            case_id: Filter by case
            entity_type: Filter by entity type
            category: Filter by category
            max_results: Maximum results

        Returns:
            List of GDCAnnotation objects
        """
        builder = GDCFilterBuilder()
        # TODO: im sorry but the builder should be much cleaner.. api request building should be not a bunch of if statments.. or atleast hide it in the builder, pass in the details and let it handle the, build... albeit for alpha build this is fine for error checking. 
        if project_id:
            builder.add("project.project_id", project_id)
        if case_id:
            builder.add("case_id", case_id)
        if entity_type:
            builder.add("entity_type", entity_type)
        if category:
            builder.add("category", category)

        hits = self._paginate(
            "annotations",
            builder.build(),
            AnnotationFields.ALL,
            max_results=max_results,
        )
        return [GDCAnnotation.from_hit(hit) for hit in hits]

    # =========================================================================
    # Manifest Generation
    # =========================================================================

    def create_manifest(
        self,
        files: List[GDCFile],
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Create a manifest file for gdc-client download.

        Args:
            files: List of GDCFile objects
            output_path: Optional path to save manifest

        Returns:
            Manifest content as string
        """
        lines = ["id\tfilename\tmd5\tsize\tstate"]

        for f in files:
            lines.append(
                f"{f.file_id}\t{f.filename}\t{f.md5sum or ''}\t{f.file_size}\t{f.state or ''}"
            )

        manifest = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(manifest)

        return manifest

    def create_manifest_from_query(
        self,
        output_path: Path,
        project_id: Optional[str] = None,
        data_type: Optional[str] = None,
        data_category: Optional[str] = None,
        access: Optional[Literal["open", "controlled"]] = None,
        **kwargs,
    ) -> str:
        """
        Create manifest directly from a query.

        Args:
            output_path: Path to save manifest
            project_id: Project filter
            data_type: Data type filter
            data_category: Data category filter
            access: Access level filter
            **kwargs: Additional filters for get_files()

        Returns:
            Manifest content
        """
        files = self.get_files(
            project_id=project_id,
            data_type=data_type,
            data_category=data_category,
            access=access,
            **kwargs,
        )
        return self.create_manifest(files, output_path)

    # =========================================================================
    # Download Methods
    # =========================================================================

    def get_download_url(self, file_id: str) -> str:
        """Get direct download URL for a file."""
        return f"{self.base_url}/data/{file_id}"

    def download_file(
        self,
        file_id: str,
        output_path: Path,
        chunk_size: int = 8192,
    ) -> Path:
        """
        Download a single file.

        Args:
            file_id: GDC file UUID
            output_path: Path to save file
            chunk_size: Download chunk size

        Returns:
            Path to downloaded file
        """
        url = self.get_download_url(file_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        response = self.session.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        return output_path

    # =========================================================================
    # Summary & Stats
    # =========================================================================

    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """
        Get detailed summary statistics for a project.

        Returns dict with file counts by type, access level, total size, etc.
        """
        project = self.get_project(project_id)
        if not project:
            return {"error": f"Project {project_id} not found"}

        files = self.get_files(project_id=project_id, fields=FileFields.BASIC)

        summary = {
            "project_id": project_id,
            "name": project.name,
            "program": project.program_name,
            "disease_type": project.disease_type,
            "primary_site": project.primary_site,
            "case_count": project.case_count,
            "total_files": len(files),
            "by_data_type": {},
            "by_data_category": {},
            "by_access": {"open": 0, "controlled": 0},
            "by_format": {},
            "total_size_bytes": 0,
        }

        for f in files:
            # By data type
            dt = f.data_type or "unknown"
            summary["by_data_type"][dt] = summary["by_data_type"].get(dt, 0) + 1

            # By data category
            dc = f.data_category or "unknown"
            summary["by_data_category"][dc] = summary["by_data_category"].get(dc, 0) + 1

            # By access
            if f.access in summary["by_access"]:
                summary["by_access"][f.access] += 1

            # By format
            fmt = f.data_format or "unknown"
            summary["by_format"][fmt] = summary["by_format"].get(fmt, 0) + 1

            # Size
            summary["total_size_bytes"] += f.file_size

        summary["total_size_gb"] = round(summary["total_size_bytes"] / (1024**3), 2)

        return summary

    def get_available_data_types(self, project_id: Optional[str] = None) -> List[str]:
        """Get list of available data types, optionally for a specific project."""
        # Use facets to get unique values
        params = {"facets": "data_type", "size": 0}

        if project_id:
            builder = GDCFilterBuilder().add("cases.project.project_id", project_id)
            params["filters"] = json.dumps(builder.build())

        response = self._request("files", params)
        facets = response.get("data", {}).get("aggregations", {}).get("data_type", {})
        return [bucket["key"] for bucket in facets.get("buckets", [])]

    def get_available_experimental_strategies(self, project_id: Optional[str] = None) -> List[str]:
        """Get list of available experimental strategies."""
        params = {"facets": "experimental_strategy", "size": 0}

        if project_id:
            builder = GDCFilterBuilder().add("cases.project.project_id", project_id)
            params["filters"] = json.dumps(builder.build())

        response = self._request("files", params)
        facets = response.get("data", {}).get("aggregations", {}).get("experimental_strategy", {})
        return [bucket["key"] for bucket in facets.get("buckets", [])]


# =============================================================================
# Test Suite
# =============================================================================
#TODO: move to dedicated testing
def run_tests():
    """
    Comprehensive test suite for GDC API Client.

    Run with: python src/data/tcga_api_wrapper.py

    Tests (13 total):
    1. List TCGA Projects
    2. Get Single Project (TCGA-BRCA)
    3. Dynamic Field Discovery
    4. Get Cases with Expanded Data
    5. Get Slide Images
    6. Get Clinical Supplement Files
    7. Get Biospecimen Files
    8. Filter Builder - Complex Query
    9. Get Annotations
    10. Create Download Manifest
    11. Get Available Data Types
    12. Get Available Experimental Strategies
    13. Case Filtering (by gender, vital status)
    """
    import traceback

    print("=" * 80)
    print("GDC API CLIENT - TEST SUITE")
    print("=" * 80)
    print("\nThis will make real API calls to GDC. Testing with open-access data only.\n")

    client = GDCClient()
    results = {}

    # -------------------------------------------------------------------------
    # TEST 1: List Projects
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 1: List TCGA Projects")
    print("-" * 80)
    try:
        projects = client.list_projects(program="TCGA")
        results["test_1_projects"] = len(projects)

        print(f"  Found {len(projects)} TCGA projects\n")
        print(f"  {'Project ID':<15} | {'Cases':>6} | {'Files':>8} | Name")
        print(f"  {'-'*15} | {'-'*6} | {'-'*8} | {'-'*40}")
        for p in projects[:10]:
            print(f"  {p.project_id:<15} | {p.case_count:>6} | {p.file_count:>8} | {p.name[:40]}")
        # if len(projects) > 10:
        #     print(f"  ... and {len(projects) - 10} more projects")

        print(f"\n  [PASS] Retrieved {len(projects)} projects")

        # Verify data structure
        sample = projects[0]
        print(f"\n  Sample project data structure:")
        print(f"    project_id: {sample.project_id}")
        print(f"    name: {sample.name}")
        print(f"    program_name: {sample.program_name}")
        print(f"    disease_type: {sample.disease_type}")
        print(f"    primary_site: {sample.primary_site}")
        print(f"    case_count: {sample.case_count}")
        print(f"    file_count: {sample.file_count}")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_1_projects"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 2: Get Single Project
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 2: Get Single Project (TCGA-BRCA)")
    print("-" * 80)
    try:
        project = client.get_project("TCGA-BRCA")
        results["test_2_single_project"] = project is not None

        if project:
            print(f"  Project ID: {project.project_id}")
            print(f"  Name: {project.name}")
            print(f"  Program: {project.program_name}")
            print(f"  Disease Type: {project.disease_type}")
            print(f"  Primary Site: {project.primary_site}")
            print(f"  Cases: {project.case_count}")
            print(f"  Files: {project.file_count}")
            print(f"  Total Size: {project.file_size / 1e12:.2f} TB")

            print(f"\n  Data Categories:")
            for cat in project.data_categories[:5]:
                print(f"    {cat.get('data_category', 'N/A')}: {cat.get('file_count', 0)} files")

            print(f"\n  [PASS] Retrieved TCGA-BRCA project")
        else:
            print(f"  [FAIL] Project not found")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_2_single_project"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 3: Dynamic Field Discovery
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 3: Dynamic Field Discovery (cases endpoint)")
    print("-" * 80)
    try:
        # Discover default fields
        default_fields = client.get_default_fields("cases")
        results["test_3_default_fields"] = len(default_fields)

        print(f"  Default fields returned by API ({len(default_fields)}):")
        for f in default_fields[:15]:
            print(f"    - {f}")
        if len(default_fields) > 15:
            print(f"    ... and {len(default_fields) - 15} more")

        # Get expandable fields
        expandable = client.get_expandable_fields("cases")
        print(f"\n  Expandable fields ({len(expandable)}):")
        for f in expandable:
            print(f"    - {f}")

        print(f"\n  [PASS] Field discovery working")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_3_default_fields"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 4: Get Cases with Expanded Data (using expand, not hardcoded fields)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 4: Get Cases with Expanded Data (TCGA-BRCA, first 5)")
    print("-" * 80)
    try:
        # Use expand to get nested data - this is the dynamic approach
        cases = client.get_cases(
            project_id="TCGA-BRCA",
            expand=["demographic", "diagnoses", "samples", "samples.portions"],
            max_results=5
        )
        results["test_4_cases"] = len(cases)

        print(f"  Retrieved {len(cases)} cases using expand parameter\n")

        for c in cases:
            print(f"  Case: {c.submitter_id}")
            print(f"    Gender: {c.gender}")
            print(f"    Race: {c.race}")
            print(f"    Ethnicity: {c.ethnicity}")
            print(f"    Primary Diagnosis: {c.primary_diagnosis}")
            print(f"    Tumor Stage: {c.tumor_stage}")
            print(f"    Vital Status: {c.vital_status}")
            print(f"    Age at Diagnosis: {c.age_at_diagnosis}")
            print(f"    Samples: {len(c.samples)} (from expanded data)")
            print(f"    Sample IDs: {len(c.sample_ids)}")
            print(f"    Slide IDs: {len(c.slide_ids)}")

            # Show raw sample data if available
            if c.samples:
                sample = c.samples[0]
                print(f"    First sample: {sample.get('sample_type', 'N/A')} - {sample.get('tissue_type', 'N/A')}")
            print()

        print(f"  [PASS] Retrieved {len(cases)} cases with expanded data")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_4_cases"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 5: Get Slide Images (Open Access)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 5: Get Slide Images (TCGA-BRCA, open access, first 5)")
    print("-" * 80)
    try:
        slides = client.get_slide_images(
            project_id="TCGA-BRCA",
            access="open",
            max_results=5
        )
        results["test_5_slides"] = len(slides)

        print(f"  Retrieved {len(slides)} slide images\n")

        print(f"  {'Filename':<55} | {'Size (MB)':>10} | Case ID")
        print(f"  {'-'*55} | {'-'*10} | {'-'*20}")
        for s in slides:
            size_mb = s.file_size / 1e6
            print(f"  {s.filename[:55]:<55} | {size_mb:>10.1f} | {s.case_submitter_id or 'N/A'}")

        # Show full data structure for first slide
        if slides:
            s = slides[0]
            print(f"\n  Sample slide data structure:")
            print(f"    file_id: {s.file_id}")
            print(f"    filename: {s.filename}")
            print(f"    data_type: {s.data_type}")
            print(f"    data_category: {s.data_category}")
            print(f"    data_format: {s.data_format}")
            print(f"    experimental_strategy: {s.experimental_strategy}")
            print(f"    file_size: {s.file_size} bytes")
            print(f"    access: {s.access}")
            print(f"    md5sum: {s.md5sum}")
            print(f"    case_id: {s.case_id}")
            print(f"    case_submitter_id: {s.case_submitter_id}")
            print(f"    project_id: {s.project_id}")

        print(f"\n  [PASS] Retrieved {len(slides)} slide images")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_5_slides"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 6: Get Clinical Files
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 6: Get Clinical Supplement Files (TCGA-BRCA, first 10)")
    print("-" * 80)
    try:
        clinical = client.get_clinical_files(
            project_id="TCGA-BRCA",
            access="open",
            max_results=10
        )
        results["test_6_clinical"] = len(clinical)

        print(f"  Retrieved {len(clinical)} clinical files\n")

        print(f"  {'Filename':<60} | {'Type':<25} | Format")
        print(f"  {'-'*60} | {'-'*25} | {'-'*10}")
        for c in clinical:
            print(f"  {c.filename[:60]:<60} | {c.data_type[:25]:<25} | {c.data_format}")

        print(f"\n  [PASS] Retrieved {len(clinical)} clinical files")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_6_clinical"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 7: Get Biospecimen Files
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 7: Get Biospecimen Files (TCGA-BRCA, first 5)")
    print("-" * 80)
    try:
        biospecimen = client.get_biospecimen_files(
            project_id="TCGA-BRCA",
            access="open",
            max_results=5
        )
        results["test_7_biospecimen"] = len(biospecimen)

        print(f"  Retrieved {len(biospecimen)} biospecimen files\n")

        for b in biospecimen:
            print(f"  {b.filename}")
            print(f"    Type: {b.data_type}, Format: {b.data_format}")

        print(f"\n  [PASS] Retrieved {len(biospecimen)} biospecimen files")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_7_biospecimen"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 8: Filter Builder
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 8: Filter Builder - Complex Query")
    print("-" * 80)
    try:
        # Build a complex filter
        builder = GDCFilterBuilder()
        builder.add("cases.project.project_id", "TCGA-BRCA")
        builder.add("access", "open")
        builder.add("data_type", ["Slide Image", "Clinical Supplement"], FilterOp.IN)

        filter_dict = builder.build()
        print(f"  Built filter:\n{json.dumps(filter_dict, indent=4)}")

        # Use it in a query
        files = client.get_files(
            custom_filter=filter_dict,
            max_results=5
        )
        results["test_8_filter_builder"] = len(files)

        print(f"\n  Query returned {len(files)} files")
        for f in files:
            print(f"    {f.data_type}: {f.filename[:50]}")

        print(f"\n  [PASS] Filter builder working")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_8_filter_builder"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 9: Get Annotations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 9: Get Annotations (TCGA-BRCA, first 5)")
    print("-" * 80)
    try:
        annotations = client.get_annotations(
            project_id="TCGA-BRCA",
            max_results=5
        )
        results["test_9_annotations"] = len(annotations)

        print(f"  Retrieved {len(annotations)} annotations\n")

        for a in annotations:
            print(f"  Annotation: {a.annotation_id}")
            print(f"    Case: {a.case_submitter_id}")
            print(f"    Entity Type: {a.entity_type}")
            print(f"    Category: {a.category}")
            print(f"    Classification: {a.classification}")
            print(f"    Notes: {(a.notes or '')[:100]}...")
            print()

        print(f"  [PASS] Retrieved {len(annotations)} annotations")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_9_annotations"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 10: Create Manifest
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 10: Create Download Manifest")
    print("-" * 80)
    try:
        # Get a few files
        files = client.get_slide_images("TCGA-BRCA", access="open", max_results=3)

        # Create manifest (don't save to file)
        manifest = client.create_manifest(files)
        results["test_10_manifest"] = len(manifest.split("\n"))

        print(f"  Created manifest for {len(files)} files:\n")
        print("  " + "-" * 76)
        for line in manifest.split("\n")[:5]:
            # Truncate long lines for display
            if len(line) > 76:
                print(f"  {line[:76]}...")
            else:
                print(f"  {line}")
        print("  " + "-" * 76)

        print(f"\n  [PASS] Manifest created with {len(files)} files")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_10_manifest"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 11: Get Available Data Types
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 11: Get Available Data Types (TCGA-BRCA)")
    print("-" * 80)
    try:
        data_types = client.get_available_data_types(project_id="TCGA-BRCA")
        results["test_11_data_types"] = len(data_types)

        print(f"  Available data types in TCGA-BRCA:\n")
        for dt in data_types:
            print(f"    - {dt}")

        print(f"\n  [PASS] Found {len(data_types)} data types")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_11_data_types"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 12: Get Available Experimental Strategies
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 12: Get Available Experimental Strategies (TCGA-BRCA)")
    print("-" * 80)
    try:
        strategies = client.get_available_experimental_strategies(project_id="TCGA-BRCA")
        results["test_12_strategies"] = len(strategies)

        print(f"  Available experimental strategies in TCGA-BRCA:\n")
        for s in strategies:
            print(f"    - {s}")

        print(f"\n  [PASS] Found {len(strategies)} experimental strategies")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_12_strategies"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # TEST 13: Case Filtering (by gender, vital status)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 13: Case Filtering (female, deceased, TCGA-BRCA)")
    print("-" * 80)
    try:
        cases = client.get_cases(
            project_id="TCGA-BRCA",
            gender="female",
            vital_status="Dead",
            max_results=5
        )
        results["test_13_case_filter"] = len(cases)

        print(f"  Retrieved {len(cases)} female deceased cases\n")

        for c in cases:
            print(f"  {c.submitter_id}: {c.gender}, {c.vital_status}, died day {c.days_to_death}")

        print(f"\n  [PASS] Case filtering working")

    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results["test_13_case_filter"] = f"ERROR: {e}"

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        if isinstance(result, str) and result.startswith("ERROR"):
            status = "FAIL"
            failed += 1
        else:
            status = "PASS"
            passed += 1

        print(f"  {test_name}: {status} (result: {result})")

    print(f"\n  Total: {passed} passed, {failed} failed")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_tests()
