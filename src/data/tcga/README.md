# TCGA / GDC Data Module

This module provides tools for querying, downloading, and managing TCGA (The Cancer Genome Atlas) data from the GDC (Genomic Data Commons).

## What is TCGA?

TCGA is a landmark cancer genomics program that molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. The data is hosted by the NCI's Genomic Data Commons.

## What is GDC?

The Genomic Data Commons (GDC) is a data sharing platform that hosts TCGA and other cancer genomics datasets. It provides a REST API to query and download data programmatically.

---

## Module Architecture

```
src/data/tcga/
├── __init__.py        # Exports all public classes
├── gdc_client.py      # Generic GDC API wrapper (queries only)
├── hierarchy.py       # Builds hierarchy index (case → sample → portion → slide)
├── etl.py             # Flat table builder with proper hierarchy broadcasting
├── config.py          # Configuration dataclass
├── manifest.py        # Manifest file generation for gdc-client
├── downloader.py      # Download orchestration
├── gene_matrix.py     # Gene-level mutation matrix from MAF files
├── README.md          # This file
└── notebooks/
    ├── tutorial_one.ipynb      # GDC Client basics
    └── tutorial_two_etl.ipynb  # ETL pipeline, downloads & gene matrix
```

### Layer Responsibilities

| Module | Responsibility |
|--------|---------------|
| `gdc_client.py` | Generic API wrapper - queries GDC, no ETL logic |
| `hierarchy.py` | Builds index: slide_id → {sample_id, case_id, ...} |
| `etl.py` | Builds flat DataFrames with hierarchy broadcasting |
| `config.py` | Configuration management (directories, project selection) |
| `manifest.py` | Creates manifest files for gdc-client downloads |
| `downloader.py` | Orchestrates downloads, tracks status |
| `gene_matrix.py` | Gene-level one-hot encoding from MAF files |

---

## Installation

```bash
pip install requests pandas


```

---

## Quick Start

### Option 1: Just Query Data

```python
from src.data.tcga import GDCClient

client = GDCClient()
projects = client.list_projects(program="TCGA")
for p in projects:
    print(f"{p.project_id}: {p.case_count} patients")
```

### Option 2: Full ETL Pipeline

```python
from pathlib import Path
from src.data.tcga import (
    TCGAConfig,
    TCGASlideETL,
    ManifestGenerator,
    TCGADownloader,
)

# 1. Configure
config = TCGAConfig(
    project_ids=["TCGA-LUAD"],
    data_dir=Path("data/tcga"),
)
config.ensure_directories()

# 2. Build flat table
etl = TCGASlideETL()
df = etl.build_slide_table(
    project_ids=config.project_ids,
    include_demographics=True,
    include_diagnosis=True,
    include_maf=True,
)
df = etl.add_local_paths(df, config)

# 3. Generate manifests
manifest_gen = ManifestGenerator()
slide_manifest = manifest_gen.create_slide_manifest(df, config.manifests_dir / "slides.txt")
maf_manifest = manifest_gen.create_maf_manifest(df, config.manifests_dir / "maf.txt")

# 4. Download (or use gdc-client directly)
downloader = TCGADownloader()
result = downloader.download_from_manifest(slide_manifest, config.slides_dir)
print(f"Downloaded: {result.files_downloaded}/{result.files_total}")

# 5. Save table
df.to_csv(config.tables_dir / "slides.csv", index=False)
```

---

## Data Structure

### TCGA Hierarchy

```
TCGA Program
└── Project (e.g., TCGA-LUAD)
    └── Case (patient)
        ├── demographic (gender, race, ethnicity)
        ├── diagnoses (cancer type, stage, survival)
        └── samples (tumor, normal tissue)
            └── portions
                ├── slides → files (SVS images)
                └── analytes → aliquots → files (MAF, sequencing)
```

**Important:** MAF files are linked at the **aliquot** level, not the sample level. The `Tumor_Sample_UUID` in MAF files is actually an aliquot UUID. GeneMatrix resolves this via the GDC API to properly link mutations to samples.

### ETL Output: Flat Table

The ETL creates a flat DataFrame where **each row is a slide image file**:

| Column Level | Columns |
|--------------|---------|
| File | file_id, filename, file_size, md5sum, slide_local_path |
| Slide | slide_id, percent_tumor_cells, percent_necrosis |
| Portion | portion_id, is_ffpe |
| Sample | sample_id, sample_type, tissue_type |
| Case | case_id, gender, race, primary_diagnosis, tumor_stage, vital_status |
| MAF | maf_file_id, maf_filename, maf_local_path, has_maf |

**Key concept:** Parent-level data is **broadcast** down to all child slides.

---

## Module Details

### TCGAConfig

Configuration dataclass for the pipeline:

```python
from src.data.tcga import TCGAConfig

config = TCGAConfig(
    project_ids=["TCGA-LUAD", "TCGA-LUSC"],  # Required
    data_dir=Path("data/tcga"),               # Base directory
    include_demographics=True,
    include_diagnosis=True,
    include_maf=True,
    access="open",  # "open" or "controlled"
)

# Computed paths
config.slides_dir     # data/tcga/slides
config.maf_dir        # data/tcga/maf
config.manifests_dir  # data/tcga/manifests
config.tables_dir     # data/tcga/tables

# Create directories
config.ensure_directories()
```

### TCGASlideETL

Builds flat tables with proper hierarchy broadcasting:

```python
from src.data.tcga import TCGASlideETL

etl = TCGASlideETL()

# Build table
df = etl.build_slide_table(
    project_ids=["TCGA-LUAD"],
    include_demographics=True,
    include_diagnosis=True,
    include_maf=True,
    access="open",
)

# Add local file paths
df = etl.add_local_paths(df, config)
```

**Output columns:** 34+ columns including file metadata, slide data, sample data, case data (demographics, diagnosis), and MAF linkage.

### ManifestGenerator

Creates manifest files for gdc-client:

```python
from src.data.tcga import ManifestGenerator

manifest_gen = ManifestGenerator()

# Full manifests
slide_manifest = manifest_gen.create_slide_manifest(df, Path("slides.txt"))
maf_manifest = manifest_gen.create_maf_manifest(df, Path("maf.txt"))

# Subset manifest for testing (first N files)
test_manifest = manifest_gen.create_subset_manifest(
    manifest_path=slide_manifest,
    output_path=Path("test.txt"),
    max_files=2,
)
```

### TCGADownloader

Orchestrates downloads using gdc-client:

```python
from src.data.tcga import TCGADownloader, DownloadStatus

downloader = TCGADownloader()

# Check status
status = downloader.check_download_status(output_dir, manifest_path)
print(f"Status: {status.status.value}")  # not_started, in_progress, completed

# Download
result = downloader.download_from_manifest(
    manifest_path=manifest_path,
    output_dir=output_dir,
    n_processes=4,
)
print(f"Downloaded: {result.files_downloaded}/{result.files_total}")
```

**Resume:** gdc-client automatically resumes interrupted downloads.

### HierarchyBuilder

Low-level utility for building hierarchy indices:

```python
from src.data.tcga import GDCClient, HierarchyBuilder

client = GDCClient()
cases = client._paginate(
    "cases",
    filters={"op": "=", "content": {"field": "project.project_id", "value": "TCGA-LUAD"}},
    expand=["samples", "samples.portions", "samples.portions.slides"],
)

builder = HierarchyBuilder()
index = builder.build_index(cases)
# index[slide_id] → HierarchyNode with case_id, sample_id, etc.
```

### GeneMatrix

Builds gene-level one-hot encoding from downloaded MAF files:

```python
from src.data.tcga import GeneMatrix, GDCClient

# Build from downloaded MAF files
# Requires GDCClient to resolve aliquot → sample mapping
client = GDCClient()
gm = GeneMatrix(client=client)
gm.build_from_maf_dir(config.maf_dir)

print(gm)  # GeneMatrix(samples=510, genes=15234)
print(gm.shape)  # (510, 15234) - samples x genes

# Save for reuse
gm.save(config.tables_dir / "gene_matrix.parquet")

# Load existing (no client needed for loading)
gm = GeneMatrix.load(config.tables_dir / "gene_matrix.parquet")

# Get subset of genes
tp53_kras = gm.subset(genes=["TP53", "KRAS", "EGFR"])

# Merge with slide table (all genes)
df_with_genes = gm.merge(slide_df)

# Merge with specific genes only
df_with_genes = gm.merge(slide_df, genes=["TP53", "KRAS"])
```

**Key features:**
- Resolves aliquot → sample via GDC API (MAF files use aliquot UUIDs, not sample UUIDs)
- Joins on `sample_id` (UUID) for proper linking with slide table
- Left join preserves all slides - slides without MAF get 0 for all genes
- Subset to specific genes of interest
- Saves to parquet for efficient storage/loading

---

## File Types

| File Type | Format | Description | Access |
|-----------|--------|-------------|--------|
| Slide Image | .svs | Whole slide microscopy images | Open |
| MAF | .maf.gz | Masked somatic mutations | Open |
| Clinical | .xml | Clinical supplement data | Open |
| Gene Expression | various | RNA-seq quantification | Mixed |

### Download Structure

gdc-client creates:
```
output_dir/
├── <file_uuid_1>/
│   └── <filename.svs>
├── <file_uuid_2>/
│   └── <filename.maf.gz>
```

**No unpacking needed** - SVS and MAF files are ready to use.

---

## Tutorials

### Tutorial 1: GDC Client Basics
`notebooks/tutorial_one.ipynb`
- List projects
- Query cases with clinical data
- Get slide images
- Generate manifests

### Tutorial 2: ETL Pipeline & Gene Matrix
`notebooks/tutorial_two_etl.ipynb`
- Configure pipeline
- Build flat slide tables
- Add local paths
- Generate manifests
- Download files (slides and MAF)
- Build gene mutation matrix from MAF files
- Merge gene data with slide table

---

## Common Tasks

### Get Slide Images for Deep Learning

```python
from src.data.tcga import TCGAConfig, TCGASlideETL

config = TCGAConfig(project_ids=["TCGA-LUAD"])
etl = TCGASlideETL()

df = etl.build_slide_table(
    project_ids=config.project_ids,
    include_maf=False,  # Don't need MAF for image analysis
)
df = etl.add_local_paths(df, config)

# Filter to tumor slides only
tumor_df = df[df['tissue_type'] == 'Tumor']
print(f"Tumor slides: {len(tumor_df)}")
```

### Link Slides to Mutations

```python
# MAF files link at SAMPLE level
df = etl.build_slide_table(["TCGA-LUAD"], include_maf=True)

# All slides from same sample share the same MAF
# Normal tissue samples don't have MAF
print(df.groupby('sample_type')['has_maf'].mean())
```

### Test Download with Subset

```python
# Create subset manifest for testing
test_manifest = manifest_gen.create_subset_manifest(
    slide_manifest,
    config.manifests_dir / "test.txt",
    max_files=2,
)

# Download just 2 files
result = downloader.download_from_manifest(test_manifest, config.slides_dir)
```

### Build Gene Mutation Matrix

```python
from src.data.tcga import GeneMatrix, GDCClient

# After downloading MAF files...
client = GDCClient()
gm = GeneMatrix(client=client)
gm.build_from_maf_dir(config.maf_dir)

# Save for reuse
gm.save(config.tables_dir / "gene_matrix.parquet")

# Merge specific genes with slide table
df_with_genes = gm.merge(slide_df, genes=["TP53", "KRAS", "EGFR"])

# Now each slide row has TP53, KRAS, EGFR columns (1=mutated, 0=not)
```

---

## External Resources

- **GDC Portal**: https://portal.gdc.cancer.gov/
- **GDC API Docs**: https://docs.gdc.cancer.gov/API/Users_Guide/
- **gdc-client**: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
- **TCGA Overview**: https://www.cancer.gov/tcga

---

## Troubleshooting

**"project_ids cannot be empty"**
- TCGAConfig requires you to specify at least one project

**"gdc-client not found"**
- Install with: `pip install gdc-client`

**"Timeout errors"**
- Increase timeout: `GDCClient(timeout=60)`
- Try smaller queries first

**"Some slides don't have MAF"**
- Normal tissue samples don't have MAF (MAF is for tumor mutations)
- Not all tumor samples were sequenced

**"Could not resolve any aliquot → sample mappings"**
- GeneMatrix needs to query GDC API to map aliquot UUIDs to sample UUIDs
- Ensure you have internet connectivity
- Check that the MAF files contain valid `Tumor_Sample_UUID` values

**"No overlapping samples after merge"**
- The sample_ids in gene matrix don't match slide table
- Verify MAF files are from the same project as your slides
- Check that aliquot → sample resolution succeeded
