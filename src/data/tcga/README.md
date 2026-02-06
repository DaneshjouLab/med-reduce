# TCGA / GDC Data Module

This module provides tools for querying and managing TCGA (The Cancer Genome Atlas) data from the GDC (Genomic Data Commons).

## What is TCGA?

TCGA is a landmark cancer genomics program that molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. The data is hosted by the NCI's Genomic Data Commons.

## What is GDC?

The Genomic Data Commons (GDC) is a data sharing platform that hosts TCGA and other cancer genomics datasets. It provides a REST API to query and download data programmatically.

**Important:** You don't need to know the exact field names or data structure ahead of time. This client can discover available fields dynamically.

---

## Installation

No additional dependencies beyond `requests` (already in requirements):

```bash
pip install requests
```

---

## Quick Start

```python
from src.data.tcga import GDCClient

# Create client (no authentication needed for open-access data)
client = GDCClient()

# What projects are available?
projects = client.list_projects(program="TCGA")
for p in projects:
    print(f"{p.project_id}: {p.name} ({p.case_count} patients)")
```

---

## Understanding the Data Structure

TCGA data is organized hierarchically. Here's what each level means in plain terms:

```
TCGA (the program)
│
├── TCGA-BRCA (Breast Cancer project)
├── TCGA-LUAD (Lung Adenocarcinoma project)
├── TCGA-SKCM (Skin Cutaneous Melanoma project)
│   ... 33 cancer types total
│
└── Each project contains CASES (patients)
         │
         ├── Who is this patient?
         │   └── demographic: age, gender, race, ethnicity
         │
         ├── What cancer do they have?
         │   └── diagnoses: cancer type, tumor stage, tumor grade
         │                  when diagnosed, survival status
         │
         ├── What tissue samples were collected?
         │   └── samples: tumor vs normal tissue, fresh vs frozen
         │        └── portions: subdivisions of the sample
         │             └── slides: the actual microscope slides
         │                         (with pathologist annotations like
         │                          % tumor cells, % necrosis)
         │
         ├── Lifestyle/environmental factors?
         │   └── exposures: smoking history, alcohol use, BMI
         │
         └── Family cancer history?
             └── family_histories: relatives with cancer
```

### What Files Are Available?

Each case has associated **files** - the actual data you can download:

| File Type | What It Is | Example Use |
|-----------|-----------|-------------|
| **Slide Image** | Whole slide microscopy images (.svs) | Deep learning on pathology |
| **Clinical Supplement** | XML files with clinical details | Extract survival data |
| **Gene Expression** | RNA-seq quantification | Gene expression analysis |
| **Somatic Mutation** | Mutation calls (MAF files) | Identify driver mutations |
| **Copy Number** | Chromosomal gains/losses | Genomic instability analysis |

### Open vs Controlled Access

- **Open access**: Anyone can download (slide images, clinical summaries)
- **Controlled access**: Requires dbGaP approval (raw sequencing, germline variants)

This module defaults to **open access** data only.

---

## Discovering What's Available

**You don't need to memorize field names.** The client can tell you what's available:

```python
client = GDCClient()

# What fields exist for cases (patients)?
fields = client.discover_fields("cases")
print(f"Found {len(fields)} available fields")

# What nested data can be expanded?
expandable = client.get_expandable_fields("cases")
print(expandable)
# ['demographic', 'diagnoses', 'samples', 'exposures', ...]
```

---

## Common Tasks

### 1. List Available Cancer Types

```python
projects = client.list_projects(program="TCGA")

for p in projects:
    print(f"{p.project_id}")
    print(f"  Cancer: {p.disease_type}")
    print(f"  Site: {p.primary_site}")
    print(f"  Patients: {p.case_count}")
    print(f"  Files: {p.file_count}")
    print()
```

### 2. Get Patient Clinical Data

```python
# Get patients from breast cancer project
# expand= tells the API to include nested data
cases = client.get_cases(
    project_id="TCGA-BRCA",
    expand=["demographic", "diagnoses", "samples"],
    max_results=10
)

for case in cases:
    print(f"Patient: {case.submitter_id}")
    print(f"  Gender: {case.gender}")
    print(f"  Age at diagnosis: {case.age_at_diagnosis}")
    print(f"  Cancer type: {case.primary_diagnosis}")
    print(f"  Stage: {case.tumor_stage}")
    print(f"  Alive/Dead: {case.vital_status}")
    print(f"  Number of samples: {len(case.samples)}")
    print()
```

### 3. Get Pathology Slide Images

```python
# Get slide images (the microscopy images)
slides = client.get_slide_images(
    project_id="TCGA-BRCA",
    access="open",  # Only open-access slides
    max_results=5
)

for slide in slides:
    print(f"File: {slide.filename}")
    print(f"  Size: {slide.file_size / 1e9:.2f} GB")
    print(f"  Patient: {slide.case_submitter_id}")
    print(f"  Type: {slide.experimental_strategy}")  # Diagnostic vs Tissue slide
    print()
```

### 4. Filter Patients by Criteria

```python
# Find deceased female patients
cases = client.get_cases(
    project_id="TCGA-BRCA",
    gender="female",
    vital_status="Dead",
    max_results=50
)

print(f"Found {len(cases)} deceased female patients")
for case in cases:
    print(f"  {case.submitter_id}: died day {case.days_to_death}")
```

### 5. Build Complex Queries

```python
from src.data.tcga import GDCFilterBuilder, FilterOp

# Find open-access files that are either slides OR clinical XMLs
filter = (
    GDCFilterBuilder()
    .add("cases.project.project_id", "TCGA-BRCA")
    .add("access", "open")
    .add("data_type", ["Slide Image", "Clinical Supplement"], FilterOp.IN)
    .build()
)

files = client.get_files(custom_filter=filter)
print(f"Found {len(files)} matching files")
```

### 6. Generate Download Manifest

The GDC provides a command-line tool (`gdc-client`) for bulk downloads. This creates the manifest file it needs:

```python
# Get the files you want
slides = client.get_slide_images("TCGA-BRCA", access="open", max_results=100)

# Create manifest
client.create_manifest(slides, output_path="my_download_manifest.txt")

# Then in terminal:
# gdc-client download -m my_download_manifest.txt
```

---

## Running the Test Suite

To verify everything works and see real data:

```bash
python src/data/tcga/gdc_client.py
```

This runs 13 tests against the live GDC API, showing you actual TCGA-BRCA data:
- Project listings
- Patient clinical data
- Slide images
- Clinical files
- And more

---

## What Each Field Actually Contains

### Patient Demographics (`demographic`)
| Field | Meaning | Example Values |
|-------|---------|----------------|
| gender | Biological sex | "female", "male" |
| race | Self-reported race | "white", "black or african american", "asian" |
| ethnicity | Hispanic/Latino origin | "not hispanic or latino", "hispanic or latino" |
| year_of_birth | Birth year | 1945, 1962 |
| year_of_death | Death year (if applicable) | 2015, None |

### Diagnosis Information (`diagnoses`)
| Field | Meaning | Example Values |
|-------|---------|----------------|
| primary_diagnosis | Cancer type | "Infiltrating duct carcinoma, NOS" |
| age_at_diagnosis | Age in days when diagnosed | 18250 (about 50 years) |
| tumor_stage | How advanced | "stage iia", "stage iiic" |
| tumor_grade | How abnormal cells look | "G1", "G2", "G3" |
| vital_status | Alive or dead | "Alive", "Dead" |
| days_to_death | Days from diagnosis to death | 365, 1825, None |

### Sample Information (`samples`)
| Field | Meaning | Example Values |
|-------|---------|----------------|
| sample_type | What was collected | "Primary Tumor", "Blood Derived Normal" |
| tissue_type | Tumor or normal | "Tumor", "Normal" |
| is_ffpe | Preserved in paraffin? | True, False |
| tumor_descriptor | Tumor characteristics | "Primary", "Metastatic" |

### Slide Annotations (`samples.portions.slides`)
| Field | Meaning | Example Values |
|-------|---------|----------------|
| percent_tumor_cells | % of slide that's tumor | 80, 60, 40 |
| percent_necrosis | % dead tissue | 5, 10, 20 |
| percent_normal_cells | % normal cells | 10, 20 |
| percent_stromal_cells | % connective tissue | 15, 30 |

---

## File Structure

```
src/data/tcga/
├── __init__.py     # Exports: GDCClient, GDCFilterBuilder, FilterOp, etc.
├── gdc_client.py   # Main client implementation + test suite
└── README.md       # This file
```

---

## External Resources

- **GDC Portal** (browse data visually): https://portal.gdc.cancer.gov/
- **GDC API Docs**: https://docs.gdc.cancer.gov/API/Users_Guide/
- **TCGA Overview**: https://www.cancer.gov/tcga
- **gdc-client download tool**: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

---

## Troubleshooting

**"No cases returned"**
- Check if the project_id is correct (e.g., "TCGA-BRCA" not "tcga-brca")
- Some filters may be too restrictive

**"Timeout errors"**
- The GDC API can be slow. Increase timeout: `GDCClient(timeout=60)`
- Try smaller `max_results`

**"Need controlled access data"**
- You need dbGaP approval and an authentication token
- Pass token: `GDCClient(token="your-token-here")`
