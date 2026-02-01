# REDUCE: Representation Transfer Under Perceptual Constraints
**R**esolution-Aware **E**valuation of **D**eep **U**nderstanding and **C**omputational **E**fficiency

**REDUCE** is a research framework for studying accuracy–efficiency trade-offs in vision models under controlled perceptual degradations, such as systematic input resolution reduction. The framework supports linear probing, two-stage probing, and distillation, enabling consistent, multi-resolution evaluation with comprehensive metric tracking and post-hoc analysis.
REDUCE introduces a family of models and evaluation protocols designed for efficiency-under-pressure settings, where representational robustness, computational cost, and predictive performance must be jointly optimized and rigorously compared.

The design emphasizes:
- **On-the-fly (lazy) input transformations** for clean experimental control
- **Reproducibility** via Hydra configs, saved seeds, and resolved configs
- **Clear separation** between training, data handling, and evaluation

---

## 📊 Datasets

Supported medical imaging domains:

- **[ISIC 2017](https://arxiv.org/pdf/1710.05006)** — Dermatology: skin lesion classification (nevus, melanoma, seborrheic keratosis)
- **[CheXpert](https://arxiv.org/pdf/1901.07031)** — Radiology: chest X-ray classification (14 findings)
- **[TCGA](https://gdc.cancer.gov/about-data)** — Pathology: histopathology images

To use your own dataset, prepare:
- **Images folder**: `{dataset}/images/` with image files (`.jpg`, `.png`, etc.)
- **Labels CSV**: `{dataset}/labels.csv` with columns `[image_id, label]` or multi-label columns
- Update config: `data_dir`, `local_label_file`, `local_label_column`, `num_labels`

---

## Quick Start

**1. Create and activate virtual environment:**
```bash
python3.10 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or on Windows:
# venv\Scripts\activate
```

**2. Install package with dependencies:**
```bash
pip install -e .
```

**3. Run an experiment:**
```bash
python -m src.cli.run_multiresolution_probe \
    --domain dermatology \
    --resolutions 512 256
```

---

## Repository Overview

At a high level, the workflow is:

```
configs → CLI (train) → DataModule → Models → Engines → Metrics → Analysis
```


```
reduced-perception/
│
├── configs/                              # Hydra configuration files
│   ├── config_segmentation.yaml          # Segmentation task config
│   ├── config_segmentation_vit.yaml      # Segmentation with ViT backbone
│   ├── probe_two_stage_dermatology.yaml  # Two-stage probing for dermatology
│   ├── probe_two_stage_radiology.yaml    # Two-stage probing for radiology
│   ├── probe_two_stage_pathology.yaml    # Two-stage probing for pathology
│   └── probe_two_stage_vit.yaml          # Two-stage probing with ViT backbone
│
├── examples/                             # Standalone example scripts
│   ├── analyze_experiment_results.py     # Post-hoc analysis of metrics & plots
│   └── load_checkpoint_example.py        # Example: loading a trained checkpoint
│
├── jobs/                                 # Container / job execution scripts
│   ├── slim_container.sh                 # Lightweight container build/run
│   └── train_container.sh                # Training entrypoint for containers / HPC
│
├── scripts/                              # One-off utilities and sanity checks
│   ├── merge_isic2017.py                 # Dataset preparation / merging utility
│   └── test_transforms.py                # Test image transformations (e.g. resolution)
│
├── src/                                  # Core library code
│   │
│   ├── cli/                              # Command-line entry points (Hydra-driven)
│   │   ├── cache_teacher_embeddings.py   # Precompute & cache teacher embeddings
│   │   ├── run_experiments.py            # Batch experiment launcher
│   │   ├── run_multiresolution_probe.py  # Sweep over input resolutions
│   │   ├── run_probe_two_stage.py        # Two-stage probing runner
│   │   └── train.py                      # MAIN training entry point (dataset → model → engine)
│   │
│   ├── data/                             # Data loading & dataset abstractions
│   │   ├── data_utils.py                 # Shared dataset helpers
│   │   ├── datamodule.py                 # BaseDataModule (dataset entry point)
│   │   ├── dataset_factory.py            # Factory for dataset selection
│   │   ├── datasets.py                   # Dataset definitions
│   │   ├── embedding_dataset.py          # Dataset backed by cached embeddings
│   │   ├── isic_datamodule.py            # ISIC datamodule (standard)
│   │   ├── tabular_datamodule_persistent.py # ISIC datamodule with persistent caching
│   │   ├── isic_feature_loader.py        # Feature-level ISIC loading
│   │   └── isic_loader.py                # Raw ISIC image loading
│   │
│   ├── engines/                          # Training & evaluation engines
│   │   ├── linear_probe_engine.py        # Linear probe on frozen features
│   │   ├── linear_probe_embedding_engine.py
│   │   │                                 # Linear probing on cached embeddings
│   │   └── training_core.py              # Shared training loop logic (epochs, logging)
│   │
│   ├── evaluation/                    # Metrics, analysis, visualization
│   │   ├── analyze_results.py         # Aggregated analysis logic
│   │   ├── metrics_collector.py       # Collects & persists metrics (JSON / CSV)
│   │   ├── metrics.py                 # Metric definitions (accuracy, AUROC, etc.)
│   │   ├── run_umap_analysis.py       # UMAP embedding visualization
│   │   ├── visualization.py           # Plotting utilities
│   │   └── visualize_results.py       # High-level result visualization scripts
│   │
│   ├── losses/                           # Loss functions
│   │   ├── __init__.py
│   │   └── classification.py             # Classification losses
│   │
│   ├── models/                           # Model definitions & factories
│   │   ├── dinov3.py                     # DINOv3 backbone
│   │   ├── dinov3_segmentation.py        # DINOv3 for segmentation
│   │   └── factory.py                    # Model factory / registry
│   │
│   ├── transformations/                  # Input-space transformations
│   │   ├── __init__.py
│   │   └── transforms.py                 # ResolutionReductionTransform (lazy, on-the-fly)
│   │
│   ├── utils/                            # General utilities (logging, helpers)
│   │
│   └── wrappers/                         # High-level experiment wrappers
│       ├── probe_cv.py                   # Cross-validation probing
│       ├── probe_two_stage.py            # Two-stage probing logic
│       └── __init__.py
│
├── requirements.txt
├── requirements.txt.licence
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

(Optional) For containerized or HPC runs, see scripts in `jobs/`.

---

## Running on HPC (Sherlock)

### 1. Build the container (first time only)

```bash
sbatch jobs/slim_container.sh
```

This pulls a lightweight Python 3.10 image and saves it to `/scratch/users/$USER/simg/`. Follow the printed instructions to enter the container interactively and install PyTorch + dependencies into the `.venv`.

### 2. Run experiments

Submit a training job with:

```bash
sbatch jobs/train_container.sh
```

Inside `train_container.sh`, change the Python command depending on the task:

**Classification (linear probing):**
```bash
python -m src.cli.run_multiresolution_probe \
    --domain dermatology \
    --model dinov3 \
    --config probe_two_stage \
    --tune-hyperparams \
    --resolutions 512 256 128 64
```

**Segmentation:**
```bash
python -m src.cli.run_multiresolution_segmentation \
    --domain dermatology \
    --model dinov3 \
    --tune-hyperparams \
    --resolutions 512 256 128 64
```

Both scripts follow the same two-step protocol:
1. `--tune-hyperparams` runs hyperparameter search via cross-validation at the highest resolution
2. `--resolutions` trains and evaluates at each resolution using the tuned hyperparameters

Results are saved to separate directories to avoid conflicts:
- Classification: `runs/probe_two_stage/hyperparam_search/`
- Segmentation: `runs/segmentation/hyperparam_search_segmentation/`

