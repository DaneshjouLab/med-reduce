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
python3.10 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or on Windows: .venv\Scripts\activate
```

**2. Install PyTorch and project:**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install project from pyproject.toml
pip install -e .
```

**3. Run an experiment:**
```bash
python -m src.cli.run_multiresolution_probe \
    --domain dermatology \
    --model dinov3 \
    --tune-hyperparams \
    --resolutions 512 256 \
    --seeds 42 123 456
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
│   ├── setup_container.sh                # One-time setup: creates venv, installs deps
│   ├── slim_container.sh                 # Pulls lightweight Python container image
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

Create a Python environment and install the package:

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install project and dependencies
pip install -e .

# For development tools (optional)
pip install -e ".[dev]"
```

For containerized or HPC runs, see [Running on HPC](#running-on-hpc-sherlock).

---

## Running on HPC (Sherlock)

### Prerequisites: HuggingFace Authentication

The DINOv3 model (`facebook/dinov3-vits16-pretrain-lvd1689m`) is a gated model that requires:

1. **Request access** on HuggingFace:
   - Go to [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
   - Click "Request access" and wait for approval

2. **Create a HuggingFace token**:
   - Go to [HuggingFace Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "Read" permissions

3. **Save the token in the project root** (on the cluster):
   ```bash
   # In the project directory, create .huggingface/token
   cd /scratch/users/$USER/reduced-perception
   mkdir -p .huggingface
   echo "hf_your_token_here" > .huggingface/token
   chmod 600 .huggingface/token
   ```

   The token file is git-ignored, so it won't be committed.

   Or set it as an environment variable before submitting jobs:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

**Alternative: Use DINOv2 (no authentication required)**

If you don't have access to DINOv3, you can use the public DINOv2 model:
```bash
python -m src.cli.run_multiresolution_probe \
    --domain dermatology \
    --model dinov2 \
    ...
```

---

### Complete Pipeline (3 steps)

```bash
# Step 1: One-time setup (creates directories, venv, installs dependencies)
sbatch jobs/setup_container.sh

# Step 2: Monitor setup (wait for completion)
tail -f logs/setup_env_*.out

# Step 3: Run training with bootstrap seeds
sbatch jobs/train_container.sh
```

That's it! The pipeline will:
1. Run hyperparameter tuning once (seed 42, highest resolution)
2. Run final probing for all seeds (42, 123, 456) at all resolutions (512, 256, 128, 64)

---

### Detailed Setup Instructions

#### 1. Setup environment (first time only)

```bash
# Create logs directory
mkdir -p logs

# Submit setup job (creates venv, installs PyTorch + dependencies)
sbatch jobs/setup_container.sh

# Monitor progress
tail -f logs/setup_env_*.out
```

The setup script automatically:
- Creates required directories (`pip_cache/`, `simg/`, `tmp/`, `huggingface/`, `torch/`)
- Pulls the Python 3.10-slim container if missing
- Creates `.venv` and installs PyTorch (CUDA 11.8)
- Installs project from `pyproject.toml` (`pip install -e .`)
- Verifies CUDA is working

#### 2. Run experiments

```bash
sbatch jobs/train_container.sh
```

Monitor progress:
```bash
# Watch output
tail -f logs/probe_3seeds_*.out

# Check job status
squeue -u $USER
```

---

### Multi-Seed Bootstrap Support

The pipeline supports running multiple bootstrap seeds to estimate variance in results. Each seed produces independent splits, embeddings, and results.

**CLI Usage:**
```bash
# Single seed (default behavior)
python -m src.cli.run_multiresolution_probe \
    --domain dermatology \
    --model dinov3 \
    --tune-hyperparams \
    --resolutions 512 256 128 64

# Multiple bootstrap seeds (recommended for papers)
python -m src.cli.run_multiresolution_probe \
    --domain dermatology \
    --model dinov3 \
    --tune-hyperparams \
    --resolutions 512 256 128 64 \
    --seeds 42 123 456
```

**How it works:**
- Hyperparameter tuning runs **once** with the first seed (e.g., 42)
- Final probing runs for **all seeds** using shared hyperparameters
- This isolates variance to train/test split differences (standard bootstrap approach)

**Directory structure with seeds:**
```
splits/
  {dataset_name}/
    seed_42/
      train_indices.npy
      test_indices.npy
      cv_folds.json
    seed_123/
      ...

cache/embeddings/
  {dataset_name}/
    {model_name}/
      seed_42/
        512px/
          train_embeddings.pt
          test_embeddings.pt
      seed_123/
        ...

runs/probe_two_stage/
  seed_42/
    hyperparam_search/
      best_hyperparameters.json
    results_dinov3_512px.json
    results_dinov3_256px.json
  seed_123/
    results_dinov3_512px.json
    ...
```

---

### Running All Domains

```bash
# Dermatology (ISIC 2017)
python -m src.cli.run_multiresolution_probe \
    --domain dermatology --model dinov3 \
    --tune-hyperparams --resolutions 512 256 128 64 \
    --seeds 42 123 456 \
    --config configs/probe_two_stage_dermatology

# Radiology (CheXpert)
python -m src.cli.run_multiresolution_probe \
    --domain radiology --model dinov3 \
    --tune-hyperparams --resolutions 512 256 128 64 \
    --seeds 42 123 456 \
    --config configs/probe_two_stage_radiology

# Pathology (TCGA)
python -m src.cli.run_multiresolution_probe \
    --domain pathology --model dinov3 \
    --tune-hyperparams --resolutions 512 256 128 64 \
    --seeds 42 123 456 \
    --config configs/probe_two_stage_pathology
```

---

### Training Budget

Default job configuration (`jobs/train_container.sh`):

| Resource | Value | Reason |
|----------|-------|--------|
| Time | 12 hours | ~6-8h estimated, with buffer |
| Memory | 48 GB | Embedding caching + data loading |
| CPUs | 8 | Matches `num_workers` in config |
| GPUs | 1 | DINOv3-ViT-S fits on single GPU |

**Workload breakdown (3 seeds, 4 resolutions):**
1. **Hyperparameter tuning** (~3-4h): 18 configs × 5-fold CV × 100 epochs
2. **Embedding extraction** (~2h): 3 seeds × 4 resolutions × 2 splits
3. **Final linear probing** (~1-2h): 12 runs × 100 epochs

---

### Troubleshooting

**Container not found:**
```bash
# Pull the container manually
./jobs/slim_container.sh
```

**Mount error (pip_cache):**
```bash
mkdir -p /scratch/users/$USER/pip_cache
```

**venv not found:**
```bash
# Run the setup script
sbatch jobs/setup_container.sh
```

**Check available containers:**
```bash
ls -la /scratch/users/$USER/simg/*.sif
```

