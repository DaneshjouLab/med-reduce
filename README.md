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
# Baseline LP (frozen DINOv3 encoder + linear probe)
python -m src.cli.run_multiresolution_probe \
    --domain dermatology \
    --model dinov3 \
    --tune-hyperparams \
    --resolutions 512 256 128 64 \
    --seeds 42 123 456

# Distillation (train ResNet18 student to match DINOv3 embeddings)
python -m src.cli.run_distillation \
    --config-name=distillation_dermatology \
    train.seed=42
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
│   ├── distillation_dermatology.yaml     # Distillation config for dermatology
│   ├── distillation_pathology.yaml       # Distillation config for pathology
│   ├── distillation_radiology.yaml       # Distillation config for radiology
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
│   ├── distill_container.sh              # Pipeline B: distillation training
│   ├── eval_distilled_container.sh       # Pipeline C: LP eval of distilled students
│   ├── setup_container.sh                # One-time setup: creates venv, installs deps
│   ├── slim_container.sh                 # Pulls lightweight Python container image
│   └── train_container.sh                # Pipeline A: baseline LP training
│
├── scripts/                              # One-off utilities and sanity checks
│   ├── merge_isic2017.py                 # Dataset preparation / merging utility
│   └── test_transforms.py                # Test image transformations (e.g. resolution)
│
├── src/                                  # Core library code
│   │
│   ├── cli/                              # Command-line entry points (Hydra-driven)
│   │   ├── cache_teacher_embeddings.py   # Precompute & cache teacher embeddings
│   │   ├── run_distillation.py           # Distillation pipeline runner
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
│   │   ├── distillation_engine.py        # Distillation training loop
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
│   │   ├── classification.py             # Classification losses
│   │   └── distillation.py              # Embedding distillation loss
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
│       ├── distillation_wrapper.py       # Distillation pipeline orchestrator
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

## Experiment Pipelines

The framework supports three experimental pipelines. All pipelines use the same persistent train/test splits (managed by `SplitManager`) to ensure fair comparison. All experiments should be run across three seeds (42, 123, 456) for variance estimation.

### Pipeline Overview

```
Pipeline A: Baseline LP
  Frozen DINOv3 @ each resolution → cache embeddings → linear probe → AUROC

Pipeline B: Distillation
  Frozen DINOv3 @ 512px → cache embeddings → train student (ResNet18/TinyViT)
  end-to-end on degraded images → save distilled_student.pt

Pipeline C: LP with Distilled Student
  Frozen distilled student @ each resolution → cache embeddings → linear probe → AUROC
```

---

### Pipeline A: Baseline Linear Probing (DINOv3)

Evaluates frozen DINOv3 embeddings at multiple resolutions via linear probing.

**Step 1 — Hyperparameter tuning** (once per domain, first seed only):

```bash
# Dermatology
python -m src.cli.run_multiresolution_probe \
    --domain dermatology --model dinov3 \
    --tune-hyperparams \
    --resolutions 512 256 128 64 \
    --seeds 42 123 456 \
    --config configs/probe_two_stage_dermatology

# Pathology (per task)
for TASK in luad_vs_lusc lgg_vs_gbm kras tp53 egfr idh; do
  python -m src.cli.run_multiresolution_probe \
      --domain pathology --model dinov3 \
      --tune-hyperparams \
      --resolutions 512 256 128 64 \
      --seeds 42 123 456 \
      --config configs/probe_two_stage_pathology \
      --extra-overrides "datamodule.task=${TASK}"
done

# Radiology
python -m src.cli.run_multiresolution_probe \
    --domain radiology --model dinov3 \
    --tune-hyperparams \
    --resolutions 512 256 128 64 \
    --seeds 42 123 456 \
    --config configs/probe_two_stage_radiology
```

This automatically:
1. Runs 5-fold CV hyperparameter search at 512px with seed 42
2. Runs final LP at all 4 resolutions for all 3 seeds using the tuned hyperparameters

**Outputs:**
```
runs/probe_two_stage/
  seed_42/
    hyperparam_search/best_hyperparameters.json
    results_dinov3_512px.json
    results_dinov3_256px.json
    results_dinov3_128px.json
    results_dinov3_64px.json
  seed_123/
    results_dinov3_*.json
  seed_456/
    results_dinov3_*.json
```

---

### Pipeline B: Distillation (Train Student Models)

Trains a student model (ResNet18 or TinyViT) to match DINOv3 embeddings on clean 512px images, while the student receives degraded inputs.

**Step 1 — Run distillation for each seed:**

```bash
# Dermatology — ResNet18 student
for SEED in 42 123 456; do
  python -m src.cli.run_distillation \
      --config-name=distillation_dermatology \
      train.seed=${SEED}
done

# Radiology — ResNet18 student
for SEED in 42 123 456; do
  python -m src.cli.run_distillation \
      --config-name=distillation_radiology \
      train.seed=${SEED}
done

# Pathology — ResNet18 student (per task)
for TASK in luad_vs_lusc lgg_vs_gbm kras tp53 egfr idh; do
  for SEED in 42 123 456; do
    python -m src.cli.run_distillation \
        --config-name=distillation_pathology \
        train.seed=${SEED} \
        datamodule.task=${TASK}
  done
done

# Any domain — TinyViT student (override student config)
for SEED in 42 123 456; do
  python -m src.cli.run_distillation \
      --config-name=distillation_dermatology \
      train.seed=${SEED} \
      student.name=tiny_vit \
      student.model_id=tiny_vit_21m_224
done
```

**What happens:**
1. Teacher embeddings are cached at 512px (reused across seeds if same data)
2. Student trains end-to-end on degraded images to match teacher embeddings
3. Loss: `alpha * MSE + (1 - alpha) * (1 - cosine_similarity)`
4. Best checkpoint saved to `runs/distillation/seed_{SEED}/distilled_student.pt`

**Config options** (`configs/distillation_dermatology.yaml`):

| Key | Default | Description |
|-----|---------|-------------|
| `distillation.alpha` | 0.5 | MSE vs cosine balance (1.0 = pure MSE, 0.0 = pure cosine) |
| `distillation.teacher_resolution` | 512 | Resolution for teacher embedding extraction |
| `student.model_id` | resnet18 | timm model ID for the student |
| `train.epochs` | 100 | Distillation training epochs |
| `train.optimizer.lr` | 1e-4 | Learning rate |

**Outputs:**
```
runs/distillation/
  seed_42/
    distilled_student.pt    # checkpoint with student_state_dict + metadata
  seed_123/
    distilled_student.pt
  seed_456/
    distilled_student.pt

cache/teacher_embeddings/
  {hash}/
    embeddings.pt, labels.pt, image_ids.json, metadata.json
```

---

### Pipeline C: LP Evaluation of Distilled Students

After distillation, freeze the student backbone and evaluate it through the same LP pipeline as Pipeline A. This allows direct AUROC comparison between DINOv3 baseline and distilled students at each resolution.

**Step 1 — Run LP with the distilled student at all resolutions:**

```bash
# Use the existing two-stage probe pipeline with the distilled student model
for SEED in 42 123 456; do
  python -m src.cli.run_multiresolution_probe \
      --domain dermatology \
      --model dinov3 \
      --resolutions 512 256 128 64 \
      --seeds ${SEED} \
      --config configs/probe_two_stage_dermatology \
      --extra-overrides \
        "model.name=resnet18_distilled" \
        "model.model_id=resnet18" \
        "model.type=timm" \
        "model.config.num_labels=3" \
        "model.config.pretrained=false"
done
```

> **Note:** To load the distilled weights (instead of random/ImageNet init), you will need to manually load the checkpoint from `runs/distillation/seed_{SEED}/distilled_student.pt` into the student model before embedding extraction. This can be done by extending `ProbeTwoStageWrapper` or writing a small script that loads the checkpoint and runs the LP pipeline.

---

### Split Consistency

All three pipelines use the same `SplitManager` with the same `split_dir` and `seed`, ensuring:
- Identical train/test splits across baseline LP, distillation, and distilled LP
- Results are directly comparable within the same seed
- Variance is estimated across seeds (42, 123, 456)

**Directory structure:**
```
splits/
  {dataset_name}/
    seed_42/
      train_indices.npy
      test_indices.npy
      cv_folds.json
    seed_123/
      ...
    seed_456/
      ...
```

---

### Multi-Seed Bootstrap

All experiments use three seeds. The pattern is consistent:

- **Hyperparameter tuning** runs once with seed 42 (first seed)
- **Final training/evaluation** runs for all seeds (42, 123, 456)
- **Distillation** runs independently per seed (each seed gets its own student checkpoint)

This isolates variance to train/test split differences (standard bootstrap approach).

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
   cd /scratch/users/$USER/reduced-perception
   mkdir -p .huggingface
   echo "hf_your_token_here" > .huggingface/token
   chmod 600 .huggingface/token
   ```

   Or set as an environment variable:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

### HPC Setup

```bash
# Step 1: One-time setup (creates directories, venv, installs dependencies)
sbatch jobs/setup_container.sh
tail -f logs/setup_env_*.out  # wait for completion
```

### Running All Three Pipelines on HPC

Each pipeline has a dedicated job script. Run them sequentially (B depends on nothing extra, C depends on B).

```bash
# ── Pipeline A: Baseline LP ──────────────────────────────────────────────────
DOMAIN=dermatology sbatch jobs/train_container.sh
DOMAIN=pathology   sbatch jobs/train_container.sh
DOMAIN=radiology   sbatch jobs/train_container.sh

# ── Pipeline B: Distillation ─────────────────────────────────────────────────
# (can run in parallel with Pipeline A)
DOMAIN=dermatology sbatch jobs/distill_container.sh
DOMAIN=pathology   sbatch jobs/distill_container.sh
DOMAIN=radiology   sbatch jobs/distill_container.sh

# TinyViT student (submit separately or after ResNet18 finishes)
STUDENT=tiny_vit_21m_224 DOMAIN=dermatology sbatch jobs/distill_container.sh

# ── Pipeline C: LP eval of distilled students ────────────────────────────────
# (run AFTER Pipeline B completes for that domain/student)
DOMAIN=dermatology sbatch jobs/eval_distilled_container.sh
DOMAIN=pathology   sbatch jobs/eval_distilled_container.sh
DOMAIN=radiology   sbatch jobs/eval_distilled_container.sh

# TinyViT eval
STUDENT=tiny_vit_21m_224 DOMAIN=dermatology sbatch jobs/eval_distilled_container.sh
```

All job scripts accept the same environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOMAIN` | (required) | `dermatology`, `radiology`, or `pathology` |
| `SEEDS` | `42 123 456` | Space-separated bootstrap seeds |
| `RESOLUTIONS` | `512 256 128 64` | Space-separated resolutions (LP only) |
| `STUDENT` | `resnet18` | timm model ID for student (distillation only) |
| `TASKS` | all 6 TCGA tasks | Pathology tasks to run (pathology only) |
| `ALPHA` | `0.5` | MSE vs cosine balance (distillation only) |
| `EPOCHS` | `100` | Training epochs (distillation only) |

Monitor progress:
```bash
tail -f logs/distill_3seeds_*.out
tail -f logs/eval_distilled_3seeds_*.out
squeue -u $USER
```

### Training Budget

| Resource | Value | Reason |
|----------|-------|--------|
| Time | 12 hours | ~6-8h for LP baseline, with buffer |
| Memory | 48 GB | Embedding caching + data loading |
| CPUs | 8 | Matches `num_workers` in config |
| GPUs | 1 | DINOv3-ViT-S fits on single GPU |

**Workload breakdown (3 seeds, 4 resolutions):**

| Pipeline | Estimate | Details |
|----------|----------|---------|
| Baseline LP (hyper-tuning) | ~3-4h | 18 configs x 5-fold CV x 100 epochs |
| Baseline LP (final probing) | ~1-2h | 3 seeds x 4 resolutions x 200 epochs |
| Distillation (per student) | ~4-6h | 3 seeds x 100 epochs end-to-end |
| LP with distilled student | ~1-2h | 3 seeds x 4 resolutions x 200 epochs |

---

### Troubleshooting

**Container not found:**
```bash
./jobs/slim_container.sh
```

**Mount error (pip_cache):**
```bash
mkdir -p /scratch/users/$USER/pip_cache
```

**venv not found:**
```bash
sbatch jobs/setup_container.sh
```

**Check available containers:**
```bash
ls -la /scratch/users/$USER/simg/*.sif
```

