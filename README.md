<!-- This source file is part of the DaneshjouLab projects

SPDX-FileCopyrightText: 2025 Stanford University

SPDX-License-Identifier: MIT
-->

# Finetuning Pretrained Models for Compressed Dermatology Image Analysis

This project explores how compressed and degraded dermatology images (from the ISIC 2019 dataset) affect classification performance using pretrained vision models. It compares fine-tuning vs. linear probing across multiple JPEG quality levels.

![System architecture diagram](<./media/CS231N Poster.png>)

## Project Goals

- Evaluate model robustness to image compression (JPEG 90/50/20)
- Compare pretrained models: ViT, DINOv2, and SimCLR
- Benchmark fine-tuning vs. linear probing
- Analyze FLOPs, GPU memory, and classification accuracy

## Models

- `ViT`: Vision Transformer from Hugging Face
- `DINOv2`: Self-supervised ViT from Meta
- `SimCLR`: Contrastive ResNet50 trained with linear classifier

## Metrics Tracked

- Accuracy, F1 Score, AUC
- FLOPs (GFLOPs)
- GPU memory usage
- Training and evaluation time

## Project Structure

```
reduced-perception/
├── configs/
│   └── example_config.yaml          # Configs for job submissions
compressed-perception/
├── README.md                    # Project overview & documentation
├── LICENSES/                    # Directory containing license files (REUSE compliance)
│   └── MIT.txt                  # MIT license text
│
├── pyproject.toml               # Python packaging config
├── setup.py                     # Installation script for the package
├── setup.cfg                    # Configuration for setup tools
│
├── scripts/                         # Lightweight utility or shell scripts
│   ├── download_unpack_isic2019.sh  # Downloads and unpacks ISIC data
│   └── submit_from_config.sh        # SLURM submission helper
│
├── jobs/                            # SLURM-related job definitions
│   └── job_template.slurm
│
├── src/                             # Source code, logically grouped
│   ├── __init__.py
│   ├── finetune/                    # Fine-tuning workflows
│   │   └── baseline_finetuning.py
│   ├── evaluation/                  # Evaluation + plotting
│   │   └── evaluate_isic_results.py
│   └── models/                      # Model-related scripts
│       ├── model_comparison.py      # Config file with constant strings
│       ├── model_comparison.py
│       └── model_comparison_2.py

│
├── results/                         # Auto-generated results
│   ├── plots/                       # Accuracy/f1/AUC plots
│   └── logs/                        # Training logs or SLURM outputs
│
├── requirements.txt
├── .gitignore
├── .github
└── README.md
```

## Quick Start

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training:
   ```bash
   python train_models.py
   ```

3. View results
   We use weights and biases for logging, so output plots can be seen there
├── requirements.txt             # Dependencies file
├── requirements.txt.license     # Dependencies file license
├── .yamllint                    # YAML linter configuration
├── .yamllint.license            # YAML linter configuration license
│
├── .github/                     # GitHub specific files
│   └── workflows/               # CI/CD workflow definitions
│       ├── build-and-test.yml
│       └── pull_request.yml
│
├── .reuse/                      # REUSE compliance configuration
│   └── dep5                     # Copyright and license information
│
├── docs/                        # Documentation
│   └── pipeline.md              # Pipeline documentation
│
├── scripts/                        # Standalone scripts
│   ├── ...
│   └── visualize_isic_results.py   # Visualize metrics for model comparison (TODO)
│
├── configs/                     # Configuration files (TODO)
│   ├── datasets/                # Dataset configs
│   │   └── isic2019.yaml        # ISIC 2019 dataset config
│   ├── models/                  # Model configs
│   │   ├── vit.yaml             # ViT model config
│   │   ├── dinov2.yaml          # DINOv2 model config
│   │   └── simclr.yaml          # SimCLR model config
│   ├── experiments/             # Experiment configs
│   │   ├── baseline.yaml        # Baseline experiment
│   │   └── lr_sweep.yaml        # Learning rate sweep experiment
│   └── example_config.yaml      # Example configuration file
│
│
├── tests/                   # Test suite (TODO)
│   ├── unit/                # Unit tests
│   │   └── test_transforms.py
│   ├── integration/         # Integration tests
│   │   └── test_pipeline.py
│   └── conftest.py          # Test fixtures and configuration
│
├── src/
│   └── compressed_perception/ # Main package
│       ├── __init__.py      # Package initialization
│       │
│       ├── models/          # Model implementations
│       │   ├── __init__.py
│       │   ├── architectures/   # Model architecture definitions
│       │   │   ├── __init__.py
│       │   │   ├── vit.py       # Vision Transformer adaptations
│       │   │   └── simclr.py    # SimCLR adaptations
│       │   │
│       │   ├── evaluation/      # Model evaluation code
│       │   │   ├── __init__.py
│       │   │   └── metrics.py   # Evaluation metrics
│       │   │
│       │   ├── comparison/      # Model comparison utilities
│       │   │   ├── __init__.py
│       │   │   ├── compare_baseline.py  # Baseline comparison
│       │   │   └── compare_lr_sweep.py  # Learning rate sweeping
│       │   │
│       │   ├── training/        # Training infrastructure
│       │   │   ├── __init__.py
│       │   │   ├── trainers.py      # Training loops
│       │   │   └── callbacks.py     # Training callbacks
│       │   │
│       │   └── utils/           # Model utilities
│       │       ├── __init__.py
│       │       ├── constants.py # Model constants
│       │       └── helpers.py   # Helper functions
│       │
│       ├── modules/          # Reusable modules
│       │   ├── __init__.py
│       │   ├── transforms/      # Image transformations
│       │   │   ├── __init__.py
│       │   │   ├── degradation.py  # Image degradation transforms
│       │   │   └── augmentation.py # Data augmentation transforms
│       │   │
│       │   └── data_preparation/ # Data preparation utilities
│       │       ├── __init__.py
│       │       └── preparation.py  # Dataset preparation
│       │
│
│
├── results/
│
├── jobs/                         # Cluster job submission files
│   ├── job_template.slurm        # SLURM job template
│   ├── run.sh                    # General run script
│   ├── rurun_compare_baseline.sh # Learning rate experiment script
│   ├── run_compare_lr_sweep.sh   # Model comparison script
│   └── configs/             # Job configurations
│
└── media/                   # Media files for documentation
    └── CS231N Poster.png    # Project poster
```

## 📦 Dataset

- [ISIC 2019 (Hugging Face)](https://huggingface.co/datasets/MKZuziak/ISIC_2019_224)
