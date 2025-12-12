# Finetuning Pretrained Models for Compressed Dermatology Image Analysis

This project explores how compressed and degraded dermatology images (from the ISIC 2019 dataset) affect classification performance using pretrained vision models. It compares fine-tuning vs. linear probing across multiple JPEG quality levels.

![System architecture diagram](<./CS231N Poster.png>)

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

## 📦 Dataset

- [ISIC 2019 (Hugging Face)](https://huggingface.co/datasets/MKZuziak/ISIC_2019_224)
