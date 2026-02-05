#!/bin/bash
#SBATCH --job-name=probe_3seeds
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# TRAINING BUDGET ESTIMATE (dermatology, dinov3, 3 seeds, 4 resolutions)
# =============================================================================
#
# 1. Hyperparameter Tuning (seed 42 only, 512px):
#    - Grid: 3 lr × 3 wd × 2 bs = 18 configs
#    - 5-fold CV × 100 epochs each
#    - Estimated: ~3-4 hours (LP on embeddings is fast)
#
# 2. Embedding Extraction:
#    - 3 seeds × 4 resolutions × ~10 min = ~2 hours
#    - DINOv3-ViT-S at 512px, batch_size=256
#
# 3. Final Linear Probing:
#    - 3 seeds × 4 resolutions × 100 epochs
#    - Estimated: ~1-2 hours
#
# TOTAL ESTIMATED: 6-8 hours (with 12h buffer for safety)
#
# RESOURCES:
#   - GPU: 1x (DINOv3-ViT-S ~4GB, embeddings fit in VRAM)
#   - RAM: 48GB (caching embeddings + data loading)
#   - CPUs: 8 (num_workers=8 in config)
# =============================================================================

# This source file is part of the compressed perception project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# Container-based execution instead of conda
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_IMAGE:-python_3.10-slim-v2.sif}"

# Check if container exists
if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "ERROR: Container not found at $SIF_STORE/$SIF_IMAGE"
    echo "Available images:"
    ls -la "$SIF_STORE"/*.sif 2>/dev/null || echo "  No .sif files found in $SIF_STORE"
    echo ""
    echo "To pull the container, run: ./jobs/slim_container.sh"
    exit 1
fi

# Run the training inside the container
"$TOOL" exec --nv \
     -B "/scratch/users/$USER/reduced-perception:/workspace" \
     -B "/scratch/users/$USER:/scratch_user" \
     -B "/scratch/users/$USER/pip_cache:/root/.cache/pip" \
     -B "/tmp:/tmp" \
     --pwd /workspace \
     "$SIF_STORE/$SIF_IMAGE" \
      bash -c "

    set -e

    cd /workspace
    echo 'INFO: Successfully navigated to /workspace.'

    source .venv/bin/activate
    echo 'INFO: Virtual environment activated.'

    export TMPDIR=/scratch/users/$USER/tmp
    export HF_HOME=/scratch/users/$USER/huggingface
    export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
    export TORCH_HOME=/scratch/users/$USER/torch
    export TRAIN_OUTPUT_DIR=/scratch/users/$USER/results
    export LOG_DIR=/scratch/users/$USER/logs
    export MODEL_DIR=/scratch/users/$USER/models
    export PLOT_DIR=/scratch/users/$USER/plots
    export WANDB_MODE=offline  # Run without wandb API key (logs saved locally)

    # HuggingFace authentication for gated models (dinov3)
    # Token is read from ~/.huggingface/token or HF_TOKEN env var
    if [ -f /scratch_user/.huggingface/token ]; then
        export HF_TOKEN=\$(cat /scratch_user/.huggingface/token)
        echo 'INFO: HuggingFace token loaded from ~/.huggingface/token'
    elif [ -n \"\$HF_TOKEN\" ]; then
        echo 'INFO: Using HF_TOKEN from environment'
    else
        echo 'WARNING: No HuggingFace token found. Gated models (dinov3) may fail.'
        echo '         Run: huggingface-cli login (or set HF_TOKEN env var)'
    fi

    # Create all directories
    echo 'INFO: Creating required directories...'
    mkdir -p \$TMPDIR \$HF_HOME \$HF_DATASETS_CACHE \$TORCH_HOME \
            \$TRAIN_OUTPUT_DIR \$LOG_DIR \$MODEL_DIR \$PLOT_DIR

    # Run the script
    echo 'INFO: Starting Hydra run...'
    export HYDRA_FULL_ERROR=1
    python -m src.cli.run_multiresolution_probe \
        --domain dermatology \
        --model dinov3 \
        --config probe_two_stage_dermatology \
        --tune-hyperparams \
        --resolutions 512 256 128 64 \
        --seeds 42 123 456

    echo 'INFO: Job finished successfully.'
  "