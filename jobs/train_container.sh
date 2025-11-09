#!/bin/bash
#SBATCH --job-name=train_container
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This source file is part of the compressed perception project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# Container-based execution instead of conda
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="/scratch/users/$USER/simg"

# Run the training inside the container
"$TOOL" exec --nv \
     -B "/home/groups/roxanad/compressed-perception:/workspace" \
     -B "/scratch/users/$USER:/scratch_user" \
     -B "/scratch/users/$USER/pip_cache:/root/.cache/pip" \
     -B "/tmp:/tmp" \
     --pwd /workspace \
     "$SIF_STORE/python_3.10-slim-copy.sif" \
      bash -c "

    set -e

    cd /workspace
    echo 'INFO: Successfully navigated to /workspace.'

    source .venv/bin/activate
    echo 'INFO: Virtual environment activated.'

    # Add src to PYTHONPATH
    export PYTHONPATH=\$PYTHONPATH:/workspace/src

    export TMPDIR=/scratch/users/$USER/tmp
    export HF_HOME=/scratch/users/$USER/huggingface
    export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
    export TORCH_HOME=/scratch/users/$USER/torch
    export TRAIN_OUTPUT_DIR=/scratch/users/$USER/results
    export LOG_DIR=/scratch/users/$USER/logs
    export MODEL_DIR=/scratch/users/$USER/models
    export PLOT_DIR=/scratch/users/$USER/plots

    # Create all directories
    echo 'INFO: Creating required directories...'
    mkdir -p \$TMPDIR \$HF_HOME \$HF_DATASETS_CACHE \$TORCH_HOME \
            \$TRAIN_OUTPUT_DIR \$LOG_DIR \$MODEL_DIR \$PLOT_DIR

    echo 'INFO: Installing hydra-core...'
    python -m pip install --no-cache-dir hydra-core

    # Run the distillation script
    echo 'INFO: Starting Hydra run...'
    hydra run -c configs/config.yaml src.cli.main.train.batch_size=64
    
    echo 'INFO: Job finished successfully.'
  "