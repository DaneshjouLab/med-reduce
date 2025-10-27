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
  -B "/home/groups/roxanad/sonnet/compressed-perception/workspace:/workspace" \
  --pwd /workspace \
  "$SIF_STORE/python_3.10-slim.sif" \
  bash -c "

    cd /workspace

    # Activate the virtual environment
    source .venv/bin/activate

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
    mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME \
            $TRAIN_OUTPUT_DIR $LOG_DIR $MODEL_DIR $PLOT_DIR

    # Run the distillation script
    python -m src.train \
      --resolution 224 \
      --batch_size 64 \
      --num_train_images 10000 \
      --num_epochs 5 \
      --eval_steps 200 \
      --learning_rate 1e-5 \
      --mode both
  "