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
  -B "/home/groups/roxanad/sonnet/compressed-perception:/workspace" \
  --pwd /workspace \
  "$SIF_STORE/python_3.10-slim.sif" \
  bash -c "

    cd /workspace
    
    # Activate the virtual environment
    source .venv/bin/activate

    # Add src to PYTHONPATH
    export PYTHONPATH=\$PYTHONPATH:/workspace/src

    # Run the distillation script
    python -m src.train
  "