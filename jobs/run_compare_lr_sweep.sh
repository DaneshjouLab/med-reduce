# This source file is part of the ARPA-H CARE LLM project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

#!/bin/bash
#SBATCH --job-name=compare_lr_sweep
#SBATCH --output=logs/compare_lr_sweep_%j.out
#SBATCH --error=logs/compare_lr_sweep_%j.err
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load Python module
ml load python/3.12.1

# Setup virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  python3.12 -m pip install uv
  uv venv
  source .venv/bin/activate

  # Install from requirements.txt
  uv pip install -r requirements.txt
else
  source .venv/bin/activate
fi

# WANDB Key
export WANDB_API_KEY="7ab80eeb87ef06298c6bca1258208b1739ad32fe"

# Running the script
python src/models/comparison/compare_lr_sweep.py --resolution 224 --batch_size 256 --num_train_images 25000 --num_epochs 6 --eval_steps 10 --learning_rate 1e-4