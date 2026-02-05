#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# ONE-TIME SETUP: Creates venv and installs all dependencies
# Run this once before using train_container.sh
# =============================================================================

set -e

# Create required directories
echo "INFO: Creating required directories..."
mkdir -p /scratch/users/$USER/pip_cache
mkdir -p /scratch/users/$USER/simg
mkdir -p /scratch/users/$USER/tmp
mkdir -p /scratch/users/$USER/huggingface
mkdir -p /scratch/users/$USER/torch
mkdir -p logs

# Container setup
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_IMAGE:-python_3.10-slim-v2.sif}"

# Pull container if it doesn't exist
if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "INFO: Pulling container image..."
    cd "$SIF_STORE"
    $TOOL pull "$SIF_IMAGE" "docker://python:3.10-slim"
    cd -
fi

echo "INFO: Using container: $SIF_STORE/$SIF_IMAGE"

# Run setup inside container
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
    echo 'INFO: Working directory: /workspace'

    # Create venv if it doesn't exist
    if [ ! -d '.venv' ]; then
        echo 'INFO: Creating virtual environment...'
        python -m venv .venv
    else
        echo 'INFO: Virtual environment already exists'
    fi

    source .venv/bin/activate
    echo 'INFO: Virtual environment activated'
    echo 'INFO: Python: '\$(which python)

    # HuggingFace authentication for gated models (dinov3)
    if [ -f /scratch_user/.huggingface/token ]; then
        export HF_TOKEN=\$(cat /scratch_user/.huggingface/token)
        echo 'INFO: HuggingFace token loaded from ~/.huggingface/token'
    elif [ -n \"\$HF_TOKEN\" ]; then
        echo 'INFO: Using HF_TOKEN from environment'
    else
        echo 'WARNING: No HuggingFace token found. Gated models (dinov3) may fail.'
    fi

    # Upgrade pip
    echo 'INFO: Upgrading pip...'
    pip install --upgrade pip wheel setuptools

    # Install PyTorch with CUDA 11.8 (skip if already installed)
    if ! python -c 'import torch' 2>/dev/null; then
        echo 'INFO: Installing PyTorch with CUDA 11.8...'
        pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    else
        echo 'INFO: PyTorch already installed, skipping...'
    fi

    # Always reinstall project to pick up any changes (uses pyproject.toml)
    echo 'INFO: Installing/reinstalling project from pyproject.toml...'
    pip install -e . --no-deps
    pip install -e .

    # Verify installation
    echo ''
    echo '=========================================='
    echo 'VERIFICATION'
    echo '=========================================='
    python -c \"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

# Verify src package imports
print()
print('Testing src package imports...')
from src.data.tabular_datamodule_persistent import TabularDataModulePersistent
from src.wrappers.probe_two_stage import ProbeTwoStageWrapper
from src.utils.split_manager import SplitManager
from src.utils.embedding_cache import EmbeddingCache
print('All imports successful!')
\"

    echo ''
    echo 'INFO: Setup completed successfully!'
    echo 'INFO: You can now run: sbatch jobs/train_container.sh'
"
