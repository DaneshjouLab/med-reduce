#!/bin/bash
#SBATCH --job-name=setup_tcga
#SBATCH --partition=roxanad
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# ONE-TIME SETUP: Pull container, create venv, install deps (incl. gdc-client)
# Run this once before build_tcga_dataset.sh
#
# Usage:
#   cd /path/to/compressed-perception
#   mkdir -p logs
#   sbatch jobs/setup_tcga.sh
# =============================================================================

set -e

SCRATCH_USER="/scratch/users/$USER"
SCRATCH_GROUP="${SCRATCH_GROUP:-/scratch/groups/roxanad}"
TCGA_DATA_DIR="${SCRATCH_GROUP}/datasets/tcga"
PROJECT_DIR="$SLURM_SUBMIT_DIR"

echo "INFO: Project directory: $PROJECT_DIR"
echo "INFO: TCGA data directory: $TCGA_DATA_DIR"

# Create required directories
echo "INFO: Creating directories..."
mkdir -p "$SCRATCH_USER/pip_cache"
mkdir -p "$SCRATCH_USER/simg"
mkdir -p "$SCRATCH_USER/tmp"
mkdir -p "$TCGA_DATA_DIR"
mkdir -p logs

# Container setup
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="$SCRATCH_USER/simg"
SIF_IMAGE="${SIF_IMAGE:-python_3.10-slim-v2.sif}"

# Pull container if it doesn't exist
if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "INFO: Pulling container image..."
    cd "$SIF_STORE"
    $TOOL pull "$SIF_IMAGE" "docker://python:3.10-slim"
    cd "$PROJECT_DIR"
else
    echo "INFO: Container already exists: $SIF_STORE/$SIF_IMAGE"
fi

# Run setup inside container
"$TOOL" exec \
    -B "$PROJECT_DIR:/workspace" \
    -B "$SCRATCH_USER:/scratch_user" \
    -B "$SCRATCH_USER/pip_cache:/root/.cache/pip" \
    -B "$TCGA_DATA_DIR:/tcga_data" \
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
    export PYTHONPATH=/workspace:\$PYTHONPATH
    echo 'INFO: Python: '\$(which python)

    # Upgrade pip
    echo 'INFO: Upgrading pip...'
    pip install --upgrade pip wheel setuptools

    # Clean stale egg-info (may have wrong paths from rsync)
    rm -rf /workspace/*.egg-info

    # Install project and all dependencies
    echo 'INFO: Installing project from pyproject.toml...'
    pip install -e .

    # Verify installation
    echo ''
    echo '=========================================='
    echo 'VERIFICATION'
    echo '=========================================='
    python -c \"
import sys
print(f'Python: {sys.version}')

print()
print('Testing TCGA pipeline imports...')
from src.data.tcga.pipeline import TCGADatasetBuilder
from src.data.tcga.etl import TCGASlideETL
from src.data.tcga.downloader import TCGADownloader
from src.data.tcga.slide_processor import SlideProcessor
from src.data.tcga.gene_matrix import GeneMatrix
from src.data.tcga.manifest import ManifestGenerator
from src.data.tcga.config import TCGAConfig
print('All TCGA pipeline imports successful!')

print()
print('Testing openslide...')
import openslide
print(f'OpenSlide version: {openslide.__version__}')
\"

    # Verify data directory is writable
    echo ''
    echo 'Testing data directory...'
    touch /tcga_data/.write_test && rm /tcga_data/.write_test
    echo 'Data directory is writable: /tcga_data'

    echo ''
    echo 'INFO: Setup completed successfully!'
    echo 'INFO: Next step: sbatch jobs/build_tcga_dataset.sh'
"
