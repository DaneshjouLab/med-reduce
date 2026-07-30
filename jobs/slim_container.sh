#!/usr/bin/env bash

# This source file is part of the compressed perception project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

set -euo pipefail

# Alternative setup script using minimal Python base image
# Use this if the PyTorch container is too large for your available memory

### ======== USER SETTINGS ========
# Use slim Python image instead of heavy PyTorch container
IMG_TAG="${IMG_TAG:-python:3.10-slim}"

### ======== PATHS ========
export CODE_DIR="${CODE_DIR:-/home/groups/roxanad/compressed-perception}"

# Store .sif and caches on SCRATCH
export SIF_STORE="${SIF_STORE:-/scratch/users/$USER/simg}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/users/$USER/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/users/$USER/apptainer_tmp}"

# venv location inside your repo (bind-mounted into container)
export VENV_DIR="${VENV_DIR:-$CODE_DIR/.venv}"

# Memory settings (much more conservative for smaller images)
export APPTAINER_SQUASHFS_THREADS="${APPTAINER_SQUASHFS_THREADS:-2}"
export APPTAINER_SQUASHFS_MEM="${APPTAINER_SQUASHFS_MEM:-1024M}"

# Back-compat for Singularity env names
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

### ======== PREP ========
mkdir -p "$SIF_STORE" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$CODE_DIR"

# Prefer apptainer if present, else singularity
if command -v apptainer >/dev/null 2>&1; then
  CTR="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CTR="singularity"
else
  echo "[error] Neither 'apptainer' nor 'singularity' found in PATH." >&2
  exit 1
fi

echo "[info] Using container tool: $CTR"
echo "[info] Using minimal Python image: $IMG_TAG"
echo "[info] Cache: $APPTAINER_CACHEDIR"
echo "[info] Tmp:   $APPTAINER_TMPDIR"
cd "$SIF_STORE"

### ======== PULL IMAGE ========
# Name the output SIF deterministically
# OUT_NAME="$(echo "$IMG_TAG" | sed 's#[:/@ ]#_#g').sif"
OUT_NAME="python_3.10-slim-v2.sif"

echo "[info] Starting image pull (should be much faster for Python-slim)"
echo "[info] This is only ~100MB instead of 8+GB for PyTorch containers"

# Pull the slim image (should not have memory issues)
$CTR pull "$OUT_NAME" "docker://$IMG_TAG"

# capture produced SIF name
SIF="$OUT_NAME"
echo "[info] SIF image ready: $SIF_STORE/$SIF"

# Quick verification
echo "[info] Verifying container contents..."
$CTR exec "$SIF_STORE/$SIF" python --version

### ======== NEXT STEPS ========
cat <<'EOF'

[info] Slim container ready! Now you need to install PyTorch manually.

Next steps:
1. Start an interactive session with GPU:
   srun -p gpu -c 4 --gres=gpu:1 --mem=32G --pty bash

2. Enter the container:
   TOOL=$(command -v apptainer || command -v singularity)
   SIF_STORE="/scratch/users/$USER/simg"

   # Create a temp directory for Python cache if it doesn't exist
   mkdir -p /scratch/users/$USER/pip_cache

   # Enter the container with proper bind mounts
   "$TOOL" shell --nv \
     -B "${CODE_DIR}:/workspace" \
     -B "/scratch/users/$USER:/scratch_user" \
     -B "/scratch/users/$USER/pip_cache:/root/.cache/pip" \
     -B "/tmp:/tmp" \
     --pwd /workspace \
     "$SIF_STORE/python_3.10-slim-copy.sif"

   # Note: The -B flags bind your host directories to the container
   # Make sure all these locations are writable by you

3. FIRST TIME ONLY - Inside the container, create venv and install PyTorch:
   # (Only do this once - the first time you set up)

   # First, make sure you're in the workspace directory where you have write permissions:
   cd /workspace

   # Create and activate the virtual environment (IMPORTANT!)
   python -m venv .venv
   source .venv/bin/activate

   # Check you're using the virtual environment's Python
   which python  # Should show /workspace/.venv/bin/python

   # Update basic tools - make sure we're using the venv pip
   python -m pip install --upgrade pip
   python -m pip install --upgrade wheel setuptools

   # Check pip location to confirm it's in the venv
   which pip  # Should show /workspace/.venv/bin/pip

   # Create site-packages directory if it doesn't exist (should not be necessary but just in case)
   mkdir -p /workspace/.venv/lib/python3.10/site-packages

   # Install PyTorch (choose version based on your CUDA):
   # For CUDA 11.8 (available versions):
   python -m pip install --no-cache-dir torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

   # Alternative: CPU-only version (if CUDA issues):
   # python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

   # For CUDA 12.1 (if you have newer drivers):
   # python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

   # Install other requirements one by one with --no-cache-dir to reduce memory usage
   # cat requirements.txt | xargs -n 1 python -m pip install --no-cache-dir

   # Verify PyTorch works:
   python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

4. EVERY TIME - To exit and re-enter the container:
   # To exit the container:
   exit

   # To get back in (from a GPU node):
   TOOL=$(command -v apptainer || command -v singularity)
   SIF_STORE="/scratch/users/$USER/simg"

   "$TOOL" shell --nv \
     -B "/scratch/users/$USER/safran:/workspace" \
     -B "/scratch/users/$USER:/scratch_user" \
     -B "/scratch/users/$USER/pip_cache:/root/.cache/pip" \
     -B "/tmp:/tmp" \
     --pwd /workspace \
     "$SIF_STORE/python_3.10-slim.sif"

   # IMPORTANT: Always activate the virtual environment first:
   source .venv/bin/activate

   # Verify you're using the virtual environment's Python:
   which python  # Should show /workspace/.venv/bin/python

5. Running your scripts:
   # Inside the container with venv activated:
   python -m src.train
   # etc.

[info] Your virtual environment (.venv) is saved in the bound directory,
       so it persists between container sessions.

[info] This approach downloads PyTorch as pip packages instead of pulling
       a massive container image, avoiding the memory issues during build.

EOF