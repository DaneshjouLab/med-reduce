#!/bin/bash
#SBATCH --job-name=probe_3seeds
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=26:00:00
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
    # Token is read from project root .huggingface/token or HF_TOKEN env var
    if [ -f /workspace/.huggingface/token ]; then
        export HF_TOKEN=\$(cat /workspace/.huggingface/token)
        echo 'INFO: HuggingFace token loaded from .huggingface/token'
    elif [ -n \"\$HF_TOKEN\" ]; then
        echo 'INFO: Using HF_TOKEN from environment'
    else
        echo 'WARNING: No HuggingFace token found. Gated models (dinov3) may fail.'
        echo '         Create .huggingface/token in project root with your HF token'
    fi

    # Create all directories
    echo 'INFO: Creating required directories...'
    mkdir -p \$TMPDIR \$HF_HOME \$HF_DATASETS_CACHE \$TORCH_HOME \
            \$TRAIN_OUTPUT_DIR \$LOG_DIR \$MODEL_DIR \$PLOT_DIR

    # ==========================================================================
    # Resource Monitoring (runs in background)
    # ==========================================================================
    START_TIME=\$(date +%s)
    SLURM_TIME_LIMIT_SEC=\$((12 * 60 * 60))  # 12 hours in seconds

    monitor_resources() {
        while true; do
            sleep 300  # Log every 5 minutes

            CURRENT_TIME=\$(date +%s)
            ELAPSED=\$((CURRENT_TIME - START_TIME))
            REMAINING=\$((SLURM_TIME_LIMIT_SEC - ELAPSED))

            ELAPSED_H=\$((ELAPSED / 3600))
            ELAPSED_M=\$(((ELAPSED % 3600) / 60))
            REMAINING_H=\$((REMAINING / 3600))
            REMAINING_M=\$(((REMAINING % 3600) / 60))

            echo ''
            echo '============================================================'
            echo \"RESOURCE MONITOR - \$(date '+%Y-%m-%d %H:%M:%S')\"
            echo '============================================================'
            echo \"Time elapsed:   \${ELAPSED_H}h \${ELAPSED_M}m\"
            echo \"Time remaining: \${REMAINING_H}h \${REMAINING_M}m\"
            echo ''

            # GPU stats (if nvidia-smi available)
            if command -v nvidia-smi &> /dev/null; then
                echo 'GPU Usage:'
                nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
                    --format=csv,noheader,nounits | while read line; do
                    echo \"  \$line\"
                done
            fi

            # CPU and memory
            echo ''
            echo 'CPU/Memory:'
            echo \"  CPU cores: \$(nproc)\"
            echo \"  Memory: \$(free -h | grep Mem | awk '{print \$3 \"/\" \$2}')\"
            echo '============================================================'
            echo ''
        done
    }

    # Start monitoring in background
    monitor_resources &
    MONITOR_PID=\$!
    echo \"INFO: Resource monitor started (PID: \$MONITOR_PID)\"

    # Cleanup function
    cleanup() {
        echo 'INFO: Stopping resource monitor...'
        kill \$MONITOR_PID 2>/dev/null || true
    }
    trap cleanup EXIT

    # ==========================================================================
    # Main Training
    # ==========================================================================
    echo 'INFO: Starting Hydra run...'
    echo \"INFO: Start time: \$(date '+%Y-%m-%d %H:%M:%S')\"
    export HYDRA_FULL_ERROR=1

    python -m src.cli.run_multiresolution_probe \
        --domain dermatology \
        --model dinov3 \
        --config probe_two_stage_dermatology \
        --tune-hyperparams \
        --resolutions 512 256 128 64


    END_TIME=\$(date +%s)
    TOTAL_ELAPSED=\$((END_TIME - START_TIME))
    TOTAL_H=\$((TOTAL_ELAPSED / 3600))
    TOTAL_M=\$(((TOTAL_ELAPSED % 3600) / 60))

    echo ''
    echo '============================================================'
    echo 'JOB COMPLETED'
    echo '============================================================'
    echo \"End time: \$(date '+%Y-%m-%d %H:%M:%S')\"
    echo \"Total runtime: \${TOTAL_H}h \${TOTAL_M}m\"
    echo '============================================================'
  "