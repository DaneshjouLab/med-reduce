#!/bin/bash
#SBATCH --job-name=probe_3seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# USAGE:
#   DOMAIN=dermatology sbatch jobs/train_container.sh
#   DOMAIN=radiology   sbatch jobs/train_container.sh
#   DOMAIN=pathology   sbatch jobs/train_container.sh
#
# Optional overrides:
#   MODEL=dinov2 DOMAIN=radiology sbatch jobs/train_container.sh
#   RESOLUTIONS="512 256" DOMAIN=pathology sbatch jobs/train_container.sh
#   SEEDS="42 123 456" DOMAIN=dermatology sbatch jobs/train_container.sh
#
# Pathology-specific: choose which TCGA tasks to run (default: all 5)
#   DOMAIN=pathology sbatch jobs/train_container.sh                          # all tasks
#   TASKS="luad_vs_lusc kras" DOMAIN=pathology sbatch jobs/train_container.sh  # subset
#
# TRAINING BUDGET ESTIMATE (per domain, dinov3, 3 seeds, 4 resolutions)
# -----------------------------------------------------------------------------
# 1. Hyperparameter Tuning (seed 42 only, highest res):
#    - Grid: 3 lr × 3 wd × 2 bs = 18 configs, 5-fold CV
#    - Estimated: ~3-4 hours
# 2. Embedding Extraction: 3 seeds × 4 resolutions × ~10 min = ~2 hours
# 3. Final Linear Probing: 3 seeds × 4 resolutions = ~1-2 hours
# TOTAL ESTIMATED: 6-8 hours (with buffer for safety)
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

# =============================================================================
# Domain configuration
# =============================================================================
DOMAIN="${DOMAIN:?ERROR: DOMAIN is required. Set DOMAIN=dermatology|radiology|pathology}"
MODEL="${MODEL:-dinov3}"
STUDENT="${STUDENT:-resnet18}"
STUDENT_NAME="${STUDENT_NAME:-$(echo "$STUDENT" | sed 's/_[0-9].*$//')}"
RESOLUTIONS="${RESOLUTIONS:-512 256 128 64}"
SEEDS="${SEEDS:-42 123 456}"
EXTRAS="${EXTRAS:-}"  # Extra Hydra overrides (e.g. EXTRAS="runtime.run_dir=/path/to/v2")
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"  # Dir with distilled checkpoints for probe_distilled mode

# Pathology-specific: TCGA tasks to run (ignored for other domains)
# Override with e.g. TASKS="kras tp53" to run a subset
TASKS="${TASKS:-luad_vs_lusc lgg_vs_gbm kras tp53 egfr}"

# Validate domain and resolve config name
case "$DOMAIN" in
    dermatology|radiology|pathology)
        CONFIG="probe_two_stage_${DOMAIN}"
        ;;
    *)
        echo "ERROR: Unknown domain '$DOMAIN'. Must be one of: dermatology, radiology, pathology"
        exit 1
        ;;
esac

echo "INFO: Domain=$DOMAIN  Model=$MODEL  Config=$CONFIG"
echo "INFO: Resolutions=$RESOLUTIONS"
echo "INFO: Seeds=$SEEDS"
if [ "$DOMAIN" = "pathology" ]; then
    echo "INFO: Tasks=$TASKS"
fi

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
    SLURM_TIME_LIMIT_SEC=\$((60 * 60 * 60))  # 60 hours in seconds

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
            if command -v free &> /dev/null; then
                echo \"  Memory: \$(free -h | grep Mem | awk '{print \$3 \"/\" \$2}')\"
            elif [ -f /proc/meminfo ]; then
                echo \"  Memory: \$(awk '/MemTotal/{t=\$2} /MemAvailable/{a=\$2} END{printf \"%.0fM / %.0fM\", (t-a)/1024, t/1024}' /proc/meminfo)\"
            fi
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
    echo \"INFO: Domain: $DOMAIN | Model: $MODEL\"
    echo \"INFO: Start time: \$(date '+%Y-%m-%d %H:%M:%S')\"
    export HYDRA_FULL_ERROR=1

    if [ \"$DOMAIN\" = 'pathology' ]; then
        # Pathology: loop over TCGA tasks (each is a separate classification problem)
        for TASK in $TASKS; do
            echo ''
            echo '============================================================'
            echo \"INFO: Starting pathology task: \$TASK\"
            echo '============================================================'

            CKPT_OVERRIDE=\"\"
            if [ -n \"$CHECKPOINT_DIR\" ]; then
                CKPT_OVERRIDE=\"+model.config.checkpoint_dir=${CHECKPOINT_DIR} +model.config.checkpoint_pattern=distilled_${STUDENT_NAME}_\${TASK}.pt\"
            fi
            python -m src.cli.run_multiresolution_probe \
                --domain $DOMAIN \
                --model $MODEL \
                --config $CONFIG \
                --tune-hyperparams \
                --resolutions $RESOLUTIONS \
                --seeds $SEEDS \
                --extra-overrides datamodule.task=\$TASK \$CKPT_OVERRIDE $EXTRAS


            echo \"INFO: Finished pathology task: \$TASK\"
        done
    else
        # Dermatology / Radiology: single run, no task override
        CKPT_OVERRIDE=\"\"
        if [ -n \"$CHECKPOINT_DIR\" ]; then
            CKPT_OVERRIDE=\"+model.config.checkpoint_dir=${CHECKPOINT_DIR} +model.config.checkpoint_pattern=distilled_${STUDENT_NAME}.pt\"
        fi
        EXTRA_ARGS=\"\"
        if [ -n \"$EXTRAS\" ] || [ -n \"\$CKPT_OVERRIDE\" ]; then
            EXTRA_ARGS=\"--extra-overrides \$CKPT_OVERRIDE $EXTRAS\"
        fi
        python -m src.cli.run_multiresolution_probe \
            --domain $DOMAIN \
            --model $MODEL \
            --config $CONFIG \
            --tune-hyperparams \
            --resolutions $RESOLUTIONS \
            --seeds $SEEDS \
            \$EXTRA_ARGS
    fi


    END_TIME=\$(date +%s)
    TOTAL_ELAPSED=\$((END_TIME - START_TIME))
    TOTAL_H=\$((TOTAL_ELAPSED / 3600))
    TOTAL_M=\$(((TOTAL_ELAPSED % 3600) / 60))

    echo ''
    echo '============================================================'
    echo \"JOB COMPLETED: $DOMAIN ($MODEL)\"
    echo '============================================================'
    echo \"End time: \$(date '+%Y-%m-%d %H:%M:%S')\"
    echo \"Total runtime: \${TOTAL_H}h \${TOTAL_M}m\"
    echo '============================================================'
  "