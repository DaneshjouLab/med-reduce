#!/bin/bash
#SBATCH --job-name=distill_3seeds
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Pipeline B: Distillation — Train student to match DINOv3 embeddings
#
# USAGE:
#   DOMAIN=dermatology sbatch jobs/distill_container.sh
#   DOMAIN=pathology   sbatch jobs/distill_container.sh
#   DOMAIN=radiology   sbatch jobs/distill_container.sh
#
# Optional overrides:
#   STUDENT=tiny_vit_21m_224 DOMAIN=dermatology sbatch jobs/distill_container.sh
#   SEEDS="42" DOMAIN=dermatology sbatch jobs/distill_container.sh
#   ALPHA=0.7 DOMAIN=dermatology sbatch jobs/distill_container.sh
#
# Pathology-specific: choose which TCGA tasks to run (default: all 6)
#   TASKS="luad_vs_lusc kras" DOMAIN=pathology sbatch jobs/distill_container.sh
#
# TRAINING BUDGET ESTIMATE (per domain, per student, 3 seeds)
# -----------------------------------------------------------------------------
# 1. Teacher embedding caching (~20 min, reused across seeds)
# 2. Student distillation: 3 seeds x 100 epochs end-to-end = ~4-6 hours
# TOTAL ESTIMATED: 5-7 hours per student model
#
# RESOURCES:
#   - GPU: 1x (student model + forward pass)
#   - RAM: 48GB (teacher embeddings in memory + data loading)
#   - CPUs: 8 (num_workers=8 in config)
# =============================================================================

# This source file is part of the compressed perception project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# =============================================================================
# Configuration
# =============================================================================
DOMAIN="${DOMAIN:?ERROR: DOMAIN is required. Set DOMAIN=dermatology|radiology|pathology}"
STUDENT="${STUDENT:-resnet18}"
SEEDS="${SEEDS:-42 123 456}"
ALPHA="${ALPHA:-0.5}"
EPOCHS="${EPOCHS:-100}"

# Pathology-specific: TCGA tasks to run (ignored for other domains)
TASKS="${TASKS:-luad_vs_lusc lgg_vs_gbm kras tp53 egfr idh}"

# Validate domain and resolve config name
case "$DOMAIN" in
    dermatology|radiology|pathology)
        CONFIG="distillation_${DOMAIN}"
        ;;
    *)
        echo "ERROR: Unknown domain '$DOMAIN'. Must be one of: dermatology, radiology, pathology"
        exit 1
        ;;
esac

echo "INFO: Domain=$DOMAIN  Student=$STUDENT  Config=$CONFIG"
echo "INFO: Seeds=$SEEDS  Alpha=$ALPHA  Epochs=$EPOCHS"
if [ "$DOMAIN" = "pathology" ]; then
    echo "INFO: Tasks=$TASKS"
fi

# Container-based execution
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

# Run inside the container
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
    export WANDB_MODE=offline

    # HuggingFace authentication for gated models (dinov3 teacher)
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
    mkdir -p \$TMPDIR \$HF_HOME \$HF_DATASETS_CACHE \$TORCH_HOME

    # ==========================================================================
    # Resource Monitoring (runs in background)
    # ==========================================================================
    START_TIME=\$(date +%s)
    SLURM_TIME_LIMIT_SEC=\$((24 * 60 * 60))

    monitor_resources() {
        while true; do
            sleep 300
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

            if command -v nvidia-smi &> /dev/null; then
                echo 'GPU Usage:'
                nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
                    --format=csv,noheader,nounits | while read line; do
                    echo \"  \$line\"
                done
            fi

            echo \"CPU cores: \$(nproc)\"
            echo \"Memory: \$(free -h | grep Mem | awk '{print \$3 \"/\" \$2}')\"
            echo '============================================================'
            echo ''
        done
    }

    monitor_resources &
    MONITOR_PID=\$!
    echo \"INFO: Resource monitor started (PID: \$MONITOR_PID)\"

    cleanup() {
        echo 'INFO: Stopping resource monitor...'
        kill \$MONITOR_PID 2>/dev/null || true
    }
    trap cleanup EXIT

    # ==========================================================================
    # Main Distillation
    # ==========================================================================
    echo 'INFO: Starting distillation pipeline...'
    echo \"INFO: Domain: $DOMAIN | Student: $STUDENT | Alpha: $ALPHA\"
    echo \"INFO: Start time: \$(date '+%Y-%m-%d %H:%M:%S')\"
    export HYDRA_FULL_ERROR=1

    if [ \"$DOMAIN\" = 'pathology' ]; then
        for TASK in $TASKS; do
            echo ''
            echo '============================================================'
            echo \"INFO: Distillation for pathology task: \$TASK\"
            echo '============================================================'

            for SEED in $SEEDS; do
                echo \"INFO: Running seed \$SEED for task \$TASK...\"
                python -m src.cli.run_distillation \
                    --config-name=$CONFIG \
                    train.seed=\$SEED \
                    distillation.alpha=$ALPHA \
                    train.epochs=$EPOCHS \
                    student.model_id=$STUDENT \
                    datamodule.task=\$TASK
                echo \"INFO: Finished seed \$SEED for task \$TASK\"
            done

            echo \"INFO: Finished pathology task: \$TASK\"
        done
    else
        for SEED in $SEEDS; do
            echo ''
            echo '============================================================'
            echo \"INFO: Distillation seed \$SEED\"
            echo '============================================================'

            python -m src.cli.run_distillation \
                --config-name=$CONFIG \
                train.seed=\$SEED \
                distillation.alpha=$ALPHA \
                train.epochs=$EPOCHS \
                student.model_id=$STUDENT

            echo \"INFO: Finished seed \$SEED\"
        done
    fi

    END_TIME=\$(date +%s)
    TOTAL_ELAPSED=\$((END_TIME - START_TIME))
    TOTAL_H=\$((TOTAL_ELAPSED / 3600))
    TOTAL_M=\$(((TOTAL_ELAPSED % 3600) / 60))

    echo ''
    echo '============================================================'
    echo \"JOB COMPLETED: Distillation $DOMAIN ($STUDENT)\"
    echo '============================================================'
    echo \"End time: \$(date '+%Y-%m-%d %H:%M:%S')\"
    echo \"Total runtime: \${TOTAL_H}h \${TOTAL_M}m\"
    echo '============================================================'
  "
