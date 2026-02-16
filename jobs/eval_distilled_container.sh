#!/bin/bash
#SBATCH --job-name=eval_distilled_3seeds
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Pipeline C: LP evaluation of distilled student models
#
# Run this AFTER distill_container.sh has completed.
# Uses the same splits and LP pipeline as the baseline (train_container.sh),
# but swaps the frozen encoder from DINOv3 to the distilled student.
#
# USAGE:
#   DOMAIN=dermatology sbatch jobs/eval_distilled_container.sh
#   DOMAIN=pathology   sbatch jobs/eval_distilled_container.sh
#   DOMAIN=radiology   sbatch jobs/eval_distilled_container.sh
#
# Optional overrides:
#   STUDENT=tiny_vit_21m_224 DOMAIN=dermatology sbatch jobs/eval_distilled_container.sh
#   RESOLUTIONS="512 256" DOMAIN=dermatology sbatch jobs/eval_distilled_container.sh
#   SEEDS="42" DOMAIN=dermatology sbatch jobs/eval_distilled_container.sh
#
# Pathology-specific:
#   TASKS="luad_vs_lusc kras" DOMAIN=pathology sbatch jobs/eval_distilled_container.sh
#
# TRAINING BUDGET ESTIMATE (3 seeds, 4 resolutions)
# -----------------------------------------------------------------------------
# 1. Embedding extraction: 3 seeds x 4 resolutions x ~5 min = ~1 hour
# 2. Linear probing: 3 seeds x 4 resolutions x 200 epochs = ~1-2 hours
# TOTAL ESTIMATED: 2-3 hours
#
# RESOURCES:
#   - GPU: 1x (student model much smaller than DINOv3)
#   - RAM: 48GB (embedding caching + data loading)
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
STUDENT_NAME="${STUDENT_NAME:-${STUDENT}_distilled}"
RESOLUTIONS="${RESOLUTIONS:-512 256 128 64}"
SEEDS="${SEEDS:-42 123 456}"

# Pathology-specific: TCGA tasks (ignored for other domains)
TASKS="${TASKS:-luad_vs_lusc lgg_vs_gbm kras tp53 egfr idh}"

# Number of classes per domain
case "$DOMAIN" in
    dermatology)
        CONFIG="probe_two_stage_dermatology"
        NUM_LABELS=3
        ;;
    radiology)
        CONFIG="probe_two_stage_radiology"
        NUM_LABELS=2
        ;;
    pathology)
        CONFIG="probe_two_stage_pathology"
        NUM_LABELS=2
        ;;
    *)
        echo "ERROR: Unknown domain '$DOMAIN'. Must be one of: dermatology, radiology, pathology"
        exit 1
        ;;
esac

echo "INFO: Domain=$DOMAIN  Student=$STUDENT  Config=$CONFIG"
echo "INFO: Resolutions=$RESOLUTIONS"
echo "INFO: Seeds=$SEEDS"
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

    # HuggingFace authentication
    if [ -f /workspace/.huggingface/token ]; then
        export HF_TOKEN=\$(cat /workspace/.huggingface/token)
        echo 'INFO: HuggingFace token loaded from .huggingface/token'
    elif [ -n \"\$HF_TOKEN\" ]; then
        echo 'INFO: Using HF_TOKEN from environment'
    else
        echo 'WARNING: No HuggingFace token found.'
    fi

    mkdir -p \$TMPDIR \$HF_HOME \$HF_DATASETS_CACHE \$TORCH_HOME

    # ==========================================================================
    # Resource Monitoring
    # ==========================================================================
    START_TIME=\$(date +%s)
    SLURM_TIME_LIMIT_SEC=\$((12 * 60 * 60))

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
    # Main LP Evaluation of Distilled Student
    # ==========================================================================
    echo 'INFO: Starting LP evaluation of distilled student...'
    echo \"INFO: Domain: $DOMAIN | Student: $STUDENT_NAME\"
    echo \"INFO: Start time: \$(date '+%Y-%m-%d %H:%M:%S')\"
    export HYDRA_FULL_ERROR=1

    # Model overrides to swap DINOv3 for the distilled student
    MODEL_OVERRIDES=\"model.name=$STUDENT_NAME model.model_id=$STUDENT model.type=timm model.config.num_labels=$NUM_LABELS model.config.pretrained=false\"

    if [ \"$DOMAIN\" = 'pathology' ]; then
        for TASK in $TASKS; do
            echo ''
            echo '============================================================'
            echo \"INFO: Evaluating distilled student on pathology task: \$TASK\"
            echo '============================================================'

            python -m src.cli.run_multiresolution_probe \
                --domain $DOMAIN \
                --model dinov3 \
                --config $CONFIG \
                --resolutions $RESOLUTIONS \
                --seeds $SEEDS \
                --extra-overrides \
                    datamodule.task=\$TASK \
                    \$MODEL_OVERRIDES

            echo \"INFO: Finished pathology task: \$TASK\"
        done
    else
        python -m src.cli.run_multiresolution_probe \
            --domain $DOMAIN \
            --model dinov3 \
            --config $CONFIG \
            --resolutions $RESOLUTIONS \
            --seeds $SEEDS \
            --extra-overrides \$MODEL_OVERRIDES
    fi

    END_TIME=\$(date +%s)
    TOTAL_ELAPSED=\$((END_TIME - START_TIME))
    TOTAL_H=\$((TOTAL_ELAPSED / 3600))
    TOTAL_M=\$(((TOTAL_ELAPSED % 3600) / 60))

    echo ''
    echo '============================================================'
    echo \"JOB COMPLETED: LP eval $DOMAIN ($STUDENT_NAME)\"
    echo '============================================================'
    echo \"End time: \$(date '+%Y-%m-%d %H:%M:%S')\"
    echo \"Total runtime: \${TOTAL_H}h \${TOTAL_M}m\"
    echo '============================================================'
  "
