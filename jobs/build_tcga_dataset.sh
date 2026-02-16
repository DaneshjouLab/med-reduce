#!/bin/bash
#SBATCH --job-name=tcga_build
#SBATCH --partition=roxanad
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# TCGA DATASET BUILD PIPELINE
# =============================================================================
#
# Downloads TCGA slide images and builds the full dataset:
#   ETL → Manifest → Download → Process Slides → Gene Matrix → Assemble
#
# Data output: /scratch/groups/roxanad/datasets/tcga
#
# RESOURCE ESTIMATES:
#   - ETL + Manifest: ~5 min (GDC API queries)
#   - Download: 4-20+ hours (depends on # slides, ~2000 SVS files ~500GB)
#   - Process slides: 1-3 hours (SVS → 512x512 JPG thumbnails)
#   - Gene matrix + Assemble: ~10 min
#
# Prerequisites:
#   sbatch jobs/setup_tcga.sh   (one-time)
#
# Usage:
#   cd /path/to/compressed-perception
#   sbatch jobs/build_tcga_dataset.sh
#
#   # Or override steps (e.g., skip download if already done):
#   sbatch jobs/build_tcga_dataset.sh --steps etl,manifest,process_slides,gene_matrix,assemble
#
#   # Or limit files for testing:
#   sbatch jobs/build_tcga_dataset.sh --max-files 5
# =============================================================================

set -e

SCRATCH_USER="/scratch/users/$USER"
SCRATCH_GROUP="/scratch/groups/roxanad"
TCGA_DATA_DIR="${SCRATCH_GROUP}/datasets/tcga"
PROJECT_DIR="$SLURM_SUBMIT_DIR"

# Parse optional args passed via: sbatch jobs/build_tcga_dataset.sh [args]
EXTRA_ARGS=""
STEPS_OVERRIDE=""
MAX_FILES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS_OVERRIDE="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --force)
            EXTRA_ARGS="$EXTRA_ARGS --force"
            shift
            ;;
        --dry-run)
            EXTRA_ARGS="$EXTRA_ARGS --dry-run"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================================"
echo "TCGA DATASET BUILD"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Project dir:   $PROJECT_DIR"
echo "Data dir:      $TCGA_DATA_DIR"
echo "Start time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Container setup
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="$SCRATCH_USER/simg"
SIF_IMAGE="${SIF_IMAGE:-python_3.10-slim-v2.sif}"

if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "ERROR: Container not found at $SIF_STORE/$SIF_IMAGE"
    echo "Run setup first: sbatch jobs/setup_tcga.sh"
    exit 1
fi

# Ensure data directories exist
mkdir -p "$TCGA_DATA_DIR"
mkdir -p logs

# Run pipeline inside container
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

    source .venv/bin/activate
    export PYTHONPATH=/workspace:\$PYTHONPATH
    echo 'INFO: Virtual environment activated.'
    echo 'INFO: Python: '\$(which python)

    export TMPDIR=/scratch_user/tmp
    mkdir -p \$TMPDIR

    # ==========================================================================
    # Resource monitoring (background)
    # ==========================================================================
    START_TIME=\$(date +%s)
    SLURM_TIME_LIMIT_SEC=\$((24 * 60 * 60))

    monitor_resources() {
        while true; do
            sleep 600  # Log every 10 minutes
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
            echo \"Time elapsed:   \${ELAPSED_H}h \${ELAPSED_M}m\"
            echo \"Time remaining: \${REMAINING_H}h \${REMAINING_M}m\"
            echo \"Disk usage (data dir): \$(du -sh /tcga_data 2>/dev/null | cut -f1)\"
            echo \"CPU/Memory: \$(free -h | grep Mem | awk '{print \$3 \"/\" \$2}')\"
            echo '============================================================'
        done
    }

    monitor_resources &
    MONITOR_PID=\$!
    cleanup() {
        kill \$MONITOR_PID 2>/dev/null || true
    }
    trap cleanup EXIT

    # ==========================================================================
    # Build CLI command
    # ==========================================================================
    CMD=\"python -m src.cli.build_tcga_dataset\"
    CMD=\"\$CMD --config configs/tcga_dataset_cluster.yaml\"

    # Steps override
    STEPS_ARG='$STEPS_OVERRIDE'
    if [ -n \"\$STEPS_ARG\" ]; then
        CMD=\"\$CMD --steps \$STEPS_ARG\"
    fi

    # Max files override
    MAX_FILES_ARG='$MAX_FILES'
    if [ -n \"\$MAX_FILES_ARG\" ]; then
        CMD=\"\$CMD download.max_files=\$MAX_FILES_ARG\"
    fi

    # Extra args (--force, --dry-run)
    CMD=\"\$CMD $EXTRA_ARGS\"

    echo ''
    echo \"INFO: Running: \$CMD\"
    echo ''
    eval \$CMD

    # ==========================================================================
    # Summary
    # ==========================================================================
    END_TIME=\$(date +%s)
    TOTAL_ELAPSED=\$((END_TIME - START_TIME))
    TOTAL_H=\$((TOTAL_ELAPSED / 3600))
    TOTAL_M=\$(((TOTAL_ELAPSED % 3600) / 60))

    echo ''
    echo '============================================================'
    echo 'JOB COMPLETED'
    echo '============================================================'
    echo \"End time:       \$(date '+%Y-%m-%d %H:%M:%S')\"
    echo \"Total runtime:  \${TOTAL_H}h \${TOTAL_M}m\"
    echo \"Data directory: /tcga_data\"
    echo \"Disk usage:     \$(du -sh /tcga_data 2>/dev/null | cut -f1)\"
    echo '============================================================'
"
