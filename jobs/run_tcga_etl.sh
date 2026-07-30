#!/bin/bash
#SBATCH --job-name=tcga_etl
#SBATCH --partition=roxanad
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# ONE COMMAND: install all packages + run the TCGA ETL + land outputs in the
# training location. Reads the raw SVS slides read-only from OAK (no 500GB copy)
# and writes thumbnails + dataset.csv directly to the scratch dataset dir that
# the LP / distillation configs point at.
#
#   sbatch jobs/run_tcga_etl.sh
#
# Optional overrides:
#   STEPS="gene_matrix,process_slides,assemble" sbatch jobs/run_tcga_etl.sh   # also rebuild mutation labels
#   REGEN_THUMBNAILS=0 sbatch jobs/run_tcga_etl.sh                            # keep existing thumbnails
#   OAK_TCGA=/oak/.../processed_tcga TCGA_DATA_DIR=/scratch/.../tcga sbatch jobs/run_tcga_etl.sh
# =============================================================================

set -euo pipefail

SCRATCH_USER="/scratch/users/$USER"
SCRATCH_GROUP="${SCRATCH_GROUP:-/scratch/groups/roxanad}"

# Source (raw ETL artifacts incl. slides) and destination (training location).
OAK_TCGA="${OAK_TCGA:-/oak/stanford/groups/roxanad/bikia/processed_tcga}"
TCGA_DATA_DIR="${TCGA_DATA_DIR:-${SCRATCH_GROUP}/datasets/tcga}"   # what the configs read
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
STEPS="${STEPS:-process_slides,assemble}"          # thumbnails + dataset.csv
REGEN_THUMBNAILS="${REGEN_THUMBNAILS:-1}"          # 1 = clear existing so they are recreated

echo "INFO: Project dir : $PROJECT_DIR"
echo "INFO: OAK source  : $OAK_TCGA"
echo "INFO: Dest (train): $TCGA_DATA_DIR"
echo "INFO: Steps       : $STEPS"

# -------------------------------------------------------
# Host-side prep (OAK + scratch are mounted on the node).
# Stage the small tables (slide_table.parquet + gene_matrix.parquet); the large
# slides/ stay on OAK and are bind-mounted read-only. Thumbnails + dataset.csv
# are (re)generated straight into $TCGA_DATA_DIR.
# -------------------------------------------------------
mkdir -p "$TCGA_DATA_DIR"/{tables,thumbnails,slides,maf} "$SCRATCH_USER"/{pip_cache,simg,tmp} logs
echo "INFO: Staging any existing tables/ parquets from OAK ..."
# Use cp, not `rsync -a`: the dest tables/ is group-owned, so preserving its
# directory times/perms fails ("Operation not permitted"). We only need the
# parquet inputs (slide_table + gene_matrix); dataset.csv is regenerated. If OAK
# has none, the 'etl' step must be in STEPS to build slide_table from GDC.
shopt -s nullglob
oak_parquets=("$OAK_TCGA"/tables/*.parquet)
shopt -u nullglob
if [ ${#oak_parquets[@]} -gt 0 ]; then
  cp -f "${oak_parquets[@]}" "$TCGA_DATA_DIR/tables/"
  echo "INFO: staged ${#oak_parquets[@]} parquet(s) from OAK tables/."
else
  echo "INFO: no parquets in OAK tables/ — 'etl' must build slide_table."
fi

# slide_table is required by process_slides/assemble. If it's absent AND 'etl'
# isn't in STEPS to build it, fail fast with a clear message.
if [ ! -f "$TCGA_DATA_DIR/tables/slide_table.parquet" ] && [[ ",$STEPS," != *",etl,"* ]]; then
  echo "ERROR: no slide_table.parquet, and 'etl' is not in STEPS='$STEPS'."
  echo "       Rebuild it (needs network/GDC):"
  echo '       STEPS="etl,manifest,gene_matrix,process_slides,assemble" sbatch jobs/run_tcga_etl.sh'
  exit 1
fi

if [ "$REGEN_THUMBNAILS" = "1" ]; then
  echo "INFO: Clearing existing thumbnails so process_slides recreates them ..."
  rm -rf "${TCGA_DATA_DIR:?}/thumbnails/"* 2>/dev/null || true
fi

# -------------------------------------------------------
# Container
# -------------------------------------------------------
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="$SCRATCH_USER/simg"
SIF_IMAGE="${SIF_IMAGE:-python_3.10-slim-v2.sif}"
if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
  echo "INFO: Pulling container image ..."
  ( cd "$SIF_STORE" && "$TOOL" pull "$SIF_IMAGE" "docker://python:3.10-slim" )
fi

# Bind: workspace, scratch, pip cache, dest dataset dir (writable) and — nested —
# the OAK slides read-only over /tcga_data/slides so recomputed SVS paths resolve.
"$TOOL" exec \
  -B "$PROJECT_DIR:/workspace" \
  -B "$SCRATCH_USER:/scratch_user" \
  -B "$SCRATCH_USER/pip_cache:/root/.cache/pip" \
  -B "$TCGA_DATA_DIR:/tcga_data" \
  -B "$OAK_TCGA/slides:/tcga_data/slides:ro" \
  -B "$OAK_TCGA/maf:/tcga_data/maf:ro" \
  -B "/tmp:/tmp" \
  --pwd /workspace \
  "$SIF_STORE/$SIF_IMAGE" \
  bash -c "
    set -e
    cd /workspace
    export PYTHONPATH=/workspace:\$PYTHONPATH
    export TMPDIR=/scratch_user/tmp && mkdir -p \$TMPDIR

    # ---- Install all packages ----
    if [ ! -d '.venv' ]; then echo 'INFO: Creating venv...'; python -m venv .venv; fi
    source .venv/bin/activate
    echo 'INFO: Python: '\$(which python)
    pip install --upgrade pip wheel setuptools
    rm -rf /workspace/*.egg-info
    echo 'INFO: Installing project + deps (incl. openslide) ...'
    pip install -e .
    python -c 'import openslide; print(\"OpenSlide:\", openslide.__version__)'

    # ---- Run the ETL (writes into /tcga_data = the training location) ----
    echo 'INFO: Running TCGA ETL steps: $STEPS'
    python -m src.cli.build_tcga_dataset \
        --config configs/tcga_dataset_cluster.yaml \
        --steps $STEPS

    echo ''
    echo 'INFO: Outputs:'
    ls -la /tcga_data/tables/dataset.csv || echo '  (dataset.csv not found!)'
    echo \"  thumbnails: \$(ls /tcga_data/thumbnails 2>/dev/null | wc -l) files\"
  "

echo ""
echo "DONE. dataset.csv → $TCGA_DATA_DIR/tables/dataset.csv ; thumbnails → $TCGA_DATA_DIR/thumbnails/"
echo "These are exactly the paths configs/probe_two_stage_pathology.yaml points at."
