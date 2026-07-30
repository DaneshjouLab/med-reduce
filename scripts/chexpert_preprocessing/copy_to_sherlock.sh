#!/bin/bash
#SBATCH --job-name=copy_chexpert
#SBATCH --partition=roxanad
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Copy CheXpert images listed in train_valid_combined.csv to a flat
# patient-structured destination on Sherlock/Oak.
#
# Source dirs (searched in order):
#   1) /oak/stanford/groups/roxanad/CheXpert/chex/chexpert-v1.0/
#   2) /oak/stanford/groups/roxanad/CheXpert/chex/chexpert-v1.0/valid/
#
# Destination:
#   $MR_CHEXPERT_DST  (default /oak/stanford/groups/roxanad/$USER/processed_chexpert/)
#
# Usage: sbatch copy_to_sherlock.sh

set -euo pipefail

# Use SLURM_SUBMIT_DIR (where sbatch was run), fall back to script dir for local runs
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

mkdir -p "$SCRIPT_DIR/logs"

CSV="$SCRIPT_DIR/train_valid_combined.csv"
SRC1="/oak/stanford/groups/roxanad/CheXpert/chex/chexpert-v1.0"
SRC2="/oak/stanford/groups/roxanad/CheXpert/chex/chexpert-v1.0/valid"
DST="${MR_CHEXPERT_DST:-/oak/stanford/groups/roxanad/$USER/processed_chexpert}/combined_train_valid_chexpert_v1.0"
LOGFILE="$SCRIPT_DIR/copy_to_sherlock.log"

# Log to both stdout and logfile
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }
log_err() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOGFILE" >&2; }

> "$LOGFILE"  # clear log

log "=== Starting copy_to_sherlock.sh ==="
log "CSV:  $CSV"
log "SRC1: $SRC1"
log "SRC2: $SRC2"
log "DST:  $DST"

# ── Validate inputs ──────────────────────────────────────────────────────────
if [ ! -f "$CSV" ]; then
    log_err "CSV file not found: $CSV"
    exit 1
fi

csv_lines=$(($(wc -l < "$CSV") - 1))
log "CSV contains $csv_lines paths (excl. header)"

if [ ! -d "$SRC1" ]; then
    log_err "Source dir 1 not found: $SRC1"
    exit 1
fi
log "SRC1 exists: OK"

if [ ! -d "$SRC2" ]; then
    log_err "Source dir 2 not found: $SRC2"
    exit 1
fi
log "SRC2 exists: OK"

mkdir -p "$DST"
log "DST created/verified: $DST"

# ── Copy loop ─────────────────────────────────────────────────────────────────
copied=0
missing=0
skipped=0
total=0
start_time=$(date +%s)

tail -n +2 "$CSV" | cut -d, -f1 | while IFS= read -r raw_path; do
    total=$((total + 1))

    # Extract patient.../study.../file from the path
    rel=$(echo "$raw_path" | grep -oiP 'patient\d+/.*' || true)

    if [ -z "$rel" ]; then
        log_err "Could not parse patient path from: $raw_path"
        skipped=$((skipped + 1))
        continue
    fi

    flat_name=$(echo "$rel" | sed 's|/|_|g')

    # Skip if already copied
    if [ -f "$DST/$flat_name" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    src_file=""
    if [ -f "$SRC1/$rel" ]; then
        src_file="$SRC1/$rel"
    elif [ -f "$SRC2/$rel" ]; then
        src_file="$SRC2/$rel"
    fi

    if [ -n "$src_file" ]; then
        if cp "$src_file" "$DST/$flat_name"; then
            copied=$((copied + 1))
        else
            log_err "Failed to copy: $src_file -> $DST/$flat_name"
        fi
    else
        log_err "MISSING in both SRC dirs: $rel"
        missing=$((missing + 1))
    fi

    # Progress every 5000 files
    if [ $((total % 5000)) -eq 0 ]; then
        elapsed=$(( $(date +%s) - start_time ))
        rate=$(( total / (elapsed + 1) ))
        remaining=$(( (csv_lines - total) / (rate + 1) ))
        log "Progress: $total/$csv_lines | copied=$copied skipped=$skipped missing=$missing | ${rate} files/s, ~${remaining}s remaining"
    fi
done

elapsed=$(( $(date +%s) - start_time ))
log "=== Done ==="
log "Total: $total | Copied: $copied | Skipped (already exist): $skipped | Missing: $missing"
log "Elapsed: ${elapsed}s"
log "Log saved to: $LOGFILE"
