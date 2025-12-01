#!/bin/bash
# Example workflow: Cache teacher embeddings and train with distillation
#
# This script demonstrates the complete workflow for knowledge distillation:
# 1. Pre-cache teacher embeddings at full resolution
# 2. Run student training with distillation using cached embeddings

set -e  # Exit on error

echo "==============================================="
echo "Knowledge Distillation Example Workflow"
echo "==============================================="
echo ""

# Configuration
CONFIG_FILE="configs/config.yaml"
CACHE_DIR="./cache/teacher_embeddings"
FULL_RESOLUTION=224

echo "Step 1: Caching teacher embeddings..."
echo "---------------------------------------"
python -m src.cli.cache_teacher_embeddings \
    --config "$CONFIG_FILE" \
    --cache-dir "$CACHE_DIR" \
    --full-resolution $FULL_RESOLUTION \
    --batch-size 256 \
    --num-workers 8 \
    --splits train

echo ""
echo "✓ Teacher embeddings cached successfully!"
echo ""