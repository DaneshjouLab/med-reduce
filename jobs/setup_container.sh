#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# ONE-TIME SETUP: Creates venv and installs all dependencies
# Run this once before using train_container.sh
# =============================================================================

set -e

# Create required directories
echo "INFO: Creating required directories..."
mkdir -p /scratch/users/$USER/pip_cache
mkdir -p /scratch/users/$USER/simg
mkdir -p /scratch/users/$USER/tmp
mkdir -p /scratch/users/$USER/huggingface
mkdir -p /scratch/users/$USER/torch
mkdir -p logs

# Container setup
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_IMAGE:-python_3.10-slim-v2.sif}"

# Pull container if it doesn't exist
if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "INFO: Pulling container image..."
    cd "$SIF_STORE"
    $TOOL pull "$SIF_IMAGE" "docker://python:3.10-slim"
    cd -
fi

echo "INFO: Using container: $SIF_STORE/$SIF_IMAGE"

# Run setup inside container
"$TOOL" exec --nv \
    -B "/scratch/users/$USER/med-reduce:/workspace" \
    -B "/scratch/users/$USER:/scratch_user" \
    -B "/scratch/users/$USER/pip_cache:/root/.cache/pip" \
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
    echo 'INFO: Virtual environment activated'
    echo 'INFO: Python: '\$(which python)

    # HuggingFace authentication for gated models (dinov3)
    # Token is read from project root .huggingface/token or HF_TOKEN env var
    if [ -f /workspace/.huggingface/token ]; then
        export HF_TOKEN=\$(cat /workspace/.huggingface/token)
        echo 'INFO: HuggingFace token loaded from .huggingface/token'
    elif [ -n \"\$HF_TOKEN\" ]; then
        echo 'INFO: Using HF_TOKEN from environment'
    else
        echo 'WARNING: No HuggingFace token found. Gated models (dinov3) may fail.'
    fi

    # Upgrade pip
    echo 'INFO: Upgrading pip...'
    pip install --upgrade pip wheel setuptools

    # Install PyTorch with CUDA 11.8 (skip if already installed)
    if ! python -c 'import torch' 2>/dev/null; then
        echo 'INFO: Installing PyTorch with CUDA 11.8...'
        pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    else
        echo 'INFO: PyTorch already installed, skipping...'
    fi

    # Always reinstall project to pick up any changes (uses pyproject.toml)
    echo 'INFO: Installing/reinstalling project from pyproject.toml...'
    pip install -e . --no-deps
    pip install -e .

    # ==========================================================================
    # Prefetch ALL model weights into the shared HF cache so GPU/compute nodes
    # (often offline) can run without downloading. Covers every teacher used by
    # the LP baseline + the timm students used by the distillation container
    # (which reads this same cache). Each download is non-fatal on failure.
    # ==========================================================================
    echo ''
    echo '=========================================='
    echo 'PREFETCHING MODEL WEIGHTS'
    echo '=========================================='
    export HF_HOME=/scratch_user/huggingface
    export TORCH_HOME=/scratch_user/torch
    mkdir -p \$HF_HOME \$TORCH_HOME
    python -c \"
def _try(name, fn):
    try:
        fn(); print('  [ok] ' + name)
    except Exception as e:
        print('  [WARN] ' + name + ': ' + type(e).__name__ + ': ' + str(e)[:160])
import transformers, timm
# Teachers
_try('dinov3 teacher (facebook/dinov3-vits16-pretrain-lvd1689m)',
     lambda: transformers.AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m'))
_try('vit teacher (google/vit-base-patch16-224)',
     lambda: transformers.ViTModel.from_pretrained('google/vit-base-patch16-224'))
_try('biomedclip teacher (open_clip hf-hub)',
     lambda: __import__('open_clip').create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'))
# Distillation students (timm, pretrained)
_try('resnet50.a1_in1k student', lambda: timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=0))
_try('tiny_vit_21m_224.dist_in22k student', lambda: timm.create_model('tiny_vit_21m_224.dist_in22k', pretrained=True, num_classes=0))
print('Prefetch done (any [WARN] above is non-fatal; DINOv3 needs a valid HF token).')
\"

    # Verify installation
    echo ''
    echo '=========================================='
    echo 'VERIFICATION'
    echo '=========================================='
    python -c \"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

# Verify src package imports
print()
print('Testing src package imports...')
from src.data.tabular_datamodule_persistent import TabularDataModulePersistent
from src.wrappers.probe_two_stage import ProbeTwoStageWrapper
from src.utils.split_manager import SplitManager
from src.utils.embedding_cache import EmbeddingCache
print('All imports successful!')
\"

    echo ''
    echo 'INFO: Setup completed successfully!'
    echo 'INFO: You can now run: sbatch jobs/train_container.sh'
"
