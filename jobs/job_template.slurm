#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodelist={{NODELIST}}
#SBATCH --gres=gpu:{{GPUS}}
#SBATCH --cpus-per-task={{CPUS_PER_TASK}}
#SBATCH --mem={{MEMORY}}
#SBATCH --time={{TIME_LIMIT}}
#SBATCH --output={{OUTPUT_FILE}}
#SBATCH --error={{ERROR_FILE}}

set -e  # Exit immediately on error

# Set up project directories
export CODE_DIR={{CODE_DIR}}
export VENV_DIR=$CODE_DIR/venv
export PROJECT_ROOT={{PROJECT_ROOT}}

ml load python/3.12.1
# Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r "$CODE_DIR/requirements.txt"
else
  echo "Using existing virtual environment"
  source "$VENV_DIR/bin/activate"
fi

# Create necessary output directories
mkdir -p $PROJECT_ROOT/{hf_cache/transformers,hf_cache/datasets,results,logs,models,plots}

# Set environment variables for Hugging Face + outputs
export TRANSFORMERS_CACHE=$PROJECT_ROOT/hf_cache/transformers
export HF_DATASETS_CACHE=$PROJECT_ROOT/hf_cache/datasets
export HF_HOME=$PROJECT_ROOT/hf_cache
export TRAIN_OUTPUT_DIR=$PROJECT_ROOT/results
export LOG_DIR=$PROJECT_ROOT/logs
export MODEL_DIR=$PROJECT_ROOT/models
export PLOT_DIR=$PROJECT_ROOT/plots

# Run training script
cd "$CODE_DIR"
echo "Starting training at $(date)"
python model_comparison_2.py
echo "Training complete at $(date)"
