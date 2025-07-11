#!/bin/bash

#SBATCH --job-name=231n_job
#SBATCH --time=2-23:59:00
#SBATCH --output=job_output_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p roxanad
#SBATCH --mem=128G


# Loading python version
ml python/3.12

# Creating venv and installing requirements
python3 -m venv venv      
source venv/bin/activate
pip install -r /home/groups/roxanad/eric/CS231N/requirements.txt

# WANDB Key
export WANDB_API_KEY="7ab80eeb87ef06298c6bca1258208b1739ad32fe"

# Running the script
python src/models/model_comparison_models.py --resolution 224 --batch_size 256 --num_train_images 25000 --num_epochs 10 --eval_steps 10