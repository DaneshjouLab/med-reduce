# Model Comparison Pipeline

## Overview

This script (`model_comparison_models.py`) provides a baseline for comparing different image classification models at various image compression levels, including the original images. It supports fine-tuning, linear probing, and optional image degradation transforms.

---

## Features

- **Model Support:** Vision Transformer (ViT), DINOv2, SimCLR (self-supervised backbone)
- **Data Augmentation:** Optional JPEG compression, Gaussian blur, and color quantization
- **Dataset Balancing:** Ensures equal samples per class for fair comparison
- **Training & Evaluation:** Uses Hugging Face Trainer for streamlined workflows
- **Experiment Tracking:** Integrated with Weights & Biases (`wandb`)
- **GPU Monitoring:** Optional support via `pynvml`

---

## Workflow

1. **Environment Setup**
   - Loads required libraries and sets up cache directories.
   - Checks for GPU availability.

2. **Dataset Loading & Balancing**
   - Loads ISIC_2019_224 dataset.
   - Balances the dataset across filtered classes.

3. **Model Initialization**
   - Initializes model and preprocessor based on configuration.

4. **Preprocessing & Augmentation**
   - Applies resizing, normalization, and optional degradation transforms.

5. **Training & Evaluation**
   - Splits data into training and validation sets.
   - Trains and evaluates each model, logging results to `wandb`.

---

## Usage

```bash
python [model_comparison_models.py](http://_vscodecontentref_/2) --resolution 224 --batch_size 256 --num_train_images 25000 --num_epochs 10 --eval_steps 10
```

## Configuration
- Models: Edit the models list in the script to add or modify model configurations.
- Transforms: Toggle apply_transforms in prepare_datasets() to enable/disable augmentations.
- Hyperparameters: Adjust arguments in the main() function for batch size, epochs, etc.

## Output
- Training and evaluation metrics are printed to the console and logged to Weights & Biases.
- Results can be used for further analysis or ablation studies.

## Extending
- Add new models by updating the models list.
- Implement new transforms in utils/transforms.py.
- Add new datasets by modifying the dataset loading logic.

## References
- Hugging Face Transformers
- PyTorch
- Weights & Biases
- ISIC 2019 Dataset