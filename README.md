# Finetuning Pretrained Models for Compressed Dermatology Image Analysis

This project explores how compressed and degraded dermatology images (from the ISIC 2019 dataset) affect classification performance using pretrained vision models. It compares fine-tuning vs. linear probing across multiple compression and degradation levels.

![Midpoint Research Poster](<./CS231N Poster.png>)

## Project Goals

- Evaluate model robustness to multiple image degradations (JPEG compression, Gaussian blur, color quantization)
- Compare pretrained models: ViT, DINOv2, and SimCLR
- Benchmark fine-tuning vs. linear probing strategies
- Analyze computational efficiency (FLOPs, GPU memory) vs. classification performance

## Models

- **ViT**: Vision Transformer (`google/vit-base-patch16-224`)
- **DINOv2**: Self-supervised ViT from Meta (`facebook/dinov2-base`)

## Metrics Tracked

- **Performance**: Accuracy, F1 Score, AUC-ROC
- **Efficiency**: GFLOPs, peak GPU memory usage
- **Training**: Time per epoch, convergence speed
- **Robustness**: Performance degradation under compression

## Project Structure

```
CS231N/
├── configs/
│   ├── config.yaml                 # SLURM job configuration
│   └── example_config.yaml 
├── jobs/              
│   ├── submit_from_config.sh       # SLURM job template
│   └── submit_from_config.sh       # Job submission script
│
├── src/                            # Main source code
│   ├── config.py                   # Configuration and constants
│   ├── utils.py                    # Environment, GPU, I/O utilities
│   ├── models.py                   # Model architectures and helpers
│   ├── transforms.py               # Image degradation transforms
│   ├── datasets.py                 # Dataset implementations
│   ├── training.py                 # Training callbacks and metrics
│   └── train.py                    # Main training script
│
├── scripts/                        # Utility scripts
│   └── download_unpack_isic2019.sh
│
├── results/                        # Generated outputs
│   ├── models/                     # Saved model checkpoints
│   ├── plots/                      # Confusion matrices, metrics
│   └── logs/                       # Training logs, SLURM outputs
│
├── requirements.txt
└── README.md   
```

## Quick Start

### Local Training

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Weights & Biases (optional):**
   ```bash
   wandb login  # Or set WANDB_API_KEY environment variable
   ```

3. **Run training:**
   ```bash
   # Basic training
   python src/train.py --batch_size 256 --num_epochs 5
   
   # With custom configuration
   python src/train.py \
     --resolution 224 \
     --batch_size 128 \
     --num_train_images 1000 \
     --num_epochs 10 \
     --learning_rate 1e-4 \
     --eval_steps 100
   ```

### Cluster Training (SLURM)

1. **Configure job settings:**
   ```bash
   # Edit config.yaml with your settings
   vim config.yaml
   ```

2. **Submit job:**
   ```bash
   ./submit_from_config.sh
   ```

3. **Monitor job:**
   ```bash
   squeue -u $USER
   tail -f train_output.log
   ```

## Configuration Options

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--resolution` | 224 | Input image size |
| `--batch_size` | 128 | Training batch size |
| `--num_train_images` | 500 | Training samples per class |
| `--num_epochs` | 3 | Training epochs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--eval_steps` | 100 | Steps between evaluations |

### Data Augmentations

The training pipeline automatically applies degradations to portions of the training data:
- **JPEG Compression**: Random quality levels (10-100)
- **Gaussian Blur**: Random radius (0.5-5.0)
- **Color Quantization**: Reduce to 4-128 colors

## Results Tracking

### Weights & Biases
Results are automatically logged to W&B with:
- Real-time loss and metrics
- Confusion matrices
- GPU memory usage
- Model checkpoints

View at: https://wandb.ai/your-entity/your-project

### Local Files
Results are also saved locally:
- **Models**: `results/models/{model_name}_{timestamp}/`
- **Metrics**: `results/logs/{model_name}_finetune_log.jsonl`
- **Plots**: `results/plots/{model_name}/conf_mat.png`

## 📦 Dataset

- **Source**: [ISIC 2019 (Hugging Face)](https://huggingface.co/datasets/MKZuziak/ISIC_2019_224)
- **Classes**: Binary classification (filtered from 8 classes)
- **Size**: ~25,000 images at 224x224 resolution
- **Format**: Pre-processed and cached via HuggingFace datasets

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU training)
- 32GB RAM recommended
- 1+ NVIDIA GPU with 8GB+ VRAM

## Citation

Forthcoming

## License

MIT License - See LICENSE file for details