# TCGA Pipeline v2

A refactored pipeline for processing and analyzing TCGA whole slide images (WSI) with modern deep learning models.

## Features

- 🔬 **Tissue Segmentation**: Automated tissue detection using HEST segmentation models
- 🧬 **Feature Extraction**: Patch encoding with UNI2 and DINOv3 vision transformers
- 🎯 **Classification**: Linear probing for downstream prediction tasks
- 📊 **Evaluation**: Comprehensive metrics and visualization tools

## Project Structure

```
tcgapipeline/
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration and constants
│   │
│   ├── data/                      # Data loading and processing
│   │   ├── __init__.py
│   │   ├── datasets.py            # PyTorch Dataset classes
│   │   ├── data_utils.py          # Data utilities
│   │   └── datamodule.py          # Data splitting and loaders
│   │
│   ├── models/                    # Model definitions
│   │   ├── __init__.py
│   │   └── factory.py             # Model factory (UNI2, DINOv3)
│   │
│   ├── engines/                   # Training and processing engines
│   │   ├── __init__.py
│   │   ├── segmentation_engine.py # Tissue segmentation
│   │   ├── encoding_engine.py     # Feature extraction
│   │   └── linear_probe_engine.py # Linear classification
│   │
│   ├── evaluation/                # Evaluation and visualization
│   │   ├── __init__.py
│   │   ├── metrics.py             # Classification metrics
│   │   └── visualization.py       # Plotting utilities
│   │
│   ├── losses/                    # Loss functions
│   │   ├── __init__.py
│   │   └── classification.py      # Classification losses
│   │
│   ├── transformation/            # Image transformations
│   │   ├── __init__.py
│   │   └── transforms.py          # Preprocessing transforms
│   │
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── utils.py               # I/O utilities
│   │   ├── constants.py           # Constants
│   │   └── logging.py             # Logging setup
│   │
│   └── wrappers/                  # Model wrappers
│       ├── __init__.py
│       └── probe.py               # Linear probe wrapper
│
├── segment.py                     # Main script for segmentation
├── patch-encode.py                # Main script for encoding
├── classify.py                    # Main script for classification
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- timm
- transformers
- openslide-python
- scikit-learn
- matplotlib
- huggingface-hub

## Usage

### 1. Tissue Segmentation

Identify tissue regions in whole slide images:

```bash
python segment.py
```

Edit `segment.py` to configure datasets and parameters:

```python
datasets = ["lgg"]
config = SegmentationConfig(
    confidence_thresh=0.5,
    patch_len=512,
    level=0,
    batch_size=64,
    num_workers=16
)
```

### 2. Feature Extraction

Encode tissue patches using pre-trained models:

```bash
python patch-encode.py univ2 gbm 0
```

Arguments:
- `model`: Encoder name (`univ2` or `dinov3`)
- `dataset`: Dataset name (e.g., `gbm`, `lgg`)
- `level`: Pyramid level (typically `0`)

### 3. Classification

Train linear classifier on extracted features:

```bash
python classify.py univ2 subtype 0 luad lusc
```

Arguments:
- `encoder`: Encoder name (`univ2` or `dinov3`)
- `variable`: Variable to predict (`subtype` or gene name)
- `level`: Pyramid level
- `datasets`: One or more dataset names

## Configuration

The pipeline uses dataclass-based configuration in `src/config.py`:

### SegmentationConfig

```python
@dataclass
class SegmentationConfig:
    confidence_thresh: float = 0.5
    patch_len: int = 512
    level: int = 0
    batch_size: int = 32
    num_workers: int = 4
```

### EncodingConfig

```python
@dataclass
class EncodingConfig:
    model_name: str = "univ2"
    level: int = 0
    patch_len: int = 224
    batch_size: int = 32
    num_workers: int = 4
    threshold: float = 0.5
```

### ClassificationConfig

```python
@dataclass
class ClassificationConfig:
    num_epochs: int = 20
    batch_size: int = 64
    num_workers: int = 4
    lr_range: tuple = (1e-6, 1e2)
    num_lr_steps: int = 33
```

## Data Organization

The pipeline expects data organized as follows:

```
$TCGA_ROOT/
├── wsi-datasets/tcga/
│   └── {dataset}/svs/           # Raw WSI files
├── rpark23/
│   ├── outputs/
│   │   ├── hest/tcga/{dataset}/ # Segmentation results
│   │   ├── {encoder}/tcga/      # Encoded features
│   │   ├── models/              # Trained models
│   │   ├── output/              # Results
│   │   └── plots/               # Visualizations
│   └── clinical_data/tcga/      # Clinical metadata
```

Set the `TCGA_ROOT` environment variable or modify `ROOT_DIR` in `src/config.py`.

## Models

### Supported Encoders

1. **UNI2** - Universal Vision Transformer from MahmoodLab
   - 1.5B parameters
   - Pre-trained on diverse pathology images

2. **DINOv3** - Self-supervised Vision Transformer from Meta
   - ViT-Large architecture
   - Pre-trained with self-distillation

## Evaluation Metrics

The pipeline computes comprehensive classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Balanced Accuracy
- ROC Curve visualization

## Output

Results are saved in structured directories:

```
$TCGA_ROOT/rpark23/outputs/
├── models/
│   └── {encoder}_{var}_{level}_{datasets}.pth
├── output/
│   └── {encoder}_{var}_{level}_{datasets}.pkl
└── plots/
    └── {encoder}_{var}_{level}_{datasets}_roc.png
```

## Development

### Architecture

The refactored codebase follows a modular architecture inspired by modern deep learning frameworks:

1. **Config Layer**: Centralized configuration with dataclasses
2. **Data Layer**: Dataset and DataLoader management
3. **Model Layer**: Model factory pattern for easy extension
4. **Engine Layer**: Training and inference pipelines
5. **Evaluation Layer**: Metrics and visualization

### Adding New Models

To add a new encoder:

1. Add configuration to `src/config.py`
2. Implement loader in `src/models/factory.py`
3. Update `SUPPORTED_ENCODERS` list

```python
def load_my_model() -> Tuple[torch.nn.Module, Any]:
    model = ...
    transforms = ...
    return model, transforms
```

### Adding New Tasks

To add a new downstream task:

1. Create engine in `src/engines/`
2. Define loss in `src/losses/`
3. Create wrapper in `src/wrappers/`
4. Add metrics in `src/evaluation/`

## Citation

If you use this pipeline, please cite:

```bibtex
@software{tcga_pipeline_v2,
  title = {TCGA Pipeline v2},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/tcga-pipeline-v2}
}
```

## License

[Add your license here]

## Acknowledgments

- UNI2 model from [MahmoodLab](https://github.com/mahmoodlab)
- DINOv3 from [Meta AI](https://github.com/facebookresearch/dinov2)
- HEST segmentation from [MahmoodLab](https://github.com/mahmoodlab/hest)
- TCGA data from [NIH/NCI](https://www.cancer.gov/tcga)

