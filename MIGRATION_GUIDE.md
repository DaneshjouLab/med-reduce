# Migration Guide: TCGA Pipeline v1 → v2

This guide helps you migrate from the old flat structure to the new modular architecture.

## Overview of Changes

### Before (v1)
```
tcgapipeline/
├── linear_classifier.py
├── patch_encoding.py
├── segmentation.py
├── utils.py
└── wsi.py
```

### After (v2)
```
tcgapipeline/
├── src/
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── engines/
│   ├── evaluation/
│   ├── losses/
│   ├── transformation/
│   ├── utils/
│   └── wrappers/
└── README.md
```

## Import Changes

### Old Imports → New Imports

#### Linear Classifier

**Old:**
```python
from tcgapipeline.linear_classifier import (
    LogisticRegressionModel,
    train_logistic_regression,
    evaluate_model,
    split_dataset_by_patient
)
```

**New:**
```python
from tcgapipeline.src.wrappers import LogisticRegressionModel
from tcgapipeline.src.engines import train_logistic_regression
from tcgapipeline.src.evaluation import evaluate_model
from tcgapipeline.src.data.datamodule import split_dataset_by_patient
```

#### Patch Encoding

**Old:**
```python
from tcgapipeline.patch_encoding import (
    load_univ2,
    load_dinov3,
    get_patch_encoder,
    patch_slides
)
```

**New:**
```python
from tcgapipeline.src.models import (
    load_univ2,
    load_dinov3,
    get_patch_encoder
)
from tcgapipeline.src.engines import encode_slides
```

#### Segmentation

**Old:**
```python
from tcgapipeline.segmentation import HESTSegmenter, segment_slides
```

**New:**
```python
from tcgapipeline.src.engines import HESTSegmenter, segment_slides
```

#### WSI and Data

**Old:**
```python
from tcgapipeline.wsi import (
    TCGASlides,
    TCGAPatches,
    TCGAPrediction,
    get_slides_loader,
    get_patient_id
)
```

**New:**
```python
from tcgapipeline.src.data import (
    TCGASlides,
    TCGAPatches,
    TCGAPrediction,
    get_slides_loader,
    get_patient_id
)
```

#### Utils

**Old:**
```python
from tcgapipeline.utils import (
    save_pickle,
    load_pickle,
    save_json,
    load_json,
    quarter_resolution
)
```

**New:**
```python
from tcgapipeline.src.utils import (
    save_pickle,
    load_pickle,
    save_json,
    load_json
)
from tcgapipeline.src.transformation import quarter_resolution
```

## Function Signature Changes

### 1. segment_slides()

**Old:**
```python
segment_slides(slides_loader, segmenter, level, batch_size, num_workers)
```

**New:**
```python
from tcgapipeline.src.config import SegmentationConfig

config = SegmentationConfig(
    level=level,
    batch_size=batch_size,
    num_workers=num_workers,
    patch_len=512,
    confidence_thresh=0.5
)
segment_slides(slides_loader, segmenter, config)
```

### 2. patch_slides() → encode_slides()

**Old:**
```python
patch_slides(
    model_name, slides_loader, level, transforms,
    THRESHOLD, patch_len, batch_size, num_workers
)
```

**New:**
```python
from tcgapipeline.src.config import EncodingConfig

config = EncodingConfig(
    model_name=model_name,
    level=level,
    patch_len=patch_len,
    batch_size=batch_size,
    num_workers=num_workers,
    threshold=THRESHOLD
)
encode_slides(model_name, slides_loader, config, transforms)
```

### 3. split_dataset_by_patient()

**Old:**
```python
train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = \
    split_dataset_by_patient(dataset, batch_size, num_workers)
```

**New:**
```python
from tcgapipeline.src.config import DataSplitConfig

split_config = DataSplitConfig(
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    seed=42
)
train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = \
    split_dataset_by_patient(
        dataset, split_config, batch_size, num_workers
    )
```

## Configuration Management

### Using Dataclass Configs

**Before:** Parameters passed directly

**After:** Use configuration dataclasses

```python
from tcgapipeline.src.config import (
    SegmentationConfig,
    EncodingConfig,
    ClassificationConfig,
    DataSplitConfig
)

# Example: Classification config
config = ClassificationConfig(
    num_epochs=20,
    batch_size=64,
    lr_range=(1e-6, 1e2),
    num_lr_steps=33
)
```

### Environment Variables

Set the `TCGA_ROOT` environment variable:

```bash
export TCGA_ROOT="/oak/stanford/groups/roxanad/"
```

Or modify `ROOT_DIR` in `src/config.py`.

## New Features in v2

### 1. Comprehensive Evaluation

```python
from tcgapipeline.src.evaluation import evaluate_model, plot_roc_curve

metrics = evaluate_model(model, test_loader)
plot_roc_curve(
    metrics["fpr"],
    metrics["tpr"],
    metrics["auc"],
    save_path="roc_curve.png"
)
```

### 2. Structured Visualization

```python
from tcgapipeline.src.evaluation.visualization import (
    plot_training_history,
    plot_confusion_matrix
)

plot_training_history(lr_list, losses, save_path="history.png")
plot_confusion_matrix(y_true, y_pred, save_path="confusion.png")
```

### 3. Modular Loss Functions

```python
from tcgapipeline.src.losses import get_classification_loss

criterion = get_classification_loss("bce")
```

### 4. Logging Utilities

```python
from tcgapipeline.src.utils.logging import setup_logger

logger = setup_logger("my_pipeline", log_file="pipeline.log")
logger.info("Processing started")
```

## Backward Compatibility

The old module files are preserved but deprecated. For full compatibility with v2 features, update your imports as shown above.

## Step-by-Step Migration

### For Segmentation Scripts

1. Update imports:
   ```python
   from tcgapipeline.src.config import SegmentationConfig
   from tcgapipeline.src.engines import HESTSegmenter, segment_slides
   ```

2. Create config:
   ```python
   config = SegmentationConfig(
       level=0, batch_size=64, num_workers=16
   )
   ```

3. Update function call:
   ```python
   segment_slides(slides_loader, segmenter, config)
   ```

### For Encoding Scripts

1. Update imports:
   ```python
   from tcgapipeline.src.config import EncodingConfig
   from tcgapipeline.src.engines import encode_slides
   ```

2. Create config:
   ```python
   config = EncodingConfig(
       model_name="univ2", level=0, threshold=0.5
   )
   ```

3. Update function call:
   ```python
   encode_slides(model_name, slides_loader, config, patch_transforms)
   ```

### For Classification Scripts

1. Update all imports following the table above

2. Create configs:
   ```python
   class_config = ClassificationConfig(...)
   split_config = DataSplitConfig(...)
   ```

3. Update function calls with new signatures

## Troubleshooting

### ImportError: No module named 'tcgapipeline.src'

Make sure you're running from the project root and the package is installed:

```bash
cd /path/to/tcga-pipeline-v2
pip install -e .
```

### Configuration Errors

Check that all required config parameters are provided:

```python
# This will raise an error if invalid
config = EncodingConfig(model_name="unsupported")  # ValueError

# Use supported models
config = EncodingConfig(model_name="univ2")  # ✓
```

### Module Resolution Issues

If imports fail, add the project root to PYTHONPATH:

```bash
export PYTHONPATH="/path/to/tcga-pipeline-v2:$PYTHONPATH"
```

## Benefits of v2

1. **Modularity**: Clear separation of concerns
2. **Maintainability**: Easier to locate and update code
3. **Extensibility**: Simple to add new models/tasks
4. **Type Safety**: Dataclass configs catch errors early
5. **Documentation**: Better organized with clear structure
6. **Testing**: Modular design enables unit testing
7. **Reusability**: Components can be imported independently

## Questions?

For issues or questions about migration, please open an issue on GitHub or contact the maintainers.

