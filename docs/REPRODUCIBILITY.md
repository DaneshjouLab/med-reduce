# Reproducibility Guide

This describes the reproducibility features and best practices for the project.

## Overview

The codebase implements reproducibility mechanisms to ensure that experiments can be reliably reproduced:

1. **Random seed management** - Consistent seeding across all libraries
2. **Image ID traceability** - Clear mapping between embeddings and source images
3. **Split persistence** - Reusable train/val/test splits
4. **Configuration tracking** - Full experiment configuration logging
5. **Embedding caching** - Cached embeddings with metadata

## Random Seed Management

### Setting Seeds

Seeds are configured in your config YAML file:

```yaml
# Global settings
seed: 42

train:
  seed: 42  # Used for split generation, CV folds, etc.
  deterministic: false  # Set to true for full determinism (slower)
```

### Deterministic Mode

For maximum reproducibility, enable deterministic mode:

```yaml
train:
  deterministic: true
```

**Note:** Deterministic mode is slower but ensures bit-exact reproducibility across runs.

### What Gets Seeded

The seed propagates to:
- Python's `random` module
- NumPy's random number generator
- PyTorch (CPU and CUDA)
- DataLoader workers (each worker gets `base_seed + worker_id`)
- Train/val/test split generation
- Cross-validation fold creation
- Data augmentation (if deterministic mode enabled)

### Seed Logging

Seeds are automatically logged in multiple locations:

1. **Console output** - Printed at training start
2. **`resolved_config.yaml`** - Full config including seed
3. **`seed_log.json`** - Detailed seed usage by component
4. **`reproducibility_settings.json`** - System reproducibility settings
5. **Checkpoint files** - Seed saved with model weights

Example seed log structure:
```json
{
  "base_seed": 42,
  "components": {
    "main": {
      "seed": 42,
      "metadata": {"deterministic": false}
    },
    "split_manager": {
      "seed": 42,
      "metadata": {"dataset": "isic2017"}
    },
    "dataloader": {
      "seed": 42,
      "metadata": {"num_workers": 4}
    }
  }
}
```

## Image ID Traceability

### Image IDs in Datasets

All datasets include `image_id` in returned samples:

```python
sample = dataset[0]
# sample = {
#     "pixel_values": PIL.Image,
#     "label": int,
#     "image_id": str  # e.g., "ISIC_0000123"
# }
```

### Embedding Cache Traceability

When embeddings are cached, image IDs are stored alongside:

```
cache/embeddings/{dataset_hash}/
├── embeddings.pt         # [N, D] embedding tensor
├── labels.pt             # [N] label tensor
├── image_ids.json        # [N] list of image IDs (CRITICAL for traceability)
├── metadata.json         # Cache metadata (model, resolution, etc.)
└── cache.pt              # Optional single-file format
```

The `image_ids.json` file maintains the exact mapping:
```json
[
  "ISIC_0000001",
  "ISIC_0000002",
  "ISIC_0000003",
  ...
]
```

Index `i` in `embeddings.pt` corresponds to `image_ids.json[i]`.

### Teacher Embedding Cache

Teacher embeddings maintain the same traceability:

```
cache/teacher_embeddings/{dataset_hash}/
├── embeddings.pt
├── labels.pt
├── image_ids.json        # Image ID mapping
├── sample_ids.pt         # Legacy format (indices)
└── metadata.json
```

### Verifying Traceability

To verify image ID traceability in your data:

```python
from src.utils.teacher_cache import TeacherEmbeddingLookup

# Load cached embeddings with ID lookup
lookup = TeacherEmbeddingLookup(
    cache_dir="./cache/teacher_embeddings",
    dataset_name="isic2017",
    split="train"
)

# Get embedding for specific image
embedding = lookup.get_by_image_id("ISIC_0000123")

# Get all image IDs
image_ids = lookup.get_all_image_ids()
```

### Fallback Behavior

If `image_id` is not present in the dataset:
- A warning is logged
- Sequential IDs are generated: `sample_0`, `sample_1`, etc.
- These are still tracked in `image_ids.json`

**Best practice:** Always ensure your datasets include an `image_id` field.

## Split Persistence

### Automatic Split Storage

Train/val/test splits are automatically saved and reused:

```
splits/{dataset_name}/
├── train_indices.npy     # Training sample indices
├── test_indices.npy      # Test sample indices
├── val_indices.npy       # Validation indices (if used)
├── cv_folds.json         # Cross-validation fold definitions
└── metadata.json         # Split metadata
```

### Split Metadata

The metadata file tracks split configuration:

```json
{
  "dataset_name": "isic2017",
  "dataset_size": 2000,
  "seed": 42,
  "use_val_split": false,
  "train_ratio": 0.8,
  "stratified": false,
  "split_sizes": {
    "train": 1600,
    "test": 400
  }
}
```

### Configuring Split Directory

Set via environment variable or config:

```bash
export COMPRESSED_PERCEPTION_SPLIT_DIR=/path/to/splits
```

Or in config:
```yaml
split_dir: /path/to/splits
```

### Cross-Validation Folds

CV folds are saved in `cv_folds.json`:

```json
{
  "seed": 42,
  "k_folds": 5,
  "folds": [
    {
      "fold": 0,
      "train_indices": [...],
      "val_indices": [...]
    },
    ...
  ]
}
```

## Configuration Tracking

### Resolved Configuration

Every run saves the complete resolved configuration:

```
runs/{experiment_name}/
├── resolved_config.yaml           # Full Hydra config
├── reproducibility_settings.json  # System settings
├── seed_log.json                  # Seed usage log
└── final_metrics.json             # Training results
```

### Hyperparameter Search Tracking

When hyperparameter search is enabled:

```
runs/{experiment_name}/hyperparam_search/
├── best_hyperparameters.json     # Best found hyperparameters
├── search_results.json            # All configurations tested
└── fold_{i}_results.json          # Per-fold results
```

## Checkpoint Metadata

Model checkpoints now include reproducibility information:

```python
checkpoint = {
    "model_state_dict": state_dict,
    "fold": fold_number,
    "metric": best_metric,
    "model_config": model_config,
    "cfg": full_config,
    "seed": 42,  # NEW: Random seed used
    "reproducibility_info": {  # NEW: System info
        "torch_version": "2.0.0",
        "cuda_available": true,
        "cudnn_deterministic": false,
        ...
    },
    "optimizer_state_dict": optimizer.state_dict(),
}
```

### Loading Checkpoints

When you load a checkpoint, seed information is automatically logged:

```python
from src.utils.checkpoint_utils import load_checkpoint

checkpoint = load_checkpoint("path/to/checkpoint.pt")
# Logs: "Checkpoint was trained with seed: 42"
```

## Best Practices

### 1. Use Consistent Seeds

```yaml
seed: 42  # Global seed

train:
  seed: 42  # Same seed for training components
```

### 2. Enable Deterministic Mode for Critical Experiments

```yaml
train:
  deterministic: true
```

### 3. Always Use Split Manager

Don't create random splits manually:

```python
# ✅ GOOD: Use SplitManager
from src.utils.split_manager import SplitManager

split_mgr = SplitManager(
    split_dir="./splits",
    dataset_name="isic2017",
    seed=42
)
train_idx, test_idx = split_mgr.get_or_create_split(
    dataset_size=len(dataset),
    train_ratio=0.8
)

# ❌ BAD: Manual random split
indices = np.random.permutation(len(dataset))
```

### 4. Use DataLoader with Worker Seeding

```python
from src.utils.reproducibility import get_worker_init_fn

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=get_worker_init_fn(seed=42)  # ✅ Ensures reproducible workers
)
```

### 5. Verify Image ID Traceability

Check that your datasets provide image IDs:

```python
sample = dataset[0]
assert "image_id" in sample, "Dataset must provide image_id for traceability"
```

### 6. Save Seed Logs

At the end of training:

```python
from src.utils.reproducibility import SeedTracker

seed_tracker = SeedTracker(base_seed=42)
seed_tracker.log_seed("component_name", seed_value)
seed_tracker.save(output_dir / "seed_log.json")
```

## Reproducibility Checklist

Before running experiments, verify:

- [ ] Seed is set in config file
- [ ] Split directory is configured (env var or config)
- [ ] Cache directories are configured
- [ ] DataLoader uses `worker_init_fn`
- [ ] Dataset includes `image_id` field
- [ ] Deterministic mode enabled (if needed)
- [ ] Config is saved with each run
- [ ] Checkpoints include seed metadata

## Troubleshooting

### Non-Reproducible Results

**Issue:** Results vary across runs despite same seed

**Solutions:**
1. Enable deterministic mode: `train.deterministic: true`
2. Set `PYTHONHASHSEED`: `export PYTHONHASHSEED=42`
3. Disable CUDA benchmark: Already done in deterministic mode
4. Check for non-deterministic operations in your code

### Missing Image IDs

**Issue:** Warning about missing image IDs

**Solutions:**
1. Ensure your dataset includes an `image_id` column
2. Check CSV/dataset format
3. Verify `local_image_id_column` config matches your data

### Different Results on Different Hardware

**Issue:** Results differ between GPU types or CPU vs GPU

**Solutions:**
1. This is expected with `deterministic: false`
2. Enable `deterministic: true` for hardware-independent results
3. Note: Deterministic mode may be slower

## Advanced: Custom Reproducibility

For custom components, use the reproducibility utilities:

```python
from src.utils.reproducibility import seed_everything, SeedTracker

# Seed everything
seed_everything(seed=42, deterministic=True)

# Track custom seeds
tracker = SeedTracker(base_seed=42)
tracker.log_seed("my_component", seed=42, metadata={"info": "value"})

# Get worker init function
from src.utils.reproducibility import get_worker_init_fn
worker_fn = get_worker_init_fn(base_seed=42)
```

## References

- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- NumPy Random Seeds: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
- Deterministic Algorithms: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
