# TCGA Pipeline Scripts

This directory contains the main executable scripts for the TCGA pipeline.

## Scripts

### 1. `segment.py`
Segment tissue regions in whole slide images.

**Usage:**
```bash
cd tcgapipeline
python scripts/segment.py
```

**Configuration:**
Edit the script to configure datasets and parameters:
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

### 2. `patch-encode.py`
Encode tissue patches using pre-trained models (UNI2 or DINOv3).

**Usage:**
```bash
cd tcgapipeline
python scripts/patch-encode.py <model> <dataset> <level>
```

**Arguments:**
- `model`: Model name (`univ2` or `dinov3`)
- `dataset`: Dataset name (e.g., `gbm`, `lgg`)
- `level`: Pyramid level (typically `0`)

**Example:**
```bash
python scripts/patch-encode.py univ2 gbm 0
```

### 3. `classify.py`
Train and evaluate linear classifier on extracted features.

**Usage:**
```bash
cd tcgapipeline
python scripts/classify.py <encoder> <variable> <level> <dataset1> [dataset2 ...]
```

**Arguments:**
- `encoder`: Encoder name (`univ2` or `dinov3`)
- `variable`: Variable to predict (`subtype` or gene name)
- `level`: Pyramid level
- `datasets`: One or more dataset names

**Example:**
```bash
python scripts/classify.py univ2 subtype 0 luad lusc
```

## Running from Project Root

You can also run these scripts from the project root using the wrapper scripts:

```bash
# From project root
./run_segment.sh
./run_encode.sh univ2 gbm 0
./run_classify.sh univ2 subtype 0 luad lusc
```

## Output Locations

Results are saved to `$TCGA_ROOT/rpark23/outputs/`:
- **Segmentation**: `hest/tcga/{dataset}/level_{level}/`
- **Encoding**: `{encoder}/tcga/{dataset}/level_{level}/`
- **Classification**: `models/` and `output/`
- **Plots**: `plots/`

## Modifying Scripts

All scripts use configuration dataclasses from `src/config.py`. To modify behavior:

1. **For one-off changes**: Edit the script directly
2. **For permanent changes**: Update default values in `src/config.py`
3. **For testing**: Create a copy in `examples/`

## Notes

- Scripts must be run from within the `tcgapipeline` directory
- Make sure `TCGA_ROOT` environment variable is set or modify `src/config.py`
- All dependencies must be installed: `pip install -r requirements.txt`

