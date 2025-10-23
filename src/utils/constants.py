"""Constants for TCGA pipeline."""
import os

# Root directory
ROOT_DIR = os.getenv("TCGA_ROOT", "/oak/stanford/groups/roxanad/")

# Data directories
OUTPUTS_DIR = os.path.join(ROOT_DIR, "rpark23/outputs/")
CLINICAL_DIR = os.path.join(ROOT_DIR, "rpark23/clinical_data/tcga/")
WSI_DATASETS_DIR = os.path.join(ROOT_DIR, "wsi-datasets/tcga/")
CACHE_DIR = os.path.join(ROOT_DIR, "rpark23/cache/hub/")

