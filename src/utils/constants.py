# src/utils/constants.py
# -*- coding: utf-8 -*-
"""Global constants and lightweight enums used across the training pipeline."""

# ---------------------------------------------------------------------------
# Model Families
# ---------------------------------------------------------------------------

# Hugging Face vision models supported by the unified factory
HF_MODELS = {"vit", "dinov2", "dinov3"}

# ---------------------------------------------------------------------------
# Dataset defaults
# ---------------------------------------------------------------------------

NUM_CLASSES = 1000  # update dynamically per dataset if needed
NUM_FILTERED_CLASSES = 8  # for ISIC filtered subset example
DEFAULT_IMAGE_SIZE = 224  # default input resolution


