# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

# src/utils/constants.py
# -*- coding: utf-8 -*-
"""Global constants and lightweight enums used across the training pipeline."""

# ---------------------------------------------------------------------------
# Model Families
# ---------------------------------------------------------------------------

# Hugging Face vision models supported by the unified factory
HF_MODELS = {"vit", "dinov2"}

# ---------------------------------------------------------------------------
# Dataset defaults
# ---------------------------------------------------------------------------

NUM_CLASSES = 1000  # update dynamically per dataset if needed
NUM_FILTERED_CLASSES = 8  # for ISIC filtered subset example


