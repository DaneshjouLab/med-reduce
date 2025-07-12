# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""Configuration constants for model training and evaluation."""

HF_MODELS = ["vit", "dinov2"]
SSL_MODEL = "simclr"
SIMCLR_BACKBONE = "resnet50"
NUM_CLASSES = 8

# Filter constants
FILTERED_CLASSES = ["0", "1"]  # Classes to use after filtering
NUM_FILTERED_CLASSES = len(FILTERED_CLASSES)  # Number of classes after filtering
