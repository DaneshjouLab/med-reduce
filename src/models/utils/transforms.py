# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""
Image transformation utilities for data augmentation and degradation.
"""

import io
import random
from PIL import Image
from torchvision import transforms

# Compatibility for LANCZOS resampling
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS # pylint: disable=no-member


class JPEGCompressionTransform:
    """
    Apply JPEG compression to an image to simulate lossy compression artifacts.
    """
    def __init__(self, quality=75):
        """
        Apply JPEG compression to an image.

        Args:
            quality (int): Compression quality (1-100, higher is better quality).
        """
        self.quality = quality

    def __call__(self, img):
        """
        Apply JPEG compression to the input image.

        Args:
            img (PIL.Image or Tensor): Input image.

        Returns:
            PIL.Image: Compressed image.
        """
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        # Store original size
        original_size = img.size

        # Apply JPEG compression
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        img = Image.open(buffer)

        # Ensure size is maintained
        if img.size != original_size:
            img = img.resize(original_size, LANCZOS)

        return img

    def get_quality(self):
        """
        Return the JPEG compression quality setting.
        """
        return self.quality


class GaussianBlurTransform:
    """Apply Gaussian blur to an image with a given probability.
    """
    def __init__(self, p=1):
        """
        Apply Gaussian blur to an image with a given probability.

        Args:
            p (float): Probability of applying the blur (0 to 1).
        """
        self.p = p

    def __call__(self, img):
        """
        Apply Gaussian blur to the input image.

        Args:
            img (PIL.Image or Tensor): Input image.

        Returns:
            PIL.Image: Blurred image.
        """
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        # Store original size
        original_size = img.size

        # Apply Gaussian blur with probability p
        if random.random() < self.p:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 2.0)
            img = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)

        # Ensure size is maintained
        if img.size != original_size:
            img = img.resize(original_size, LANCZOS)

        return img

    def get_probability(self):
        """
        Return the probability of applying Gaussian blur.
        """
        return self.p


class ColorQuantizationTransform:
    """
    Apply color quantization to an image with a given probability.
    """
    def __init__(self, p=1):
        """
        Args:
            p (float): Probability of applying the quantization (0 to 1).
        """
        self.p = p

    def __call__(self, img):
        """
        Apply color quantization to the input image.

        Args:
            img (PIL.Image or Tensor): Input image.

        Returns:
            PIL.Image: Quantized image.
        """
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        # Store original size
        original_size = img.size

        # Apply color quantization with probability p
        if random.random() < self.p:
            num_colors = random.randint(16, 64)
            img = img.quantize(colors=num_colors, method=Image.Quantize.MAXCOVERAGE).convert("RGB")

        # Ensure size is maintained
        if img.size != original_size:
            img = img.resize(original_size, LANCZOS)

        return img

    def get_probability(self):
        """
        Return the probability of applying color quantization.
        """
        return self.p
