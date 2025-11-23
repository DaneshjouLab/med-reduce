# This source file is part of the Daneshjou Lab projects
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

"""Image transformation utilities."""
# Standard library imports
import io
from typing import Optional, Tuple

# Third-party imports
import numpy as np  # pylint: disable=import-error
from PIL import Image, ImageFilter  # pylint: disable=import-error
from torchvision import transforms

class ResolutionReductionTransform:  # pylint: disable=too-few-public-methods
    """Reduce image resolution by factor or target resolution."""

    def __init__(
        self,
        reduction_factor: Optional[float] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        restore_original_size: bool = False,
    ):
        self.reduction_factor = reduction_factor
        self.target_resolution = target_resolution
        self.restore_original_size = restore_original_size

    def __call__(self, img: Image.Image) -> Image.Image:
        ow, oh = img.size

        if self.target_resolution is not None:
            nw, nh = self.target_resolution
        else:
            factor = (
                np.random.uniform(0.2, 0.8)
                if self.reduction_factor is None
                else self.reduction_factor
            )
            factor = max(0.1, min(1.0, factor))
            nw, nh = max(1, int(ow * factor)), max(1, int(oh * factor))

        # Downsample
        reduced = img.resize((nw, nh), Image.Resampling.LANCZOS)

        # Either return the reduced image as-is, or restore to original size
        if self.restore_original_size:
            return reduced.resize((ow, oh), Image.Resampling.LANCZOS)
        return reduced



class JPEGCompressionTransform:  # pylint: disable=too-few-public-methods
    """Apply JPEG compression to images."""

    def __init__(self, quality: Optional[int] = None):
        """
        Args:
            quality: JPEG quality (1-100). If None, random quality is used.
        """
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression."""
        if self.quality is None:
            quality = np.random.randint(10, 100)
        else:
            quality = self.quality

        # Save to bytes with JPEG compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        # Load back
        return Image.open(buffer)

class GaussianBlurTransform:  # pylint: disable=too-few-public-methods
    """Apply Gaussian blur to images."""

    def __init__(self, radius: Optional[float] = None):
        """
        Args:
            radius: Blur radius. If None, random radius is used.
        """
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian blur."""
        if self.radius is None:
            radius = np.random.uniform(0.5, 5.0)
        else:
            radius = self.radius

        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class ColorQuantizationTransform:  # pylint: disable=too-few-public-methods
    """Reduce color palette of images."""

    def __init__(self, n_colors: Optional[int] = None):
        """
        Args:
            n_colors: Number of colors. If None, random value is used.
        """
        self.n_colors = n_colors

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply color quantization."""
        if self.n_colors is None:
            n_colors = np.random.randint(4, 128)
        else:
            n_colors = self.n_colors

        return img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT).convert("RGB")

class SegmentationTransform:
            """Applies deterministic transforms (Resize, ToTensor) to both image and mask."""
            def __init__(self, target_size=256):
                self.val_test_tfs_img = transforms.Compose([
                    transforms.Resize((target_size, target_size)),
                    transforms.ToTensor(), # Image typically uses 3 channels
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.val_test_tfs_mask = transforms.Compose([
                    transforms.Resize((target_size, target_size), interpolation=Image.Resampling.NEAREST),
                    transforms.ToTensor(), # Mask typically uses 1 channel (or C channels for multi-class)
                ])

            def __call__(self, image, mask):
                # Normalization is only applied to the image, not the mask
                image = self.val_test_tfs_img(image)
                mask = self.val_test_tfs_mask(mask)
                # Ensure mask is integer-like if required by the loss function
                return image, mask

def get_degradation_transforms():
    """Get default list of degradation transforms."""
    return [
        ResolutionReductionTransform(),
    ]
