"""Image transformation utilities."""
import io
from typing import Optional
import numpy as np
from PIL import Image, ImageFilter

class ResolutionReductionTransform:  # pylint: disable=too-few-public-methods
    """Reduce spatial resolution of images."""

    def __init__(self, reduction_factor: Optional[float] = None):
        """
        Args:
            reduction_factor: Factor to reduce resolution by (0.1-1.0).
                            For example, 0.5 reduces to half resolution.
                            If None, random factor is used.
        """
        self.reduction_factor = reduction_factor

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply resolution reduction."""
        if self.reduction_factor is None:
            # Random reduction factor between 0.2 and 0.8
            reduction_factor = np.random.uniform(0.2, 0.8)
        else:
            reduction_factor = self.reduction_factor

        # Clamp reduction factor to valid range
        reduction_factor = max(0.1, min(1.0, reduction_factor))

        # Calculate new size
        original_width, original_height = img.size
        new_width = int(original_width * reduction_factor)
        new_height = int(original_height * reduction_factor)

        # Ensure minimum size of 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Downsample and then upsample back to original size
        downsampled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return downsampled.resize((original_width, original_height), Image.Resampling.LANCZOS)

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

class ColorQuantizationTransform:
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


def get_degradation_transforms():
    """Get default list of degradation transforms."""
    return [
        ResolutionReductionTransform(),
    ]
