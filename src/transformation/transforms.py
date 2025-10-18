"""Image transformation utilities."""
import io
from typing import Optional
import numpy as np
from PIL import Image, ImageFilter

from typing import Optional, Tuple
import numpy as np
from PIL import Image

from typing import Optional, Tuple
import numpy as np
from PIL import Image

class ResolutionReductionTransform:
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
