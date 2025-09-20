"""Image transformation utilities."""
import io
from typing import Optional
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

class JPEGCompressionTransform:
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

class GaussianBlurTransform:
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
            
        return img.quantize(colors=n_colors, method=Image.MEDIANCUT).convert("RGB")

def get_degradation_transforms():
    """Get default list of degradation transforms."""
    return [
        JPEGCompressionTransform(),
        GaussianBlurTransform(),
        ColorQuantizationTransform(),
    ]