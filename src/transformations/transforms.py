"""Image transformation utilities."""
import io
from typing import Optional, Tuple

# Third-party imports
import numpy as np  # pylint: disable=import-error
from PIL import Image, ImageFilter  # pylint: disable=import-error
from torchvision import transforms

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

class FeatureDetectionTransform:
    """
    Applies deterministic transforms (Resize, ToTensor, Normalize) to both image and superpixel mask.

    For feature detection, we need to:
    1. Resize image and superpixel mask together
    2. Normalize image (but not superpixel mask)
    3. Keep superpixel mask as integer IDs (use nearest neighbor interpolation)
    """
    def __init__(self, target_size=256):
        self.target_size = target_size
        self.img_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # For superpixel mask, use nearest neighbor to preserve integer IDs
        self.mask_resize = transforms.Resize((target_size, target_size), interpolation=Image.Resampling.NEAREST)

    def __call__(self, image, superpixel_mask):
        import torch

        # Transform image
        image = self.img_transform(image)

        # Transform superpixel mask (keep as PIL for resize, then convert to tensor)
        if isinstance(superpixel_mask, np.ndarray):
            # Convert numpy array to PIL Image for resizing
            superpixel_mask = Image.fromarray(superpixel_mask.astype(np.int32), mode='I')

        superpixel_mask = self.mask_resize(superpixel_mask)

        # Convert to tensor (keep as long integers for indexing)
        superpixel_mask = torch.from_numpy(np.array(superpixel_mask)).long()

        return image, superpixel_mask


def get_degradation_transforms():
    """Get default list of degradation transforms."""
    return [
        ResolutionReductionTransform(),
    ]
