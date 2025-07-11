"""
Image transformation utilities for data augmentation and degradation.
"""

import io
import random
from PIL import Image
from torchvision import transforms

class JPEGCompressionTransform:
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        # Store original size
        original_size = img.size
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        img = Image.open(buffer)
        # Ensure size is maintained
        if img.size != original_size:
            img = img.resize(original_size, Image.LANCZOS)
        return img

class GaussianBlurTransform:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        # Store original size
        original_size = img.size
        if random.random() < self.p:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 2.0)
            img = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
        # Ensure size is maintained
        if img.size != original_size:
            img = img.resize(original_size, Image.LANCZOS)
        return img

class ColorQuantizationTransform:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        # Store original size
        original_size = img.size
        if random.random() < self.p:
            num_colors = random.randint(16, 64)
            img = img.quantize(colors=num_colors, method=Image.Quantize.MAXCOVERAGE).convert("RGB")
        # Ensure size is maintained
        if img.size != original_size:
            img = img.resize(original_size, Image.LANCZOS)
        return img 
