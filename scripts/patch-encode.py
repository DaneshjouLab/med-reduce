"""Encode WSI patches using pre-trained models."""
import sys

from src.config import EncodingConfig
from src.data import get_slides_loader
from src.transformation import quarter_resolution
from src.engines.encoding_engine import encode_slides


if __name__ == "__main__":
    # Parse command line arguments
    model = sys.argv[1]
    datasets = [sys.argv[2]]
    level = int(sys.argv[3])
    
    # Configuration
    config = EncodingConfig(
        model_name=model,
        level=level,
        patch_len=512,
        batch_size=64,
        num_workers=15,
        threshold=0.5
    )
    
    print(f"🧬 Encoding {', '.join(datasets).upper()} slides using {model.upper()}...")
    
    # Additional transforms
    patch_transforms = [quarter_resolution]
    
    # Load slides and encode
    slides_loader = get_slides_loader(datasets)
    encode_slides(model, slides_loader, config, patch_transforms)