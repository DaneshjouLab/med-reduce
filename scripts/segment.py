"""Segment tissue in whole slide images."""
print("⏳ Loading dependencies ...")

from src.config import SegmentationConfig
from src.data import get_slides_loader
from src.engines.segmentation_engine import HESTSegmenter, segment_slides

# Configuration
datasets = ["lgg"]
config = SegmentationConfig(
    confidence_thresh=0.5,
    patch_len=512,
    level=0,
    batch_size=64,
    num_workers=16
)

print(f"🔬 Segmenting tissue in {', '.join(datasets).upper()} slides...")

# Load segmenter and slides
segmenter = HESTSegmenter(confidence_thresh=config.confidence_thresh)
slides_loader = get_slides_loader(datasets)

# Run segmentation
segment_slides(slides_loader, segmenter, config)
