"""
Simple example demonstrating the TCGA pipeline v2 API.

This script shows how to:
1. Segment tissue in WSI
2. Encode patches with a pre-trained model
3. Train a linear classifier
4. Evaluate the model
"""

from tcgapipeline.src.config import (
    SegmentationConfig,
    EncodingConfig,
    ClassificationConfig,
    DataSplitConfig,
)
from tcgapipeline.src.data import get_slides_loader, TCGAPrediction
from tcgapipeline.src.data.datamodule import split_dataset_by_patient
from tcgapipeline.src.engines import (
    HESTSegmenter,
    segment_slides,
    encode_slides,
    train_logistic_regression,
)
from tcgapipeline.src.evaluation import evaluate_model, plot_roc_curve
from tcgapipeline.src.transformation import quarter_resolution


def step1_segment(datasets):
    """Step 1: Segment tissue in whole slide images."""
    print("\n" + "="*60)
    print("STEP 1: TISSUE SEGMENTATION")
    print("="*60)
    
    config = SegmentationConfig(
        confidence_thresh=0.5,
        patch_len=512,
        level=0,
        batch_size=64,
        num_workers=16
    )
    
    segmenter = HESTSegmenter(confidence_thresh=config.confidence_thresh)
    slides_loader = get_slides_loader(datasets)
    segment_slides(slides_loader, segmenter, config)
    
    print("✅ Segmentation complete!")


def step2_encode(model_name, datasets):
    """Step 2: Encode patches using pre-trained model."""
    print("\n" + "="*60)
    print(f"STEP 2: PATCH ENCODING WITH {model_name.upper()}")
    print("="*60)
    
    config = EncodingConfig(
        model_name=model_name,
        level=0,
        patch_len=512,
        batch_size=64,
        num_workers=15,
        threshold=0.5
    )
    
    patch_transforms = [quarter_resolution]
    slides_loader = get_slides_loader(datasets)
    encode_slides(model_name, slides_loader, config, patch_transforms)
    
    print("✅ Encoding complete!")


def step3_classify(encoder_name, var, level, datasets):
    """Step 3: Train and evaluate linear classifier."""
    print("\n" + "="*60)
    print(f"STEP 3: CLASSIFICATION - {var.upper()}")
    print("="*60)
    
    # Configuration
    class_config = ClassificationConfig(
        num_epochs=20,
        batch_size=16,
        num_workers=16,
        lr_range=(1e-6, 1e2),
        num_lr_steps=33
    )
    
    split_config = DataSplitConfig(
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=42
    )
    
    # Load data
    print("Loading data...")
    dataset = TCGAPrediction(encoder_name, level, datasets, var)
    
    # Split data
    print("Splitting dataset...")
    train_loader, val_loader, test_loader, _, _, _ = split_dataset_by_patient(
        dataset, split_config,
        class_config.batch_size,
        class_config.num_workers
    )
    
    # Train model
    print("Training model...")
    import numpy as np
    lr_list = np.logspace(
        np.log10(class_config.lr_range[0]),
        np.log10(class_config.lr_range[1]),
        num=class_config.num_lr_steps
    )
    model, losses = train_logistic_regression(train_loader, val_loader, lr_list)
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader)
    
    # Visualize
    plot_roc_curve(
        metrics["fpr"],
        metrics["tpr"],
        metrics["auc"],
        save_path=f"{encoder_name}_{var}_roc.png",
        show=False
    )
    
    print("✅ Classification complete!")
    print(f"\n📊 Results:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  AUC: {metrics['auc']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    
    return model, metrics


def main():
    """Run the complete pipeline."""
    print("\n" + "="*60)
    print("TCGA PIPELINE V2 - EXAMPLE")
    print("="*60)
    
    # Configuration
    datasets = ["lgg"]
    model_name = "univ2"
    var = "subtype"
    level = 0
    
    # Run pipeline
    # step1_segment(datasets)
    # step2_encode(model_name, datasets)
    # model, metrics = step3_classify(model_name, var, level, datasets)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()

