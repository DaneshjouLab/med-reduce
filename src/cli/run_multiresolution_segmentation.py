#!/usr/bin/env python3
"""
Multi-resolution segmentation experiment runner.

This script orchestrates segmentation training across multiple resolutions,
following the paper's Train@R, Test@R protocol:

1. Hyperparameter tuning at highest resolution (e.g., 512px)
2. Final evaluation at all resolutions using tuned hyperparameters

Usage:
    # Step 1: Tune hyperparameters at highest resolution
    python -m src.cli.run_multiresolution_segmentation \
        --domain dermatology \
        --model dinov3 \
        --tune-hyperparams

    # Step 2: Evaluate at multiple resolutions with tuned params
    python -m src.cli.run_multiresolution_segmentation \
        --domain dermatology \
        --model dinov3 \
        --resolutions 512 256 128 64

    # Or do both in one run:
    python -m src.cli.run_multiresolution_segmentation \
        --domain dermatology \
        --model dinov3 \
        --tune-hyperparams \
        --resolutions 512 256 128 64
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
from omegaconf import OmegaConf


# Model configurations
MODELS = ["dinov3"]
MODEL_CONFIGS = {
    "dinov3": {
        "model.name": "dinov3_segmentation",
        "model.model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "model.type": "dinov3_segmentation",
    }
}

# Resolution configuration per domain
DOMAIN_CONFIG = {
    "dermatology": {
        "highest_resolution": 512,
        "default_resolutions": [512, 256, 128, 64],
        "config": "config_segmentation",
        "image_dir": "/scratch/groups/roxanad/datasets/isic/challenges/2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data",
        "mask_dir": "/scratch/groups/roxanad/datasets/isic/challenges/2017/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth",
    },
}


def run_hyperparameter_tuning(
    domain: str,
    model_key: str = "dinov3",
):
    """
    Run hyperparameter tuning at the highest resolution for a domain.

    This uses 5-fold CV on the 80% training set to find the best hyperparameters.
    The best params are saved to hyperparam_search/best_hyperparameters.json.

    Args:
        domain: Domain name (dermatology, radiology, pathology)
        model_key: Model to use (currently only dinov3)
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Choose from {list(DOMAIN_CONFIG.keys())}")

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")

    domain_cfg = DOMAIN_CONFIG[domain]
    highest_res = domain_cfg["highest_resolution"]
    config_name = domain_cfg.get("config", "config_segmentation")
    image_dir = domain_cfg.get("image_dir")
    mask_dir = domain_cfg.get("mask_dir")

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING: {domain.upper()} SEGMENTATION at {highest_res}px with {model_key.upper()}")
    print(f"{'='*80}\n")
    print(f"Using 5-fold CV on 80% training set")
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}\n")

    # Build command with model overrides
    cmd = [
        "python", "-m", "src.cli.train",
        f"--config-name={config_name}",
        f"data.image_size={highest_res}",
        "train.hyperparam_search.enabled=true",
        f"logging.run_name={model_key}_segmentation_{domain}_{highest_res}px_search",
    ]

    # Add dataset paths
    if image_dir:
        cmd.append(f"datamodule.image_dir={image_dir}")
    if mask_dir:
        cmd.append(f"datamodule.mask_dir={mask_dir}")

    # Add model configuration overrides
    for key, value in MODEL_CONFIGS[model_key].items():
        cmd.append(f"{key}={value}")

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Hyperparameter tuning failed for {domain} with {model_key}")
        sys.exit(1)

    print(f"\n✅ Hyperparameter tuning completed for {domain} with {model_key}")
    print(f"   Best params saved to: runs/.../hyperparam_search/best_hyperparameters.json")
    print(f"   Note the path above and use it with --hyperparam-file for final evaluation")


def run_final_segmentation(
    domain: str,
    resolutions: List[int],
    model_key: str = "dinov3",
    hyperparam_file: str = None,
):
    """
    Run final segmentation evaluation at multiple resolutions with tuned hyperparameters.

    This trains ONE model on full 80% training set and evaluates on held-out 20% test set
    for each resolution (following the paper's Train@R, Test@R protocol).

    Args:
        domain: Domain name
        resolutions: List of resolutions to evaluate
        model_key: Model to use (currently only dinov3)
        hyperparam_file: Path to hyperparameter file (if None, uses most recent)
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}")

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")

    domain_cfg = DOMAIN_CONFIG[domain]
    config_name = domain_cfg.get("config", "config_segmentation")
    image_dir = domain_cfg.get("image_dir")
    mask_dir = domain_cfg.get("mask_dir")

    # Find hyperparameter file if not specified
    if hyperparam_file is None:
        project_root = Path(__file__).parent.parent.parent
        runs_dir = project_root / "runs"

        # Find most recent hyperparam search results
        search_dirs = sorted(runs_dir.glob("**/hyperparam_search"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not search_dirs:
            print(f"⚠️  No hyperparameter search results found in {runs_dir}")
            print(f"   Run hyperparameter tuning first with --tune-hyperparams")
            sys.exit(1)

        hyperparam_file = str(search_dirs[0] / "best_hyperparameters.json")

    # Check if hyperparameter file exists
    if not Path(hyperparam_file).exists():
        print(f"⚠️  Hyperparameter file not found: {hyperparam_file}")
        print(f"   Run hyperparameter tuning first with --tune-hyperparams")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION: {domain.upper()} SEGMENTATION at resolutions {resolutions} with {model_key.upper()}")
    print(f"Using hyperparameters from: {hyperparam_file}")
    print(f"{'='*80}\n")
    print(f"Train@R, Test@R protocol:")
    print(f"  - Train on full 80% training set")
    print(f"  - Evaluate on held-out 20% test set")
    print(f"  - Report test set performance\n")

    results = []

    for resolution in resolutions:
        print(f"\n{'-'*80}")
        print(f"Training at {resolution}px with {model_key}")
        print(f"{'-'*80}\n")

        # Build command
        cmd = [
            "python", "-m", "src.cli.train",
            f"--config-name={config_name}",
            f"data.image_size={resolution}",
            "train.hyperparam_search.enabled=false",
            f"train.hyperparam_search.load_from_file={hyperparam_file}",
            f"logging.run_name={model_key}_segmentation_{domain}_{resolution}px_eval",
        ]

        # Add dataset paths
        if image_dir:
            cmd.append(f"datamodule.image_dir={image_dir}")
        if mask_dir:
            cmd.append(f"datamodule.mask_dir={mask_dir}")

        # Add model configuration overrides
        for key, value in MODEL_CONFIGS[model_key].items():
            cmd.append(f"{key}={value}")

        print(f"Running: {' '.join(cmd)}\n")

        # Run
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"\n❌ Training failed at {resolution}px")
            results.append((resolution, "FAILED"))
        else:
            print(f"\n✅ Training completed at {resolution}px")
            results.append((resolution, "SUCCESS"))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    for resolution, status in results:
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_icon} {resolution}px: {status}")

    print(f"\n{'='*80}\n")

    if all(status == "SUCCESS" for _, status in results):
        print("🎉 All resolutions completed successfully!")
        print("\nNext steps:")
        print("  1. Check runs/ directory for final_model/ checkpoints")
        print("  2. Review test set metrics in final_metrics.json")
        print("  3. Compare performance across resolutions")
    else:
        print("⚠️  Some resolutions failed. Check logs above for details.")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-resolution segmentation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=list(DOMAIN_CONFIG.keys()),
        help="Domain to run experiments on",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dinov3",
        choices=MODELS,
        help="Model to use (default: dinov3)",
    )

    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Run hyperparameter tuning at highest resolution",
    )

    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        help="Resolutions to run final evaluation at (e.g., 512 256 128 64)",
    )

    parser.add_argument(
        "--hyperparam-file",
        type=str,
        default=None,
        help="Path to hyperparameter file (default: auto-detect from most recent tuning run)",
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Override image directory path",
    )

    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Override mask directory path",
    )

    args = parser.parse_args()

    # Override dataset paths if provided
    if args.image_dir or args.mask_dir:
        domain_cfg = DOMAIN_CONFIG[args.domain]
        if args.image_dir:
            domain_cfg["image_dir"] = args.image_dir
        if args.mask_dir:
            domain_cfg["mask_dir"] = args.mask_dir

    # Validate arguments
    if not args.tune_hyperparams and not args.resolutions:
        parser.error("Must specify either --tune-hyperparams or --resolutions (or both)")

    # Run hyperparameter tuning
    if args.tune_hyperparams:
        run_hyperparameter_tuning(args.domain, args.model)

    # Run final evaluation
    if args.resolutions:
        run_final_segmentation(
            args.domain,
            args.resolutions,
            args.model,
            args.hyperparam_file,
        )


if __name__ == "__main__":
    main()
