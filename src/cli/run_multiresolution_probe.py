#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


# Model configurations
MODELS = ["vit", "dinov3"]
MODEL_CONFIGS = {
    "vit": {
        "model.name": "vit",
        "model.model_id": "google/vit-base-patch16-224",
        "model.type": "vit",
        "+model.dtype": "bfloat16"
    },
    "dinov3": {
        "model.name": "dinov3",
        "model.model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "model.type": "dinov3",
    }
}

# Resolution configuration per domain
DOMAIN_CONFIG = {
    "dermatology": {
        "highest_resolution": 512,
        "default_resolutions": [512, 256, 128, 64],
        "dataset": "isic",
        "data_dir": "./data/isic",
    },
}


def run_hyperparameter_tuning(
    domain: str,
    model_key: str = "dinov3",
    config_path: str = "configs/probe_two_stage.yaml"
):
    """
    Run hyperparameter tuning at the highest resolution for a domain.

    Args:
        domain: Domain name (dermatology, radiology, pathology)
        model_key: Model to use (vit, dinov3)
        config_path: Path to config file
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Choose from {list(DOMAIN_CONFIG.keys())}")

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")

    domain_cfg = DOMAIN_CONFIG[domain]
    highest_res = domain_cfg["highest_resolution"]
    dataset = domain_cfg["dataset"]
    data_dir = domain_cfg["data_dir"]

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING: {domain.upper()} at {highest_res}px with {model_key.upper()}")
    print(f"{'='*80}\n")

    # Build command with model overrides
    cmd = [
        "python", "-m", "src.wrappers.probe_two_stage",
        f"--config-name={config_path}",
        f"domain={domain}",
        f"data.image_size={highest_res}",
        f"datamodule.dataset_name={dataset}",
        f"datamodule.data_dir={data_dir}",
        "train.hyperparam_search.enabled=true",
        f"logging.run_name=hyperparam_tune_{model_key}_{domain}_{highest_res}px",
    ]

    # Add model configuration overrides
    for key, value in MODEL_CONFIGS[model_key].items():
        cmd.append(f"{key}={value}")

    print(f"Running: {' '.join(cmd)}\n")

    # Run
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n❌ Hyperparameter tuning failed for {domain} with {model_key}")
        sys.exit(1)

    print(f"\n✅ Hyperparameter tuning completed for {domain} with {model_key}")
    print(f"   Results saved to: ./runs/probe_two_stage/hyperparam_search/best_hyperparameters.json")


def run_final_probing(
    domain: str,
    resolutions: List[int],
    model_key: str = "dinov3",
    config_path: str = "configs/probe_two_stage.yaml",
    hyperparam_file: str = None,
):
    """
    Run final linear probing at multiple resolutions with tuned hyperparameters.

    Args:
        domain: Domain name
        resolutions: List of resolutions to evaluate
        model_key: Model to use (vit, dinov3)
        config_path: Path to config file
        hyperparam_file: Path to hyperparameter file (if None, uses default location)
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}")

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")

    domain_cfg = DOMAIN_CONFIG[domain]
    dataset = domain_cfg["dataset"]
    data_dir = domain_cfg["data_dir"]

    # Default hyperparameter file location
    if hyperparam_file is None:
        hyperparam_file = "./runs/probe_two_stage/hyperparam_search/best_hyperparameters.json"

    # Check if hyperparameter file exists
    if not Path(hyperparam_file).exists():
        print(f"⚠️  Hyperparameter file not found: {hyperparam_file}")
        print(f"   Run hyperparameter tuning first with --tune-hyperparams")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"FINAL PROBING: {domain.upper()} at resolutions {resolutions} with {model_key.upper()}")
    print(f"Using hyperparameters from: {hyperparam_file}")
    print(f"{'='*80}\n")

    results = []

    for resolution in resolutions:
        print(f"\n{'-'*80}")
        print(f"Training at {resolution}px with {model_key}")
        print(f"{'-'*80}\n")

        # Build command
        cmd = [
            "python", "-m", "src.wrappers.probe_two_stage",
            f"--config-name={config_path}",
            f"domain={domain}",
            f"data.image_size={resolution}",
            f"datamodule.dataset_name={dataset}",
            f"datamodule.data_dir={data_dir}",
            "train.hyperparam_search.enabled=false",
            f"train.hyperparam_search.load_from_file={hyperparam_file}",
            f"logging.run_name={model_key}_{domain}_{resolution}px_final",
        ]

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


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-resolution linear probing experiments",
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
        help="Resolutions to run final probing at (e.g., 512 256 128 64)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/probe_two_stage.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--hyperparam-file",
        type=str,
        default=None,
        help="Path to hyperparameter file (default: auto-detect from tuning run)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.tune_hyperparams and not args.resolutions:
        parser.error("Must specify either --tune-hyperparams or --resolutions")

    # Run hyperparameter tuning
    if args.tune_hyperparams:
        run_hyperparameter_tuning(args.domain, args.model, args.config)

    # Run final probing
    if args.resolutions:
        run_final_probing(
            args.domain,
            args.resolutions,
            args.model,
            args.config,
            args.hyperparam_file,
        )


if __name__ == "__main__":
    main()
