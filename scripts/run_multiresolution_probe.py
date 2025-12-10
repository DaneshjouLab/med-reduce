#!/usr/bin/env python3
"""
Run multi-resolution linear probing experiments with the two-stage approach.

Usage:
    # Step 1: Run hyperparameter tuning at highest resolution (512px for derm/rad, 1024px for path)
    python scripts/run_multiresolution_probe.py --domain dermatology --tune-hyperparams

    # Step 2: Run final probing at all resolutions with tuned hyperparameters
    python scripts/run_multiresolution_probe.py --domain dermatology --resolutions 512 256 128 64

Example workflow:
    # Dermatology
    python scripts/run_multiresolution_probe.py --domain dermatology --tune-hyperparams
    python scripts/run_multiresolution_probe.py --domain dermatology --resolutions 512 256 128 64
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


# Resolution configuration per domain
DOMAIN_CONFIG = {
    "dermatology": {
        "highest_resolution": 512,
        "default_resolutions": [512, 256, 128, 64],
        "dataset": "isic",
        "data_dir": "./data/isic",
    },
}


def run_hyperparameter_tuning(domain: str, config_path: str = "configs/probe_two_stage.yaml"):
    """
    Run hyperparameter tuning at the highest resolution for a domain.

    Args:
        domain: Domain name (dermatology, radiology, pathology)
        config_path: Path to config file
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Choose from {list(DOMAIN_CONFIG.keys())}")

    domain_cfg = DOMAIN_CONFIG[domain]
    highest_res = domain_cfg["highest_resolution"]
    dataset = domain_cfg["dataset"]
    data_dir = domain_cfg["data_dir"]

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING: {domain.upper()} at {highest_res}px")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        "python", "-m", "src.wrappers.probe_two_stage",
        f"--config-name={config_path}",
        f"domain={domain}",
        f"data.image_size={highest_res}",
        f"datamodule.dataset_name={dataset}",
        f"datamodule.data_dir={data_dir}",
        "train.hyperparam_search.enabled=true",
        f"logging.run_name=hyperparam_tune_{domain}_{highest_res}px",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    # Run
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n❌ Hyperparameter tuning failed for {domain}")
        sys.exit(1)

    print(f"\n✅ Hyperparameter tuning completed for {domain}")
    print(f"   Results saved to: ./runs/probe_two_stage/hyperparam_search/best_hyperparameters.json")


def run_final_probing(
    domain: str,
    resolutions: List[int],
    config_path: str = "configs/probe_two_stage.yaml",
    hyperparam_file: str = None,
):
    """
    Run final linear probing at multiple resolutions with tuned hyperparameters.

    Args:
        domain: Domain name
        resolutions: List of resolutions to evaluate
        config_path: Path to config file
        hyperparam_file: Path to hyperparameter file (if None, uses default location)
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}")

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
    print(f"FINAL PROBING: {domain.upper()} at resolutions {resolutions}")
    print(f"Using hyperparameters from: {hyperparam_file}")
    print(f"{'='*80}\n")

    results = []

    for resolution in resolutions:
        print(f"\n{'-'*80}")
        print(f"Training at {resolution}px")
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
            f"logging.run_name={domain}_{resolution}px_final",
        ]

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
        run_hyperparameter_tuning(args.domain, args.config)

    # Run final probing
    if args.resolutions:
        run_final_probing(
            args.domain,
            args.resolutions,
            args.config,
            args.hyperparam_file,
        )


if __name__ == "__main__":
    main()
