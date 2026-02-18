#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
from omegaconf import OmegaConf


# Model configurations
MODELS = ["vit", "dinov2", "dinov3", "resnet18", "tiny_vit_21m_224"]
MODEL_CONFIGS = {
    "vit": {
        "model.name": "vit",
        "model.model_id": "google/vit-base-patch16-224",
        "model.type": "vit",
        "+model.dtype": "bfloat16"
    },
    "dinov2": {
        "model.name": "dinov2",
        "model.model_id": "facebook/dinov2-small",
        "model.type": "dinov2",
    },
    "dinov3": {
        "model.name": "dinov3",
        "model.model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "model.type": "dinov3",
    },
    "resnet18": {
        "model.name": "resnet18",
        "model.model_id": "resnet18",
        "model.type": "timm",
        "+model.config.pretrained": "false",
    },
    "tiny_vit_21m_224": {
        "model.name": "tiny_vit",
        "model.model_id": "tiny_vit_21m_224",
        "model.type": "timm",
        "+model.config.pretrained": "false",
    }
}

# Resolution configuration per domain
DOMAIN_CONFIG = {
    "dermatology": {
        "highest_resolution": 512,
        "default_resolutions": [512, 256, 128, 64],
    },
    "radiology": {
        "highest_resolution": 512,
        "default_resolutions": [512, 256, 128, 64],
    },
    "pathology": {
        "highest_resolution": 512,
        "default_resolutions": [512, 256, 128, 64],
    },
}


def run_hyperparameter_tuning(
    domain: str,
    model_key: str = "dinov3",
    config_path: str = "configs/probe_two_stage.yaml",
    seed: int = 42,
    extra_overrides: List[str] = None,
):
    """
    Run hyperparameter tuning at the highest resolution for a domain.

    Args:
        domain: Domain name (dermatology, radiology, pathology)
        model_key: Model to use (vit, dinov3)
        config_path: Path to config file
        seed: Random seed for reproducibility
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Choose from {list(DOMAIN_CONFIG.keys())}")

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")

    domain_cfg = DOMAIN_CONFIG[domain]
    highest_res = domain_cfg["highest_resolution"]
    dataset = domain_cfg.get("dataset")
    data_dir = domain_cfg.get("data_dir")

    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING: {domain.upper()} at {highest_res}px with {model_key.upper()} (seed={seed})")
    print(f"{'='*80}\n")

    # Build command with model overrides
    cmd = [
        "python", "-m", "src.cli.run_probe_two_stage",
        f"--config-name={config_path}",
        f"domain={domain}",
        f"data.image_size={highest_res}",
        "train.hyperparam_search.enabled=true",
        f"train.seed={seed}",
        f"++logging.run_name={model_key}_{domain}_{highest_res}px_seed{seed}_R*search",
    ]

    if dataset is not None:
        cmd.append(f"datamodule.dataset_name={dataset}")
    if data_dir is not None:
        cmd.append(f"datamodule.data_dir={data_dir}")

    # Add model configuration overrides
    for key, value in MODEL_CONFIGS[model_key].items():
        cmd.append(f"{key}={value}")

    # Add any extra Hydra overrides
    if extra_overrides:
        cmd.extend(extra_overrides)

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n❌ Hyperparameter tuning failed for {domain} with {model_key} (seed={seed})")
        sys.exit(1)

    print(f"\n✅ Hyperparameter tuning completed for {domain} with {model_key} (seed={seed})")


def run_final_probing(
    domain: str,
    resolutions: List[int],
    model_key: str = "dinov3",
    config_path: str = "configs/probe_two_stage.yaml",
    hyperparam_file: str = None,
    seed: int = 42,
    extra_overrides: List[str] = None,
):
    """
    Run final linear probing at multiple resolutions with tuned hyperparameters.

    Args:
        domain: Domain name
        resolutions: List of resolutions to evaluate
        model_key: Model to use (vit, dinov3)
        config_path: Path to config file
        hyperparam_file: Path to hyperparameter file (if None, uses default location)
        seed: Random seed for reproducibility
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}")

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")

    domain_cfg = DOMAIN_CONFIG[domain]
    dataset = domain_cfg.get("dataset")
    data_dir = domain_cfg.get("data_dir")

    if hyperparam_file is None:
        project_root = Path(__file__).parent.parent.parent
        final_config_path = config_path

        if not final_config_path.startswith("configs/"):
            final_config_path = f"configs/{final_config_path}"

        if not final_config_path.endswith(".yaml"):
            final_config_path = f"{final_config_path}.yaml"

        resolved_config_path = project_root / final_config_path
        config = OmegaConf.load(resolved_config_path)

        # Check if hyperparam_search.load_from_file is specified
        hyperparam_file = config.train.hyperparam_search.get('load_from_file')
        if not hyperparam_file:
            run_dir = config.runtime.get('run_dir', './runs/probe_two_stage')
            # Use seed in the path to hyperparam file
            hyperparam_file = f"{run_dir}/seed_{seed}/hyperparam_search/best_hyperparameters.json"

        # Resolve hyperparam_file relative to project root
        hyperparam_file = str((project_root / hyperparam_file).resolve())

    # Check if hyperparameter file exists
    if not Path(hyperparam_file).exists():
        print(f"⚠️  Hyperparameter file not found: {hyperparam_file}")
        print(f"   Run hyperparameter tuning first with --tune-hyperparams")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"FINAL PROBING: {domain.upper()} at resolutions {resolutions} with {model_key.upper()} (seed={seed})")
    print(f"Using hyperparameters from: {hyperparam_file}")
    print(f"{'='*80}\n")

    results = []

    for resolution in resolutions:
        print(f"\n{'-'*80}")
        print(f"Training at {resolution}px with {model_key} (seed={seed})")
        print(f"{'-'*80}\n")

        # Build command
        cmd = [
            "python", "-m", "src.cli.run_probe_two_stage",
            f"--config-name={config_path}",
            f"domain={domain}",
            f"data.image_size={resolution}",
            "train.hyperparam_search.enabled=false",
            f"++train.hyperparam_search.load_from_file={hyperparam_file}",
            f"train.seed={seed}",
            f"++logging.run_name={model_key}_{domain}_{resolution}px_seed{seed}_eval",
        ]

        if dataset is not None:
            cmd.append(f"datamodule.dataset_name={dataset}")
        if data_dir is not None:
            cmd.append(f"datamodule.data_dir={data_dir}")

        # Add model configuration overrides
        for key, value in MODEL_CONFIGS[model_key].items():
            cmd.append(f"{key}={value}")

        # Add any extra Hydra overrides
        if extra_overrides:
            cmd.extend(extra_overrides)

        print(f"Running: {' '.join(cmd)}\n")

        # Run
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"\n❌ Training failed at {resolution}px (seed={seed})")
            results.append((resolution, seed, "FAILED"))
        else:
            print(f"\n✅ Training completed at {resolution}px (seed={seed})")
            results.append((resolution, seed, "SUCCESS"))

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY (seed={seed})")
    print(f"{'='*80}\n")

    for resolution, seed_val, status in results:
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_icon} {resolution}px (seed={seed_val}): {status}")

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
        choices=list(MODEL_CONFIGS.keys()),
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
        default="configs/probe_two_stage",
        help="Path to config file",
    )

    parser.add_argument(
        "--hyperparam-file",
        type=str,
        default=None,
        help="Path to hyperparameter file (default: auto-detect from tuning run)",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for bootstrap runs (e.g., --seeds 42 123 456)",
    )

    parser.add_argument(
        "--extra-overrides",
        type=str,
        nargs="*",
        default=[],
        help="Additional Hydra overrides passed to run_probe_two_stage (e.g., datamodule.task=kras)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.tune_hyperparams and not args.resolutions:
        parser.error("Must specify either --tune-hyperparams or --resolutions")

    first_seed = args.seeds[0]
    hyperparam_file = args.hyperparam_file

    # Run hyperparameter tuning (only once with the first seed)
    if args.tune_hyperparams:
        run_hyperparameter_tuning(args.domain, args.model, args.config, seed=first_seed, extra_overrides=args.extra_overrides)

        # Construct path to best_hyperparameters.json from first seed
        if hyperparam_file is None:
            project_root = Path(__file__).parent.parent.parent
            final_config_path = args.config

            if not final_config_path.startswith("configs/"):
                final_config_path = f"configs/{final_config_path}"

            if not final_config_path.endswith(".yaml"):
                final_config_path = f"{final_config_path}.yaml"

            resolved_config_path = project_root / final_config_path
            config = OmegaConf.load(resolved_config_path)

            run_dir = config.runtime.get('run_dir', './runs/probe_two_stage')
            hyperparam_file = str((project_root / f"{run_dir}/seed_{first_seed}/hyperparam_search/best_hyperparameters.json").resolve())

    # Run final probing for ALL seeds using shared hyperparams
    if args.resolutions:
        for seed in args.seeds:
            run_final_probing(
                args.domain,
                args.resolutions,
                args.model,
                args.config,
                hyperparam_file=hyperparam_file,
                seed=seed,
                extra_overrides=args.extra_overrides,
            )


if __name__ == "__main__":
    main()
