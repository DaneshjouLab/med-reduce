#!/usr/bin/env python3
"""
Experiment runner using Hydra's override system.
Usage: python run_experiments.py
"""

import subprocess
import itertools
from typing import List

# Define experiment grid
MODELS = ["vit", "dinov3"]
RESOLUTIONS = [224, 112, 56]
MODES = ["finetune", "probe"]

# Model configurations
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


def build_hydra_overrides(model_key: str, resolution: int, mode: str) -> List[str]:
    """Build list of Hydra override arguments."""
    overrides = []
    
    # Model overrides
    for key, value in MODEL_CONFIGS[model_key].items():
        overrides.append(f"{key}={value}")
    
    # Resolution overrides
    overrides.extend([
        f"dataset.image_size={resolution}",
        f"datamodule.image_size={resolution}",
    ])
    
    # Mode override
    overrides.append(f"train.mode={mode}")
    
    # Logging overrides
    run_name = f"{model_key}_{mode}_{resolution}"
    overrides.extend([
        f"logging.run_name={run_name}",
        f"runtime.run_dir=./runs/{run_name}",
    ])
    
    return overrides


def run_experiment(overrides: List[str], dry_run: bool = False):
    """Run training with Hydra overrides."""
    cmd = ["python", "-m", "src.cli.train"] + overrides
    
    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return 0
    
    print(f"\n{'='*80}")
    print(f"Running: python -m src.cli.train \\")
    for override in overrides:
        print(f"  {override} \\")
    print(f"{'='*80}\n")
      
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"⚠️  Experiment failed with return code {result.returncode}")
    else:
        print(f"✓ Experiment completed successfully")
    
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run grid search experiments")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without running")
    parser.add_argument("--models", nargs="+", choices=MODELS,
                       default=MODELS, help="Models to run")
    parser.add_argument("--resolutions", nargs="+", type=int,
                       choices=RESOLUTIONS, default=RESOLUTIONS,
                       help="Resolutions to run")
    parser.add_argument("--modes", nargs="+", choices=MODES,
                       default=MODES, help="Modes to run")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue running experiments even if one fails")
    
    args = parser.parse_args()
    
    # Generate all experiment combinations
    experiments = list(itertools.product(args.models, args.resolutions, args.modes))
    
    print(f"\n🚀 Planned experiments: {len(experiments)}")
    print(f"{'='*80}")
    for i, (model, res, mode) in enumerate(experiments, 1):
        print(f"  {i:2d}. {model:8s} @ {res:3d}px, {mode:8s}")
    print(f"{'='*80}\n")
    
    # Run experiments
    failed_experiments = []
    for i, (model_key, resolution, mode) in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(experiments)}: {model_key} @ {resolution}px, {mode}")
        print(f"{'#'*80}")
        
        overrides = build_hydra_overrides(model_key, resolution, mode)
        return_code = run_experiment(overrides, dry_run=args.dry_run)
        
        if return_code != 0:
            failed_experiments.append((model_key, resolution, mode))
            if not args.continue_on_error and not args.dry_run:
                print(f"\n❌ Stopping due to failed experiment")
                break
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Completed: {len(experiments) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print(f"\n❌ Failed experiments:")
        for model, res, mode in failed_experiments:
            print(f"  - {model} @ {res}px, {mode}")
    else:
        print(f"\n✅ All experiments completed successfully!")


if __name__ == "__main__":
    main()