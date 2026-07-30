#!/usr/bin/env python3
"""CLI entry point for two-stage linear probing."""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


@hydra.main(config_path="../../configs", config_name="probe_two_stage_dermatology", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for two-stage linear probing."""
    OmegaConf.set_struct(cfg, False)

    from src.wrappers.probe_two_stage import run

    print("\n" + "="*80)
    print("TWO-STAGE LINEAR PROBING")
    print("="*80)
    print("\nConfiguration:\n", OmegaConf.to_yaml(cfg), flush=True)

    # Update runtime.run_dir with Hydra's output directory if not explicitly configured
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Use configured run_dir if it exists and is not empty, otherwise use Hydra's output dir
    configured_run_dir = cfg.get('runtime', {}).get('run_dir', None)

    if configured_run_dir and str(configured_run_dir).strip() and str(configured_run_dir) != '{}':
        print(f"\n📁 Using configured run directory: {configured_run_dir}")
    else:
        if 'runtime' not in cfg:
            cfg.runtime = {}
        cfg.runtime.run_dir = str(hydra_output_dir)
        print(f"\n📁 Using Hydra output directory: {hydra_output_dir}")

    # Run the wrapper
    result = run(cfg)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best metric: {result.get('best_metric', 'N/A'):.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
