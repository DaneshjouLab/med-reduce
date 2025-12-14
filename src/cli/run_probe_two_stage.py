#!/usr/bin/env python3
"""CLI entry point for two-stage linear probing."""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


@hydra.main(config_path="../../configs", config_name="probe_two_stage", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for two-stage linear probing."""
    OmegaConf.set_struct(cfg, False)

    from src.wrappers.probe_two_stage import run

    print("\n" + "="*80)
    print("TWO-STAGE LINEAR PROBING")
    print("="*80)
    print("\nConfiguration:\n", OmegaConf.to_yaml(cfg), flush=True)

    # Update runtime.run_dir with Hydra's output directory if it's still set to the default
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # If run_dir is set to a server path, keep it; otherwise use Hydra's output dir
    if hasattr(cfg, 'runtime') and hasattr(cfg.runtime, 'run_dir'):
        configured_run_dir = str(cfg.runtime.run_dir)
        # Check if it's a local/relative path or server path
        if not configured_run_dir.startswith('/scratch'):
            # Use Hydra's output directory for local runs
            cfg.runtime.run_dir = str(hydra_output_dir)
            print(f"\n📁 Using Hydra output directory: {hydra_output_dir}")
    else:
        # No runtime.run_dir specified, use Hydra's
        if not hasattr(cfg, 'runtime'):
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
