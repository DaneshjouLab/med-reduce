#!/usr/bin/env python3
"""CLI entry point for distillation pipeline."""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


@hydra.main(config_path="../../configs", config_name="distillation_dermatology", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for distillation."""
    OmegaConf.set_struct(cfg, False)

    from src.wrappers.distillation_wrapper import run

    print("\n" + "=" * 80)
    print("DISTILLATION PIPELINE")
    print("=" * 80)
    print("\nConfiguration:\n", OmegaConf.to_yaml(cfg), flush=True)

    # Update runtime.run_dir with Hydra's output directory if not explicitly configured
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    configured_run_dir = cfg.get("runtime", {}).get("run_dir", None)

    if configured_run_dir and str(configured_run_dir).strip() and str(configured_run_dir) != "{}":
        print(f"\nUsing configured run directory: {configured_run_dir}")
    else:
        if "runtime" not in cfg:
            cfg.runtime = {}
        cfg.runtime.run_dir = str(hydra_output_dir)
        print(f"\nUsing Hydra output directory: {hydra_output_dir}")

    result = run(cfg)

    print("\n" + "=" * 80)
    print("DISTILLATION COMPLETE")
    print("=" * 80)
    print(f"Best val loss: {result.get('best_val_loss', 'N/A'):.6f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
