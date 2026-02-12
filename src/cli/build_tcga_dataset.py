"""
CLI entry point for building a TCGA dataset end-to-end.

Usage:
    # Default config
    python -m src.cli.build_tcga_dataset

    # Custom config
    python -m src.cli.build_tcga_dataset --config configs/tcga_dataset.yaml

    # Override steps (comma-separated)
    python -m src.cli.build_tcga_dataset --steps etl,manifest

    # Override config values (positional key=value)
    python -m src.cli.build_tcga_dataset projects="[TCGA-BRCA]" download.max_files=2

    # Combine overrides and step selection
    python -m src.cli.build_tcga_dataset --steps etl,manifest download.max_files=2

    # Dry run (show resolved config, don't execute)
    python -m src.cli.build_tcga_dataset --dry-run

    # Force re-run (ignore existing artifacts)
    python -m src.cli.build_tcga_dataset --force
"""

import argparse
import logging
import sys

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/tcga_dataset.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a TCGA dataset end-to-end from GDC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --dry-run\n"
            "  %(prog)s --steps etl,manifest\n"
            '  %(prog)s projects="[TCGA-BRCA]" download.max_files=2\n'
            "  %(prog)s --force --steps etl\n"
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Override which pipeline steps to run (comma-separated, e.g. etl,manifest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and steps without executing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if artifacts exist",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help='Config overrides (key=value), e.g. projects="[TCGA-BRCA]"',
    )
    return parser.parse_args(argv)


def load_config(args):
    """Load YAML config and merge CLI overrides via OmegaConf."""
    cfg = OmegaConf.load(args.config)

    # Apply dotlist overrides from positional args
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Override steps if provided via --steps (comma-separated string)
    if args.steps is not None:
        cfg.steps = [s.strip() for s in args.steps.split(",")]

    return cfg


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args(argv)
    cfg = load_config(args)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — resolved config:")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print(f"Steps: {list(cfg.steps)}")
        print("=" * 60)
        return

    from src.data.tcga.pipeline import TCGADatasetBuilder

    builder = TCGADatasetBuilder(cfg, force=args.force)
    dataset_path = builder.run()
    logger.info("Done. Dataset at: %s", dataset_path)


if __name__ == "__main__":
    main()
