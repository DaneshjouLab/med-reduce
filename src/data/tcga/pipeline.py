"""
TCGA Dataset Builder — end-to-end pipeline orchestrator.

Chains the existing TCGA modules into a single YAML-driven pipeline:
query GDC → build slide table → generate manifests → download →
process slides → build gene matrix → assemble final dataset.

Usage:
    from src.data.tcga import TCGADatasetBuilder

    builder = TCGADatasetBuilder(cfg)
    builder.run()
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig

from src.data.tcga.config import TCGAConfig
from src.data.tcga.downloader import DownloadStatus, TCGADownloader
from src.data.tcga.etl import TCGASlideETL
from src.data.tcga.gene_matrix import GeneMatrix
from src.data.tcga.manifest import ManifestGenerator
from src.data.tcga.slide_processor import SlideProcessor

logger = logging.getLogger(__name__)

VALID_STEPS = ["etl", "manifest", "download", "process_slides", "gene_matrix", "assemble"]


class TCGADatasetBuilder:
    """Orchestrates the full TCGA dataset build pipeline.

    Each step is a method that reads / writes intermediate artifacts under
    ``data_dir`` so the pipeline is resumable.  Pass ``force=True`` to
    re-run steps even when their artifacts already exist.

    Args:
        cfg: A resolved OmegaConf ``DictConfig`` (or plain dict) matching
            the schema of ``configs/tcga_dataset.yaml``.
        force: If True, ignore existing artifacts and re-run every step.
    """

    def __init__(self, cfg: DictConfig, force: bool = False):
        self.cfg = cfg
        self.force = force

        # Build TCGAConfig from the YAML fields
        self.tcga_config = TCGAConfig(
            project_ids=list(cfg.projects),
            data_dir=Path(cfg.data_dir),
            include_demographics=cfg.etl.include_demographics,
            include_diagnosis=cfg.etl.include_diagnosis,
            include_maf=cfg.etl.include_maf,
            access=cfg.access,
        )
        self.tcga_config.ensure_directories()

        # Convenience aliases
        self._tables_dir = self.tcga_config.tables_dir
        self._manifests_dir = self.tcga_config.manifests_dir

    # ── public entry point ────────────────────────────────────────────

    def run(self, steps: Optional[List[str]] = None) -> Path:
        """Run the pipeline.

        Args:
            steps: Override which steps to execute.  Defaults to
                ``cfg.steps``.

        Returns:
            Path to the final dataset CSV.
        """
        steps = steps or list(self.cfg.steps)
        self._validate_steps(steps)

        logger.info("Pipeline steps: %s", steps)
        logger.info("Projects: %s", list(self.cfg.projects))
        logger.info("Data dir: %s", self.tcga_config.data_dir)

        dispatch = {
            "etl": self.run_etl,
            "manifest": self.run_manifest,
            "download": self.run_download,
            "process_slides": self.run_process_slides,
            "gene_matrix": self.run_gene_matrix,
            "assemble": self.run_assemble,
        }

        for step in steps:
            logger.info("=" * 60)
            logger.info("STEP: %s", step)
            logger.info("=" * 60)
            dispatch[step]()

        dataset_path = self._tables_dir / "dataset.csv"
        self._print_summary(dataset_path)
        return dataset_path

    # ── step 1: ETL ───────────────────────────────────────────────────

    def run_etl(self) -> pd.DataFrame:
        """Query GDC API and build flat slide table."""
        output = self._tables_dir / "slide_table.parquet"

        if output.exists() and not self.force:
            logger.info("Slide table already exists at %s — skipping ETL", output)
            return pd.read_parquet(output)

        etl = TCGASlideETL()
        df = etl.build_slide_table(
            project_ids=self.tcga_config.project_ids,
            include_demographics=self.tcga_config.include_demographics,
            include_diagnosis=self.tcga_config.include_diagnosis,
            include_maf=self.tcga_config.include_maf,
            access=self.tcga_config.access,
        )
        df = etl.add_local_paths(df, self.tcga_config)
        logger.info("Slide table: %d rows, %d columns", len(df), len(df.columns))

        self._stringify_paths(df).to_parquet(output, index=False)
        self._stringify_paths(df).to_csv(output.with_suffix(".csv"), index=False)
        logger.info("Saved slide table → %s", output)
        return df

    # ── step 2: manifests ─────────────────────────────────────────────

    def run_manifest(self) -> Dict[str, Optional[Path]]:
        """Generate download manifests from the slide table."""
        slide_manifest = self._manifests_dir / "slides_manifest.txt"
        maf_manifest = self._manifests_dir / "maf_manifest.txt"

        if slide_manifest.exists() and not self.force:
            logger.info("Manifests already exist — skipping manifest generation")
            return {"slides": slide_manifest, "maf": maf_manifest if maf_manifest.exists() else None}

        df = self._load_slide_table()
        gen = ManifestGenerator()

        # Slide manifest
        gen.create_slide_manifest(df, slide_manifest)
        logger.info("Slide manifest → %s", slide_manifest)

        # MAF manifest (only if MAF data present)
        maf_path: Optional[Path] = None
        if self.tcga_config.include_maf:
            maf_path = gen.create_maf_manifest(df, maf_manifest)
            if maf_path:
                logger.info("MAF manifest   → %s", maf_path)
            else:
                logger.info("No MAF files found — skipping MAF manifest")

        # Subset manifests for testing
        max_files = self.cfg.download.get("max_files")
        if max_files:
            subset_slide = self._manifests_dir / "slides_manifest_subset.txt"
            gen.create_subset_manifest(slide_manifest, subset_slide, max_files)
            logger.info("Subset slide manifest (%d files) → %s", max_files, subset_slide)

            if maf_path:
                subset_maf = self._manifests_dir / "maf_manifest_subset.txt"
                gen.create_subset_manifest(maf_manifest, subset_maf, max_files)
                logger.info("Subset MAF manifest (%d files) → %s", max_files, subset_maf)

        return {"slides": slide_manifest, "maf": maf_path}

    # ── step 3: download ──────────────────────────────────────────────

    def run_download(self) -> None:
        """Download slides and MAF files via gdc-client."""
        dl_cfg = self.cfg.download
        if not dl_cfg.get("enabled", True):
            logger.info("Downloads disabled in config — skipping")
            return

        downloader = TCGADownloader()
        max_files = dl_cfg.get("max_files")
        token_path = dl_cfg.get("token_path")
        if token_path:
            token_path = Path(token_path)
        n_processes = dl_cfg.get("n_processes", 4)

        # --- Slides ---
        if dl_cfg.get("slides", True):
            manifest = self._resolve_manifest("slides_manifest", max_files)
            status = downloader.check_download_status(self.tcga_config.slides_dir, manifest)

            if status.status == DownloadStatus.COMPLETED and not self.force:
                logger.info("Slides already downloaded (%d/%d) — skipping",
                            status.files_downloaded, status.files_total)
            else:
                logger.info("Downloading slides from %s …", manifest)
                result = downloader.download_from_manifest(
                    manifest, self.tcga_config.slides_dir,
                    token_path=token_path, n_processes=n_processes,
                )
                logger.info("Slide download %s: %d/%d files",
                            result.status.value, result.files_downloaded, result.files_total)
                if result.error_message:
                    logger.error("Download error: %s", result.error_message)

        # --- MAF ---
        if dl_cfg.get("maf", True) and self.tcga_config.include_maf:
            maf_manifest = self._resolve_manifest("maf_manifest", max_files)
            if maf_manifest and maf_manifest.exists():
                status = downloader.check_download_status(self.tcga_config.maf_dir, maf_manifest)

                if status.status == DownloadStatus.COMPLETED and not self.force:
                    logger.info("MAF files already downloaded (%d/%d) — skipping",
                                status.files_downloaded, status.files_total)
                else:
                    logger.info("Downloading MAF files from %s …", maf_manifest)
                    result = downloader.download_from_manifest(
                        maf_manifest, self.tcga_config.maf_dir,
                        token_path=token_path, n_processes=n_processes,
                    )
                    logger.info("MAF download %s: %d/%d files",
                                result.status.value, result.files_downloaded, result.files_total)
                    if result.error_message:
                        logger.error("Download error: %s", result.error_message)
            else:
                logger.info("No MAF manifest found — skipping MAF download")

    # ── step 4: process slides ────────────────────────────────────────

    def run_process_slides(self) -> pd.DataFrame:
        """Create JPG thumbnails from SVS whole-slide images."""
        df = self._load_slide_table()
        slides_cfg = self.cfg.slides
        size = tuple(slides_cfg.get("thumbnail_size", [512, 512]))
        n_workers = slides_cfg.get("n_workers", 4)

        processor = SlideProcessor(n_workers=n_workers)
        result = processor.process_slides(
            df=df,
            output_dir=self.tcga_config.thumbnails_dir,
            size=size,
        )

        logger.info("Slide processing: %d processed, %d skipped, %d failed, %d missing",
                     result.processed, result.skipped, result.failed, result.missing)

        # Persist the updated table with jpg_path
        output = self._tables_dir / "slide_table.parquet"
        self._stringify_paths(result.df).to_parquet(output, index=False)
        self._stringify_paths(result.df).to_csv(output.with_suffix(".csv"), index=False)
        logger.info("Updated slide table with jpg_path → %s", output)

        return result.df

    # ── step 5: gene matrix ───────────────────────────────────────────

    def run_gene_matrix(self) -> GeneMatrix:
        """Build gene mutation matrix from downloaded MAF files."""
        gm_cfg = self.cfg.gene_matrix
        if not gm_cfg.get("enabled", True):
            logger.info("Gene matrix disabled in config — skipping")
            return GeneMatrix()

        output = self._tables_dir / "gene_matrix.parquet"
        if output.exists() and not self.force:
            logger.info("Gene matrix already exists at %s — loading", output)
            return GeneMatrix.load(output)

        gm = GeneMatrix()
        gm.build_from_maf_dir(self.tcga_config.maf_dir)
        gm.save(output)
        logger.info("Gene matrix %s → %s", gm.shape, output)
        return gm

    # ── step 6: assemble ──────────────────────────────────────────────

    def run_assemble(self) -> pd.DataFrame:
        """Merge slide table + gene matrix → final dataset."""
        df = self._load_slide_table()

        # Validate local paths (adds slide_exists / maf_exists)
        etl = TCGASlideETL()
        df = etl.validate_local_paths(df)

        # Merge gene matrix if it exists
        gm_path = self._tables_dir / "gene_matrix.parquet"
        if gm_path.exists():
            gm = GeneMatrix.load(gm_path)
            genes = self.cfg.gene_matrix.get("genes")
            genes = list(genes) if genes else None
            df = gm.merge(df, genes=genes)
            logger.info("Merged gene matrix (%d genes) into slide table", len(gm.genes) if genes is None else len(genes))

        # Save final dataset
        csv_out = self._tables_dir / "dataset.csv"
        parquet_out = self._tables_dir / "dataset.parquet"
        df_out = self._stringify_paths(df)
        df_out.to_csv(csv_out, index=False)
        df_out.to_parquet(parquet_out, index=False)
        logger.info("Final dataset: %d rows × %d columns", len(df), len(df.columns))
        logger.info("Saved → %s", csv_out)
        logger.info("Saved → %s", parquet_out)
        return df

    # ── helpers ───────────────────────────────────────────────────────

    def _load_slide_table(self) -> pd.DataFrame:
        """Load the slide table from parquet (must have been built by run_etl)."""
        path = self._tables_dir / "slide_table.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Slide table not found at {path}. Run the 'etl' step first."
            )
        return pd.read_parquet(path)

    def _resolve_manifest(self, name: str, max_files: Optional[int] = None) -> Path:
        """Return the subset manifest if max_files is set, else the full one."""
        if max_files:
            subset = self._manifests_dir / f"{name}_subset.txt"
            if subset.exists():
                return subset
        return self._manifests_dir / f"{name}.txt"

    def _print_summary(self, dataset_path: Path) -> None:
        """Print a human-readable summary of the pipeline run."""
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        slide_table = self._tables_dir / "slide_table.parquet"
        if slide_table.exists():
            df = pd.read_parquet(slide_table)
            logger.info("Slide table : %d rows", len(df))

        gm_path = self._tables_dir / "gene_matrix.parquet"
        if gm_path.exists():
            gm = GeneMatrix.load(gm_path)
            logger.info("Gene matrix : %d samples × %d genes", *gm.shape)

        if dataset_path.exists():
            ds = pd.read_csv(dataset_path, nrows=0)
            n_rows = sum(1 for _ in open(dataset_path)) - 1
            logger.info("Dataset     : %d rows × %d columns", n_rows, len(ds.columns))
            logger.info("Output      : %s", dataset_path)

    @staticmethod
    def _validate_steps(steps: List[str]) -> None:
        for s in steps:
            if s not in VALID_STEPS:
                raise ValueError(
                    f"Unknown step '{s}'. Valid steps: {VALID_STEPS}"
                )

    @staticmethod
    def _stringify_paths(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Path objects to strings so parquet/csv serialisation works."""
        df = df.copy()
        path_cols = [c for c in df.columns if c.endswith("_path")]
        for col in path_cols:
            df[col] = df[col].apply(lambda v: str(v) if v is not None else None)
        return df
