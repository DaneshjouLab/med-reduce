"""Parallel processing of whole slide images.

Creates thumbnail images from SVS files using multiprocessing.

Usage:
    from src.data.tcga import SlideProcessor

    processor = SlideProcessor(n_workers=4)
    result = processor.process_slides(
        df=slide_df,
        output_dir=config.thumbnails_dir,
        size=(512, 512),
    )

    print(f"Processed: {result.processed}")
    print(f"Skipped (exists): {result.skipped}")
    print(f"Failed: {result.failed}")

    # DataFrame now has jpg_path column
    df_with_paths = result.df
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from multiprocessing import Pool, cpu_count
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of slide processing batch."""
    df: pd.DataFrame
    processed: int = 0
    skipped: int = 0  # Already existed
    failed: int = 0   # Errors during processing
    missing: int = 0  # Source file not found


@dataclass
class SlideTask:
    """Single slide processing task."""
    slide_id: str
    source_path: Path
    output_path: Path
    size: Tuple[int, int]


@dataclass
class SlideTaskResult:
    """Result of processing a single slide."""
    slide_id: str
    jpg_path: Optional[Path]
    status: str  # "processed", "skipped", "failed", "missing"
    error: Optional[str] = None


def _process_single_slide(task: SlideTask) -> SlideTaskResult:
    """Process a single slide (runs in worker process).

    This function is defined at module level for pickling with multiprocessing.
    """
    try:
        # Check if output already exists
        if task.output_path.exists():
            return SlideTaskResult(
                slide_id=task.slide_id,
                jpg_path=task.output_path,
                status="skipped",
            )

        # Check if source exists
        if not task.source_path.exists():
            return SlideTaskResult(
                slide_id=task.slide_id,
                jpg_path=None,
                status="missing",
            )

        # Import openslide here (in worker process)
        try:
            import openslide
        except ImportError:
            return SlideTaskResult(
                slide_id=task.slide_id,
                jpg_path=None,
                status="failed",
                error="openslide not installed: pip install openslide-python",
            )

        # Open slide and create thumbnail
        slide = openslide.OpenSlide(str(task.source_path))
        thumbnail = slide.get_thumbnail(task.size)
        slide.close()

        # Convert to RGB if necessary (some slides have RGBA)
        if thumbnail.mode == 'RGBA':
            thumbnail = thumbnail.convert('RGB')

        # Save as JPEG
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        thumbnail.save(task.output_path, 'JPEG', quality=85)

        return SlideTaskResult(
            slide_id=task.slide_id,
            jpg_path=task.output_path,
            status="processed",
        )

    except Exception as e:
        return SlideTaskResult(
            slide_id=task.slide_id,
            jpg_path=None,
            status="failed",
            error=str(e),
        )


class SlideProcessor:
    """Parallel processing of whole slide images.

    Creates thumbnail images from SVS files using multiprocessing.
    Each slide is processed by a separate worker.

    Usage:
        processor = SlideProcessor(n_workers=4)
        result = processor.process_slides(
            df=slide_df,
            output_dir=config.thumbnails_dir,
        )
    """

    def __init__(self, n_workers: Optional[int] = None):
        """
        Args:
            n_workers: Number of worker processes. Defaults to CPU count.
        """
        self.n_workers = n_workers or cpu_count()

    def process_slides(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        size: Tuple[int, int] = (512, 512),
        slide_path_col: str = "slide_local_path",
        slide_id_col: str = "slide_id",
    ) -> ProcessingResult:
        """Process slides in parallel, creating thumbnails.

        Required DataFrame columns:
            - slide_local_path: Path to SVS file (from etl.add_local_paths())
            - slide_id: Unique identifier for naming output JPG

        Args:
            df: DataFrame with required columns (see above)
            output_dir: Directory to save JPG thumbnails
            size: Thumbnail size (width, height)
            slide_path_col: Column name for slide file path (default: slide_local_path)
            slide_id_col: Column name for slide ID (default: slide_id)

        Returns:
            ProcessingResult with:
                - df: DataFrame with new 'jpg_path' column added
                - processed: Count of newly created thumbnails
                - skipped: Count of existing thumbnails (not recreated)
                - failed: Count of errors during processing
                - missing: Count of source files not found
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build task list
        tasks = self._build_tasks(df, output_dir, size, slide_path_col, slide_id_col)

        logger.info(f"Processing {len(tasks)} slides with {self.n_workers} workers")

        # Process in parallel
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(_process_single_slide, tasks)

        # Aggregate results
        return self._aggregate_results(df, results, slide_id_col)

    def _build_tasks(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        size: Tuple[int, int],
        slide_path_col: str,
        slide_id_col: str,
    ) -> List[SlideTask]:
        """Build list of processing tasks from DataFrame."""
        tasks = []

        for _, row in df.iterrows():
            slide_id = row[slide_id_col]
            source_path = row[slide_path_col]

            # Handle Path objects or strings
            if source_path is not None:
                source_path = Path(source_path)
            else:
                source_path = Path("")  # Will fail exists() check

            output_path = output_dir / f"{slide_id}.jpg"

            tasks.append(SlideTask(
                slide_id=slide_id,
                source_path=source_path,
                output_path=output_path,
                size=size,
            ))

        return tasks

    def _aggregate_results(
        self,
        df: pd.DataFrame,
        results: List[SlideTaskResult],
        slide_id_col: str,
    ) -> ProcessingResult:
        """Aggregate worker results and update DataFrame."""
        # Build lookup: slide_id -> result
        result_lookup: Dict[str, SlideTaskResult] = {
            r.slide_id: r for r in results
        }

        # Count statistics
        processed = sum(1 for r in results if r.status == "processed")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")
        missing = sum(1 for r in results if r.status == "missing")

        # Log summary
        logger.info(f"Processing complete: {processed} processed, {skipped} skipped, {failed} failed, {missing} missing")

        # Log failures
        for r in results:
            if r.status == "failed" and r.error:
                logger.error(f"Failed to process {r.slide_id}: {r.error}")

        # Add jpg_path column to DataFrame
        df = df.copy()
        df["jpg_path"] = df[slide_id_col].map(
            lambda sid: result_lookup.get(sid, SlideTaskResult(sid, None, "unknown")).jpg_path
        )

        return ProcessingResult(
            df=df,
            processed=processed,
            skipped=skipped,
            failed=failed,
            missing=missing,
        )
