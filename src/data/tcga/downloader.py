"""Download orchestration for TCGA data."""

import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class DownloadStatus(Enum):
    """Status of a download operation."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DownloadResult:
    """Result of a download operation."""
    status: DownloadStatus
    manifest_path: Path
    output_dir: Path
    files_total: int
    files_downloaded: int = 0
    error_message: Optional[str] = None


class TCGADownloader:
    """Orchestrates downloads using gdc-client.

    Handles:
    - Running gdc-client with manifests
    - Tracking download status
    - Resume/continuation of interrupted downloads (gdc-client handles this)

    Usage:
        downloader = TCGADownloader()

        # Download from manifest
        result = downloader.download_from_manifest(
            manifest_path=Path("manifests/slides_manifest.txt"),
            output_dir=Path("data/slides"),
        )
        print(f"Status: {result.status.value}")
        print(f"Downloaded: {result.files_downloaded}/{result.files_total}")

        # Check status without downloading
        status = downloader.check_download_status(output_dir, manifest_path)
    """

    def __init__(self, gdc_client_path: str = "gdc-client"):
        """
        Args:
            gdc_client_path: Path to gdc-client executable
        """
        self.gdc_client_path = gdc_client_path

    def download_from_manifest(
        self,
        manifest_path: Path,
        output_dir: Path,
        token_path: Optional[Path] = None,
        n_processes: int = 4,
    ) -> DownloadResult:
        """Download files from a manifest.

        If download was previously interrupted, gdc-client will automatically
        resume from where it left off.

        Args:
            manifest_path: Path to manifest file
            output_dir: Directory to download to
            token_path: Path to GDC token file (for controlled access data)
            n_processes: Number of parallel downloads

        Returns:
            DownloadResult with status and counts
        """
        manifest_path = Path(manifest_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Count files in manifest
        files_total = self._count_manifest_files(manifest_path)

        # Build command
        cmd = [
            self.gdc_client_path,
            "download",
            "-m", str(manifest_path),
            "-d", str(output_dir),
            "-n", str(n_processes),
        ]
        if token_path:
            cmd.extend(["-t", str(token_path)])

        try:
            # Run download (this will resume automatically if interrupted)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            files_downloaded = self._count_downloaded_files(output_dir)

            if result.returncode == 0:
                return DownloadResult(
                    status=DownloadStatus.COMPLETED,
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    files_total=files_total,
                    files_downloaded=files_downloaded,
                )
            else:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    files_total=files_total,
                    files_downloaded=files_downloaded,
                    error_message=result.stderr,
                )

        except FileNotFoundError:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                manifest_path=manifest_path,
                output_dir=output_dir,
                files_total=files_total,
                error_message=f"gdc-client not found at: {self.gdc_client_path}. "
                              f"Install with: pip install gdc-client",
            )

    def check_download_status(
        self,
        output_dir: Path,
        manifest_path: Path,
    ) -> DownloadResult:
        """Check status of a download (for resume detection).

        Args:
            output_dir: Download directory
            manifest_path: Manifest file

        Returns:
            DownloadResult with current status
        """
        manifest_path = Path(manifest_path)
        output_dir = Path(output_dir)

        files_total = self._count_manifest_files(manifest_path)
        files_downloaded = self._count_downloaded_files(output_dir)

        if files_downloaded == 0:
            status = DownloadStatus.NOT_STARTED
        elif files_downloaded < files_total:
            status = DownloadStatus.IN_PROGRESS
        else:
            status = DownloadStatus.COMPLETED

        return DownloadResult(
            status=status,
            manifest_path=manifest_path,
            output_dir=output_dir,
            files_total=files_total,
            files_downloaded=files_downloaded,
        )

    def _count_manifest_files(self, manifest_path: Path) -> int:
        """Count files in manifest (excluding header)."""
        with open(manifest_path) as f:
            return sum(1 for _ in f) - 1  # Subtract header

    def _count_downloaded_files(self, output_dir: Path) -> int:
        """Count completed downloads.

        gdc-client creates: <output_dir>/<uuid>/<filename>
        A download is complete if the UUID folder has a non-partial file.
        """
        if not output_dir.exists():
            return 0

        count = 0
        for uuid_dir in output_dir.iterdir():
            if uuid_dir.is_dir():
                # Check if there's a completed file inside (not .partial, not logs)
                for f in uuid_dir.iterdir():
                    if f.is_file() and f.name != "logs" and not f.suffix == ".partial":
                        count += 1
                        break
        return count
