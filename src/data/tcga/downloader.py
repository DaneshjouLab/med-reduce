"""Download orchestration for TCGA data via the GDC REST API."""

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


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
    """Downloads TCGA files directly from the GDC REST API.

    Uses https://api.gdc.cancer.gov/data/<uuid> to download files
    listed in a manifest. Supports parallel downloads, resume on
    interruption, and optional MD5 verification.

    Usage:
        downloader = TCGADownloader()

        result = downloader.download_from_manifest(
            manifest_path=Path("manifests/slides_manifest.txt"),
            output_dir=Path("data/slides"),
        )
        print(f"Status: {result.status.value}")
        print(f"Downloaded: {result.files_downloaded}/{result.files_total}")
    """

    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size

    def download_from_manifest(
        self,
        manifest_path: Path,
        output_dir: Path,
        token_path: Optional[Path] = None,
        n_processes: int = 4,
    ) -> DownloadResult:
        """Download files listed in a manifest via the GDC API.

        Skips files that are already downloaded. Supports parallel
        downloads and optional MD5 verification.
        """
        manifest_path = Path(manifest_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        entries = self._parse_manifest(manifest_path)
        files_total = len(entries)

        headers = {}
        if token_path:
            token = Path(token_path).read_text().strip()
            headers["X-Auth-Token"] = token

        # Filter to only files not yet downloaded
        to_download = []
        for file_id, filename, md5, size, state in entries:
            file_dir = output_dir / file_id
            file_path = file_dir / filename
            if file_path.exists():
                continue
            to_download.append((file_id, filename, md5, size))

        if not to_download:
            logger.info("All %d files already downloaded", files_total)
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                manifest_path=manifest_path,
                output_dir=output_dir,
                files_total=files_total,
                files_downloaded=files_total,
            )

        logger.info("Downloading %d/%d files (%d already exist)",
                     len(to_download), files_total, files_total - len(to_download))

        failed = []
        downloaded = 0
        # TODO: this is erroneous and will work for threading, but not for slide processing as generally the slide tools control one FD at a time. 
        with ThreadPoolExecutor(max_workers=n_processes) as pool:
            futures = {
                pool.submit(
                    self._download_file, file_id, filename, md5, output_dir, headers
                ): (file_id, filename)
                for file_id, filename, md5, size in to_download
            }
            for future in as_completed(futures):
                file_id, filename = futures[future]
                try:
                    success = future.result()
                    if success:
                        downloaded += 1
                    else:
                        failed.append(file_id)
                except Exception as e:
                    logger.error("Failed %s: %s", file_id, e)
                    failed.append(file_id)

                total_done = downloaded + (files_total - len(to_download))
                if downloaded % 50 == 0 and downloaded > 0:
                    logger.info("Progress: %d/%d downloaded", total_done, files_total)

        files_downloaded = self._count_downloaded_files(output_dir)
        error_msg = None
        if failed:
            error_msg = f"{len(failed)} files failed to download"
            logger.error(error_msg)

        status = DownloadStatus.COMPLETED if not failed else DownloadStatus.FAILED

        return DownloadResult(
            status=status,
            manifest_path=manifest_path,
            output_dir=output_dir,
            files_total=files_total,
            files_downloaded=files_downloaded,
            error_message=error_msg,
        )

    def _download_file(
        self,
        file_id: str,
        filename: str,
        md5: str,
        output_dir: Path,
        headers: dict,
    ) -> bool:
        """Download a single file from the GDC API."""
        file_dir = output_dir / file_id
        file_path = file_dir / filename
        partial_path = file_dir / f"{filename}.partial"

        file_dir.mkdir(parents=True, exist_ok=True)

        url = f"{GDC_DATA_URL}/{file_id}"

        try:
            with requests.get(url, headers=headers, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(partial_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=self.chunk_size):
                        f.write(chunk)

            # MD5 verification if available
            if md5:
                actual_md5 = self._md5sum(partial_path)
                if actual_md5 != md5:
                    logger.error("MD5 mismatch for %s: expected %s, got %s",
                                 filename, md5, actual_md5)
                    partial_path.unlink(missing_ok=True)
                    return False

            partial_path.rename(file_path)
            return True

        except Exception as e:
            logger.error("Download error for %s (%s): %s", filename, file_id, e)
            partial_path.unlink(missing_ok=True)
            return False

    def check_download_status(
        self,
        output_dir: Path,
        manifest_path: Path,
    ) -> DownloadResult:
        """Check status of a download (for resume detection)."""
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

    def _parse_manifest(self, manifest_path: Path) -> List[Tuple[str, str, str, int, str]]:
        """Parse manifest file into list of (id, filename, md5, size, state)."""
        entries = []
        # TOOD: fix this record of maintenance.. weak and flimsy, especiaaly if corrupted and does not currently support multi access reads and writes, ie no locking or atomic writes
        with open(manifest_path) as f:
            next(f)  # skip header
            # Construction is weak should outsource to an object.. not guaranteed to be consistently parsing and should split the logical operations into their own class .. ie i should plugin a schema for this stuff and scale. 
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                file_id = parts[0]
                filename = parts[1] if len(parts) > 1 else ""
                md5 = parts[2] if len(parts) > 2 else ""
                size = int(float(parts[3])) if len(parts) > 3 and parts[3] else 0
                state = parts[4] if len(parts) > 4 else ""
                entries.append((file_id, filename, md5, size, state))
        return entries

    def _count_manifest_files(self, manifest_path: Path) -> int:
        """Count files in manifest (excluding header)."""
        with open(manifest_path) as f:
            return sum(1 for _ in f) - 1

    def _count_downloaded_files(self, output_dir: Path) -> int:
        """Count completed downloads (uuid dirs with a non-partial file)."""
        if not output_dir.exists():
            return 0

        count = 0
        for uuid_dir in output_dir.iterdir():
            if uuid_dir.is_dir():
                for f in uuid_dir.iterdir():
                    if f.is_file() and f.name != "logs" and not f.suffix == ".partial":
                        count += 1
                        break
        return count

    def _md5sum(self, path: Path) -> str:
        """Compute MD5 of a file."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
