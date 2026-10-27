#!/usr/bin/env python3
"""
Dataset Downloader for Smart Lemon Harvester.

This module provides automated and guided downloading for lemon detection
and quality classification datasets:
1. Kaggle Lemon Quality Dataset (via Google Drive mirror with gdown)
2. Custom Citrus / Lemon Detection Dataset (via Roboflow Universe)

Usage Examples:
    # Download Kaggle Lemon Quality Dataset
    python data/download_dataset.py --dataset kaggle-lemon-quality

    # Print Roboflow download instructions and code snippet
    python data/download_dataset.py --dataset roboflow-citrus

    # Run all dataset download routines
    python data/download_dataset.py --dataset all

    # Custom output directory
    python data/download_dataset.py --dataset kaggle-lemon-quality --output-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lemon_dataset_downloader")

# Default Google Drive File ID for Kaggle Lemon Quality Dataset mirror
# Users may override via --drive-id or --drive-url
DEFAULT_LEMON_QUALITY_DRIVE_ID = "1gM5m_Q_5k6PzE9I8cM7Q0jK2xL3vN4pO"


def download_with_gdown(
    file_id: Optional[str],
    url: Optional[str],
    destination_zip: Path,
) -> bool:
    """Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID string.
        url: Full Google Drive shareable URL.
        destination_zip: Local destination file path.

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    try:
        import gdown
    except ImportError:
        logger.error(
            "The 'gdown' package is not installed. Please install it with:\n"
            "    pip install gdown"
        )
        return False

    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    target_str = str(destination_zip)

    try:
        logger.info("Connecting to Google Drive...")
        if url:
            logger.info("Downloading from URL: %s", url)
            output = gdown.download(url=url, output=target_str, quiet=False, fuzzy=True)
        elif file_id:
            logger.info("Downloading using Drive ID: %s", file_id)
            gdown_url = f"https://drive.google.com/uc?id={file_id}"
            output = gdown.download(url=gdown_url, output=target_str, quiet=False, fuzzy=True)
        else:
            logger.error("No valid Google Drive URL or File ID provided.")
            return False

        if output and Path(output).exists() and Path(output).stat().st_size > 0:
            logger.info("Download completed successfully: %s", output)
            return True
        else:
            logger.error("Download failed or output file is empty.")
            return False
    except Exception as exc:
        logger.error("Failed to download via gdown: %s", exc)
        return False


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Safely extract a ZIP archive to a target directory.

    Args:
        zip_path: Path to the .zip archive.
        extract_dir: Destination folder path.

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    if not zip_path.exists():
        logger.error("ZIP archive does not exist: %s", zip_path)
        return False

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting '%s' to '%s'...", zip_path.name, extract_dir)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = archive.namelist()
            total_files = len(members)
            logger.info("Archive contains %d files/folders.", total_files)

            # Extract all files
            archive.extractall(extract_dir)

        logger.info("Extraction complete! Extracted to: %s", extract_dir)
        return True
    except zipfile.BadZipFile:
        logger.error("The downloaded file '%s' is not a valid ZIP file.", zip_path)
        return False
    except Exception as exc:
        logger.error("Unexpected error during archive extraction: %s", exc)
        return False


def download_kaggle_lemon_quality(
    output_dir: Path,
    drive_id: Optional[str] = None,
    drive_url: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Download and extract the Kaggle Lemon Quality Dataset.

    Target directory layout:
        output_dir/lemon-quality/
            ├── good_quality/
            ├── bad_quality/
            └── empty_background/

    Args:
        output_dir: Root raw data directory (e.g. data/raw).
        drive_id: Google Drive file ID.
        drive_url: Google Drive file URL.
        force: Whether to re-download if dataset directory already exists.

    Returns:
        bool: True if downloaded and extracted successfully.
    """
    dataset_extract_dir = output_dir / "lemon-quality"
    zip_dest = output_dir / "lemon-quality-dataset.zip"

    if dataset_extract_dir.exists() and not force:
        existing_files = list(dataset_extract_dir.glob("**/*.*"))
        if len(existing_files) > 10:
            logger.info(
                "Dataset already exists at '%s' with %d files. Use --force to re-download.",
                dataset_extract_dir,
                len(existing_files),
            )
            return True

    logger.info("Preparing to download Kaggle Lemon Quality Dataset...")
    active_drive_id = drive_id or (DEFAULT_LEMON_QUALITY_DRIVE_ID if not drive_url else None)

    success = download_with_gdown(
        file_id=active_drive_id,
        url=drive_url,
        destination_zip=zip_dest,
    )

    if not success:
        logger.warning(
            "\n"
            "====================================================================\n"
            "NOTE: Automated Google Drive mirror download was not completed.\n"
            "Alternative download methods:\n"
            "1. Kaggle CLI:\n"
            "   pip install kaggle\n"
            "   kaggle datasets download -d yusufemir/lemon-quality-dataset -p %s\n"
            "   unzip %s/lemon-quality-dataset.zip -d %s\n"
            "\n"
            "2. Manual Download:\n"
            "   Visit https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset\n"
            "   Download archive and unzip into: %s\n"
            "\n"
            "3. Custom Google Drive link:\n"
            "   python data/download_dataset.py --dataset kaggle-lemon-quality --drive-id <YOUR_DRIVE_ID>\n"
            "====================================================================",
            output_dir,
            output_dir,
            dataset_extract_dir,
            dataset_extract_dir,
        )
        return False

    extract_success = extract_zip(zip_dest, dataset_extract_dir)
    if extract_success:
        logger.info("Kaggle Lemon Quality dataset is ready at: %s", dataset_extract_dir)
    return extract_success


def print_roboflow_instructions(output_dir: Path) -> None:
    """Print step-by-step instructions for downloading detection datasets from Roboflow.

    Args:
        output_dir: Path where raw datasets are stored.
    """
    citrus_dir = output_dir / "roboflow-citrus"
    banner = "=" * 76
    print("\n" + banner)
    print(" ROBOFLOW CITRUS / LEMON DETECTION DATASET SETUP INSTRUCTIONS ")
    print(banner)
    print(
        f"""
The Lemon Harvester detection model uses YOLOv8 format bounding box annotations.
You can download annotated lemon / citrus detection datasets from Roboflow Universe.

Recommended datasets on Roboflow Universe:
  * Citrus Fruit Detection (lemon, orange, grapefruit)
  * Lemon Fruit Ripeness Detection

Quick Setup Steps:
1. Install the Roboflow Python SDK:
       pip install roboflow

2. Sign in to https://universe.roboflow.com and obtain your API key from
   your Account Settings (Roboflow API Key).

3. Use the following Python snippet to download the dataset in YOLOv8 format:

----------------------------------------------------------------------------
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("citrus-farm").project("lemon-detection")
version = project.version(1)
dataset = version.download("yolov8", location="{citrus_dir.as_posix()}")
----------------------------------------------------------------------------

4. Alternatively, download the YOLOv8 ZIP directly from Roboflow Universe
   and extract it to:
       {citrus_dir.as_posix()}

5. Next step - Prepare YOLO splits & data.yaml:
       python data/prepare_dataset.py --task detection --source-dir {citrus_dir.as_posix()}
"""
    )
    print(banner + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download datasets for Smart Lemon Harvester AI pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["kaggle-lemon-quality", "roboflow-citrus", "all"],
        default="kaggle-lemon-quality",
        help="Dataset to download or configure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded raw datasets.",
    )
    parser.add_argument(
        "--drive-id",
        type=str,
        default=None,
        help="Optional custom Google Drive File ID for kaggle-lemon-quality.",
    )
    parser.add_argument(
        "--drive-url",
        type=str,
        default=None,
        help="Optional custom Google Drive URL for kaggle-lemon-quality.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if target dataset already exists.",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function.

    Returns:
        int: Exit status code (0 for success, non-zero for failure).
    """
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting dataset acquisition: dataset=%s, output_dir=%s", args.dataset, output_dir)

    status = 0
    if args.dataset in ("kaggle-lemon-quality", "all"):
        success = download_kaggle_lemon_quality(
            output_dir=output_dir,
            drive_id=args.drive_id,
            drive_url=args.drive_url,
            force=args.force,
        )
        if not success and args.dataset != "all":
            status = 1

    if args.dataset in ("roboflow-citrus", "all"):
        print_roboflow_instructions(output_dir=output_dir)

    logger.info("Process finished with status code %d.", status)
    return status


if __name__ == "__main__":
    sys.exit(main())
