#!/usr/bin/env python3
"""
Dataset Preparation Pipeline for Smart Lemon Harvester.

This module processes raw downloaded or collected lemon datasets and prepares
them for machine learning training:
1. Detection Task: Prepares YOLOv8 formatted datasets with images/, labels/, and data.yaml
2. Classification Task: Organizes images into PyTorch ImageFolder format (raw/semi_ripe/ripe)
3. Split Task: Splits existing datasets into train, val, and test subsets
4. Both Task: Executes both detection and classification data preparation pipelines

Usage Examples:
    # Prepare both detection and classification datasets
    python data/prepare_dataset.py --task both

    # Prepare only YOLOv8 detection dataset
    python data/prepare_dataset.py --task detection --source-dir data/raw/roboflow-citrus

    # Prepare classification ImageFolder dataset with custom split ratios
    python data/prepare_dataset.py --task classification --source-dir data/raw/lemon-quality --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

    # Split a generic dataset into train/val/test splits
    python data/prepare_dataset.py --task split --source-dir data/raw/lemon-quality --output-dir data/classification
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lemon_dataset_preparer")

# Graceful tqdm fallback if not installed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc: Optional[str] = None, total: Optional[int] = None, **kwargs):
        """Minimal fallback iterator when tqdm is unavailable."""
        items = list(iterable) if not hasattr(iterable, "__len__") else iterable
        total_count = total or len(items)
        if desc:
            logger.info("%s (total: %d)", desc, total_count)
        for item in items:
            yield item


# Valid image file extensions
IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Canonical ripeness/quality classes for the Smart Lemon Harvester classifier
CANONICAL_CLASSES: List[str] = ["raw", "semi_ripe", "ripe"]

# Intelligent folder-to-class mapping heuristics
CLASS_NAME_MAPPINGS: Dict[str, str] = {
    # Raw / Unripe / Green lemon mappings
    "raw": "raw",
    "unripe": "raw",
    "green": "raw",
    "green_lemon": "raw",
    "green_lemons": "raw",
    "bad_quality": "raw",  # In kaggle-lemon-quality, bad quality / immature
    "unripened": "raw",
    # Semi-ripe / Turning / Medium mappings
    "semi_ripe": "semi_ripe",
    "semi-ripe": "semi_ripe",
    "semiripe": "semi_ripe",
    "yellow_green": "semi_ripe",
    "green_yellow": "semi_ripe",
    "partially_ripe": "semi_ripe",
    "turning": "semi_ripe",
    "medium": "semi_ripe",
    # Ripe / Mature / Good quality mappings
    "ripe": "ripe",
    "yellow": "ripe",
    "yellow_lemon": "ripe",
    "yellow_lemons": "ripe",
    "good_quality": "ripe",
    "mature": "ripe",
    "fully_ripe": "ripe",
}


def find_image_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all image files in a directory.

    Args:
        directory: Directory path to scan.
        recursive: Whether to scan subdirectories recursively.

    Returns:
        List of Path objects for all matching images.
    """
    if not directory.exists() or not directory.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    return [
        path for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def split_items(
    items: Sequence[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Deterministically split a sequence of items into train, val, and test subsets.

    Args:
        items: List of items to partition.
        train_ratio: Proportion for training split.
        val_ratio: Proportion for validation split.
        test_ratio: Proportion for testing split.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_items, val_items, test_items).
    """
    item_list = list(items)
    if not item_list:
        return [], [], []

    rng = random.Random(seed)
    rng.shuffle(item_list)

    total_len = len(item_list)
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    # Give remainder to test split to ensure all items are preserved
    n_test = total_len - n_train - n_val

    train_items = item_list[:n_train]
    val_items = item_list[n_train : n_train + n_val]
    test_items = item_list[n_train + n_val :]

    return train_items, val_items, test_items


def normalize_class_name(folder_name: str) -> Optional[str]:
    """Map a folder name to one of the canonical classes (raw, semi_ripe, ripe).

    Args:
        folder_name: Source directory or class label name.

    Returns:
        Canonical class name string, or None if folder should be ignored (e.g. background).
    """
    normalized = folder_name.strip().lower().replace(" ", "_")
    if "empty" in normalized or "background" in normalized:
        return None  # Background samples handled separately if needed
    return CLASS_NAME_MAPPINGS.get(normalized, normalized)


def prepare_classification_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> bool:
    """Organize raw images into PyTorch ImageFolder structure with train/val/test splits.

    Target structure:
        output_dir/classification/
            ├── train/
            │   ├── raw/
            │   ├── semi_ripe/
            │   └── ripe/
            ├── val/
            │   ├── raw/
            │   ├── semi_ripe/
            │   └── ripe/
            └── test/
                ├── raw/
                ├── semi_ripe/
                └── ripe/

    Args:
        source_dir: Path to directory containing class subfolders.
        output_dir: Root data directory (e.g. data).
        train_ratio: Ratio of data for train split.
        val_ratio: Ratio of data for val split.
        test_ratio: Ratio of data for test split.
        seed: Random seed.

    Returns:
        bool: True if dataset preparation was successful.
    """
    logger.info("--- Preparing Classification Dataset (ImageFolder Format) ---")
    classification_dir = output_dir / "classification"

    # Search for source directory candidates if source_dir is generic 'data/raw'
    search_dirs = [source_dir]
    lemon_quality_dir = source_dir / "lemon-quality"
    if lemon_quality_dir.exists():
        search_dirs.insert(0, lemon_quality_dir)

    effective_source: Optional[Path] = None
    for candidate in search_dirs:
        if candidate.exists() and any(p.is_dir() for p in candidate.iterdir()):
            effective_source = candidate
            break

    if not effective_source:
        logger.warning(
            "Classification source directory '%s' contains no subdirectories or does not exist.",
            source_dir,
        )
        # Create standard directory structure so the project layout is complete
        for split in ["train", "val", "test"]:
            for cls_name in CANONICAL_CLASSES:
                (classification_dir / split / cls_name).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Created empty classification directory structure at: %s", classification_dir
        )
        return True

    logger.info("Scanning source classes in: %s", effective_source)

    # Group files by canonical class
    class_images: Dict[str, List[Path]] = defaultdict(list)
    for subfolder in effective_source.iterdir():
        if not subfolder.is_dir():
            continue

        target_class = normalize_class_name(subfolder.name)
        if target_class is None:
            logger.info("Skipping background folder: %s", subfolder.name)
            continue

        images = find_image_files(subfolder, recursive=True)
        logger.info(
            "Found %d images in '%s' -> mapped to class '%s'",
            len(images),
            subfolder.name,
            target_class,
        )
        class_images[target_class].extend(images)

    # Ensure all canonical classes exist in output directory
    for split in ["train", "val", "test"]:
        for cls_name in CANONICAL_CLASSES:
            (classification_dir / split / cls_name).mkdir(parents=True, exist_ok=True)

    # Partition each class and copy files
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_copied = 0

    for cls_name, images in class_images.items():
        train_imgs, val_imgs, test_imgs = split_items(
            images, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )

        split_dict = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
        for split_name, split_files in split_dict.items():
            dest_dir = classification_dir / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            desc = f"Copying {cls_name} [{split_name}]"
            for src_file in tqdm(split_files, desc=desc):
                dest_file = dest_dir / src_file.name
                # Avoid duplicate names by prefixing parent name if clash occurs
                if dest_file.exists():
                    dest_file = dest_dir / f"{src_file.parent.name}_{src_file.name}"
                shutil.copy2(src_file, dest_file)
                stats[split_name][cls_name] += 1
                total_copied += 1

    logger.info("Classification dataset prepared successfully at: %s", classification_dir)
    logger.info("Summary Statistics:")
    for split_name in ["train", "val", "test"]:
        split_total = sum(stats[split_name].values())
        class_breakdown = ", ".join(f"{c}: {stats[split_name][c]}" for c in stats[split_name])
        logger.info("  %s split: %d images (%s)", split_name, split_total, class_breakdown or "0 images")

    return True


def create_yolo_data_yaml(
    detection_dir: Path,
    class_names: Optional[List[str]] = None,
) -> Path:
    """Generate YOLOv8 data.yaml dataset configuration file.

    Args:
        detection_dir: Directory where YOLO dataset is located.
        class_names: List of class names (defaults to ['lemon']).

    Returns:
        Path to the generated data.yaml file.
    """
    classes = class_names or ["lemon"]
    yaml_path = detection_dir / "data.yaml"

    # Use forward slashes for cross-platform compatibility with Ultralytics YOLO
    dataset_root = detection_dir.resolve().as_posix()
    formatted_names = "[" + ", ".join(f"'{c}'" for c in classes) + "]"

    yaml_content = f"""# Ultralytics YOLOv8 Dataset Configuration
# Smart Lemon Harvester Vision Pipeline

path: {dataset_root}  # dataset root directory
train: images/train   # train images (relative to 'path')
val: images/val       # val images (relative to 'path')
test: images/test     # test images (relative to 'path')

# Classes
nc: {len(classes)}
names: {formatted_names}
"""
    with open(yaml_path, "w", encoding="utf-8") as file:
        file.write(yaml_content)

    logger.info("Generated YOLO data.yaml configuration at: %s", yaml_path)
    return yaml_path


def prepare_detection_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
    generate_pseudo_labels: bool = True,
) -> bool:
    """Convert and organize images/labels into YOLOv8 detection directory format.

    Target structure:
        output_dir/detection/
            ├── data.yaml
            ├── images/
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── labels/
                ├── train/
                ├── val/
                └── test/

    Args:
        source_dir: Source directory containing images and optional YOLO annotations.
        output_dir: Root data directory.
        train_ratio: Ratio of data for train split.
        val_ratio: Ratio of data for val split.
        test_ratio: Ratio of data for test split.
        seed: Random seed.
        class_names: Detection class list.
        generate_pseudo_labels: If True, generates default bounding boxes for unannotated lemons.

    Returns:
        bool: True if detection dataset prepared successfully.
    """
    logger.info("--- Preparing Detection Dataset (YOLOv8 Format) ---")
    detection_dir = output_dir / "detection"

    # Create directory structure
    for split in ["train", "val", "test"]:
        (detection_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (detection_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Search for candidate source directories
    candidate_dirs = [source_dir]
    for sub in ["roboflow-citrus", "lemon-quality", "detection"]:
        if (source_dir / sub).exists():
            candidate_dirs.insert(0, source_dir / sub)

    effective_source: Optional[Path] = None
    for candidate in candidate_dirs:
        if candidate.exists() and (find_image_files(candidate) or any(candidate.iterdir())):
            effective_source = candidate
            break

    if not effective_source:
        logger.warning(
            "Detection source directory '%s' not found. Creating empty YOLO structure.",
            source_dir,
        )
        create_yolo_data_yaml(detection_dir, class_names=class_names)
        return True

    logger.info("Searching for images in: %s", effective_source)
    image_files = find_image_files(effective_source, recursive=True)

    if not image_files:
        logger.warning("No image files found in %s.", effective_source)
        create_yolo_data_yaml(detection_dir, class_names=class_names)
        return True

    logger.info("Found %d images to prepare for YOLO detection.", len(image_files))

    train_imgs, val_imgs, test_imgs = split_items(
        image_files, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    split_dict = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    stats: Dict[str, int] = defaultdict(int)

    for split_name, img_paths in split_dict.items():
        img_dest_dir = detection_dir / "images" / split_name
        lbl_dest_dir = detection_dir / "labels" / split_name

        desc = f"Processing YOLO [{split_name}]"
        for img_path in tqdm(img_paths, desc=desc):
            # Target image path
            dest_img = img_dest_dir / img_path.name
            if dest_img.exists():
                dest_img = img_dest_dir / f"{img_path.parent.name}_{img_path.name}"
            shutil.copy2(img_path, dest_img)

            # Look for existing corresponding label (.txt) file
            label_candidate = img_path.with_suffix(".txt")
            alt_label_candidate = img_path.parent.parent / "labels" / (img_path.stem + ".txt")

            dest_label = lbl_dest_dir / (dest_img.stem + ".txt")

            if label_candidate.exists():
                shutil.copy2(label_candidate, dest_label)
            elif alt_label_candidate.exists():
                shutil.copy2(alt_label_candidate, dest_label)
            elif generate_pseudo_labels:
                # If image is an empty background, create an empty label file (YOLO negative sample)
                is_background = (
                    "empty" in img_path.stem.lower()
                    or "background" in img_path.parent.name.lower()
                    or "empty_background" in str(img_path).lower()
                )
                if is_background:
                    dest_label.write_text("", encoding="utf-8")
                else:
                    # Single centered lemon bounding box (class 0, x_center=0.5, y_center=0.5, w=0.75, h=0.75)
                    pseudo_bbox = "0 0.500000 0.500000 0.750000 0.750000\n"
                    dest_label.write_text(pseudo_bbox, encoding="utf-8")

            stats[split_name] += 1

    # Write data.yaml configuration
    create_yolo_data_yaml(detection_dir, class_names=class_names)

    logger.info("Detection dataset prepared successfully at: %s", detection_dir)
    logger.info(
        "Summary Statistics: train=%d, val=%d, test=%d images",
        stats["train"],
        stats["val"],
        stats["test"],
    )
    return True


def split_generic_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> bool:
    """Split an existing folder of images or subfolders into train/val/test splits.

    Args:
        source_dir: Directory containing images or class folders.
        output_dir: Destination directory.
        train_ratio: Ratio for train split.
        val_ratio: Ratio for val split.
        test_ratio: Ratio for test split.
        seed: Random seed.

    Returns:
        bool: True if split was performed successfully.
    """
    logger.info("--- Performing Generic Dataset Split ---")
    if not source_dir.exists():
        logger.error("Source directory does not exist: %s", source_dir)
        return False

    subdirs = [p for p in source_dir.iterdir() if p.is_dir() and p.name not in ("train", "val", "test")]

    if subdirs:
        # Source has class subfolders: split each class subfolder
        logger.info("Found %d class directories in %s to split.", len(subdirs), source_dir)
        for class_dir in subdirs:
            images = find_image_files(class_dir, recursive=False)
            if not images:
                images = find_image_files(class_dir, recursive=True)

            logger.info("Class '%s': %d images", class_dir.name, len(images))
            train_imgs, val_imgs, test_imgs = split_items(
                images, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
            )
            split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

            for split_name, img_list in split_map.items():
                split_class_dir = output_dir / split_name / class_dir.name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                for img in tqdm(img_list, desc=f"Splitting {class_dir.name} -> {split_name}"):
                    shutil.copy2(img, split_class_dir / img.name)
    else:
        # Source is a flat image folder
        images = find_image_files(source_dir, recursive=False)
        logger.info("Found %d flat images in %s to split.", len(images), source_dir)
        train_imgs, val_imgs, test_imgs = split_items(
            images, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

        for split_name, img_list in split_map.items():
            dest_dir = output_dir / split_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in tqdm(img_list, desc=f"Splitting -> {split_name}"):
                shutil.copy2(img, dest_dir / img.name)

    logger.info("Generic dataset splitting completed at: %s", output_dir)
    return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Dataset preparation and splitting pipeline for Smart Lemon Harvester.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["detection", "classification", "both", "split"],
        default="both",
        help="Preparation task to execute.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to source directory containing downloaded/raw data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root output directory where prepared datasets will be stored.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of dataset used for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proportion of dataset used for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of dataset used for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable data splits.",
    )
    parser.add_argument(
        "--no-pseudo-labels",
        action="store_true",
        help="Disable auto-generation of pseudo bounding box labels for unannotated images in detection task.",
    )

    return parser.parse_args()


def main() -> int:
    """Main pipeline execution function.

    Returns:
        int: Status code (0 for success, non-zero for error).
    """
    args = parse_args()

    # Validate split ratios
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-4:
        logger.error(
            "Split ratios must sum to 1.0 (received train=%.2f, val=%.2f, test=%.2f, sum=%.2f)",
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            ratio_sum,
        )
        return 1

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    logger.info("Initializing dataset preparation:")
    logger.info("  Task:        %s", args.task)
    logger.info("  Source:      %s", source_dir)
    logger.info("  Output:      %s", output_dir)
    logger.info(
        "  Ratios:      Train=%.2f, Val=%.2f, Test=%.2f (seed=%d)",
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    success = True
    try:
        if args.task in ("classification", "both"):
            success = prepare_classification_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            ) and success

        if args.task in ("detection", "both"):
            success = prepare_detection_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                generate_pseudo_labels=not args.no_pseudo_labels,
            ) and success

        if args.task == "split":
            success = split_generic_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            ) and success

        if success:
            logger.info("Pipeline executed successfully for task '%s'!", args.task)
            return 0
        else:
            logger.error("Pipeline encountered errors during execution.")
            return 1
    except Exception as exc:
        logger.exception("Unexpected error in dataset pipeline: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
