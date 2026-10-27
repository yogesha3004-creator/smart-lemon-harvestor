"""Training script for multi-task lemon ripeness and quality classifier.

Trains an EfficientNet-B0 backbone with dual classification heads:
1. Ripeness Head: raw, semi_ripe, ripe
2. Quality Head: grade_a, grade_b, grade_c

Features:
- CLI and YAML configuration support (CLI overrides YAML)
- Cross-entropy loss for both heads with configurable loss weighting
- Adam optimizer with Cosine Annealing learning rate schedule
- Early stopping based on validation loss
- Model checkpointing saving model_state_dict, optimizer_state_dict, epoch, best_val_loss
- Training and validation metrics logging with tqdm progress bars
- Resumption from existing checkpoints
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import yaml
from tqdm import tqdm

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.classifier import MultiTaskLemonClassifier, RipenessClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_classifier")

RIPENESS_CLASSES = ["raw", "semi_ripe", "ripe"]
QUALITY_CLASSES = ["grade_a", "grade_b", "grade_c"]

RIPENESS_MAP = {
    "raw": 0,
    "unripe": 0,
    "green": 0,
    "semi_ripe": 1,
    "semi-ripe": 1,
    "semiripe": 1,
    "turning": 1,
    "ripe": 2,
    "yellow": 2,
}

QUALITY_MAP = {
    "grade_a": 0,
    "gradea": 0,
    "good_quality": 0,
    "good": 0,
    "grade_b": 1,
    "gradeb": 1,
    "medium_quality": 1,
    "grade_c": 2,
    "gradec": 2,
    "bad_quality": 2,
    "bad": 2,
    "damaged": 2,
}


class LemonClassificationDataset(Dataset):
    """Custom Dataset for multi-task lemon classification.

    Loads images and resolves both ripeness and quality labels from:
    1. Two-level class subdirectories (e.g. root/ripe/grade_a/img.jpg)
    2. Single-level class subdirectories with filename or default parsing
    3. Kaggle Lemon Quality directory structures (good_quality, bad_quality)
    4. Optional annotations.csv or labels.json
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """Initialize dataset from root directory.

        Args:
            root_dir: Directory containing images or class subfolders.
            transform: Torchvision image transformation pipeline.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self._index_dataset()

        if len(self.samples) == 0:
            logger.warning(f"No image files found in {self.root_dir}")

    def _index_dataset(self) -> None:
        """Scan directory and index all images with resolved (ripeness, quality) targets."""
        # Check for annotations file
        csv_file = self.root_dir / "annotations.csv"
        if csv_file.exists():
            self._load_from_csv(csv_file)
            return

        # Scan filesystem recursively
        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                ripeness_idx, quality_idx = self._resolve_labels(path)
                self.samples.append((path, ripeness_idx, quality_idx))

    def _resolve_labels(self, path: Path) -> Tuple[int, int]:
        """Infer ripeness and quality integer labels from file path and filename.

        Args:
            path: Absolute or relative file path.

        Returns:
            Tuple of (ripeness_idx, quality_idx).
        """
        path_str = str(path).lower().replace("\\", "/")
        parts = path.parts

        ripeness_idx = 2  # default: ripe
        quality_idx = 0   # default: grade_a

        # Search for ripeness hints in path parts
        for part in reversed(parts[:-1]):
            clean_part = part.lower().strip()
            if clean_part in RIPENESS_MAP:
                ripeness_idx = RIPENESS_MAP[clean_part]
                break

        # Search for quality hints in path parts
        for part in reversed(parts[:-1]):
            clean_part = part.lower().strip()
            if clean_part in QUALITY_MAP:
                quality_idx = QUALITY_MAP[clean_part]
                break

        # Also inspect filename for keywords if not found in folder names
        filename = path.stem.lower()
        for k, v in RIPENESS_MAP.items():
            if k in filename:
                ripeness_idx = v
                break

        for k, v in QUALITY_MAP.items():
            if k in filename:
                quality_idx = v
                break

        return ripeness_idx, quality_idx

    def _load_from_csv(self, csv_file: Path) -> None:
        """Load samples from CSV annotations file.

        Args:
            csv_file: Path to annotations.csv.
        """
        import csv

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get("image_path", row.get("filename", ""))
                img_path = self.root_dir / rel_path
                if not img_path.exists():
                    continue

                rip_val = row.get("ripeness", "ripe").lower()
                qual_val = row.get("quality", "grade_a").lower()

                rip_idx = RIPENESS_MAP.get(rip_val, 2)
                qual_idx = QUALITY_MAP.get(qual_val, 0)
                self.samples.append((img_path, rip_idx, qual_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        img_path, ripeness_target, quality_target = self.samples[idx]

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Fallback to black image in case of corrupted file
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, ripeness_target, quality_target


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for classifier training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train Multi-Task Lemon Classifier (Ripeness & Quality).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to classification dataset directory (containing train/val subdirectories).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML training configuration (e.g., configs/classifier_config.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume training from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (default: 0 on Windows, 2 on Linux).",
    )
    parser.add_argument(
        "--ripeness-weight",
        type=float,
        default=1.0,
        help="Loss weight multiplier for ripeness head.",
    )
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.5,
        help="Loss weight multiplier for quality head.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load classifier configuration from YAML.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        logger.warning(f"Config file not found: {path}. Using default parameters.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            logger.info(f"Loaded classifier configuration from {path}")
            return cfg
    except Exception as e:
        logger.error(f"Failed to read config file {path}: {e}")
        return {}


def build_transforms(
    input_size: int = 224,
    cfg_aug: Optional[Dict[str, Any]] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build data augmentation transforms for training and validation.

    Args:
        input_size: Square image resolution.
        cfg_aug: Augmentation config dictionary.

    Returns:
        Tuple of (train_transform, val_transform).
    """
    cfg_aug = cfg_aug or {}

    # Augmentation parameters
    rot_deg = float(cfg_aug.get("random_rotation", 15.0))
    scale_range = cfg_aug.get("random_resized_crop", {}).get("scale", [0.8, 1.0])
    jitter_cfg = cfg_aug.get("color_jitter", {})
    brightness = float(jitter_cfg.get("brightness", 0.2))
    contrast = float(jitter_cfg.get("contrast", 0.2))
    saturation = float(jitter_cfg.get("saturation", 0.2))
    hue = float(jitter_cfg.get("hue", 0.1))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(scale_range[0], scale_range[1])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=rot_deg),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def build_model(
    num_ripeness: int = 3,
    num_quality: int = 3,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> MultiTaskLemonClassifier:
    """Build the multi-task EfficientNet-B0 model.

    Args:
        num_ripeness: Number of ripeness categories.
        num_quality: Number of quality grades.
        dropout: Dropout rate in classification heads.
        pretrained: Whether to load ImageNet pretrained backbone weights.

    Returns:
        Instantiated MultiTaskLemonClassifier.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    backbone = models.efficientnet_b0(weights=weights)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    model = MultiTaskLemonClassifier(
        backbone=backbone,
        in_features=in_features,
        num_ripeness=num_ripeness,
        num_quality=num_quality,
        dropout=dropout,
    )
    return model


def prepare_data_loaders(
    data_dir_path: Path,
    batch_size: int,
    input_size: int,
    num_workers: int,
    cfg_aug: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build PyTorch DataLoaders for train, val, and test splits.

    Args:
        data_dir_path: Root classification directory.
        batch_size: Batch size for DataLoader.
        input_size: Input resolution.
        num_workers: Worker threads for data loading.
        cfg_aug: Augmentation config dictionary.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform, val_transform = build_transforms(input_size=input_size, cfg_aug=cfg_aug)

    train_dir = data_dir_path / "train"
    val_dir = data_dir_path / "val"
    if not val_dir.exists():
        val_dir = data_dir_path / "validation"
    test_dir = data_dir_path / "test"

    # If explicit train and val subdirectories exist
    if train_dir.exists() and val_dir.exists():
        train_dataset = LemonClassificationDataset(train_dir, transform=train_transform)
        val_dataset = LemonClassificationDataset(val_dir, transform=val_transform)
    else:
        # Load all from data_dir and split 80/20
        logger.info(
            f"Explicit train/val subdirectories not found. Loading all from {data_dir_path} and splitting 80/20."
        )
        full_dataset = LemonClassificationDataset(data_dir_path, transform=None)
        if len(full_dataset) == 0:
            raise ValueError(f"No valid images found in {data_dir_path}")

        val_size = max(1, int(0.2 * len(full_dataset)))
        train_size = len(full_dataset) - val_size

        from torch.utils.data import random_split
        train_sub, val_sub = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Wrap subsets with respective transforms
        class TransformedSubset(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                img_path, rip_target, qual_target = self.subset.dataset.samples[self.subset.indices[idx]]
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                tensor = self.transform(img_rgb)
                return tensor, rip_target, qual_target

        train_dataset = TransformedSubset(train_sub, train_transform)
        val_dataset = TransformedSubset(val_sub, val_transform)

    test_loader = None
    if test_dir.exists():
        test_dataset = LemonClassificationDataset(test_dir, transform=val_transform)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) > batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"DataLoaders prepared: Train={len(train_dataset)} samples ({len(train_loader)} batches), "
        f"Val={len(val_dataset)} samples ({len(val_loader)} batches)"
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_ripeness: nn.Module,
    criterion_quality: nn.Module,
    ripeness_weight: float,
    quality_weight: float,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float, float]:
    """Execute one training epoch.

    Args:
        model: MultiTaskLemonClassifier model.
        loader: Training DataLoader.
        optimizer: PyTorch optimizer.
        criterion_ripeness: Loss function for ripeness.
        criterion_quality: Loss function for quality.
        ripeness_weight: Loss weight multiplier for ripeness.
        quality_weight: Loss weight multiplier for quality.
        device: Active torch device.
        epoch: Current epoch index (1-based).
        total_epochs: Total number of epochs.

    Returns:
        Tuple of (average_train_loss, train_ripeness_accuracy, train_quality_accuracy).
    """
    model.train()
    running_loss = 0.0
    rip_correct = 0
    qual_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", leave=False)

    for images, rip_targets, qual_targets in pbar:
        images = images.to(device, non_blocking=True)
        rip_targets = rip_targets.to(device, non_blocking=True)
        qual_targets = qual_targets.to(device, non_blocking=True)

        batch_size = images.size(0)
        optimizer.zero_grad()

        rip_logits, qual_logits = model(images)

        loss_rip = criterion_ripeness(rip_logits, rip_targets)
        loss_qual = criterion_quality(qual_logits, qual_targets)
        total_batch_loss = (ripeness_weight * loss_rip) + (quality_weight * loss_qual)

        total_batch_loss.backward()
        optimizer.step()

        running_loss += total_batch_loss.item() * batch_size

        # Predictions
        rip_preds = torch.argmax(rip_logits, dim=1)
        qual_preds = torch.argmax(qual_logits, dim=1)

        rip_correct += (rip_preds == rip_targets).sum().item()
        qual_correct += (qual_preds == qual_targets).sum().item()
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{total_batch_loss.item():.4f}",
            "rip_acc": f"{(rip_correct / total_samples):.2%}",
            "qual_acc": f"{(qual_correct / total_samples):.2%}",
        })

    epoch_loss = running_loss / max(1, total_samples)
    epoch_rip_acc = rip_correct / max(1, total_samples)
    epoch_qual_acc = qual_correct / max(1, total_samples)

    return epoch_loss, epoch_rip_acc, epoch_qual_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_ripeness: nn.Module,
    criterion_quality: nn.Module,
    ripeness_weight: float,
    quality_weight: float,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float, float]:
    """Evaluate model on validation dataset.

    Args:
        model: MultiTaskLemonClassifier model.
        loader: Validation DataLoader.
        criterion_ripeness: Loss function for ripeness.
        criterion_quality: Loss function for quality.
        ripeness_weight: Loss weight multiplier for ripeness.
        quality_weight: Loss weight multiplier for quality.
        device: Active torch device.
        epoch: Current epoch index (1-based).
        total_epochs: Total number of epochs.

    Returns:
        Tuple of (average_val_loss, val_ripeness_accuracy, val_quality_accuracy).
    """
    model.eval()
    running_loss = 0.0
    rip_correct = 0
    qual_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Valid]", leave=False)

    for images, rip_targets, qual_targets in pbar:
        images = images.to(device, non_blocking=True)
        rip_targets = rip_targets.to(device, non_blocking=True)
        qual_targets = qual_targets.to(device, non_blocking=True)

        batch_size = images.size(0)

        rip_logits, qual_logits = model(images)

        loss_rip = criterion_ripeness(rip_logits, rip_targets)
        loss_qual = criterion_quality(qual_logits, qual_targets)
        total_batch_loss = (ripeness_weight * loss_rip) + (quality_weight * loss_qual)

        running_loss += total_batch_loss.item() * batch_size

        rip_preds = torch.argmax(rip_logits, dim=1)
        qual_preds = torch.argmax(qual_logits, dim=1)

        rip_correct += (rip_preds == rip_targets).sum().item()
        qual_correct += (qual_preds == qual_targets).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_rip_acc = rip_correct / max(1, total_samples)
    epoch_qual_acc = qual_correct / max(1, total_samples)

    return epoch_loss, epoch_rip_acc, epoch_qual_acc


def print_training_summary(
    best_epoch: int,
    best_val_loss: float,
    best_rip_acc: float,
    best_qual_acc: float,
    total_epochs: int,
    output_dir: Path,
) -> None:
    """Print formatted summary of the completed training run.

    Args:
        best_epoch: Epoch that achieved lowest validation loss.
        best_val_loss: Lowest validation loss achieved.
        best_rip_acc: Ripeness accuracy at best epoch.
        best_qual_acc: Quality accuracy at best epoch.
        total_epochs: Total epochs executed.
        output_dir: Directory containing saved checkpoints.
    """
    sep = "=" * 64
    logger.info("\n" + sep)
    logger.info("           LEMON CLASSIFIER TRAINING SUMMARY")
    logger.info(sep)
    logger.info(f" Epochs Completed       : {total_epochs}")
    logger.info(f" Best Epoch             : {best_epoch}")
    logger.info(f" Best Validation Loss   : {best_val_loss:.4f}")
    logger.info(f" Ripeness Accuracy      : {best_rip_acc:.2%}")
    logger.info(f" Quality Accuracy       : {best_qual_acc:.2%}")
    logger.info(f" Checkpoints Directory  : {output_dir.resolve()}")
    logger.info(f" Best Checkpoint File   : {output_dir / 'best_classifier.pth'}")
    logger.info(f" Last Checkpoint File   : {output_dir / 'last_classifier.pth'}")
    logger.info(sep + "\n")


def main() -> int:
    """Main execution function for classifier training.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()
    config = load_yaml_config(args.config)

    cfg_model = config.get("model", {})
    cfg_training = config.get("training", {})
    cfg_data = config.get("data", {})
    cfg_paths = config.get("paths", {})
    cfg_aug = config.get("augmentation", {})

    # Resolve parameters (CLI overrides YAML config)
    epochs = args.epochs if args.epochs is not None else cfg_training.get("epochs", 50)
    batch_size = (
        args.batch_size if args.batch_size is not None else cfg_training.get("batch_size", 32)
    )
    lr = args.lr if args.lr is not None else cfg_training.get("learning_rate", 0.001)
    weight_decay = (
        args.weight_decay if args.weight_decay is not None else cfg_training.get("weight_decay", 1e-4)
    )
    patience = (
        args.patience if args.patience is not None else cfg_training.get("patience", 10)
    )
    dropout = float(cfg_model.get("dropout", 0.3))
    input_size = int(cfg_model.get("input_size", 224))
    num_ripeness = int(cfg_model.get("num_ripeness_classes", 3))
    num_quality = int(cfg_model.get("num_quality_classes", 3))

    # Workers resolution
    if args.num_workers is not None:
        num_workers = args.num_workers
    elif "num_workers" in cfg_training:
        num_workers = int(cfg_training["num_workers"])
    else:
        # Default 0 on Windows to avoid multiprocessing issues
        num_workers = 0 if os.name == "nt" else 2

    # Output directory
    output_dir_str = (
        args.output_dir
        if args.output_dir is not None
        else cfg_paths.get("weights_dir", "weights/classifier")
    )
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset directory
    data_dir_str = (
        args.data_dir
        if args.data_dir is not None
        else cfg_data.get("dataset_path", "data/classification")
    )
    data_dir = Path(data_dir_str)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    if not data_dir.exists():
        logger.error(f"Classification data directory not found: {data_dir}")
        logger.error(
            "Please download and prepare the dataset first:\n"
            "  python data/download_dataset.py\n"
            "  python data/prepare_dataset.py"
        )
        return 1

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Build DataLoaders
    try:
        train_loader, val_loader, test_loader = prepare_data_loaders(
            data_dir_path=data_dir,
            batch_size=batch_size,
            input_size=input_size,
            num_workers=num_workers,
            cfg_aug=cfg_aug,
        )
    except Exception as e:
        logger.error(f"Failed to prepare data loaders: {e}", exc_info=True)
        return 1

    # Instantiate model
    pretrained_backbone = cfg_model.get("pretrained", True)
    model = build_model(
        num_ripeness=num_ripeness,
        num_quality=num_quality,
        dropout=dropout,
        pretrained=pretrained_backbone,
    )
    model.to(device)

    # Loss functions & Optimizer
    criterion_ripeness = nn.CrossEntropyLoss()
    criterion_quality = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 1
    best_rip_acc = 0.0
    best_qual_acc = 0.0
    patience_counter = 0

    # Resumption
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path
        if resume_path.exists():
            logger.info(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            logger.info(f"Resumed at epoch {start_epoch} with best_val_loss={best_val_loss:.4f}")
        else:
            logger.warning(f"Resume checkpoint not found: {resume_path}. Starting from scratch.")

    logger.info(f"Starting training for {epochs} epochs (learning rate: {lr})...")
    history: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_loss, train_rip_acc, train_qual_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion_ripeness=criterion_ripeness,
            criterion_quality=criterion_quality,
            ripeness_weight=args.ripeness_weight,
            quality_weight=args.quality_weight,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        val_loss, val_rip_acc, val_qual_acc = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion_ripeness=criterion_ripeness,
            criterion_quality=criterion_quality,
            ripeness_weight=args.ripeness_weight,
            quality_weight=args.quality_weight,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] "
            f"Train Loss: {train_loss:.4f} (Rip: {train_rip_acc:.2%}, Qual: {train_qual_acc:.2%}) | "
            f"Val Loss: {val_loss:.4f} (Rip: {val_rip_acc:.2%}, Qual: {val_qual_acc:.2%}) | "
            f"LR: {current_lr:.6f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ripeness_acc": train_rip_acc,
            "train_quality_acc": train_qual_acc,
            "val_loss": val_loss,
            "val_ripeness_acc": val_rip_acc,
            "val_quality_acc": val_qual_acc,
            "lr": current_lr,
        }
        history.append(epoch_record)

        # Checkpoint: Last epoch
        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "val_ripeness_acc": val_rip_acc,
            "val_quality_acc": val_qual_acc,
            "classes": {
                "ripeness": RIPENESS_CLASSES,
                "quality": QUALITY_CLASSES,
            },
        }
        torch.save(last_ckpt, output_dir / "last_classifier.pth")

        # Checkpoint: Best validation loss
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            best_rip_acc = val_rip_acc
            best_qual_acc = val_qual_acc
            patience_counter = 0

            best_ckpt = copy.deepcopy(last_ckpt)
            best_ckpt["best_val_loss"] = best_val_loss
            torch.save(best_ckpt, output_dir / "best_classifier.pth")
            logger.info(f"--> Saved new best model (improved by {improvement:.4f}) to {output_dir / 'best_classifier.pth'}")
        else:
            patience_counter += 1
            logger.info(f"Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs without validation improvement.")
                break

    # Save training history
    history_file = output_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_file}")

    print_training_summary(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_rip_acc=best_rip_acc,
        best_qual_acc=best_qual_acc,
        total_epochs=epoch,
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
