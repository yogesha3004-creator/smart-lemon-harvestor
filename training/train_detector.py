"""Training script for YOLOv8 lemon detector.

Supports CLI configuration, YAML configuration files, fine-tuning YOLOv8
models on lemon detection datasets, and logging best model metrics.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

# Add repository root to Python path for script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.detector import LemonDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_detector")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for detector training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 Lemon Detector for Smart Lemon Harvester.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to YOLO data.yaml configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLOv8 model variant or weights path (e.g., yolov8n, yolov8s, yolov8m, yolov8n.pt).",
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
        "--img-size",
        type=int,
        default=None,
        help="Input image resolution in pixels (square).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Directory to save experiment runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment run name.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML training configuration file (e.g., configs/detection_config.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training on ('cuda', '0', 'cpu').",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs without validation improvement).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of data loader worker threads.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent run checkpoint.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from a YAML file if provided.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary or empty dict if not provided.
    """
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not path.exists():
        logger.warning(f"Config file not found: {path}. Proceeding with default values.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            logger.info(f"Loaded training configuration from {path}")
            return cfg
    except Exception as e:
        logger.error(f"Failed to read config file {path}: {e}")
        return {}


def resolve_training_parameters(
    args: argparse.Namespace, config: Dict[str, Any]
) -> Tuple[str, str, int, int, int, str, str, Dict[str, Any]]:
    """Merge CLI arguments and YAML configuration with priority to CLI.

    Args:
        args: Parsed CLI arguments.
        config: Loaded YAML configuration dictionary.

    Returns:
        Tuple of (data_path, model_path, epochs, batch_size, img_size, project, name, extra_kwargs).
    """
    # 1. Model resolution
    cfg_model = config.get("model", {})
    cfg_training = config.get("training", {})
    cfg_data = config.get("data", {})
    cfg_paths = config.get("paths", {})
    cfg_aug = config.get("augmentation", {})

    model = args.model
    if model is None:
        arch = cfg_model.get("architecture", "yolov8n")
        model = f"{arch}.pt" if not arch.endswith(".pt") else arch
    elif not model.endswith(".pt") and not model.endswith(".yaml"):
        model = f"{model}.pt"

    # 2. Epochs, batch size, img_size
    epochs = args.epochs if args.epochs is not None else cfg_training.get("epochs", 100)
    batch_size = (
        args.batch_size if args.batch_size is not None else cfg_training.get("batch_size", 16)
    )
    img_size = args.img_size if args.img_size is not None else cfg_model.get("input_size", 640)

    # 3. Project and Run Name
    project = (
        args.project
        if args.project is not None
        else cfg_paths.get("results_dir", "runs/detection")
    )
    name = args.name if args.name is not None else "train"

    # 4. Data YAML resolution
    data_path = args.data
    if data_path is None:
        # Check config data settings
        dataset_path = cfg_data.get("dataset_path", "data/detection")
        default_yaml = Path(dataset_path) / "data.yaml"
        if not default_yaml.is_absolute():
            default_yaml = REPO_ROOT / default_yaml
        data_path = str(default_yaml)
    else:
        p = Path(data_path)
        if not p.is_absolute():
            data_path = str(REPO_ROOT / p)

    # 5. Extra kwargs for YOLO.train()
    extra_kwargs: Dict[str, Any] = {}

    if "learning_rate" in cfg_training:
        extra_kwargs["lr0"] = float(cfg_training["learning_rate"])
    if "optimizer" in cfg_training:
        extra_kwargs["optimizer"] = cfg_training["optimizer"]
    if "save_period" in cfg_training:
        extra_kwargs["save_period"] = int(cfg_training["save_period"])

    patience = args.patience if args.patience is not None else cfg_training.get("patience", 20)
    extra_kwargs["patience"] = patience

    if args.device is not None:
        extra_kwargs["device"] = args.device

    if args.workers is not None:
        extra_kwargs["workers"] = args.workers
    elif "num_workers" in cfg_training:
        extra_kwargs["workers"] = int(cfg_training["num_workers"])

    if args.resume:
        extra_kwargs["resume"] = True

    # Augmentation hyperparameters if present
    for k, v in cfg_aug.items():
        extra_kwargs[k] = v

    return data_path, model, epochs, batch_size, img_size, project, name, extra_kwargs


def extract_metrics(results: Any) -> Dict[str, float]:
    """Safely extract key evaluation metrics from YOLO training results.

    Args:
        results: Ultralytics Results / DetMetrics object or dict.

    Returns:
        Dictionary of extracted metrics (mAP50, mAP50_95, precision, recall).
    """
    metrics: Dict[str, float] = {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    if results is None:
        return metrics

    # Try accessing attributes from DetMetrics
    try:
        box = getattr(results, "box", None)
        if box is not None:
            metrics["mAP50"] = float(getattr(box, "map50", 0.0) or 0.0)
            metrics["mAP50_95"] = float(getattr(box, "map", 0.0) or 0.0)
            metrics["precision"] = float(getattr(box, "mp", 0.0) or 0.0)
            metrics["recall"] = float(getattr(box, "mr", 0.0) or 0.0)
            return metrics
    except Exception as e:
        logger.debug(f"Could not parse box metrics object: {e}")

    # Try results_dict
    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics["mAP50"] = float(
            results_dict.get("metrics/mAP50(B)", results_dict.get("mAP50", 0.0))
        )
        metrics["mAP50_95"] = float(
            results_dict.get("metrics/mAP50-95(B)", results_dict.get("mAP50-95", 0.0))
        )
        metrics["precision"] = float(
            results_dict.get("metrics/precision(B)", results_dict.get("precision", 0.0))
        )
        metrics["recall"] = float(
            results_dict.get("metrics/recall(B)", results_dict.get("recall", 0.0))
        )

    return metrics


def print_training_summary(
    model_name: str,
    data_yaml: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    save_dir: Path,
    best_pt: Path,
    last_pt: Path,
    metrics: Dict[str, float],
) -> None:
    """Print a formatted training summary table.

    Args:
        model_name: Base model weights / name.
        data_yaml: Path to data.yaml.
        epochs: Number of epochs trained.
        batch_size: Batch size used.
        img_size: Image size used.
        save_dir: Directory containing experiment artifacts.
        best_pt: Path to best.pt weights.
        last_pt: Path to last.pt weights.
        metrics: Dictionary of final metrics.
    """
    separator = "=" * 64
    subsep = "-" * 64

    logger.info("\n" + separator)
    logger.info("             LEMON DETECTOR TRAINING SUMMARY")
    logger.info(separator)
    logger.info(f" Model Architecture : {model_name}")
    logger.info(f" Dataset YAML       : {data_yaml}")
    logger.info(f" Epochs             : {epochs}")
    logger.info(f" Batch Size         : {batch_size}")
    logger.info(f" Image Size         : {img_size}x{img_size}")
    logger.info(f" Run Directory      : {save_dir}")
    logger.info(subsep)
    logger.info(f" Best Weights       : {best_pt} {'(Found)' if best_pt.exists() else '(Not Found)'}")
    logger.info(f" Last Weights       : {last_pt} {'(Found)' if last_pt.exists() else '(Not Found)'}")
    logger.info(subsep)
    logger.info(f" Best mAP@0.5       : {metrics.get('mAP50', 0.0):.4f}")
    logger.info(f" Best mAP@0.5:0.95  : {metrics.get('mAP50_95', 0.0):.4f}")
    logger.info(f" Precision          : {metrics.get('precision', 0.0):.4f}")
    logger.info(f" Recall             : {metrics.get('recall', 0.0):.4f}")
    logger.info(separator + "\n")


def save_best_weights_info(
    save_dir: Path,
    best_pt: Path,
    metrics: Dict[str, float],
    model_name: str,
    data_yaml: str,
) -> Path:
    """Save metadata and info for best.pt weights and copy to central weights directory.

    Args:
        save_dir: Output experiment directory.
        best_pt: Path to the generated best.pt weights.
        metrics: Dictionary of best validation metrics.
        model_name: Model architecture or initial weights name.
        data_yaml: Dataset YAML path.

    Returns:
        Path to the saved info JSON file.
    """
    info = {
        "model": model_name,
        "data_yaml": str(data_yaml),
        "best_weights_path": str(best_pt.resolve()) if best_pt.exists() else None,
        "metrics": metrics,
        "run_dir": str(save_dir.resolve()),
    }

    info_path = save_dir / "best_model_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Saved best model info to: {info_path}")

    # Copy to standard weights directory (weights/detection/best.pt) for convenience
    if best_pt.exists():
        standard_weights_dir = REPO_ROOT / "weights" / "detection"
        standard_weights_dir.mkdir(parents=True, exist_ok=True)
        dest_best = standard_weights_dir / "best.pt"
        try:
            shutil.copy2(best_pt, dest_best)
            logger.info(f"Copied best weights to standard deployment path: {dest_best}")
        except Exception as e:
            logger.warning(f"Could not copy best weights to {dest_best}: {e}")

    return info_path


def main() -> int:
    """Execute YOLOv8 Lemon Detector training workflow.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()
    config = load_yaml_config(args.config)

    data_path, model_name, epochs, batch_size, img_size, project, name, extra_kwargs = (
        resolve_training_parameters(args, config)
    )

    logger.info("Initializing LemonDetector training...")
    logger.info(f"Base model: {model_name}")
    logger.info(f"Dataset YAML: {data_path}")
    logger.info(f"Configuration: epochs={epochs}, batch={batch_size}, imgsz={img_size}")

    data_file = Path(data_path)
    if not data_file.exists():
        logger.error(f"Dataset YAML file not found: {data_file}")
        logger.error(
            "Please ensure dataset is downloaded and prepared. Run:\n"
            "  python data/download_dataset.py\n"
            "  python data/prepare_dataset.py"
        )
        return 1

    try:
        detector = LemonDetector(model_path=model_name)
    except Exception as e:
        logger.error(f"Failed to initialize LemonDetector with model '{model_name}': {e}")
        return 1

    try:
        logger.info("Starting training loop...")
        results = detector.train(
            data_yaml=str(data_file),
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            project=project,
            name=name,
            **extra_kwargs,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1

    # Locate output artifacts
    save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    metrics = extract_metrics(results)
    print_training_summary(
        model_name=model_name,
        data_yaml=str(data_file),
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        save_dir=save_dir,
        best_pt=best_pt,
        last_pt=last_pt,
        metrics=metrics,
    )

    save_best_weights_info(
        save_dir=save_dir,
        best_pt=best_pt,
        metrics=metrics,
        model_name=model_name,
        data_yaml=str(data_file),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
