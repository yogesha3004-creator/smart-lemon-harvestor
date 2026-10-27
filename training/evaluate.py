"""Evaluation script for lemon detection and classification models.

Evaluates YOLOv8 lemon detector and/or multi-task EfficientNet classifier.
Computes mAP, per-class precision, recall, F1, and accuracy.
Generates confusion matrix heatmaps and saves all metrics to JSON.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.detector import LemonDetector
from models.classifier import MultiTaskLemonClassifier, RipenessClassifier
from training.train_classifier import (
    LemonClassificationDataset,
    RIPENESS_CLASSES,
    QUALITY_CLASSES,
    build_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model evaluation.

    Returns:
        Parsed CLI arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Lemon Detection and Classification Models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["detection", "classification", "both"],
        default="both",
        help="Evaluation task to execute: 'detection', 'classification', or 'both'.",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default=None,
        help="Path to trained YOLOv8 weights (.pt file).",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        help="Path to trained classifier checkpoint (.pth file).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset root, data.yaml (for detection), or ImageFolder directory (for classification).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/evaluation",
        help="Directory where evaluation metrics JSON, text reports, and plots are saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on ('test' or 'val').",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference resolution for detection evaluation.",
    )
    parser.add_argument(
        "--classifier-img-size",
        type=int,
        default=224,
        help="Inference resolution for classification evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for classification evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu'). Auto-detected if omitted.",
    )

    return parser.parse_args()


def resolve_detector_weights(custom_weights: Optional[str]) -> Path:
    """Locate existing detector weights with fallback heuristics.

    Args:
        custom_weights: User-provided weights path or None.

    Returns:
        Path to weights file.
    """
    if custom_weights:
        p = Path(custom_weights)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p

    candidates = [
        REPO_ROOT / "weights" / "detection" / "best.pt",
        REPO_ROOT / "runs" / "detection" / "train" / "weights" / "best.pt",
        REPO_ROOT / "yolov8n.pt",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    return Path("yolov8n.pt")


def resolve_classifier_weights(custom_weights: Optional[str]) -> Optional[Path]:
    """Locate existing classifier weights with fallback heuristics.

    Args:
        custom_weights: User-provided weights path or None.

    Returns:
        Path to weights file or None if not found.
    """
    if custom_weights:
        p = Path(custom_weights)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p

    candidates = [
        REPO_ROOT / "weights" / "classifier" / "best_classifier.pth",
        REPO_ROOT / "runs" / "classifier" / "best_classifier.pth",
        REPO_ROOT / "weights" / "classifier" / "last_classifier.pth",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    return None


def resolve_data_paths(
    data_arg: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve detection data.yaml path and classification directory path.

    Args:
        data_arg: Path passed via --data or None.

    Returns:
        Tuple of (detection_yaml_path, classification_dir_path).
    """
    detection_yaml: Optional[Path] = None
    classification_dir: Optional[Path] = None

    if data_arg:
        arg_path = Path(data_arg)
        if not arg_path.is_absolute():
            arg_path = REPO_ROOT / arg_path

        if arg_path.is_file() and arg_path.suffix.lower() in [".yaml", ".yml"]:
            detection_yaml = arg_path
        elif arg_path.is_dir():
            # Check if this dir has data.yaml
            if (arg_path / "data.yaml").exists():
                detection_yaml = arg_path / "data.yaml"
            elif (arg_path / "detection" / "data.yaml").exists():
                detection_yaml = arg_path / "detection" / "data.yaml"

            # Check for classification
            if (arg_path / "classification").exists():
                classification_dir = arg_path / "classification"
            else:
                classification_dir = arg_path

    # Fallbacks to standard repo structure
    if detection_yaml is None:
        default_det = REPO_ROOT / "data" / "detection" / "data.yaml"
        if default_det.exists():
            detection_yaml = default_det

    if classification_dir is None:
        default_cls = REPO_ROOT / "data" / "classification"
        if default_cls.exists():
            classification_dir = default_cls

    return detection_yaml, classification_dir


def evaluate_detector(
    weights_path: Path,
    data_yaml: Path,
    img_size: int = 640,
    split: str = "test",
) -> Dict[str, Any]:
    """Run evaluation for YOLOv8 lemon detector.

    Args:
        weights_path: Path to YOLO weights (.pt).
        data_yaml: Path to data.yaml.
        img_size: Evaluation image size.
        split: Dataset split ('val' or 'test').

    Returns:
        Dictionary of detection evaluation metrics.
    """
    logger.info(f"Evaluating LemonDetector on {data_yaml} (split: {split})...")
    logger.info(f"Detector weights: {weights_path}")

    detector = LemonDetector(model_path=str(weights_path))
    results = detector.evaluate(
        data_yaml=str(data_yaml),
        img_size=img_size,
        split=split,
    )

    metrics_summary = {
        "mAP50": results.get("mAP50", 0.0),
        "mAP50_95": results.get("mAP50_95", 0.0),
        "precision": results.get("precision", 0.0),
        "recall": results.get("recall", 0.0),
    }

    sep = "=" * 60
    logger.info("\n" + sep)
    logger.info("           DETECTION EVALUATION RESULTS")
    logger.info(sep)
    logger.info(f" Weights       : {weights_path}")
    logger.info(f" Split         : {split}")
    logger.info(f" mAP@0.5       : {metrics_summary['mAP50']:.4f}")
    logger.info(f" mAP@0.5:0.95  : {metrics_summary['mAP50_95']:.4f}")
    logger.info(f" Precision     : {metrics_summary['precision']:.4f}")
    logger.info(f" Recall        : {metrics_summary['recall']:.4f}")
    logger.info(sep + "\n")

    return {
        "task": "detection",
        "weights": str(weights_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "metrics": metrics_summary,
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: Path,
    cmap: str = "Blues",
) -> None:
    """Plot confusion matrix with seaborn heatmap and save to file.

    Args:
        cm: Confusion matrix square array.
        class_names: Ordered list of class labels.
        title: Plot title.
        output_path: File path to save PNG.
        cmap: Seaborn colormap name.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("True Class", fontsize=11, fontweight="medium")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="medium")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot: {output_path}")


def evaluate_classifier(
    weights_path: Optional[Path],
    data_dir: Path,
    output_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    split: str = "test",
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation for Multi-Task Lemon Classifier.

    Args:
        weights_path: Path to trained checkpoint (.pth) or None.
        data_dir: Root classification directory.
        output_dir: Directory to save plots and reports.
        img_size: Input image resolution.
        batch_size: Batch size for DataLoader.
        split: Dataset split ('test' or 'val').
        device_str: Compute device string.

    Returns:
        Dictionary of classification metrics and classification reports.
    """
    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Evaluating Multi-Task Classifier on device: {device}")

    # Determine split folder
    target_dir = data_dir / split
    if not target_dir.exists():
        logger.warning(f"Split directory {target_dir} not found. Trying 'val' split.")
        target_dir = data_dir / "val"
    if not target_dir.exists():
        logger.warning(f"Split directory 'val' not found. Evaluating on full directory {data_dir}.")
        target_dir = data_dir

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = LemonClassificationDataset(target_dir, transform=val_transform)
    if len(dataset) == 0:
        raise ValueError(f"No evaluation samples found in {target_dir}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
    )
    logger.info(f"Loaded {len(dataset)} evaluation samples ({len(loader)} batches)")

    # Build and load model
    model = build_model(num_ripeness=3, num_quality=3, pretrained=False)
    if weights_path and weights_path.exists():
        logger.info(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        logger.warning("No trained weights provided or file not found. Evaluating randomly initialized/backbone weights.")

    model.to(device)
    model.eval()

    all_rip_true: List[int] = []
    all_rip_pred: List[int] = []
    all_qual_true: List[int] = []
    all_qual_pred: List[int] = []

    with torch.no_grad():
        for images, rip_targets, qual_targets in loader:
            images = images.to(device)
            rip_logits, qual_logits = model(images)

            rip_preds = torch.argmax(rip_logits, dim=1).cpu().numpy().tolist()
            qual_preds = torch.argmax(qual_logits, dim=1).cpu().numpy().tolist()

            all_rip_true.extend(rip_targets.numpy().tolist())
            all_rip_pred.extend(rip_preds)
            all_qual_true.extend(qual_targets.numpy().tolist())
            all_qual_pred.extend(qual_preds)

    # 1. Ripeness Metrics
    rip_acc = float(accuracy_score(all_rip_true, all_rip_pred))
    rip_p, rip_r, rip_f1, _ = precision_recall_fscore_support(
        all_rip_true, all_rip_pred, average="weighted", zero_division=0
    )
    rip_report_dict = classification_report(
        all_rip_true,
        all_rip_pred,
        target_names=RIPENESS_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    rip_report_text = classification_report(
        all_rip_true,
        all_rip_pred,
        target_names=RIPENESS_CLASSES,
        zero_division=0,
    )
    rip_cm = confusion_matrix(all_rip_true, all_rip_pred, labels=[0, 1, 2])

    # 2. Quality Metrics
    qual_acc = float(accuracy_score(all_qual_true, all_qual_pred))
    qual_p, qual_r, qual_f1, _ = precision_recall_fscore_support(
        all_qual_true, all_qual_pred, average="weighted", zero_division=0
    )
    qual_report_dict = classification_report(
        all_qual_true,
        all_qual_pred,
        target_names=QUALITY_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    qual_report_text = classification_report(
        all_qual_true,
        all_qual_pred,
        target_names=QUALITY_CLASSES,
        zero_division=0,
    )
    qual_cm = confusion_matrix(all_qual_true, all_qual_pred, labels=[0, 1, 2])

    # Generate and save confusion matrix plots
    output_dir.mkdir(parents=True, exist_ok=True)
    rip_cm_path = output_dir / "ripeness_confusion_matrix.png"
    qual_cm_path = output_dir / "quality_confusion_matrix.png"

    plot_and_save_confusion_matrix(
        cm=rip_cm,
        class_names=RIPENESS_CLASSES,
        title="Lemon Ripeness Confusion Matrix",
        output_path=rip_cm_path,
        cmap="YlOrBr",
    )

    plot_and_save_confusion_matrix(
        cm=qual_cm,
        class_names=QUALITY_CLASSES,
        title="Lemon Quality Grade Confusion Matrix",
        output_path=qual_cm_path,
        cmap="GnBu",
    )

    # Save text report
    report_file = output_dir / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== LEMON RIPENESS CLASSIFICATION REPORT ===\n")
        f.write(rip_report_text)
        f.write("\n\n=== LEMON QUALITY CLASSIFICATION REPORT ===\n")
        f.write(qual_report_text)
    logger.info(f"Saved text classification report to: {report_file}")

    sep = "=" * 60
    logger.info("\n" + sep)
    logger.info("       RIPENESS CLASSIFICATION REPORT")
    logger.info(sep)
    logger.info(f"\n{rip_report_text}")
    logger.info(sep)
    logger.info("        QUALITY CLASSIFICATION REPORT")
    logger.info(sep)
    logger.info(f"\n{qual_report_text}")
    logger.info(sep + "\n")

    return {
        "task": "classification",
        "weights": str(weights_path) if weights_path else None,
        "dataset_dir": str(target_dir),
        "split": split,
        "num_samples": len(dataset),
        "ripeness": {
            "accuracy": rip_acc,
            "weighted_precision": float(rip_p),
            "weighted_recall": float(rip_r),
            "weighted_f1": float(rip_f1),
            "report": rip_report_dict,
            "confusion_matrix": rip_cm.tolist(),
            "plot_path": str(rip_cm_path),
        },
        "quality": {
            "accuracy": qual_acc,
            "weighted_precision": float(qual_p),
            "weighted_recall": float(qual_r),
            "weighted_f1": float(qual_f1),
            "report": qual_report_dict,
            "confusion_matrix": qual_cm.tolist(),
            "plot_path": str(qual_cm_path),
        },
    }


def print_overall_summary(
    detection_res: Optional[Dict[str, Any]],
    classification_res: Optional[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Print master summary of all evaluation tasks executed.

    Args:
        detection_res: Output from evaluate_detector or None.
        classification_res: Output from evaluate_classifier or None.
        output_dir: Directory where results are stored.
    """
    sep = "=" * 64
    subsep = "-" * 64
    logger.info("\n" + sep)
    logger.info("             OVERALL EVALUATION SUMMARY")
    logger.info(sep)

    if detection_res:
        det_m = detection_res.get("metrics", {})
        logger.info("[DETECTION - YOLOv8]")
        logger.info(f"  Weights          : {detection_res.get('weights')}")
        logger.info(f"  mAP@0.5          : {det_m.get('mAP50', 0.0):.4f}")
        logger.info(f"  mAP@0.5:0.95     : {det_m.get('mAP50_95', 0.0):.4f}")
        logger.info(f"  Precision        : {det_m.get('precision', 0.0):.4f}")
        logger.info(f"  Recall           : {det_m.get('recall', 0.0):.4f}")
        logger.info(subsep)

    if classification_res:
        rip_m = classification_res.get("ripeness", {})
        qual_m = classification_res.get("quality", {})
        logger.info("[CLASSIFICATION - EfficientNet-B0]")
        logger.info(f"  Weights          : {classification_res.get('weights')}")
        logger.info(f"  Samples          : {classification_res.get('num_samples')}")
        logger.info(f"  Ripeness Acc     : {rip_m.get('accuracy', 0.0):.2%}")
        logger.info(f"  Ripeness F1      : {rip_m.get('weighted_f1', 0.0):.4f}")
        logger.info(f"  Quality Acc      : {qual_m.get('accuracy', 0.0):.2%}")
        logger.info(f"  Quality F1       : {qual_m.get('weighted_f1', 0.0):.4f}")
        logger.info(subsep)

    logger.info(f" Artifacts Directory: {output_dir.resolve()}")
    logger.info(sep + "\n")


def main() -> int:
    """Main execution function for evaluate.py.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_yaml, classification_dir = resolve_data_paths(args.data)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": args.task,
        "output_dir": str(output_dir),
    }

    # Task 1: Detection
    if args.task in ["detection", "both"]:
        det_weights = resolve_detector_weights(args.detector_weights)
        if detection_yaml and detection_yaml.exists():
            try:
                det_res = evaluate_detector(
                    weights_path=det_weights,
                    data_yaml=detection_yaml,
                    img_size=args.img_size,
                    split=args.split,
                )
                results["detection"] = det_res
            except Exception as e:
                logger.error(f"Detector evaluation failed: {e}", exc_info=True)
                if args.task == "detection":
                    return 1
        else:
            logger.warning(
                f"Detection data YAML not found (looked for: {detection_yaml}). "
                "Skipping detection evaluation."
            )
            if args.task == "detection":
                logger.error("Detection task requested but no valid data.yaml found.")
                return 1

    # Task 2: Classification
    if args.task in ["classification", "both"]:
        cls_weights = resolve_classifier_weights(args.classifier_weights)
        if classification_dir and classification_dir.exists():
            try:
                cls_res = evaluate_classifier(
                    weights_path=cls_weights,
                    data_dir=classification_dir,
                    output_dir=output_dir,
                    img_size=args.classifier_img_size,
                    batch_size=args.batch_size,
                    split=args.split,
                    device_str=args.device,
                )
                results["classification"] = cls_res
            except Exception as e:
                logger.error(f"Classifier evaluation failed: {e}", exc_info=True)
                if args.task == "classification":
                    return 1
        else:
            logger.warning(
                f"Classification dataset directory not found (looked for: {classification_dir}). "
                "Skipping classification evaluation."
            )
            if args.task == "classification":
                logger.error("Classification task requested but no data directory found.")
                return 1

    # Save combined metrics JSON
    metrics_json_path = output_dir / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_json_path}")

    # Summary
    print_overall_summary(
        detection_res=results.get("detection"),
        classification_res=results.get("classification"),
        output_dir=output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
