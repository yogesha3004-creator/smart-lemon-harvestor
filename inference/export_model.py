"""Model export utility script for Smart Lemon Harvester.

Exports the YOLOv8 lemon detector and MultiTask lemon classifier into optimized
deployment formats including ONNX, TorchScript, and TensorRT engine.
Generates an export summary reporting artifact locations and file sizes.
"""

import argparse
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional

import torch

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classifier import RipenessClassifier
from models.detector import LemonDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_model")


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def export_detector(
    weights_path: str,
    export_format: str,
    img_size: int,
    output_dir: Path,
    device: str = "cpu",
) -> Path:
    """Export YOLOv8 detector using Ultralytics export pipeline.

    Args:
        weights_path: Path to YOLO weights (.pt).
        export_format: Target format ('onnx', 'torchscript', 'engine').
        img_size: Input resolution.
        output_dir: Destination folder.
        device: Device string.

    Returns:
        Path to exported detector file.
    """
    logger.info(f"Exporting detector ({weights_path}) to format '{export_format}'...")
    detector = LemonDetector(model_path=weights_path)
    
    # Run YOLO export
    yolo_format = "torchscript" if export_format == "torchscript" else export_format
    exported_file_str = detector.export(format=yolo_format, img_size=img_size, device=device)
    exported_src = Path(exported_file_str)

    # Move/copy to output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / exported_src.name
    if exported_src.resolve() != destination.resolve():
        shutil.copy2(str(exported_src), str(destination))
        logger.info(f"Copied detector artifact to: {destination.resolve()}")
    else:
        destination = exported_src

    return destination


def export_classifier(
    weights_path: Optional[str],
    export_format: str,
    img_size: int,
    output_dir: Path,
    device: str = "cpu",
    opset: int = 12,
) -> Path:
    """Export MultiTask lemon classifier to ONNX or TorchScript.

    Args:
        weights_path: Path to classifier weights (.pth), or None for pretrained.
        export_format: Target format ('onnx', 'torchscript', 'engine').
        img_size: Input image dimension (e.g. 224).
        output_dir: Destination directory.
        device: Device string ('cpu' or 'cuda').
        opset: ONNX opset version.

    Returns:
        Path to exported classifier artifact.
    """
    logger.info(f"Exporting classifier (weights={weights_path}) to '{export_format}'...")
    classifier = RipenessClassifier(weights_path=weights_path, input_size=img_size, device=device)
    model = classifier.get_model()
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    if export_format == "onnx":
        out_file = output_dir / "classifier.onnx"
        logger.info(f"Writing ONNX graph to: {out_file.resolve()} (opset={opset})")
        torch.onnx.export(
            model,
            dummy_input,
            str(out_file),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["ripeness_logits", "quality_logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "ripeness_logits": {0: "batch_size"},
                "quality_logits": {0: "batch_size"},
            },
        )
    elif export_format == "torchscript":
        out_file = output_dir / "classifier.torchscript.pt"
        logger.info(f"Tracing TorchScript model to: {out_file.resolve()}")
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(out_file))
    elif export_format == "engine":
        # First export to ONNX, then inform regarding TensorRT builder
        out_file = output_dir / "classifier.onnx"
        logger.info("TensorRT export requires ONNX intermediate graph. Exporting ONNX first...")
        torch.onnx.export(
            model,
            dummy_input,
            str(out_file),
            input_names=["input"],
            output_names=["ripeness_logits", "quality_logits"],
            opset_version=opset,
        )
        logger.warning(
            "For full TensorRT .engine generation, ensure NVIDIA TensorRT and trtexec are installed: "
            f"`trtexec --onnx={out_file} --saveEngine={output_dir / 'classifier.engine'}`"
        )
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    return out_file


def print_export_summary(artifacts: List[Dict[str, Any]]) -> None:
    """Print an aligned summary table of exported artifacts with sizes."""
    print("\n" + "=" * 90)
    print("                        MODEL DEPLOYMENT EXPORT SUMMARY")
    print("=" * 90)
    print(
        f"{'Component':<15} | {'Format':<12} | {'Artifact Path':<45} | {'File Size'}"
    )
    print("-" * 90)
    for art in artifacts:
        comp = art.get("component", "Unknown")
        fmt = art.get("format", "").upper()
        path = str(art.get("path", ""))
        size = format_file_size(art.get("size_bytes", 0))

        # Truncate path if too long for display
        display_path = path if len(path) <= 45 else ("..." + path[-42:])
        print(f"{comp:<15} | {fmt:<12} | {display_path:<45} | {size}")
    print("=" * 90 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Lemon Harvester - Model Export Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 detector weights (.pt) to export.",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        help="Path to trained MultiTask classifier weights (.pth) to export.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "engine"],
        help="Deployment export format target.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/exported",
        help="Directory to place the exported model artifacts.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image resolution for detector export.",
    )
    parser.add_argument(
        "--classifier-img-size",
        type=int,
        default=224,
        help="Input image resolution for classifier export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version for export.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def run_export(
    detector_weights: Optional[str] = "yolov8n.pt",
    classifier_weights: Optional[str] = None,
    export_format: str = "onnx",
    output_dir: str = "weights/exported",
    img_size: int = 640,
    classifier_img_size: int = 224,
    device: str = "cpu",
    opset: int = 12,
) -> List[Dict[str, Any]]:
    """Export detector and/or classifier models to optimized formats.

    Args:
        detector_weights: YOLO detector weights path.
        classifier_weights: Classifier weights path.
        export_format: Target format ('onnx', 'torchscript', 'engine').
        output_dir: Target directory path.
        img_size: Detector image size.
        classifier_img_size: Classifier image size.
        device: Device to run export.
        opset: ONNX opset version.

    Returns:
        List of dicts describing exported artifacts.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[Dict[str, Any]] = []

    # 1. Export Detector if requested
    if detector_weights:
        try:
            det_path = export_detector(
                weights_path=detector_weights,
                export_format=export_format,
                img_size=img_size,
                output_dir=out_dir,
                device=device,
            )
            size = det_path.stat().st_size if det_path.is_file() else 0
            artifacts.append({
                "component": "Lemon Detector",
                "format": export_format,
                "path": det_path.resolve(),
                "size_bytes": size,
            })
        except Exception as exc:
            logger.error(f"Failed to export detector: {exc}", exc_info=True)

    # 2. Export Classifier
    try:
        cls_path = export_classifier(
            weights_path=classifier_weights,
            export_format=export_format,
            img_size=classifier_img_size,
            output_dir=out_dir,
            device=device,
            opset=opset,
        )
        size = cls_path.stat().st_size if cls_path.is_file() else 0
        artifacts.append({
            "component": "Lemon Classifier",
            "format": export_format,
            "path": cls_path.resolve(),
            "size_bytes": size,
        })
    except Exception as exc:
        logger.error(f"Failed to export classifier: {exc}", exc_info=True)

    # Print summary table
    print_export_summary(artifacts)
    return artifacts


def main() -> None:
    """CLI entry point for model export."""
    args = parse_args()
    try:
        run_export(
            detector_weights=args.detector_weights,
            classifier_weights=args.classifier_weights,
            export_format=args.format,
            output_dir=args.output_dir,
            img_size=args.img_size,
            classifier_img_size=args.classifier_img_size,
            device=args.device,
            opset=args.opset,
        )
    except Exception as exc:
        logger.error(f"Export execution failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
