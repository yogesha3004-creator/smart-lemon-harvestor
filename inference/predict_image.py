"""Single image inference script for Smart Lemon Harvester.

Runs lemon detection, ripeness and quality classification, and harvest decision
making on a single input image. Annotates the image with visual cues and prints
a detailed summary report to stdout.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import LemonHarvestPipeline
from utils.visualization import create_summary_overlay, draw_pipeline_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict_image")


def print_detection_table(pipeline_results: Dict[str, Any]) -> None:
    """Print an aligned CLI summary table of detection and harvest decisions.

    Args:
        pipeline_results: Output dictionary from LemonHarvestPipeline.process_image.
    """
    detections = pipeline_results.get("detections", [])
    summary = pipeline_results.get("summary", {})

    print("\n" + "=" * 88)
    print("                     SMART LEMON HARVESTER - DETECTION SUMMARY")
    print("=" * 88)
    print(
        f"{'Lemon #':<8} | {'Bounding Box [x1,y1,x2,y2]':<25} | {'Ripeness':<10} | "
        f"{'Quality':<9} | {'Size':<8} | {'Harvest?'}"
    )
    print("-" * 88)

    if not detections:
        print("  No lemons detected in the input image.")
    else:
        for idx, det in enumerate(detections, start=1):
            bbox = det.get("bbox", [])
            bbox_str = f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]" if len(bbox) >= 4 else "N/A"
            ripeness = det.get("ripeness", "unknown")
            r_conf = det.get("ripeness_confidence")
            r_str = f"{ripeness} ({r_conf*100:.0f}%)" if r_conf is not None else ripeness
            quality = det.get("quality", "unknown")
            size = det.get("size", "unknown")
            harvest = "YES (Pick)" if det.get("harvest", False) else "NO (Skip)"

            print(
                f"{idx:<8} | {bbox_str:<25} | {r_str:<10} | "
                f"{quality:<9} | {size:<8} | {harvest}"
            )

    print("-" * 88)
    print(
        f"Total Lemons: {summary.get('total_lemons', len(detections))} | "
        f"Harvestable: {summary.get('harvestable', 0)} | "
        f"Not Ready: {summary.get('not_ready', 0)}"
    )

    ripe_dist = summary.get("ripeness_distribution", {})
    if ripe_dist:
        parts = [f"{k}: {v}" for k, v in ripe_dist.items()]
        print(f"Ripeness Distribution: {', '.join(parts)}")
    print("=" * 88 + "\n")


def save_image_safe(image: np.ndarray, output_path: Path) -> Path:
    """Save an image to disk safely handling cross-platform paths."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError(f"Failed to encode image to format: {ext}")
    encoded.tofile(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Lemon Harvester - Single Image Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline_config.yaml configuration file.",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 detector weights (.pt).",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        help="Path to trained MultiTask classifier weights (.pth).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for YOLOv8 lemon detection.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory or file path for the annotated result image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated result window interactively using cv2.imshow.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detection details and harvest summary to a JSON file.",
    )
    return parser.parse_args()


def run_image_inference(
    source: str,
    config_path: Optional[str] = "configs/pipeline_config.yaml",
    detector_weights: str = "yolov8n.pt",
    classifier_weights: Optional[str] = None,
    confidence: float = 0.5,
    output: str = "results/",
    show: bool = False,
    save_json: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end inference on a single image.

    Args:
        source: Path to source image.
        config_path: Path to configuration YAML.
        detector_weights: YOLOv8 weights path.
        classifier_weights: Classifier weights path.
        confidence: Detection threshold.
        output: Destination directory or file path.
        show: If True, displays GUI window with cv2.imshow.
        save_json: If True, writes JSON results file.

    Returns:
        Pipeline results dictionary.
    """
    src_path = Path(source)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source image does not exist: {src_path.resolve()}")

    logger.info(f"Loading pipeline with detector={detector_weights}, config={config_path}")
    pipeline = LemonHarvestPipeline(
        config_path=config_path if config_path and Path(config_path).exists() else None,
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        confidence=confidence,
    )

    # Process image
    logger.info(f"Processing image: {src_path.name}")
    results = pipeline.process_image(str(src_path))

    # Print summary table
    print_detection_table(results)

    # Draw annotations
    annotated = draw_pipeline_results(results["image"], results)
    annotated = create_summary_overlay(annotated, results["summary"])

    # Determine output path
    out_arg = Path(output)
    if out_arg.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        out_file = out_arg
    else:
        out_arg.mkdir(parents=True, exist_ok=True)
        out_file = out_arg / f"{src_path.stem}_harvested.jpg"

    save_image_safe(annotated, out_file)
    logger.info(f"Saved annotated image to: {out_file.resolve()}")

    # Optional JSON export
    if save_json:
        json_file = out_file.with_suffix(".json")
        serializable_results = {
            "source": str(src_path.resolve()),
            "detections": results.get("detections", []),
            "summary": results.get("summary", {}),
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved results metadata to: {json_file.resolve()}")

    # Display image if requested
    if show:
        win_name = f"Smart Lemon Harvester - {src_path.name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, annotated)
        logger.info("Displaying image window. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def main() -> None:
    """CLI entry point for image inference."""
    args = parse_args()
    try:
        run_image_inference(
            source=args.source,
            config_path=args.config,
            detector_weights=args.detector_weights,
            classifier_weights=args.classifier_weights,
            confidence=args.confidence,
            output=args.output,
            show=args.show,
            save_json=args.save_json,
        )
    except Exception as exc:
        logger.error(f"Inference failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
