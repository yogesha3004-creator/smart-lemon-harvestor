"""Video and webcam inference script for Smart Lemon Harvester.

Processes video files or live camera streams frame-by-frame, applying lemon
detection, ripeness grading, quality evaluation, and harvest recommendations
with real-time FPS estimation and HUD overlays.
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Optional, Union

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
logger = logging.getLogger("predict_video")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Lemon Harvester - Video & Webcam Stream Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video file (e.g. 'orchard.mp4') or camera device index (e.g. '0').",
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
        default=None,
        help="Optional path to save annotated output video (e.g. 'results/output.mp4').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the processed frames in a real-time GUI window. Press 'q' to quit.",
    )
    return parser.parse_args()


def get_video_capture(source_str: str) -> cv2.VideoCapture:
    """Initialize cv2.VideoCapture from a camera index or file path.

    Args:
        source_str: Either an integer camera index ('0', '1') or path to video file.

    Returns:
        Configured cv2.VideoCapture object.

    Raises:
        FileNotFoundError: If a video file path does not exist.
        RuntimeError: If the video capture device cannot be opened.
    """
    if source_str.isdigit():
        cam_idx = int(source_str)
        logger.info(f"Connecting to webcam index {cam_idx} (trying DirectShow for virtual webcams)...")
        # On Windows, try cv2.CAP_DSHOW first to support virtual webcams (Iriun, DroidCam, OBS)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx)
        else:
            cap = cv2.VideoCapture(cam_idx)
    elif source_str.startswith(("http://", "https://", "rtsp://")):
        logger.info(f"Connecting to network camera / IP stream: {source_str}...")
        cap = cv2.VideoCapture(source_str)
    else:
        src_path = Path(source_str)
        if not src_path.is_file():
            raise FileNotFoundError(f"Video file not found: {src_path.resolve()}")
        logger.info(f"Opening video file: {src_path.resolve()}")
        cap = cv2.VideoCapture(str(src_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_str}")

    return cap


def get_video_writer(
    output_path: str,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """Create a configured cv2.VideoWriter for output video saving.

    Args:
        output_path: Target video file path.
        fps: Target frames-per-second.
        frame_size: (width, height) tuple of video frames.

    Returns:
        Configured cv2.VideoWriter object.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Choose codec based on extension
    ext = out_file.suffix.lower()
    if ext in {".mp4", ".m4v"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(out_file), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for path: {out_file}")

    return writer


def run_video_inference(
    source: str,
    config_path: Optional[str] = "configs/pipeline_config.yaml",
    detector_weights: str = "yolov8n.pt",
    classifier_weights: Optional[str] = None,
    confidence: float = 0.5,
    output: Optional[str] = None,
    show: bool = False,
) -> None:
    """Process video source frame-by-frame with FPS measurement and HUD overlays.

    Args:
        source: Video file path or camera index.
        config_path: Path to configuration YAML.
        detector_weights: YOLOv8 weights path.
        classifier_weights: Classifier weights path.
        confidence: Detection threshold.
        output: Optional destination video file path.
        show: If True, opens GUI display window.
    """
    cap = get_video_capture(source)

    # Query source video properties
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or np.isnan(src_fps):
        src_fps = 25.0  # Fallback for webcam streams

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        f"Video source opened: {src_width}x{src_height} @ {src_fps:.1f} FPS "
        f"({total_frames if total_frames > 0 else 'live stream'} frames)"
    )

    # Initialize Harvest Pipeline
    logger.info(f"Initializing pipeline (detector={detector_weights}, conf={confidence})...")
    pipeline = LemonHarvestPipeline(
        config_path=config_path if config_path and Path(config_path).exists() else None,
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        confidence=confidence,
    )

    # Setup VideoWriter if requested
    writer = None
    if output:
        writer = get_video_writer(output, src_fps, (src_width, src_height))
        logger.info(f"Saving output video to: {Path(output).resolve()}")

    win_name = "Smart Lemon Harvester - Video Inference"
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(1280, src_width), min(720, src_height))

    frame_count = 0
    start_time = time.time()
    smooth_fps = 0.0

    logger.info("Starting inference loop. Press 'q' to quit...")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("Reached end of video stream.")
                break

            frame_count += 1

            # Run full harvest pipeline on current frame
            pipeline_results = pipeline.process_image(frame)

            # Draw visual bounding boxes and harvest indicators
            annotated = draw_pipeline_results(frame, pipeline_results)

            # Draw HUD summary overlay
            annotated = create_summary_overlay(annotated, pipeline_results["summary"], position="top-left")

            # Calculate and overlay real-time FPS
            frame_time = time.time() - t0
            instant_fps = 1.0 / max(1e-5, frame_time)
            smooth_fps = (0.9 * smooth_fps + 0.1 * instant_fps) if smooth_fps > 0 else instant_fps

            # Real-time FPS badge on top-right corner
            fps_text = f"FPS: {smooth_fps:.1f}"
            (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            fps_bg_x1 = src_width - tw - 24
            fps_bg_y1 = 16
            fps_bg_x2 = src_width - 12
            fps_bg_y2 = fps_bg_y1 + th + 14

            # Semi-transparent FPS badge
            badge_overlay = annotated.copy()
            cv2.rectangle(
                badge_overlay,
                (fps_bg_x1, fps_bg_y1),
                (fps_bg_x2, fps_bg_y2),
                (20, 24, 28),
                thickness=-1,
            )
            cv2.addWeighted(badge_overlay, 0.8, annotated, 0.2, 0, annotated)
            cv2.putText(
                annotated,
                fps_text,
                (fps_bg_x1 + 6, fps_bg_y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Write frame if recording
            if writer is not None:
                writer.write(annotated)

            # Show frame interactively
            if show:
                cv2.imshow(win_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    logger.info("User requested exit ('q' key pressed).")
                    break

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames | Current speed: {smooth_fps:.1f} FPS")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        total_time = max(1e-3, time.time() - start_time)
        avg_fps = frame_count / total_time
        logger.info(
            f"Finished processing: {frame_count} frames in {total_time:.2f}s "
            f"(Average Speed: {avg_fps:.1f} FPS)"
        )

        cap.release()
        if writer is not None:
            writer.release()
            logger.info(f"Video saved successfully to: {Path(output).resolve()}")
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    """CLI entry point for video inference."""
    args = parse_args()
    try:
        run_video_inference(
            source=args.source,
            config_path=args.config,
            detector_weights=args.detector_weights,
            classifier_weights=args.classifier_weights,
            confidence=args.confidence,
            output=args.output,
            show=args.show,
        )
    except Exception as exc:
        logger.error(f"Video inference failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
