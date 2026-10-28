"""Flask Web Application for Smart Lemon Harvesting System.

Provides a responsive mobile web interface allowing any smartphone camera
to stream video or send snapshots for real-time AI lemon detection,
ripeness grading, and harvesting decisions.
"""

import base64
import io
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

# Ensure project root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.pipeline import LemonHarvestPipeline
from utils.visualization import draw_pipeline_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lemon_web")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Initialize AI Pipeline
pipeline: LemonHarvestPipeline | None = None
latest_stream_stats = {
    "summary": {"total_lemons": 0, "harvestable": 0, "not_ready": 0},
    "detections": []
}
active_capture_lock = threading.Lock()
current_capture: cv2.VideoCapture | None = None


def get_pipeline() -> LemonHarvestPipeline:
    """Lazy-load the Lemon Harvest Pipeline."""
    global pipeline
    if pipeline is None:
        config_path = REPO_ROOT / "configs" / "pipeline_config.yaml"
        logger.info("Initializing Lemon Harvest Pipeline for Web App...")
        pipeline = LemonHarvestPipeline(
            config_path=str(config_path) if config_path.exists() else None,
            detector_weights="yolov8n.pt",
            confidence=0.25,  # Optimized sensitivity for indoor & mobile lighting
        )
        logger.info("Pipeline ready for inference.")
    return pipeline


def get_local_ip() -> str:
    """Detect the local LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.route("/")
def index():
    """Render the main mobile camera detection UI."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "ip": get_local_ip(),
        "model": "YOLOv8 + EfficientNet-B0",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run AI inference on an image frame sent from the mobile camera.
    
    Accepts:
        JSON with base64 encoded 'image', OR multipart form-data 'file'.
        
    Returns:
        JSON with detected lemons, ripeness, quality, size, harvest decision,
        and annotated image preview.
    """
    pipe = get_pipeline()
    img_bgr = None

    try:
        if request.is_json:
            data = request.get_json() or {}
            img_b64 = data.get("image", "")
            if not img_b64:
                return jsonify({"error": "No image data provided"}), 400

            # Strip base64 header if present (e.g. data:image/jpeg;base64,...)
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(img_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        elif "file" in request.files:
            file = request.files["file"]
            img_bytes = file.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        else:
            return jsonify({"error": "Unsupported media format"}), 400

        if img_bgr is None or img_bgr.size == 0:
            return jsonify({"error": "Failed to decode image"}), 400

        # Run pipeline
        results = pipe.process_image(img_bgr)

        # Draw annotated result image
        annotated_bgr = draw_pipeline_results(img_bgr, results)

        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "summary": results["summary"],
            "detections": results["detections"],
            "annotated_image": annotated_b64,
        })

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500


class CameraStreamManager:
    """Thread-safe camera stream manager that isolates OpenCV capture from Flask threads."""

    def __init__(self):
        self.cap: cv2.VideoCapture | None = None
        self.source: str | None = None
        self.running: bool = False
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.thread: threading.Thread | None = None

    def set_source(self, source: str):
        """Switch video source safely."""
        with self.lock:
            if self.source == source and self.running:
                return
            self._stop_capture()
            self.source = source
            self.running = True
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def _stop_capture(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _worker(self):
        pipe = get_pipeline()
        source = self.source or "2"
        logger.info(f"CameraStreamManager starting capture for source: {source}")

        if source.isdigit():
            cam_idx = int(source)
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(cam_idx)
            else:
                self.cap = cv2.VideoCapture(cam_idx)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap or not self.cap.isOpened():
            logger.error(f"CameraStreamManager failed to open source: {source}")
            self.running = False
            return

        frame_count = 0
        last_results = None

        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    break
                success, frame = self.cap.read()
                if not success or frame is None or frame.size == 0:
                    time.sleep(0.04)
                    continue

                frame_count += 1
                if frame_count % 2 == 0 or last_results is None:
                    last_results = pipe.process_image(frame)
                    latest_stream_stats["summary"] = last_results["summary"]
                    latest_stream_stats["detections"] = last_results["detections"]

                annotated = draw_pipeline_results(frame, last_results)
                ret, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    with self.lock:
                        self.latest_jpeg = buffer.tobytes()
            except Exception as e:
                logger.error(f"Camera worker frame error: {e}")
                time.sleep(0.05)
            time.sleep(0.015)

        self._stop_capture()
        logger.info("CameraStreamManager worker stopped.")


stream_manager = CameraStreamManager()


@app.route("/api/video_feed")
def video_feed():
    """Stream annotated AI video safely to any number of browser clients."""
    source_url = request.args.get("source", "2").strip()
    stream_manager.set_source(source_url)

    def generate_frames():
        while True:
            with stream_manager.lock:
                jpeg = stream_manager.latest_jpeg

            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.035)

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stream_stats")
def stream_stats():
    """Return the latest stats from the active live video stream."""
    return jsonify(latest_stream_stats)


def run_server(host: str = "0.0.0.0", port: int = 5000):
    """Run the Flask development server."""
    local_ip = get_local_ip()
    print("=" * 65)
    print("🍋 SMART LEMON HARVESTER — MOBILE AI WEB APP")
    print("=" * 65)
    print(f"💻 On this PC:    http://localhost:{port}")
    print(f"📱 On your phone: http://{local_ip}:{port}")
    print("=" * 65)
    print("💡 Tip: Make sure your phone and laptop are on the same Wi-Fi")
    print("   or connect your laptop to your phone's Mobile Hotspot!")
    print("=" * 65)
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_server()
