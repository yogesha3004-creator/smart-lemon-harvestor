"""Tests for the lemon detector module."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.detector import LemonDetector


class TestLemonDetector:
    """Test suite for LemonDetector."""

    def test_initialization_default(self):
        """Test detector initializes with default YOLOv8n model."""
        detector = LemonDetector(model_path="yolov8n.pt")
        assert detector.model is not None
        assert detector.confidence == LemonDetector.DEFAULT_CONFIDENCE
        assert detector.iou_threshold == LemonDetector.DEFAULT_IOU

    def test_initialization_custom_thresholds(self):
        """Test detector with custom confidence and IoU thresholds."""
        detector = LemonDetector(
            model_path="yolov8n.pt",
            confidence=0.7,
            iou_threshold=0.3,
        )
        assert detector.confidence == 0.7
        assert detector.iou_threshold == 0.3

    def test_predict_returns_list(self):
        """Test that predict returns a list of detections."""
        detector = LemonDetector(model_path="yolov8n.pt")
        # Create a dummy image (random noise)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = detector.predict(dummy_image)
        assert isinstance(results, list)

    def test_predict_detection_format(self):
        """Test that each detection has the expected keys."""
        detector = LemonDetector(model_path="yolov8n.pt", confidence=0.01)
        # Create an image that might trigger some detections
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = detector.predict(dummy_image)
        for det in results:
            assert "bbox" in det
            assert "confidence" in det
            assert "class_name" in det
            assert len(det["bbox"]) == 4
            assert 0 <= det["confidence"] <= 1

    def test_class_names(self):
        """Test class names are properly defined."""
        assert "lemon" in LemonDetector.CLASS_NAMES
