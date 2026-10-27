"""Unit tests for utility modules in Smart Lemon Harvester."""

from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.visualization import (
    draw_detections,
    draw_pipeline_results,
    create_summary_overlay,
)
from utils.preprocessing import (
    preprocess_image,
    preprocess_for_classifier,
    apply_augmentation,
    validate_image,
)
from utils.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    compute_detection_metrics,
    format_metrics_table,
)


class TestVisualization:
    """Test suite for utils/visualization.py."""

    @pytest.fixture
    def dummy_image(self) -> np.ndarray:
        return np.full((480, 640, 3), 128, dtype=np.uint8)

    def test_draw_detections_list(self, dummy_image):
        detections = [
            {"bbox": [50, 60, 200, 220], "confidence": 0.88, "class_name": "lemon"},
            {"bbox": [250, 100, 400, 300], "confidence": 0.75, "class_name": "lemon"},
        ]
        result = draw_detections(dummy_image, detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == dummy_image.shape
        assert result.dtype == np.uint8

    def test_draw_detections_numpy(self, dummy_image):
        det_arr = np.array([
            [50, 60, 200, 220, 0.90, 0],
            [250, 100, 400, 300, 0.82, 0],
        ], dtype=np.float32)
        result = draw_detections(dummy_image, det_arr, class_names=["lemon"])
        assert isinstance(result, np.ndarray)
        assert result.shape == dummy_image.shape

    def test_draw_detections_empty(self, dummy_image):
        result = draw_detections(dummy_image, [])
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, dummy_image)

    def test_draw_pipeline_results(self, dummy_image):
        pipeline_results = {
            "detections": [
                {
                    "bbox": [50, 50, 150, 150],
                    "ripeness": "ripe",
                    "ripeness_confidence": 0.95,
                    "quality": "grade_a",
                    "size": "medium",
                    "harvest": True,
                },
                {
                    "bbox": [200, 200, 350, 350],
                    "ripeness": "raw",
                    "ripeness_confidence": 0.91,
                    "quality": "grade_c",
                    "size": "small",
                    "harvest": False,
                },
                {
                    "bbox": [380, 100, 480, 220],
                    "ripeness": "semi_ripe",
                    "ripeness_confidence": 0.84,
                    "quality": "grade_b",
                    "size": "large",
                    "harvest": False,
                },
            ],
            "summary": {
                "total_lemons": 3,
                "harvestable": 1,
                "not_ready": 2,
                "ripeness_distribution": {"ripe": 1, "raw": 1, "semi_ripe": 1},
            },
        }
        result = draw_pipeline_results(dummy_image, pipeline_results)
        assert isinstance(result, np.ndarray)
        assert result.shape == dummy_image.shape

    def test_create_summary_overlay(self, dummy_image):
        summary = {
            "total_lemons": 5,
            "harvestable": 3,
            "not_ready": 2,
            "ripeness_distribution": {"ripe": 3, "semi_ripe": 1, "raw": 1},
            "quality_distribution": {"grade_a": 3, "grade_b": 2},
        }
        for pos in ["top-left", "top-right", "bottom-left", "bottom-right"]:
            overlay = create_summary_overlay(dummy_image, summary, position=pos)
            assert isinstance(overlay, np.ndarray)
            assert overlay.shape == dummy_image.shape


class TestPreprocessing:
    """Test suite for utils/preprocessing.py."""

    @pytest.fixture
    def sample_image_file(self) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = np.full((100, 150, 3), 200, dtype=np.uint8)
            cv2.imwrite(tmp.name, img)
            tmp_path = Path(tmp.name)
        yield tmp_path
        if tmp_path.exists():
            tmp_path.unlink()

    def test_preprocess_image(self, sample_image_file):
        # Default load
        img = preprocess_image(sample_image_file)
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 150, 3)

        # Resize tuple
        resized = preprocess_image(sample_image_file, target_size=(64, 64))
        assert resized.shape == (64, 64, 3)

        # Resize int
        resized_sq = preprocess_image(sample_image_file, target_size=80)
        assert resized_sq.shape == (80, 80, 3)

        # Color conversion
        rgb = preprocess_image(sample_image_file, to_rgb=True)
        assert rgb.shape == (100, 150, 3)

    def test_preprocess_image_not_found(self):
        with pytest.raises(FileNotFoundError):
            preprocess_image("nonexistent_lemon_file_123.jpg")

    def test_preprocess_for_classifier_numpy(self):
        crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        tensor = preprocess_for_classifier(crop, input_size=224)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_preprocess_for_classifier_pil(self):
        pil_img = Image.new("RGB", (90, 90), color=(255, 200, 0))
        tensor = preprocess_for_classifier(pil_img, input_size=224)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_apply_augmentation(self):
        img = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
        aug = apply_augmentation(img, config={"hflip_prob": 1.0, "vflip_prob": 1.0})
        assert isinstance(aug, np.ndarray)
        assert aug.shape == img.shape

    def test_validate_image(self, sample_image_file):
        assert validate_image(sample_image_file) is True
        assert validate_image(np.zeros((10, 10, 3), dtype=np.uint8)) is True
        assert validate_image("invalid_path_456.jpg") is False
        assert validate_image(None) is False
        assert validate_image(b"not an image") is False


class TestMetrics:
    """Test suite for utils/metrics.py."""

    def test_compute_classification_metrics(self):
        y_true = ["ripe", "ripe", "semi_ripe", "raw", "raw"]
        y_pred = ["ripe", "semi_ripe", "semi_ripe", "raw", "raw"]
        class_names = ["raw", "semi_ripe", "ripe"]

        res = compute_classification_metrics(y_true, y_pred, class_names=class_names)
        assert "accuracy" in res
        assert res["accuracy"] == 0.8
        assert "macro_avg" in res
        assert "weighted_avg" in res
        assert "per_class" in res
        assert "confusion_matrix" in res
        assert res["total_samples"] == 5

    def test_plot_confusion_matrix(self):
        y_true = ["ripe", "raw", "ripe", "semi_ripe"]
        y_pred = ["ripe", "raw", "semi_ripe", "semi_ripe"]
        class_names = ["raw", "semi_ripe", "ripe"]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            saved = plot_confusion_matrix(y_true, y_pred, class_names, tmp_path)
            assert Path(saved).exists()
            assert Path(saved).stat().st_size > 0
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_compute_detection_metrics(self):
        preds = [
            {"bbox": [10, 10, 50, 50], "confidence": 0.95, "class_name": "lemon"},
            {"bbox": [60, 60, 100, 100], "confidence": 0.80, "class_name": "lemon"},
        ]
        gts = [
            {"bbox": [12, 11, 49, 52], "class_name": "lemon"},
            {"bbox": [200, 200, 250, 250], "class_name": "lemon"},
        ]
        metrics = compute_detection_metrics(preds, gts, iou_threshold=0.5)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "mAP50" in metrics
        assert metrics["tp"] == 1
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1

    def test_format_metrics_table(self):
        det_metrics = {
            "precision": 0.85,
            "recall": 0.75,
            "f1_score": 0.796,
            "mAP50": 0.82,
            "tp": 10,
            "fp": 2,
            "fn": 3,
            "iou_threshold": 0.5,
            "per_class": {
                "lemon": {"precision": 0.85, "recall": 0.75, "f1_score": 0.796, "ap": 0.82}
            },
        }
        table_str = format_metrics_table(det_metrics)
        assert "DETECTION EVALUATION REPORT" in table_str
        assert "lemon" in table_str
