"""Unit tests for inference scripts in Smart Lemon Harvester."""

from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.export_model import export_classifier, format_file_size, parse_args as parse_export_args
from inference.predict_image import parse_args as parse_image_args, print_detection_table
from inference.predict_video import parse_args as parse_video_args


class TestInferenceArgs:
    """Test CLI argument parsers for inference scripts."""

    def test_predict_image_args(self):
        test_argv = ["predict_image.py", "--source", "test.jpg", "--confidence", "0.65", "--show"]
        with patch.object(sys, "argv", test_argv):
            args = parse_image_args()
            assert args.source == "test.jpg"
            assert args.confidence == 0.65
            assert args.show is True
            assert args.output == "results/"

    def test_predict_video_args(self):
        test_argv = ["predict_video.py", "--source", "0", "--output", "out.mp4", "--show"]
        with patch.object(sys, "argv", test_argv):
            args = parse_video_args()
            assert args.source == "0"
            assert args.output == "out.mp4"
            assert args.show is True

    def test_export_model_args(self):
        test_argv = ["export_model.py", "--format", "torchscript", "--output-dir", "exports"]
        with patch.object(sys, "argv", test_argv):
            args = parse_export_args()
            assert args.format == "torchscript"
            assert args.output_dir == "exports"


class TestExportModel:
    """Test suite for model export functionality."""

    def test_format_file_size(self):
        assert format_file_size(500) == "500 B"
        assert format_file_size(2048) == "2.00 KB"
        assert format_file_size(5 * 1024 * 1024) == "5.00 MB"

    def test_export_classifier_torchscript(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            out_file = export_classifier(
                weights_path=None,
                export_format="torchscript",
                img_size=64,  # small size for fast test
                output_dir=out_dir,
                device="cpu",
            )
            assert out_file.exists()
            assert out_file.stat().st_size > 0

    def test_export_classifier_onnx(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            out_file = export_classifier(
                weights_path=None,
                export_format="onnx",
                img_size=64,
                output_dir=out_dir,
                device="cpu",
            )
            assert out_file.exists()
            assert out_file.stat().st_size > 0


class TestPrintDetectionTable:
    """Test print detection summary table."""

    def test_print_detection_table_runs(self, capsys):
        mock_results = {
            "detections": [
                {
                    "bbox": [10, 20, 100, 120],
                    "ripeness": "ripe",
                    "ripeness_confidence": 0.94,
                    "quality": "grade_a",
                    "size": "medium",
                    "harvest": True,
                }
            ],
            "summary": {
                "total_lemons": 1,
                "harvestable": 1,
                "not_ready": 0,
                "ripeness_distribution": {"ripe": 1},
            },
        }
        print_detection_table(mock_results)
        captured = capsys.readouterr()
        assert "SMART LEMON HARVESTER" in captured.out
        assert "ripe (94%)" in captured.out
        assert "YES (Pick)" in captured.out
