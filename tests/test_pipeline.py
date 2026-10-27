"""Tests for the end-to-end lemon harvest pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pipeline import LemonHarvestPipeline


class TestLemonHarvestPipeline:
    """Test suite for LemonHarvestPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        return LemonHarvestPipeline(
            detector_weights="yolov8n.pt",
            classifier_weights=None,  # No trained weights
            confidence=0.5,
            device="cpu",
        )

    def test_initialization(self, pipeline):
        """Test pipeline initializes correctly."""
        assert pipeline.detector is not None
        assert pipeline.classifier is not None

    def test_process_image_returns_dict(self, pipeline):
        """Test that process_image returns expected structure."""
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_image(dummy_image)
        
        assert isinstance(result, dict)
        assert "image" in result
        assert "detections" in result
        assert "summary" in result

    def test_summary_structure(self, pipeline):
        """Test that summary has expected keys."""
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipeline.process_image(dummy_image)
        summary = result["summary"]
        
        assert "total_lemons" in summary
        assert "harvestable" in summary
        assert "not_ready" in summary
        assert "ripeness_distribution" in summary
        assert "quality_distribution" in summary
        assert "size_distribution" in summary

    def test_harvest_decision_logic(self, pipeline):
        """Test the harvest decision criteria."""
        # Ripe + good quality + medium size = should harvest
        assert pipeline._should_harvest("ripe", "grade_a", "large") is True
        assert pipeline._should_harvest("ripe", "grade_b", "medium") is True
        
        # Raw = should not harvest
        assert pipeline._should_harvest("raw", "grade_a", "large") is False
        
        # Poor quality = should not harvest
        assert pipeline._should_harvest("ripe", "grade_c", "large") is False
        
        # Too small = should not harvest
        assert pipeline._should_harvest("ripe", "grade_a", "small") is False

    def test_count_distribution(self):
        """Test the distribution counting helper."""
        results = [
            {"ripeness": "ripe"},
            {"ripeness": "ripe"},
            {"ripeness": "raw"},
        ]
        dist = LemonHarvestPipeline._count_distribution(results, "ripeness")
        assert dist == {"ripe": 2, "raw": 1}

    def test_repr(self, pipeline):
        """Test string representation."""
        repr_str = repr(pipeline)
        assert "LemonHarvestPipeline" in repr_str

    def test_invalid_image_path(self, pipeline):
        """Test error handling for invalid image path."""
        with pytest.raises(FileNotFoundError):
            pipeline.process_image("nonexistent_image.jpg")
