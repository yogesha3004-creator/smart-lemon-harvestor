"""Tests for the ripeness classifier module."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.classifier import RipenessClassifier, MultiTaskLemonClassifier


class TestRipenessClassifier:
    """Test suite for RipenessClassifier."""

    def test_initialization(self):
        """Test classifier initializes without trained weights."""
        classifier = RipenessClassifier(device="cpu")
        assert classifier.model is not None
        assert classifier.input_size == 224

    def test_predict_returns_dict(self):
        """Test that predict returns a properly structured dict."""
        classifier = RipenessClassifier(device="cpu")
        # Create a dummy cropped lemon image (BGR)
        dummy_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = classifier.predict(dummy_crop)
        
        assert isinstance(result, dict)
        assert "ripeness" in result
        assert "ripeness_confidence" in result
        assert "quality" in result
        assert "quality_confidence" in result
        assert "ripeness_probs" in result
        assert "quality_probs" in result

    def test_predict_valid_classes(self):
        """Test that predictions are valid class names."""
        classifier = RipenessClassifier(device="cpu")
        dummy_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = classifier.predict(dummy_crop)
        
        assert result["ripeness"] in RipenessClassifier.RIPENESS_CLASSES
        assert result["quality"] in RipenessClassifier.QUALITY_CLASSES

    def test_predict_confidence_range(self):
        """Test confidence scores are in valid range."""
        classifier = RipenessClassifier(device="cpu")
        dummy_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = classifier.predict(dummy_crop)
        
        assert 0.0 <= result["ripeness_confidence"] <= 1.0
        assert 0.0 <= result["quality_confidence"] <= 1.0

    def test_estimate_size_small(self):
        """Test size estimation for small bounding box."""
        classifier = RipenessClassifier(device="cpu")
        bbox = [0, 0, 30, 30]  # area = 900
        assert classifier.estimate_size(bbox) == "small"

    def test_estimate_size_medium(self):
        """Test size estimation for medium bounding box."""
        classifier = RipenessClassifier(device="cpu")
        bbox = [0, 0, 80, 50]  # area = 4000
        assert classifier.estimate_size(bbox) == "medium"

    def test_estimate_size_large(self):
        """Test size estimation for large bounding box."""
        classifier = RipenessClassifier(device="cpu")
        bbox = [0, 0, 200, 100]  # area = 20000
        assert classifier.estimate_size(bbox) == "large"

    def test_probability_sums(self):
        """Test that probability distributions sum to ~1."""
        classifier = RipenessClassifier(device="cpu")
        dummy_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = classifier.predict(dummy_crop)
        
        ripeness_sum = sum(result["ripeness_probs"].values())
        quality_sum = sum(result["quality_probs"].values())
        assert abs(ripeness_sum - 1.0) < 1e-5
        assert abs(quality_sum - 1.0) < 1e-5


class TestMultiTaskModel:
    """Test the underlying PyTorch model."""

    def test_forward_pass_shape(self):
        """Test model output shapes."""
        from torchvision import models
        backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = torch.nn.Identity()
        
        model = MultiTaskLemonClassifier(
            backbone=backbone,
            in_features=in_features,
            num_ripeness=3,
            num_quality=3,
        )
        model.eval()
        
        dummy_input = torch.randn(2, 3, 224, 224)
        ripeness_out, quality_out = model(dummy_input)
        
        assert ripeness_out.shape == (2, 3)
        assert quality_out.shape == (2, 3)
