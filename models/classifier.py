"""Ripeness and quality classifier for detected lemons."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)


class RipenessClassifier:
    """Multi-task classifier for lemon ripeness and quality grading.
    
    Uses EfficientNet-B0 backbone with two classification heads:
    - Ripeness: raw, semi_ripe, ripe
    - Quality: grade_a, grade_b, grade_c
    """

    RIPENESS_CLASSES = ["raw", "semi_ripe", "ripe"]
    QUALITY_CLASSES = ["grade_a", "grade_b", "grade_c"]

    # Size thresholds based on bounding box area (pixels²)
    SIZE_THRESHOLDS = {"small": 2000, "medium": 5000, "large": 10000}

    def __init__(
        self,
        weights_path: Optional[str] = None,
        num_ripeness_classes: int = 3,
        num_quality_classes: int = 3,
        input_size: int = 224,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the ripeness classifier.
        
        Args:
            weights_path: Path to trained weights (.pth). If None, uses pretrained backbone.
            num_ripeness_classes: Number of ripeness classes.
            num_quality_classes: Number of quality classes.
            input_size: Input image size for the classifier.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        self.input_size = input_size
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Build model
        self.model = self._build_model(num_ripeness_classes, num_quality_classes)
        
        # Load trained weights if provided
        if weights_path and Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded classifier weights from {weights_path}")
        else:
            logger.info("Initialized classifier with pretrained EfficientNet-B0 backbone")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _build_model(
        self, num_ripeness: int, num_quality: int
    ) -> nn.Module:
        """Build the multi-task EfficientNet-B0 model."""
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Get the feature dimension from the classifier
        in_features = backbone.classifier[1].in_features
        
        # Remove the original classifier
        backbone.classifier = nn.Identity()
        
        model = MultiTaskLemonClassifier(
            backbone=backbone,
            in_features=in_features,
            num_ripeness=num_ripeness,
            num_quality=num_quality,
        )
        return model

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        """Classify a cropped lemon image.
        
        Args:
            image: Cropped lemon image as numpy array (BGR from OpenCV).
            
        Returns:
            Dictionary containing:
                - ripeness: Predicted ripeness class
                - ripeness_confidence: Confidence score for ripeness
                - quality: Predicted quality grade
                - quality_confidence: Confidence score for quality
                - ripeness_probs: Probabilities for all ripeness classes
                - quality_probs: Probabilities for all quality classes
        """
        # Convert BGR to RGB PIL Image
        if isinstance(image, np.ndarray):
            image_rgb = image[:, :, ::-1] if image.ndim == 3 else image
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        else:
            pil_image = image
        
        # Preprocess
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Forward pass
        ripeness_logits, quality_logits = self.model(tensor)
        
        # Get predictions
        ripeness_probs = torch.softmax(ripeness_logits, dim=1)[0].cpu().numpy()
        quality_probs = torch.softmax(quality_logits, dim=1)[0].cpu().numpy()
        
        ripeness_idx = int(np.argmax(ripeness_probs))
        quality_idx = int(np.argmax(quality_probs))
        
        return {
            "ripeness": self.RIPENESS_CLASSES[ripeness_idx],
            "ripeness_confidence": float(ripeness_probs[ripeness_idx]),
            "quality": self.QUALITY_CLASSES[quality_idx],
            "quality_confidence": float(quality_probs[quality_idx]),
            "ripeness_probs": {c: float(p) for c, p in zip(self.RIPENESS_CLASSES, ripeness_probs)},
            "quality_probs": {c: float(p) for c, p in zip(self.QUALITY_CLASSES, quality_probs)},
        }

    def estimate_size(self, bbox: list[float]) -> str:
        """Estimate lemon size from bounding box area.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2].
            
        Returns:
            Size category: 'small', 'medium', or 'large'.
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < self.SIZE_THRESHOLDS["small"]:
            return "small"
        elif area < self.SIZE_THRESHOLDS["large"]:
            return "medium"
        else:
            return "large"

    def get_model(self) -> nn.Module:
        """Return the underlying PyTorch model for training."""
        return self.model


class MultiTaskLemonClassifier(nn.Module):
    """Multi-task classifier with shared EfficientNet backbone.
    
    Two separate heads for ripeness classification and quality grading.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_ripeness: int = 3,
        num_quality: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        
        # Ripeness classification head
        self.ripeness_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_ripeness),
        )
        
        # Quality classification head
        self.quality_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_quality),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            
        Returns:
            Tuple of (ripeness_logits, quality_logits).
        """
        features = self.backbone(x)
        ripeness_out = self.ripeness_head(features)
        quality_out = self.quality_head(features)
        return ripeness_out, quality_out
