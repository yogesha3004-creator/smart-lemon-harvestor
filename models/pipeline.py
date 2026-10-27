"""End-to-end lemon harvesting pipeline combining detection and classification."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import yaml

from .detector import LemonDetector
from .classifier import RipenessClassifier

logger = logging.getLogger(__name__)


class LemonHarvestPipeline:
    """Complete pipeline for lemon detection, classification, and harvest decision.
    
    Stage 1: Detect lemons using YOLOv8
    Stage 2: Classify each detected lemon for ripeness and quality
    Stage 3: Make harvest recommendation based on configurable criteria
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        detector_weights: str = "yolov8n.pt",
        classifier_weights: Optional[str] = None,
        confidence: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the harvest pipeline.
        
        Args:
            config_path: Path to pipeline_config.yaml. Overrides other params if provided.
            detector_weights: Path to YOLOv8 detection weights.
            classifier_weights: Path to classifier weights (.pth).
            confidence: Detection confidence threshold.
            device: Device for classifier ('cuda', 'cpu', or None for auto).
        """
        # Load config if provided
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            detector_weights = self.config.get("detector", {}).get("weights", detector_weights)
            classifier_weights = self.config.get("classifier", {}).get("weights", classifier_weights)
            confidence = self.config.get("detector", {}).get("confidence_threshold", confidence)

        # Fallback to pretrained weights if custom trained checkpoints don't exist yet
        if detector_weights and not Path(detector_weights).exists() and not detector_weights.endswith(".pt"):
            pass
        elif detector_weights and not Path(detector_weights).exists() and detector_weights != "yolov8n.pt":
            logger.info(
                f"Trained detector weights '{detector_weights}' not found. "
                "Falling back to pretrained 'yolov8n.pt'."
            )
            detector_weights = "yolov8n.pt"

        if classifier_weights and not Path(classifier_weights).exists():
            logger.info(
                f"Trained classifier weights '{classifier_weights}' not found. "
                "Using pretrained EfficientNet backbone."
            )
            classifier_weights = None
        
        # Initialize models
        self.detector = LemonDetector(
            model_path=detector_weights,
            confidence=confidence,
        )
        self.classifier = RipenessClassifier(
            weights_path=classifier_weights,
            device=device,
        )
        
        # Harvest decision criteria
        harvest_cfg = self.config.get("harvest_decision", {})
        self.harvest_ripeness = set(harvest_cfg.get("harvest_ripeness", ["ripe"]))
        self.min_quality = harvest_cfg.get("min_quality", "grade_b")
        self.min_size = harvest_cfg.get("min_size", "medium")
        
        logger.info("Lemon Harvest Pipeline initialized")

    def process_image(self, image: Union[str, np.ndarray]) -> dict:
        """Process a single image through the full pipeline.
        
        Args:
            image: Image path or numpy array (BGR).
            
        Returns:
            Dictionary containing:
                - image: Original image (numpy array)
                - detections: List of detected lemons with classifications
                - summary: Overall harvest summary
        """
        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Stage 1: Detect lemons
        detections = self.detector.predict(img)
        
        # Stage 2 & 3: Process detections by target type (lemon, human, object)
        results = []
        for det in detections:
            bbox = det["bbox"]
            target_type = det.get("target_type", "lemon")
            class_name = det.get("class_name", "lemon")
            conf = det.get("confidence", 0.0)

            # If it's a Human / Operator
            if target_type == "human":
                results.append({
                    "bbox": bbox,
                    "target_type": "human",
                    "class_name": class_name,
                    "detection_confidence": conf,
                    "harvest": False,
                    "ripeness": "n/a",
                    "ripeness_confidence": 0.0,
                    "quality": "human",
                    "quality_confidence": 0.0,
                    "size": "n/a",
                    "status_note": "Human Operator in Frame - Safety Hold",
                })
                continue

            # If it's a Non-Harvest Object (phone, cup, chair, etc.)
            elif target_type == "object":
                results.append({
                    "bbox": bbox,
                    "target_type": "object",
                    "class_name": class_name,
                    "detection_confidence": conf,
                    "harvest": False,
                    "ripeness": "n/a",
                    "ripeness_confidence": 0.0,
                    "quality": "object",
                    "quality_confidence": 0.0,
                    "size": "n/a",
                    "status_note": f"Non-Harvest Object ({class_name})",
                })
                continue

            # Target is a Lemon / Citrus -> Run Ripeness & Quality Pipeline
            x1, y1, x2, y2 = [int(c) for c in bbox]
            pad = 5
            h, w = img.shape[:2]
            x1_pad = max(0, x1 - pad)
            y1_pad = max(0, y1 - pad)
            x2_pad = min(w, x2 + pad)
            y2_pad = min(h, y2 + pad)
            crop = img[y1_pad:y2_pad, x1_pad:x2_pad]

            if crop.size == 0:
                continue

            # Classify ripeness and quality
            classification = self.classifier.predict(crop)
            size = self.classifier.estimate_size(bbox)
            should_harvest = self._should_harvest(
                classification["ripeness"],
                classification["quality"],
                size,
            )

            result = {
                "bbox": bbox,
                "target_type": "lemon",
                "class_name": "lemon",
                "detection_confidence": conf,
                "ripeness": classification["ripeness"],
                "ripeness_confidence": classification["ripeness_confidence"],
                "quality": classification["quality"],
                "quality_confidence": classification["quality_confidence"],
                "size": size,
                "harvest": should_harvest,
                "status_note": "Lemon Target Verified",
            }
            results.append(result)

        # Summary Metrics
        lemon_results = [r for r in results if r["target_type"] == "lemon"]
        human_results = [r for r in results if r["target_type"] == "human"]
        object_results = [r for r in results if r["target_type"] == "object"]

        total_lemons = len(lemon_results)
        harvestable = sum(1 for r in lemon_results if r["harvest"])
        not_ready = total_lemons - harvestable

        # Scene status determination
        if len(human_results) > 0:
            scene_status = "HUMAN_ALERT"
            safety_interlock = "ENGAGED (OPERATOR IN VIEW)"
        elif total_lemons > 0:
            scene_status = "LEMON_TARGET"
            safety_interlock = "DISENGAGED (SAFE)"
        elif len(object_results) > 0:
            scene_status = "OBJECT_DETECTED"
            safety_interlock = "DISENGAGED (SAFE)"
        else:
            scene_status = "SCANNING"
            safety_interlock = "DISENGAGED (SAFE)"

        return {
            "image": img,
            "detections": results,
            "summary": {
                "total_items": len(results),
                "total_lemons": total_lemons,
                "humans_count": len(human_results),
                "objects_count": len(object_results),
                "harvestable": harvestable,
                "not_ready": not_ready,
                "scene_status": scene_status,
                "safety_interlock": safety_interlock,
                "ripeness_distribution": self._count_distribution(lemon_results, "ripeness"),
                "quality_distribution": self._count_distribution(lemon_results, "quality"),
                "size_distribution": self._count_distribution(lemon_results, "size"),
            },
        }

    def _should_harvest(self, ripeness: str, quality: str, size: str) -> bool:
        """Determine if a lemon should be harvested based on criteria."""
        quality_order = ["grade_c", "grade_b", "grade_a"]
        size_order = ["small", "medium", "large"]
        
        ripeness_ok = ripeness in self.harvest_ripeness
        quality_ok = (
            quality_order.index(quality) >= quality_order.index(self.min_quality)
            if quality in quality_order and self.min_quality in quality_order
            else False
        )
        size_ok = (
            size_order.index(size) >= size_order.index(self.min_size)
            if size in size_order and self.min_size in size_order
            else False
        )
        
        return ripeness_ok and quality_ok and size_ok

    @staticmethod
    def _count_distribution(results: list[dict], key: str) -> dict[str, int]:
        """Count distribution of a field across results."""
        counts: dict[str, int] = {}
        for r in results:
            val = r[key]
            counts[val] = counts.get(val, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"LemonHarvestPipeline("
            f"harvest_ripeness={self.harvest_ripeness}, "
            f"min_quality={self.min_quality}, "
            f"min_size={self.min_size})"
        )
