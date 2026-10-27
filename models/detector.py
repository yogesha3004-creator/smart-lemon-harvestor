"""YOLOv8-based lemon detector for the Smart Lemon Harvester."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class LemonDetector:
    """Lemon detection using YOLOv8.
    
    Wraps the Ultralytics YOLOv8 model for detecting lemons in images.
    Supports training, inference, evaluation, and model export.
    """

    DEFAULT_CONFIDENCE = 0.25
    DEFAULT_IOU = 0.45
    CLASS_NAMES = ["lemon"]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU,
    ) -> None:
        """Initialize the lemon detector.
        
        Args:
            model_path: Path to YOLOv8 weights (.pt file) or model variant name.
            confidence: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for Non-Maximum Suppression.
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_path)
        logger.info(f"Loaded YOLOv8 model from {model_path}")

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        project: str = "runs/detection",
        name: str = "train",
        **kwargs: Any,
    ) -> dict:
        """Train the detection model on a lemon dataset.
        
        Args:
            data_yaml: Path to the YOLO-format data.yaml file.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            img_size: Input image size.
            project: Directory to save training results.
            name: Experiment name.
            **kwargs: Additional arguments passed to YOLO.train().
            
        Returns:
            Dictionary containing training results and metrics.
        """
        logger.info(f"Starting detection training: {epochs} epochs, batch_size={batch_size}")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=project,
            name=name,
            patience=kwargs.pop("patience", 20),
            save_period=kwargs.pop("save_period", 10),
            verbose=True,
            **kwargs,
        )
        logger.info("Training complete.")
        return results

    def predict(
        self,
        source: Union[str, np.ndarray],
        img_size: int = 640,
        save: bool = False,
        **kwargs: Any,
    ) -> list[dict]:
        """Run inference on an image or video source.
        
        Args:
            source: Image path, numpy array (BGR), video path, or camera index.
            img_size: Inference image size.
            save: Whether to save annotated results.
            **kwargs: Additional arguments passed to YOLO.predict().
            
        Returns:
            List of detection dictionaries, each containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - class_name: Detected class name ('lemon')
        """
        results = self.model.predict(
            source=source,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=img_size,
            save=save,
            verbose=False,
            **kwargs,
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    raw_name = result.names.get(cls_id, "unknown") if hasattr(result, "names") else "unknown"
                    
                    # Target classification logic:
                    # 1. Human / Operator Detection
                    if raw_name in ["person", "human"]:
                        target_type = "human"
                        display_name = "Human / Operator"
                    # 2. Lemon / Citrus / Fruit Detection (including round fruits & yellow items)
                    elif raw_name in ["lemon", "orange", "apple", "banana", "sports ball"]:
                        target_type = "lemon"
                        display_name = "lemon"
                    # 3. Other Non-Harvest Object Detection
                    else:
                        target_type = "object"
                        display_name = raw_name

                    det = {
                        "bbox": box.xyxy[0].cpu().numpy().tolist(),
                        "confidence": conf,
                        "class_name": display_name,
                        "raw_class": raw_name,
                        "target_type": target_type,
                    }
                    detections.append(det)
        
        logger.info(f"Detections: {len(detections)} item(s) found")
        return detections

    def evaluate(
        self,
        data_yaml: str,
        img_size: int = 640,
        split: str = "test",
    ) -> dict:
        """Evaluate model on a dataset split.
        
        Args:
            data_yaml: Path to data.yaml.
            img_size: Image size for evaluation.
            split: Dataset split to evaluate on ('val' or 'test').
            
        Returns:
            Dictionary of evaluation metrics (mAP, precision, recall).
        """
        logger.info(f"Evaluating on {split} split...")
        metrics = self.model.val(
            data=data_yaml,
            imgsz=img_size,
            split=split,
        )
        results = {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        logger.info(f"Results: {results}")
        return results

    def export(
        self,
        format: str = "onnx",
        img_size: int = 640,
        **kwargs: Any,
    ) -> str:
        """Export model to deployment format.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'engine' for TensorRT).
            img_size: Input image size for export.
            **kwargs: Additional export arguments.
            
        Returns:
            Path to the exported model file.
        """
        logger.info(f"Exporting model to {format} format...")
        path = self.model.export(format=format, imgsz=img_size, **kwargs)
        logger.info(f"Model exported to: {path}")
        return str(path)

    def get_raw_results(self, source: Union[str, np.ndarray], **kwargs: Any):
        """Get raw Ultralytics Results objects for advanced usage."""
        return self.model.predict(
            source=source,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            **kwargs,
        )
