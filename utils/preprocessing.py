"""Image preprocessing and data augmentation utilities for Smart Lemon Harvester.

Includes image loading/resizing, tensor preparation for classification models,
OpenCV-based augmentation pipelines, and image format validation.
"""

from pathlib import Path
import random
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch


# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VALID_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Optional[Union[Tuple[int, int], int]] = None,
    to_rgb: bool = False,
) -> np.ndarray:
    """Load and optionally resize an image from disk.

    Uses a Windows-compatible file reader to safely handle file paths with
    spaces or non-ASCII characters.

    Args:
        image_path: Path to the image file.
        target_size: Optional (width, height) tuple or single integer (square).
        to_rgb: If True, convert from BGR to RGB color space.

    Returns:
        Loaded image as a NumPy array (dtype uint8).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at path: {path.resolve()}")

    # Safe binary read + cv2 decode (works reliably across Windows/Linux)
    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError(f"Failed to decode image at {path}: {exc}") from exc

    if img is None or img.size == 0:
        raise ValueError(f"Corrupted or unsupported image file: {path}")

    # Resize if requested
    if target_size is not None:
        if isinstance(target_size, int):
            new_size = (target_size, target_size)
        else:
            new_size = (int(target_size[0]), int(target_size[1]))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def preprocess_for_classifier(
    crop: Union[np.ndarray, Image.Image],
    input_size: int = 224,
) -> torch.Tensor:
    """Preprocess a cropped lemon region into a normalized PyTorch tensor.

    Applies resizing, RGB conversion, scaling to [0, 1], ImageNet mean/std
    normalization, and dimension reshaping to (1, 3, H, W).

    Args:
        crop: Cropped lemon region as a BGR NumPy array or PIL Image.
        input_size: Target square image dimension expected by the classifier.

    Returns:
        Preprocessed PyTorch FloatTensor of shape (1, 3, input_size, input_size).

    Raises:
        ValueError: If crop input is empty or invalid.
    """
    if isinstance(crop, Image.Image):
        crop_rgb = np.array(crop.convert("RGB"))
    elif isinstance(crop, np.ndarray):
        if crop.size == 0:
            raise ValueError("Input crop array is empty (size 0).")
        # Handle grayscale or BGR
        if crop.ndim == 2:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.ndim == 3 and crop.shape[2] == 3:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        elif crop.ndim == 3 and crop.shape[2] == 4:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            crop_rgb = crop
    else:
        raise TypeError(f"Unsupported crop type: {type(crop)}. Expected np.ndarray or PIL.Image.")

    # Resize to classifier input dimensions
    resized = cv2.resize(crop_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # Scale to [0.0, 1.0] float32
    float_img = resized.astype(np.float32) / 255.0

    # ImageNet normalization: (x - mean) / std
    normalized = (float_img - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose from (H, W, C) to (C, H, W)
    chw = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension (1, C, H, W) and convert to torch Tensor
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    return tensor


def apply_augmentation(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Apply stochastic data augmentation to an image using OpenCV operations.

    Supported augmentations:
    - Random Horizontal Flip
    - Random Vertical Flip
    - Random Rotation (-30 deg to +30 deg)
    - Brightness and Contrast Scaling
    - HSV Color Jitter (Hue, Saturation, Value)
    - Gaussian Blur

    Args:
        image: Source BGR image array.
        config: Optional dictionary to customize augmentation probabilities
            and ranges. Keys include:
            - 'hflip_prob' (default: 0.5)
            - 'vflip_prob' (default: 0.2)
            - 'rotate_prob' (default: 0.5)
            - 'max_rotate_angle' (default: 25.0)
            - 'color_jitter_prob' (default: 0.5)
            - 'blur_prob' (default: 0.3)

    Returns:
        Augmented BGR image as a new NumPy array.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid input image for augmentation.")

    cfg = config or {}
    aug = image.copy()
    h, w = aug.shape[:2]

    # 1. Random Horizontal Flip
    hflip_prob = cfg.get("hflip_prob", 0.5)
    if random.random() < hflip_prob:
        aug = cv2.flip(aug, 1)

    # 2. Random Vertical Flip
    vflip_prob = cfg.get("vflip_prob", 0.2)
    if random.random() < vflip_prob:
        aug = cv2.flip(aug, 0)

    # 3. Random Rotation
    rotate_prob = cfg.get("rotate_prob", 0.5)
    if random.random() < rotate_prob:
        max_angle = float(cfg.get("max_rotate_angle", 25.0))
        angle = random.uniform(-max_angle, max_angle)
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        aug = cv2.warpAffine(aug, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)

    # 4. Color Jitter (HSV adjustments)
    jitter_prob = cfg.get("color_jitter_prob", 0.5)
    if random.random() < jitter_prob and aug.ndim == 3 and aug.shape[2] == 3:
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_shift = random.uniform(-10.0, 10.0)
        s_scale = random.uniform(0.8, 1.25)
        v_scale = random.uniform(0.8, 1.25)

        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0.0, 255.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0.0, 255.0)

        aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 5. Brightness & Contrast Adjustment
    contrast_prob = cfg.get("contrast_prob", 0.4)
    if random.random() < contrast_prob:
        alpha = random.uniform(0.85, 1.15)  # Contrast factor
        beta = random.uniform(-20, 20)      # Brightness bias
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # 6. Gaussian Blur
    blur_prob = cfg.get("blur_prob", 0.3)
    if random.random() < blur_prob:
        ksize = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    return aug


def validate_image(source: Union[str, Path, np.ndarray, bytes]) -> bool:
    """Validate whether the source represents a readable, non-empty image.

    Args:
        source: Image file path (str/Path), raw image bytes, or NumPy array.

    Returns:
        True if valid and decodable, False otherwise.
    """
    if source is None:
        return False

    # Check NumPy array
    if isinstance(source, np.ndarray):
        return (
            source.size > 0
            and source.ndim in (2, 3)
            and source.shape[0] > 0
            and source.shape[1] > 0
        )

    # Check raw bytes
    if isinstance(source, (bytes, bytearray)):
        if len(source) == 0:
            return False
        try:
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            return img is not None and img.size > 0
        except Exception:
            return False

    # Check file path
    try:
        path = Path(source)
        if not path.is_file():
            return False
        if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            return False
        if path.stat().st_size == 0:
            return False
        # Verify decodability
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
        return img is not None and img.size > 0
    except Exception:
        return False
