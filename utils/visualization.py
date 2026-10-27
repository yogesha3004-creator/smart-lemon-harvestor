"""Visualization utilities for Smart Lemon Harvester.

Provides functions for rendering detection bounding boxes, ripeness and quality
labels, harvest decision indicators (checkmark/cross icons), and real-time HUD
summary overlays.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# Standard BGR color constants
COLOR_RIPE = (36, 179, 36)        # Vibrant Green
COLOR_SEMI_RIPE = (0, 215, 255)    # Vibrant Amber / Yellow
COLOR_RAW = (40, 40, 220)          # Red
COLOR_HARVEST_YES = (36, 179, 36)  # Green for harvestable
COLOR_HARVEST_NO = (0, 69, 255)    # Orange/Red for reject
COLOR_DEFAULT = (255, 178, 50)     # Orange
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_PANEL_BG = (20, 24, 28)      # Dark slate for HUD panel

RIPENESS_COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "ripe": COLOR_RIPE,
    "semi_ripe": COLOR_SEMI_RIPE,
    "raw": COLOR_RAW,
    "unripe": COLOR_RAW,
}


def _get_scale_factors(image: np.ndarray) -> Tuple[float, int, int]:
    """Calculate adaptive font scale, text thickness, and box thickness.

    Args:
        image: Input image array (H, W, C).

    Returns:
        Tuple of (font_scale, text_thickness, box_thickness).
    """
    h, w = image.shape[:2]
    diagonal = math.hypot(h, w)
    # Calibrated against a standard 1280x720 frame
    scale = max(0.45, min(1.3, (diagonal / 1468.0) * 0.72))
    text_thickness = max(1, int(round(scale * 2.0)))
    box_thickness = max(2, int(round(scale * 3.2)))
    return scale, text_thickness, box_thickness


def _draw_corner_brackets(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int,
    length: int = 18,
) -> None:
    """Draw high-tech corner brackets around an object bounding box."""
    bracket_len = min(length, max(8, int((x2 - x1) * 0.25)))
    thick = max(2, thickness + 1)
    # Top-Left
    cv2.line(image, (x1, y1), (x1 + bracket_len, y1), color, thick, cv2.LINE_AA)
    cv2.line(image, (x1, y1), (x1, y1 + bracket_len), color, thick, cv2.LINE_AA)
    # Top-Right
    cv2.line(image, (x2, y1), (x2 - bracket_len, y1), color, thick, cv2.LINE_AA)
    cv2.line(image, (x2, y1), (x2, y1 + bracket_len), color, thick, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(image, (x1, y2), (x1 + bracket_len, y2), color, thick, cv2.LINE_AA)
    cv2.line(image, (x1, y2), (x1, y2 - bracket_len), color, thick, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(image, (x2, y2), (x2 - bracket_len, y2), color, thick, cv2.LINE_AA)
    cv2.line(image, (x2, y2), (x2, y2 - bracket_len), color, thick, cv2.LINE_AA)


def _draw_harvest_icon(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    is_harvestable: bool,
) -> None:
    """Draw a clean vector checkmark or cross icon inside a circular badge.

    Using OpenCV lines ensures vector quality across all OS platforms and
    avoids Unicode rendering artifacts with Hershey fonts.

    Args:
        image: BGR image to draw on (modified in-place).
        center: (x, y) center coordinates of the badge.
        radius: Radius of the badge in pixels.
        is_harvestable: If True draws green checkmark, else red cross.
    """
    cx, cy = center
    badge_color = COLOR_HARVEST_YES if is_harvestable else COLOR_HARVEST_NO
    border_color = (255, 255, 255)

    # Filled circular background badge
    cv2.circle(image, (cx, cy), radius, badge_color, thickness=-1, lineType=cv2.LINE_AA)
    # White border ring
    cv2.circle(image, (cx, cy), radius, border_color, thickness=max(1, radius // 7), lineType=cv2.LINE_AA)

    thickness = max(2, int(round(radius * 0.22)))
    if is_harvestable:
        # Checkmark (✓) coordinates: 3 key points
        p1 = (int(cx - radius * 0.45), int(cy - radius * 0.05))
        p2 = (int(cx - radius * 0.10), int(cy + radius * 0.35))
        p3 = (int(cx + radius * 0.45), int(cy - radius * 0.35))
        cv2.line(image, p1, p2, COLOR_WHITE, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.line(image, p2, p3, COLOR_WHITE, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        # Cross (✗) coordinates: 2 diagonal lines
        offset = int(radius * 0.38)
        cv2.line(image, (cx - offset, cy - offset), (cx + offset, cy + offset),
                 COLOR_WHITE, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.line(image, (cx - offset, cy + offset), (cx + offset, cy - offset),
                 COLOR_WHITE, thickness=thickness, lineType=cv2.LINE_AA)


def draw_detections(
    image: np.ndarray,
    detections: Union[List[Dict[str, Any]], np.ndarray],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw bounding boxes and class/confidence labels on detected lemons.

    Args:
        image: Source BGR image (H, W, 3).
        detections: List of detection dictionaries (with 'bbox'/'box',
            'confidence'/'score', and optional 'class_name'/'class_id') or
            a NumPy array of shape (N, 4), (N, 5), or (N, 6).
        class_names: Optional list of class names indexed by class_id.

    Returns:
        Annotated BGR image as a new NumPy array.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    font_scale, text_thick, box_thick = _get_scale_factors(annotated)

    # Convert numpy detections to standard list of dicts if needed
    det_list: List[Dict[str, Any]] = []
    if isinstance(detections, np.ndarray):
        for row in detections:
            if len(row) >= 4:
                bbox = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                conf = float(row[4]) if len(row) > 4 else 1.0
                cls_id = int(row[5]) if len(row) > 5 else 0
                name = class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else "lemon"
                det_list.append({"bbox": bbox, "confidence": conf, "class_name": name})
    elif isinstance(detections, list):
        det_list = detections

    for det in det_list:
        # Parse bounding box
        bbox = det.get("bbox") or det.get("box") or det.get("xyxy")
        if bbox is None or len(bbox) < 4:
            continue
        x1 = max(0, min(w - 1, int(round(bbox[0]))))
        y1 = max(0, min(h - 1, int(round(bbox[1]))))
        x2 = max(0, min(w - 1, int(round(bbox[2]))))
        y2 = max(0, min(h - 1, int(round(bbox[3]))))

        if x2 <= x1 or y2 <= y1:
            continue

        conf = det.get("confidence", det.get("score", 1.0))
        name = det.get("class_name")
        if not name:
            cls_id = det.get("class_id", 0)
            name = class_names[cls_id] if class_names and 0 <= cls_id < len(class_names) else "lemon"

        label = f"{name} {conf:.2f}"
        box_color = COLOR_DEFAULT

        # Draw bounding rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness=box_thick, lineType=cv2.LINE_AA)

        # Draw label background box
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick
        )
        bg_y1 = max(0, y1 - label_h - baseline - 4)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(w, x1 + label_w + 8)

        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, thickness=-1)
        text_y = bg_y2 - baseline - 2
        cv2.putText(
            annotated,
            label,
            (bg_x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLOR_BLACK,
            thickness=text_thick,
            lineType=cv2.LINE_AA,
        )

    return annotated


def draw_pipeline_results(
    image: np.ndarray,
    pipeline_results: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> np.ndarray:
    """Draw end-to-end lemon harvest pipeline results on an image.

    Features:
    - Ripeness color-coded boxes:
        * Green for ripe
        * Yellow for semi_ripe
        * Red for raw
    - Informative badges with ripeness, quality grade, and size estimation.
    - Prominent harvest recommendation vector badge: (✓) Green or (✗) Red.

    Args:
        image: Source BGR image.
        pipeline_results: Either pipeline output dictionary containing 'detections'
            and optional 'summary', or direct list of detection dicts.

    Returns:
        Annotated BGR image as a new NumPy array.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    font_scale, text_thick, box_thick = _get_scale_factors(annotated)

    # Extract detections list
    if isinstance(pipeline_results, dict):
        detections = pipeline_results.get("detections", [])
    elif isinstance(pipeline_results, list):
        detections = pipeline_results
    else:
        detections = []

    for idx, det in enumerate(detections, start=1):
        bbox = det.get("bbox") or det.get("box") or det.get("xyxy")
        if bbox is None or len(bbox) < 4:
            continue

        x1 = max(0, min(w - 1, int(round(bbox[0]))))
        y1 = max(0, min(h - 1, int(round(bbox[1]))))
        x2 = max(0, min(w - 1, int(round(bbox[2]))))
        y2 = max(0, min(h - 1, int(round(bbox[3]))))

        if x2 <= x1 or y2 <= y1:
            continue

        # Extract attributes
        target_type = det.get("target_type", "lemon")
        class_name = det.get("class_name", "lemon")
        conf = det.get("detection_confidence", 0.0)

        # ----------------------------------------------------
        # BRANCH 1: Human / Operator
        # ----------------------------------------------------
        if target_type == "human":
            color = (255, 200, 0)  # Neon Cyan / Sky Blue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=box_thick, lineType=cv2.LINE_AA)
            _draw_corner_brackets(annotated, x1, y1, x2, y2, color, box_thick)
            line1 = f"HUMAN / OPERATOR ({conf*100:.0f}%)"
            line2 = "SAFETY INTERLOCK: HOLD"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w1, h1), b1 = cv2.getTextSize(line1, font, font_scale, text_thick)
            (w2, h2), b2 = cv2.getTextSize(line2, font, font_scale * 0.85, max(1, text_thick - 1))
            banner_w = max(w1, w2) + 12
            banner_h = h1 + h2 + b1 + b2 + 14
            banner_y1 = max(0, y1 - banner_h)
            banner_y2 = y1 if y1 - banner_h >= 0 else min(h, y1 + banner_h)
            banner_x2 = min(w, x1 + banner_w)
            sub_img = annotated[banner_y1:banner_y2, x1:banner_x2]
            dark_bg = np.full_like(sub_img, (30, 20, 10))
            cv2.addWeighted(sub_img, 0.25, dark_bg, 0.75, 0, sub_img)
            cv2.putText(annotated, line1, (x1 + 6, banner_y1 + h1 + 4), font, font_scale, color, text_thick, cv2.LINE_AA)
            cv2.putText(annotated, line2, (x1 + 6, banner_y1 + h1 + h2 + b1 + 8), font, font_scale * 0.85, (200, 240, 255), max(1, text_thick - 1), cv2.LINE_AA)
            continue

        # ----------------------------------------------------
        # BRANCH 2: Non-Harvest Object (cup, phone, etc.)
        # ----------------------------------------------------
        if target_type == "object":
            color = (180, 140, 140)  # Slate / Muted Lavender
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=box_thick, lineType=cv2.LINE_AA)
            _draw_corner_brackets(annotated, x1, y1, x2, y2, color, box_thick)
            line1 = f"OBJECT: {class_name.upper()}"
            line2 = "NON-HARVEST TARGET (IGNORED)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w1, h1), b1 = cv2.getTextSize(line1, font, font_scale, text_thick)
            (w2, h2), b2 = cv2.getTextSize(line2, font, font_scale * 0.82, max(1, text_thick - 1))
            banner_w = max(w1, w2) + 12
            banner_h = h1 + h2 + b1 + b2 + 14
            banner_y1 = max(0, y1 - banner_h)
            banner_y2 = y1 if y1 - banner_h >= 0 else min(h, y1 + banner_h)
            banner_x2 = min(w, x1 + banner_w)
            sub_img = annotated[banner_y1:banner_y2, x1:banner_x2]
            dark_bg = np.full_like(sub_img, (25, 25, 25))
            cv2.addWeighted(sub_img, 0.25, dark_bg, 0.75, 0, sub_img)
            cv2.putText(annotated, line1, (x1 + 6, banner_y1 + h1 + 4), font, font_scale, (230, 230, 230), text_thick, cv2.LINE_AA)
            cv2.putText(annotated, line2, (x1 + 6, banner_y1 + h1 + h2 + b1 + 8), font, font_scale * 0.82, color, max(1, text_thick - 1), cv2.LINE_AA)
            continue

        # ----------------------------------------------------
        # BRANCH 3: Lemon Target (Harvest Verified)
        # ----------------------------------------------------
        ripeness = str(det.get("ripeness", "unknown")).lower()
        ripeness_conf = det.get("ripeness_confidence")
        quality = str(det.get("quality", "n/a"))
        size = str(det.get("size", "n/a"))
        is_harvestable = bool(det.get("harvest", False))

        # Color based on ripeness
        color = RIPENESS_COLOR_MAP.get(ripeness, COLOR_DEFAULT)

        # 1. Draw bounding box and high-tech targeting brackets
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=box_thick, lineType=cv2.LINE_AA)
        _draw_corner_brackets(annotated, x1, y1, x2, y2, color, box_thick)

        # 2. Draw Harvest Icon badge on top-right of bounding box
        badge_radius = max(10, min(24, int(round((x2 - x1) * 0.12))))
        badge_center = (
            max(badge_radius, min(w - badge_radius, x2)),
            max(badge_radius, min(h - badge_radius, y1)),
        )
        _draw_harvest_icon(annotated, badge_center, badge_radius, is_harvestable)

        # 3. Build text lines
        r_conf_str = f" {ripeness_conf * 100:.0f}%" if ripeness_conf is not None else ""
        line1 = f"#{idx} {ripeness.upper()}{r_conf_str} | {quality}"
        harvest_str = "HARVEST [YES]" if is_harvestable else "SKIP [NO]"
        line2 = f"Size: {size.capitalize()} | {harvest_str}"

        # Calculate text banner dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w1, h1), b1 = cv2.getTextSize(line1, font, font_scale, text_thick)
        (w2, h2), b2 = cv2.getTextSize(line2, font, font_scale * 0.88, max(1, text_thick - 1))

        banner_w = max(w1, w2) + 12
        banner_h = h1 + h2 + b1 + b2 + 14

        # Place banner above box if space permits, otherwise inside/below
        if y1 - banner_h >= 0:
            banner_y1 = y1 - banner_h
            banner_y2 = y1
        elif y2 + banner_h <= h:
            banner_y1 = y2
            banner_y2 = y2 + banner_h
        else:
            banner_y1 = y1
            banner_y2 = min(h, y1 + banner_h)

        banner_x1 = max(0, min(w - banner_w, x1))
        banner_x2 = min(w, banner_x1 + banner_w)

        # Draw semi-transparent background for readability
        overlay = annotated.copy()
        cv2.rectangle(overlay, (banner_x1, banner_y1), (banner_x2, banner_y2), COLOR_PANEL_BG, thickness=-1)
        # Small colored accent bar on left side of banner
        accent_w = max(3, box_thick + 1)
        cv2.rectangle(
            overlay,
            (banner_x1, banner_y1),
            (banner_x1 + accent_w, banner_y2),
            color,
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # Draw text lines
        text_x = banner_x1 + accent_w + 6
        text_y1 = banner_y1 + h1 + 4
        text_y2 = text_y1 + h2 + 6

        cv2.putText(
            annotated,
            line1,
            (text_x, text_y1),
            font,
            font_scale,
            COLOR_WHITE,
            thickness=text_thick,
            lineType=cv2.LINE_AA,
        )

        harvest_text_color = COLOR_RIPE if is_harvestable else (150, 150, 255)
        cv2.putText(
            annotated,
            line2,
            (text_x, text_y2),
            font,
            font_scale * 0.88,
            harvest_text_color,
            thickness=max(1, text_thick - 1),
            lineType=cv2.LINE_AA,
        )

    return annotated


def create_summary_overlay(
    image: np.ndarray,
    summary: Dict[str, Any],
    position: str = "top-left",
    alpha: float = 0.75,
) -> np.ndarray:
    """Add a semi-transparent HUD summary panel to an image.

    Displays:
    - Total lemons detected
    - Harvestable vs. not ready count
    - Ripeness distribution breakdown
    - Quality and size statistics (if present in summary)

    Args:
        image: Source BGR image.
        summary: Summary dict (e.g. from LemonHarvestPipeline.process_image).
        position: Placement of panel: 'top-left', 'top-right', 'bottom-left', or 'bottom-right'.
        alpha: Opacity factor for the overlay background (0.0 to 1.0).

    Returns:
        Annotated BGR image with HUD overlay as a new NumPy array.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    total = summary.get("total_lemons", 0)
    harvestable = summary.get("harvestable", 0)
    not_ready = summary.get("not_ready", max(0, total - harvestable))
    ripeness_dist = summary.get("ripeness_distribution", {})
    quality_dist = summary.get("quality_distribution", {})

    # Adaptive font scaling for HUD
    scale = max(0.40, min(0.75, max(h, w) / 1600.0 * 0.65))
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_spacing = int(22 * (scale / 0.5))
    thick = 1

    lines: List[Tuple[str, Tuple[int, int, int]]] = [
        ("🍋 SMART LEMON HARVESTER", COLOR_SEMI_RIPE),
        ("----------------------------------", (100, 100, 100)),
        (f"Total Detected:  {total}", COLOR_WHITE),
        (f"Harvest Ready:   {harvestable}", COLOR_RIPE if harvestable > 0 else COLOR_WHITE),
        (f"Not Ready:       {not_ready}", (160, 160, 255) if not_ready > 0 else COLOR_WHITE),
    ]

    # Add Ripeness Breakdown
    if ripeness_dist:
        ripe_cnt = ripeness_dist.get("ripe", 0)
        semi_cnt = ripeness_dist.get("semi_ripe", 0)
        raw_cnt = ripeness_dist.get("raw", 0)
        lines.append((f"Ripeness: Ripe:{ripe_cnt} | Semi:{semi_cnt} | Raw:{raw_cnt}", COLOR_WHITE))

    # Add Quality Breakdown
    if quality_dist:
        q_parts = [f"{k.replace('grade_', 'Gr.').upper()}:{v}" for k, v in quality_dist.items()]
        lines.append((f"Quality:  {' | '.join(q_parts)}", (200, 200, 200)))

    # Compute panel dimensions
    max_line_w = 0
    for text, _ in lines:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        if tw > max_line_w:
            max_line_w = tw

    margin = 16
    pad = 12
    panel_w = max_line_w + (pad * 2)
    panel_h = (len(lines) * line_spacing) + (pad * 2)

    # Determine coordinates based on position
    if position == "top-right":
        px1 = w - panel_w - margin
        py1 = margin
    elif position == "bottom-left":
        px1 = margin
        py1 = h - panel_h - margin
    elif position == "bottom-right":
        px1 = w - panel_w - margin
        py1 = h - panel_h - margin
    else:  # default 'top-left'
        px1 = margin
        py1 = margin

    px1 = max(0, px1)
    py1 = max(0, py1)
    px2 = min(w, px1 + panel_w)
    py2 = min(h, py1 + panel_h)

    # Draw semi-transparent rounded rectangle card
    overlay = annotated.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_PANEL_BG, thickness=-1)
    # Thin border around HUD
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (70, 80, 90), thickness=1)
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

    # Render lines of text
    curr_y = py1 + pad + int(line_spacing * 0.75)
    for text, text_color in lines:
        cv2.putText(
            annotated,
            text,
            (px1 + pad, curr_y),
            font,
            scale,
            text_color,
            thickness=thick,
            lineType=cv2.LINE_AA,
        )
        curr_y += line_spacing

    return annotated
