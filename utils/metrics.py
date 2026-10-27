"""Evaluation and performance metrics utilities for Smart Lemon Harvester.

Includes functions for calculating classification metrics (accuracy, precision,
recall, F1-score), plotting annotated confusion matrices, computing object
detection IoU-based precision/recall/mAP metrics, and formatting clean summary
tables for CLI output.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics.

    Computes overall accuracy, macro-averaged and weighted-averaged precision,
    recall, F1-score, per-class metrics, support counts, and confusion matrix.

    Args:
        y_true: True ground truth class labels or indices.
        y_pred: Predicted class labels or indices.
        class_names: Optional explicit list/ordering of class names.

    Returns:
        Dictionary containing:
            - 'accuracy': Overall classification accuracy.
            - 'macro_avg': Dict with macro precision, recall, f1.
            - 'weighted_avg': Dict with weighted precision, recall, f1.
            - 'per_class': Dict mapping each class to its metrics.
            - 'confusion_matrix': 2D list containing raw counts.
            - 'class_names': List of unique class labels evaluated.
            - 'total_samples': Number of evaluated samples.

    Raises:
        ValueError: If y_true and y_pred have differing lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    # Determine unique class labels
    if class_names is not None:
        labels = list(class_names)
    else:
        labels = sorted(list(set(list(y_true) + list(y_pred))))

    # Compute overall accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Compute per-class precision, recall, f1, support
    prec_per_cls, rec_per_cls, f1_per_cls, supp_per_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Compute macro and weighted averages
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_dict = {}
    for i, name in enumerate(labels):
        per_class_dict[str(name)] = {
            "precision": float(prec_per_cls[i]),
            "recall": float(rec_per_cls[i]),
            "f1_score": float(f1_per_cls[i]),
            "support": int(supp_per_cls[i]),
        }

    return {
        "accuracy": acc,
        "macro_avg": {
            "precision": float(macro_prec),
            "recall": float(macro_rec),
            "f1_score": float(macro_f1),
        },
        "weighted_avg": {
            "precision": float(weighted_prec),
            "recall": float(weighted_rec),
            "f1_score": float(weighted_f1),
        },
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": [str(c) for c in labels],
        "total_samples": len(y_true),
    }


def plot_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    class_names: Sequence[str],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
) -> str:
    """Plot and save a styled confusion matrix heatmap to disk.

    Args:
        y_true: True ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class labels for tick markings.
        save_path: Output file path (.png).
        title: Title of the generated plot.
        normalize: If True, displays normalized fractions alongside raw counts.
        cmap: Matplotlib colormap name.

    Returns:
        Absolute string path to the saved PNG image.
    """
    labels = list(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        # Create dual annotations (percentage + raw count)
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm_norm[i, j] * 100.0
                annot[i, j] = f"{pct:.1f}%\n({cm[i, j]})"
        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap=cmap,
            cbar=True,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=True,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Truth", fontsize=11, labelpad=8)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(out_file.resolve())


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_detection_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate object detection metrics (Precision, Recall, F1, and mAP).

    Args:
        predictions: List of dicts, each having:
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': float score
            - 'class_name' (optional, default 'lemon')
        ground_truth: List of dicts, each having:
            - 'bbox': [x1, y1, x2, y2]
            - 'class_name' (optional, default 'lemon')
        iou_threshold: IoU matching threshold (default: 0.5).

    Returns:
        Dictionary containing:
            - 'precision': Overall precision score.
            - 'recall': Overall recall score.
            - 'f1_score': Harmonic mean of precision and recall.
            - 'mAP50': Mean Average Precision at IoU threshold.
            - 'tp': Total true positives.
            - 'fp': Total false positives.
            - 'fn': Total false negatives.
            - 'iou_threshold': IoU threshold used.
            - 'per_class': Metrics broken down per class.
    """
    # Group by class
    pred_by_cls: Dict[str, List[Dict[str, Any]]] = {}
    gt_by_cls: Dict[str, List[Dict[str, Any]]] = {}

    for p in predictions:
        cls_name = p.get("class_name", "lemon")
        pred_by_cls.setdefault(cls_name, []).append(p)

    for g in ground_truth:
        cls_name = g.get("class_name", "lemon")
        gt_by_cls.setdefault(cls_name, []).append(g)

    all_classes = sorted(list(set(list(pred_by_cls.keys()) + list(gt_by_cls.keys()))))
    if not all_classes:
        all_classes = ["lemon"]

    per_class_results = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ap_sum = 0.0

    for cls_name in all_classes:
        preds = pred_by_cls.get(cls_name, [])
        gts = gt_by_cls.get(cls_name, [])

        num_gts = len(gts)
        num_preds = len(preds)

        if num_gts == 0 and num_preds == 0:
            per_class_results[cls_name] = {
                "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "ap": 1.0,
                "tp": 0, "fp": 0, "fn": 0,
            }
            ap_sum += 1.0
            continue

        if num_gts == 0 and num_preds > 0:
            per_class_results[cls_name] = {
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "ap": 0.0,
                "tp": 0, "fp": num_preds, "fn": 0,
            }
            total_fp += num_preds
            continue

        if num_preds == 0 and num_gts > 0:
            per_class_results[cls_name] = {
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "ap": 0.0,
                "tp": 0, "fp": 0, "fn": num_gts,
            }
            total_fn += num_gts
            continue

        # Sort predictions by confidence score descending
        sorted_preds = sorted(preds, key=lambda x: x.get("confidence", 0.0), reverse=True)
        matched_gt_indices = set()
        tp_list = []
        fp_list = []

        for p in sorted_preds:
            p_box = p.get("bbox", p.get("box", []))
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt_indices:
                    continue
                g_box = gt.get("bbox", gt.get("box", []))
                iou = _compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp_list.append(1)
                fp_list.append(0)
                matched_gt_indices.add(best_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_count = sum(tp_list)
        fp_count = sum(fp_list)
        fn_count = num_gts - tp_count

        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count

        cls_prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        cls_rec = tp_count / num_gts if num_gts > 0 else 0.0
        cls_f1 = (2 * cls_prec * cls_rec / (cls_prec + cls_rec)) if (cls_prec + cls_rec) > 0 else 0.0

        # Calculate Average Precision (AP) using standard cumsum PR curve
        cum_tp = np.cumsum(tp_list)
        cum_fp = np.cumsum(fp_list)
        precisions = cum_tp / (cum_tp + cum_fp)
        recalls = cum_tp / num_gts

        # 11-point interpolation AP
        ap = 0.0
        for r_thresh in np.linspace(0.0, 1.0, 11):
            prec_at_r = precisions[recalls >= r_thresh]
            p_max = np.max(prec_at_r) if len(prec_at_r) > 0 else 0.0
            ap += p_max / 11.0

        ap_sum += ap
        per_class_results[cls_name] = {
            "precision": float(cls_prec),
            "recall": float(cls_rec),
            "f1_score": float(cls_f1),
            "ap": float(ap),
            "tp": int(tp_count),
            "fp": int(fp_count),
            "fn": int(fn_count),
        }

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        (2 * overall_prec * overall_rec / (overall_prec + overall_rec))
        if (overall_prec + overall_rec) > 0
        else 0.0
    )
    mAP50 = ap_sum / len(all_classes) if len(all_classes) > 0 else 0.0

    return {
        "precision": float(overall_prec),
        "recall": float(overall_rec),
        "f1_score": float(overall_f1),
        "mAP50": float(mAP50),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "iou_threshold": iou_threshold,
        "per_class": per_class_results,
    }


def format_metrics_table(metrics_dict: Dict[str, Any]) -> str:
    """Format any metrics dictionary into an aligned, readable table string.

    Supports classification reports, detection evaluations, and general stats.

    Args:
        metrics_dict: Dictionary returned by compute_classification_metrics,
            compute_detection_metrics, or similar evaluation tools.

    Returns:
        Formatted ASCII string representation of the metrics table.
    """
    lines: List[str] = []

    # Detection Metrics Table Format
    if "mAP50" in metrics_dict:
        lines.append("+----------------------------------------------------------------+")
        lines.append("|                   DETECTION EVALUATION REPORT                  |")
        lines.append("+----------------------+-----------+-----------+----------+------+")
        lines.append("| Metric               | Precision | Recall    | F1-Score | mAP50|")
        lines.append("+----------------------+-----------+-----------+----------+------+")
        p = metrics_dict.get("precision", 0.0)
        r = metrics_dict.get("recall", 0.0)
        f = metrics_dict.get("f1_score", 0.0)
        m = metrics_dict.get("mAP50", 0.0)
        lines.append(f"| Overall (IoU={metrics_dict.get('iou_threshold', 0.5):.2f})    | {p:9.3f} | {r:9.3f} | {f:8.3f} |{m:6.3f}|")
        lines.append("+----------------------+-----------+-----------+----------+------+")

        per_cls = metrics_dict.get("per_class", {})
        if per_cls:
            lines.append("| Class Breakdown:                                               |")
            lines.append("+----------------------+-----------+-----------+----------+------+")
            for c_name, c_m in per_cls.items():
                cp = c_m.get("precision", 0.0)
                cr = c_m.get("recall", 0.0)
                cf = c_m.get("f1_score", 0.0)
                cap = c_m.get("ap", 0.0)
                lines.append(f"| {c_name:<20} | {cp:9.3f} | {cr:9.3f} | {cf:8.3f} |{cap:6.3f}|")
            lines.append("+----------------------+-----------+-----------+----------+------+")
        lines.append(f"| TP: {metrics_dict.get('tp', 0):<5} | FP: {metrics_dict.get('fp', 0):<5} | FN: {metrics_dict.get('fn', 0):<5}                                |")
        lines.append("+----------------------------------------------------------------+")
        return "\n".join(lines)

    # Classification Metrics Table Format
    if "per_class" in metrics_dict and "accuracy" in metrics_dict:
        lines.append("+--------------------------------------------------------------------+")
        lines.append("|                  CLASSIFICATION EVALUATION REPORT                  |")
        lines.append("+----------------------+-----------+-----------+----------+----------+")
        lines.append("| Class                | Precision | Recall    | F1-Score |  Support |")
        lines.append("+----------------------+-----------+-----------+----------+----------+")
        for c_name, vals in metrics_dict["per_class"].items():
            cp = vals.get("precision", 0.0)
            cr = vals.get("recall", 0.0)
            cf = vals.get("f1_score", 0.0)
            cs = vals.get("support", 0)
            lines.append(f"| {c_name:<20} | {cp:9.3f} | {cr:9.3f} | {cf:8.3f} | {cs:8d} |")
        lines.append("+----------------------+-----------+-----------+----------+----------+")

        macro = metrics_dict.get("macro_avg", {})
        lines.append(
            f"| Macro Average        | {macro.get('precision', 0.0):9.3f} | "
            f"{macro.get('recall', 0.0):9.3f} | {macro.get('f1_score', 0.0):8.3f} | "
            f"{metrics_dict.get('total_samples', 0):8d} |"
        )
        weighted = metrics_dict.get("weighted_avg", {})
        lines.append(
            f"| Weighted Average     | {weighted.get('precision', 0.0):9.3f} | "
            f"{weighted.get('recall', 0.0):9.3f} | {weighted.get('f1_score', 0.0):8.3f} | "
            f"{metrics_dict.get('total_samples', 0):8d} |"
        )
        lines.append("+----------------------+-----------+-----------+----------+----------+")
        lines.append(f"| Overall Accuracy     : {metrics_dict.get('accuracy', 0.0):.4f}                                     |")
        lines.append("+--------------------------------------------------------------------+")
        return "\n".join(lines)

    # Generic Fallback Table Format
    lines.append("+-----------------------------------+--------------------+")
    lines.append("| Metric Key                        | Value              |")
    lines.append("+-----------------------------------+--------------------+")
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| {str(k):<33} | {val_str:>18} |")
    lines.append("+-----------------------------------+--------------------+")
    return "\n".join(lines)
