"""Comprehensive metrics and evaluation engine for GTAE-IDS anomaly detectors."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_detector(
    y_true: Union[np.ndarray, torch.Tensor],
    scores: Union[np.ndarray, torch.Tensor],
    threshold: float,
    inference_time: float = 0.0,
    training_time: float = 0.0,
    max_curve_points: int = 100,
) -> Dict[str, Any]:
    """
    Calculate full evaluation metrics and curve coordinates.

    Args:
        y_true: Binary ground-truth labels (0 = benign, 1 = malicious anomaly)
        scores: Continuous standardized anomaly scores (higher = more anomalous)
        threshold: Frozen decision threshold
        inference_time: Time taken for scoring/inference
        training_time: Time taken for model fitting
        max_curve_points: Number of points to downsample ROC/PR curves for frontend rendering

    Returns:
        Dictionary of formatted evaluation metrics, counts, and curve coordinates.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    y_true = y_true.astype(np.int8)
    scores = scores.astype(np.float64)

    # Binary predictions using the frozen threshold
    predictions = (scores >= threshold).astype(np.int8)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    # Rates and metrics
    acc = float(accuracy_score(y_true, predictions))
    prec = float(precision_score(y_true, predictions, zero_division=0))
    rec = float(recall_score(y_true, predictions, zero_division=0))
    f1 = float(f1_score(y_true, predictions, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Area Under Curves (computed from continuous scores)
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except ValueError:
        pr_auc = 0.0

    # ROC curve points
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
    roc_points = _downsample_curve(fpr_arr, tpr_arr, max_curve_points)

    # Precision-Recall curve points
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
    pr_points = _downsample_curve(rec_arr, prec_arr, max_curve_points)

    total_flows = int(len(y_true))
    benign_total = int(tn + fp)
    attack_total = int(fn + tp)
    detected_anomalies = int(tp + fp)

    results = {
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "threshold": round(float(threshold), 6),
        },
        "counts": {
            "total_flows": total_flows,
            "benign_total": benign_total,
            "attack_total": attack_total,
            "detected_anomalies": detected_anomalies,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "timing": {
            "training_time_seconds": round(training_time, 4),
            "inference_time_seconds": round(inference_time, 4),
            "flows_per_second": round(total_flows / max(1e-4, inference_time), 1),
        },
        "curves": {
            "roc": roc_points,
            "pr": pr_points,
        },
    }

    return results


def _downsample_curve(
    x_arr: np.ndarray, y_arr: np.ndarray, max_points: int = 100
) -> List[Dict[str, float]]:
    """Downsample (x, y) curve coordinates evenly for lightweight web charting."""
    if len(x_arr) <= max_points:
        indices = np.arange(len(x_arr))
    else:
        indices = np.linspace(0, len(x_arr) - 1, max_points, dtype=int)

    points = []
    for idx in indices:
        points.append({
            "x": round(float(x_arr[idx]), 4),
            "y": round(float(y_arr[idx]), 4),
        })
    return points
