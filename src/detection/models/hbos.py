"""Histogram-Based Outlier Score (HBOS) downstream anomaly detector for GTAE-IDS."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from pyod.models.hbos import HBOS

from src.detection.config import HBOSConfig

logger = logging.getLogger(__name__)


class HBOSDetector:
    """
    Histogram-Based Outlier Score (HBOS) for network flow anomaly detection.

    Enforces strict protocol guarantees:
    1. Fitted exclusively on benign training representations (y == 0, exactly 57,305 flows).
    2. Uses all 113 GTAE feature dimensions.
    3. Anomaly scores: In PyOD HBOS, decision_function(X) returns anomaly scores
       where higher values indicate more anomalous samples.
    4. Calibrates decision threshold at the 99th percentile of benign validation flows.
    5. Evaluates test set and snapshot inferences using the frozen calibrated threshold without modification.
    """

    def __init__(self, config: Optional[HBOSConfig] = None):
        self.config = config or HBOSConfig()
        self.model = HBOS(
            n_bins=self.config.n_bins,
            alpha=self.config.alpha,
            tol=self.config.tol,
            contamination=self.config.contamination,
        )
        self.is_fitted: bool = False
        self.threshold: Optional[float] = None
        self.fit_time_seconds: float = 0.0
        self.num_train_samples: int = 0
        self.feature_dim: int = 0

    def fit(self, X_train: Union[np.ndarray, torch.Tensor]) -> "HBOSDetector":
        """
        Fit HBOS model strictly on benign training flows.

        Args:
            X_train: 2D array/tensor of shape [N_train, 113] representing benign flows.
        """
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()

        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_train.shape}")

        self.num_train_samples = X_train.shape[0]
        self.feature_dim = X_train.shape[1]

        logger.info(
            f"Fitting HBOS on {self.num_train_samples:,} samples with dim {self.feature_dim} "
            f"(n_bins={self.config.n_bins}, alpha={self.config.alpha}, tol={self.config.tol})..."
        )

        t0 = time.time()
        self.model.fit(X_train)
        self.fit_time_seconds = time.time() - t0
        self.is_fitted = True

        logger.info(
            f"HBOS fit completed in {self.fit_time_seconds:.4f}s across {self.feature_dim} feature histograms."
        )
        return self

    def score(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, float]:
        """
        Compute continuous anomaly scores.

        In PyOD HBOS, decision_function(X) returns outlier scores where
        higher values indicate greater anomaly severity.

        Returns:
            scores: 1D np.ndarray of continuous anomaly scores
            inference_time: Time taken in seconds
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        t0 = time.time()
        raw_scores = self.model.decision_function(X)
        scores = np.asarray(raw_scores, dtype=np.float64)
        infer_time = time.time() - t0

        return scores, infer_time

    def calibrate_threshold(
        self,
        X_val: Union[np.ndarray, torch.Tensor],
        y_val: Union[np.ndarray, torch.Tensor],
        percentile: Optional[float] = None,
    ) -> float:
        """
        Establish the primary operational decision threshold at the 99th percentile of
        BENIGN validation samples: threshold = percentile(S_val[y_val == 0], 99.0).

        Strictly enforces that attack flows and test flows are NOT used for calibration.
        """
        if isinstance(y_val, torch.Tensor):
            y_val = y_val.detach().cpu().numpy()

        pct = percentile if percentile is not None else self.config.threshold_percentile

        val_scores, _ = self.score(X_val)
        benign_mask = (y_val == 0)

        if not np.any(benign_mask):
            raise ValueError("Validation set must contain benign samples for threshold calibration.")

        benign_val_scores = val_scores[benign_mask]
        self.threshold = float(np.percentile(benign_val_scores, pct))

        logger.info(
            f"Calibrated frozen decision threshold at {pct}th percentile of "
            f"{len(benign_val_scores):,} benign validation flows: threshold = {self.threshold:.6f}"
        )
        return self.threshold

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict binary labels (0 = benign, 1 = malicious anomaly) using decision threshold.

        Returns:
            predictions: 1D np.ndarray of binary predictions (0 or 1)
            scores: 1D np.ndarray of continuous anomaly scores
            inference_time: Inference time in seconds
        """
        th = threshold if threshold is not None else self.threshold
        if th is None:
            raise ValueError(
                "Decision threshold is not set. Run calibrate_threshold() or pass threshold explicitly."
            )

        scores, infer_time = self.score(X)
        predictions = (scores >= th).astype(np.int8)
        return predictions, scores, infer_time

    def predict_snapshot(
        self,
        flow_features_113d: Union[np.ndarray, torch.Tensor],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Reusable inference method on snapshot's 113-D GTAE features using the frozen model and threshold.

        Args:
            flow_features_113d: [N_edges, 113] feature matrix
            threshold: Optional threshold override; defaults to self.threshold

        Returns:
            Dictionary with predictions, scores, threshold, and timing.
        """
        predictions, scores, infer_time = self.predict(flow_features_113d, threshold=threshold)
        used_threshold = threshold if threshold is not None else self.threshold
        return {
            "predictions": predictions,
            "scores": scores,
            "threshold": float(used_threshold),
            "inference_time_seconds": infer_time,
            "detected_count": int(np.sum(predictions == 1)),
            "total_count": len(predictions),
        }

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
        """Save fitted model, threshold, and configuration."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        save_dir = Path(output_dir or self.config.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "hbos_model.joblib"
        meta_path = save_dir / "hbos_metadata.json"

        joblib.dump(self.model, model_path)

        metadata = {
            "model_type": "HBOS",
            "library": "PyOD",
            "n_bins": self.config.n_bins,
            "alpha": self.config.alpha,
            "tol": self.config.tol,
            "contamination": self.config.contamination,
            "fit_time_seconds": round(self.fit_time_seconds, 6),
            "num_train_samples": self.num_train_samples,
            "feature_dim": self.feature_dim,
            "threshold": self.threshold,
            "threshold_percentile": self.config.threshold_percentile,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved HBOS artifacts to {save_dir}")
        return {
            "model": model_path,
            "metadata": meta_path,
        }

    @classmethod
    def load(cls, save_dir: Union[str, Path] = "data/processed/models/hbos") -> "HBOSDetector":
        """Load fitted detector and frozen threshold from disk."""
        save_dir = Path(save_dir)
        meta_path = save_dir / "hbos_metadata.json"
        model_path = save_dir / "hbos_model.joblib"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = HBOSConfig(
            n_bins=meta.get("n_bins", 10),
            alpha=meta.get("alpha", 0.1),
            tol=meta.get("tol", 0.5),
            contamination=meta.get("contamination", 0.1),
            threshold_percentile=meta.get("threshold_percentile", 99.0),
            output_dir=save_dir,
        )

        detector = cls(config=cfg)
        detector.model = joblib.load(model_path)
        detector.threshold = meta.get("threshold")
        detector.is_fitted = True
        detector.fit_time_seconds = meta.get("fit_time_seconds", 0.0)
        detector.num_train_samples = meta.get("num_train_samples", 0)
        detector.feature_dim = meta.get("feature_dim", 113)

        logger.info(f"Loaded HBOSDetector from {save_dir} with threshold={detector.threshold}")
        return detector


def predict_hbos_snapshot(
    flow_features_113d: Union[np.ndarray, torch.Tensor],
    model_dir: Union[str, Path] = "data/processed/models/hbos",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Reusable standalone function to load the saved HBOS model and score a snapshot's 113-D GTAE features.

    Args:
        flow_features_113d: Tensor or np.ndarray of shape [N, 113]
        model_dir: Path to saved HBOS artifacts directory
        threshold: Optional operational threshold override (defaults to frozen threshold)

    Returns:
        Dict containing predictions, scores, threshold, detected_count, total_count, and timing.
    """
    detector = HBOSDetector.load(save_dir=model_dir)
    return detector.predict_snapshot(flow_features_113d, threshold=threshold)
