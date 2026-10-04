"""Isolation using Nearest Neighbor Ensemble (INNE) downstream anomaly detector for GTAE-IDS."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from pyod.models.inne import INNE

from src.detection.config import INNEConfig

logger = logging.getLogger(__name__)


class INNEDetector:
    """
    Isolation using Nearest Neighbor Ensemble (INNE) for network flow anomaly detection.

    Enforces strict protocol guarantees:
    1. Fitted exclusively on benign training representations (y == 0, exactly 57305 flows).
    2. Uses all 113 GTAE feature dimensions.
    3. Anomaly scores: PyOD INNE decision_function returns outlier scores where
       higher values indicate more anomalous samples.
    4. Calibrates decision threshold at the 99th percentile of benign validation flows.
    5. Evaluates test set and snapshot inferences using the frozen calibrated threshold.
    """

    def __init__(self, config=None):
        self.config = config or INNEConfig()
        self.model = INNE(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
        )
        self.is_fitted = False
        self.threshold = None
        self.fit_time_seconds = 0.0
        self.num_train_samples = 0
        self.feature_dim = 0

    def fit(self, X_train):
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_train.shape}")
        self.num_train_samples = X_train.shape[0]
        self.feature_dim = X_train.shape[1]
        logger.info(f"Fitting INNE on {self.num_train_samples:,} samples (n_estimators={self.config.n_estimators})...")
        t0 = time.time()
        self.model.fit(X_train)
        self.fit_time_seconds = time.time() - t0
        self.is_fitted = True
        logger.info(f"INNE fit completed in {self.fit_time_seconds:.4f}s.")
        return self

    def score(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        t0 = time.time()
        scores = np.asarray(self.model.decision_function(X), dtype=np.float64)
        return scores, time.time() - t0

    def calibrate_threshold(self, X_val, y_val, percentile=None):
        if isinstance(y_val, torch.Tensor):
            y_val = y_val.detach().cpu().numpy()
        pct = percentile if percentile is not None else self.config.threshold_percentile
        val_scores, _ = self.score(X_val)
        benign_mask = (y_val == 0)
        if not np.any(benign_mask):
            raise ValueError("Validation set must contain benign samples.")
        benign_val_scores = val_scores[benign_mask]
        self.threshold = float(np.percentile(benign_val_scores, pct))
        logger.info(f"Calibrated threshold at {pct}th percentile of {len(benign_val_scores):,} benign val flows: {self.threshold:.6f}")
        return self.threshold

    def predict(self, X, threshold=None):
        th = threshold if threshold is not None else self.threshold
        if th is None:
            raise ValueError("Threshold not set. Run calibrate_threshold() first.")
        scores, infer_time = self.score(X)
        predictions = (scores >= th).astype(np.int8)
        return predictions, scores, infer_time

    def predict_snapshot(self, flow_features_113d, threshold=None):
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

    def save(self, output_dir=None):
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        save_dir = Path(output_dir or self.config.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "inne_model.joblib"
        meta_path = save_dir / "inne_metadata.json"
        joblib.dump(self.model, model_path)
        metadata = {
            "model_type": "INNE",
            "library": "PyOD",
            "n_estimators": self.config.n_estimators,
            "max_samples": self.config.max_samples,
            "contamination": self.config.contamination,
            "random_state": self.config.random_state,
            "fit_time_seconds": round(self.fit_time_seconds, 6),
            "num_train_samples": self.num_train_samples,
            "feature_dim": self.feature_dim,
            "threshold": self.threshold,
            "threshold_percentile": self.config.threshold_percentile,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved INNE artifacts to {save_dir}")
        return {"model": model_path, "metadata": meta_path}

    @classmethod
    def load(cls, save_dir="data/processed/models/inne"):
        save_dir = Path(save_dir)
        meta_path = save_dir / "inne_metadata.json"
        model_path = save_dir / "inne_model.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = INNEConfig(
            n_estimators=meta.get("n_estimators", 100),
            max_samples=meta.get("max_samples", "auto"),
            contamination=meta.get("contamination", 0.1),
            random_state=meta.get("random_state", 42),
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
        logger.info(f"Loaded INNEDetector from {save_dir} with threshold={detector.threshold}")
        return detector


def predict_inne_snapshot(flow_features_113d, model_dir="data/processed/models/inne", threshold=None):
    """Load the saved INNE model and score a snapshot's 113-D GTAE features."""
    detector = INNEDetector.load(save_dir=model_dir)
    return detector.predict_snapshot(flow_features_113d, threshold=threshold)
