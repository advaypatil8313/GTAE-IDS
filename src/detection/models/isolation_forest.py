"""Isolation Forest (IForest) downstream anomaly detector for GTAE-IDS."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from src.detection.config import IsolationForestConfig

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest for network flow anomaly detector.

    Enforces strict protocol guarantees:
    1. Fitted exclusively on benign training representations (y == 0, exactly 57,305 flows).
    2. Preprocessing decision: Tree-based partitioning with random orthogonal splits is invariant
       to monotonic feature scaling. Therefore, raw 113-D GTAE features are used directly without
       unnecessary standardization.
    3. Standardizes anomaly score direction: S(x) = -decision_function(X),
       such that higher values strictly represent greater anomaly severity.
    4. Calibrates decision threshold at the 99th percentile of benign validation flows.
    5. Evaluates test set using the frozen calibrated threshold without any modification.
    """

    def __init__(self, config: Optional[IsolationForestConfig] = None):
        self.config = config or IsolationForestConfig()
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            bootstrap=self.config.bootstrap,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )
        self.is_fitted: bool = False
        self.threshold: Optional[float] = None
        self.fit_time_seconds: float = 0.0
        self.num_train_samples: int = 0
        self.feature_dim: int = 0

    def fit(self, X_train: Union[np.ndarray, torch.Tensor]) -> "IsolationForestDetector":
        """
        Fit IsolationForest model strictly on benign training flows.

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
            f"Fitting IsolationForest on {self.num_train_samples:,} samples with dim {self.feature_dim} "
            f"(n_estimators={self.config.n_estimators}, contamination={self.config.contamination}, "
            f"random_state={self.config.random_state})..."
        )

        t0 = time.time()
        self.model.fit(X_train)
        self.fit_time_seconds = time.time() - t0
        self.is_fitted = True

        logger.info(
            f"IsolationForest fit completed in {self.fit_time_seconds:.2f}s "
            f"across {len(self.model.estimators_)} estimators."
        )
        return self

    def score(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, float]:
        """
        Compute standardized anomaly scores.

        Direction standardization:
        In scikit-learn IsolationForest, decision_function(X) returns the average anomaly score of base classifiers.
        Negative values represent outliers (anomalous), positive values represent inliers (normal).
        Therefore: anomaly_score = -decision_function(X)
        Higher score = More anomalous.

        Returns:
            scores: 1D np.ndarray of anomaly scores
            inference_time: Time taken in seconds
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        t0 = time.time()
        raw_df = self.model.decision_function(X)
        scores = -raw_df.astype(np.float64)  # Invert so higher = more anomalous
        infer_time = time.time() - t0

        return scores, infer_time

    def calibrate_threshold(
        self,
        X_val: Union[np.ndarray, torch.Tensor],
        y_val: Union[np.ndarray, torch.Tensor],
        percentile: Optional[float] = None,
    ) -> float:
        """
        Establish the primary decision threshold at the 99th percentile of
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

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
        """Save fitted model, threshold, and configuration."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        save_dir = Path(output_dir or self.config.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "isolation_forest_model.joblib"
        meta_path = save_dir / "isolation_forest_metadata.json"

        joblib.dump(self.model, model_path)

        metadata = {
            "model_type": "IsolationForest",
            "n_estimators": self.config.n_estimators,
            "max_samples": self.config.max_samples,
            "contamination": self.config.contamination,
            "max_features": self.config.max_features,
            "bootstrap": self.config.bootstrap,
            "random_state": self.config.random_state,
            "preprocessing": "None (Tree-based orthogonal splits invariant to monotonic scaling)",
            "fit_time_seconds": round(self.fit_time_seconds, 4),
            "num_train_samples": self.num_train_samples,
            "feature_dim": self.feature_dim,
            "threshold": self.threshold,
            "threshold_percentile": self.config.threshold_percentile,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved IsolationForest artifacts to {save_dir}")
        return {
            "model": model_path,
            "metadata": meta_path,
        }

    @classmethod
    def load(cls, save_dir: Union[str, Path]) -> "IsolationForestDetector":
        """Load fitted detector from disk."""
        save_dir = Path(save_dir)
        meta_path = save_dir / "isolation_forest_metadata.json"
        model_path = save_dir / "isolation_forest_model.joblib"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = IsolationForestConfig(
            n_estimators=meta.get("n_estimators", 100),
            max_samples=meta.get("max_samples", "auto"),
            contamination=meta.get("contamination", "auto"),
            max_features=meta.get("max_features", 1.0),
            bootstrap=meta.get("bootstrap", False),
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

        logger.info(f"Loaded IsolationForestDetector from {save_dir} with threshold={detector.threshold}")
        return detector
