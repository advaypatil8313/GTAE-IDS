"""One-Class SVM (OCSVM) downstream anomaly detector for GTAE-IDS."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.detection.config import OCSVMConfig

logger = logging.getLogger(__name__)


class OCSVMDetector:
    """
    One-Class Support Vector Machine for network flow anomaly detection.

    Enforces strict protocol guarantees:
    1. Fitted exclusively on benign training representations (y == 0).
    2. Uses a StandardScaler fitted strictly on training data.
    3. Standardizes anomaly score direction: S(x) = -decision_function(X_scaled),
       such that higher values strictly represent greater anomaly severity.
    4. Calibrates decision threshold at the 99th percentile of benign validation flows.
    5. Evaluates test set using the frozen calibrated threshold without any modification.
    """

    def __init__(self, config: Optional[OCSVMConfig] = None):
        self.config = config or OCSVMConfig()
        self.scaler = StandardScaler()
        self.model = OneClassSVM(
            kernel=self.config.kernel,
            nu=self.config.nu,
            gamma=self.config.gamma,
            cache_size=self.config.cache_size,
            max_iter=self.config.max_iter,
        )
        self.is_fitted: bool = False
        self.threshold: Optional[float] = None
        self.fit_time_seconds: float = 0.0
        self.num_train_samples: int = 0
        self.num_support_vectors: int = 0

    def fit(self, X_train: Union[np.ndarray, torch.Tensor]) -> "OCSVMDetector":
        """
        Fit StandardScaler and OneClassSVM model strictly on benign training flows.

        Args:
            X_train: 2D array/tensor of shape [N_train, 113] representing benign flows.
        """
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()

        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_train.shape}")

        self.num_train_samples = X_train.shape[0]
        logger.info(
            f"Fitting OCSVM on {self.num_train_samples:,} samples with dim {X_train.shape[1]} "
            f"(kernel={self.config.kernel}, nu={self.config.nu}, gamma={self.config.gamma})..."
        )

        t0 = time.time()
        # 1. Fit scaler strictly on training samples
        X_scaled = self.scaler.fit_transform(X_train)

        # 2. Fit OneClassSVM model
        self.model.fit(X_scaled)
        self.fit_time_seconds = time.time() - t0
        self.num_support_vectors = len(self.model.support_)
        self.is_fitted = True

        logger.info(
            f"OCSVM fit completed in {self.fit_time_seconds:.2f}s. "
            f"Support vectors: {self.num_support_vectors:,} "
            f"({self.num_support_vectors / self.num_train_samples * 100:.2f}% of training set)."
        )
        return self

    def score(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, float]:
        """
        Compute standardized anomaly scores.

        Direction standardization:
        In scikit-learn OneClassSVM, decision_function(X) is positive for inliers (normal)
        and negative for outliers (anomalous).
        Therefore: anomaly_score = -decision_function(X_scaled)
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
        X_scaled = self.scaler.transform(X)
        raw_df = self.model.decision_function(X_scaled)
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
        """Save fitted model, scaler, threshold, and configuration."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        save_dir = Path(output_dir or self.config.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "ocsvm_model.joblib"
        scaler_path = save_dir / "ocsvm_scaler.joblib"
        meta_path = save_dir / "ocsvm_metadata.json"

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        metadata = {
            "model_type": "OneClassSVM",
            "kernel": self.config.kernel,
            "nu": self.config.nu,
            "gamma": self.config.gamma,
            "cache_size": self.config.cache_size,
            "fit_time_seconds": round(self.fit_time_seconds, 4),
            "num_train_samples": self.num_train_samples,
            "num_support_vectors": self.num_support_vectors,
            "support_vector_ratio": round(
                self.num_support_vectors / max(1, self.num_train_samples), 4
            ),
            "threshold": self.threshold,
            "threshold_percentile": self.config.threshold_percentile,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved OCSVM artifacts to {save_dir}")
        return {
            "model": model_path,
            "scaler": scaler_path,
            "metadata": meta_path,
        }

    @classmethod
    def load(cls, save_dir: Union[str, Path]) -> "OCSVMDetector":
        """Load fitted detector from disk."""
        save_dir = Path(save_dir)
        meta_path = save_dir / "ocsvm_metadata.json"
        model_path = save_dir / "ocsvm_model.joblib"
        scaler_path = save_dir / "ocsvm_scaler.joblib"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = OCSVMConfig(
            kernel=meta.get("kernel", "rbf"),
            nu=meta.get("nu", 0.01),
            gamma=meta.get("gamma", "scale"),
            threshold_percentile=meta.get("threshold_percentile", 99.0),
            output_dir=save_dir,
        )

        detector = cls(config=cfg)
        detector.model = joblib.load(model_path)
        detector.scaler = joblib.load(scaler_path)
        detector.threshold = meta.get("threshold")
        detector.is_fitted = True
        detector.fit_time_seconds = meta.get("fit_time_seconds", 0.0)
        detector.num_train_samples = meta.get("num_train_samples", 0)
        detector.num_support_vectors = meta.get("num_support_vectors", 0)

        logger.info(f"Loaded OCSVMDetector from {save_dir} with threshold={detector.threshold}")
        return detector
