"""Configuration for downstream anomaly detectors in GTAE-IDS."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class OCSVMConfig:
    """Configuration for One-Class SVM downstream anomaly detector."""

    # Model hyperparameters
    kernel: str = "rbf"
    nu: float = 0.01  # Upper bound on training fraction of margin errors
    gamma: str = "scale"  # 1 / (n_features * X.var())
    cache_size: int = 2000  # Kernel cache size in MB (enables fast fitting on 57k flows)
    max_iter: int = -1  # No iteration limit (full convergence)

    # Threshold calibration policy
    # Calibrate decision threshold at the specified percentile of BENIGN validation flows
    threshold_percentile: float = 99.0

    # Reproducibility
    random_state: int = 42

    # Paths
    artifacts_dir: Path = Path("data/processed/artifacts")
    output_dir: Path = Path("data/processed/models/ocsvm")

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class IsolationForestConfig:
    """Configuration for Isolation Forest downstream anomaly detector."""

    # Model hyperparameters
    n_estimators: int = 100  # Number of base isolation trees
    max_samples: str = "auto"  # min(256, n_samples) subsampling per tree
    contamination: str = "auto"  # Offset calibration; operational threshold is calibrated separately
    max_features: float = 1.0  # Fraction of features drawn to train each base estimator
    bootstrap: bool = False  # If True, individual trees are fit on random subsets with replacement
    n_jobs: int = -1  # Use all available CPU cores for fitting and inference

    # Threshold calibration policy
    # Calibrate decision threshold at the specified percentile of BENIGN validation flows
    threshold_percentile: float = 99.0

    # Reproducibility
    random_state: int = 42

    # Paths
    artifacts_dir: Path = Path("data/processed/artifacts")
    output_dir: Path = Path("data/processed/models/isolation_forest")

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class HBOSConfig:
    """Configuration for Histogram-Based Outlier Score (HBOS) downstream anomaly detector."""

    # Model hyperparameters
    n_bins: int = 10  # Number of histogram bins per feature dimension
    alpha: float = 0.1  # Regularizer to prevent zero probability bins
    tol: float = 0.5  # Tolerance parameter for dynamic bin width estimation
    contamination: float = 0.1  # Proportion of outliers; final operational threshold is calibrated separately

    # Threshold calibration policy
    # Calibrate decision threshold at the specified percentile of BENIGN validation flows
    threshold_percentile: float = 99.0

    # Paths
    artifacts_dir: Path = Path("data/processed/artifacts")
    output_dir: Path = Path("data/processed/models/hbos")

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class INNEConfig:
    """Configuration for Isolation using Nearest Neighbor Ensemble (INNE) downstream anomaly detector."""

    # Model hyperparameters
    n_estimators: int = 100  # Number of hypersphere isolation estimators
    max_samples: Union[str, int] = "auto"  # Subsampling size per estimator
    contamination: float = 0.1  # Outlier proportion; operational threshold calibrated separately
    random_state: int = 42

    # Threshold calibration policy
    # Calibrate decision threshold at the specified percentile of BENIGN validation flows
    threshold_percentile: float = 99.0

    # Paths
    artifacts_dir: Path = Path("data/processed/artifacts")
    output_dir: Path = Path("data/processed/models/inne")

    def __post_init__(self):
        self.artifacts_dir = Path(self.artifacts_dir)
        self.output_dir = Path(self.output_dir)


