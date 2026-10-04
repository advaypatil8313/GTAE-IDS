"""Downstream anomaly-detection module for GTAE-IDS."""

from src.detection.config import OCSVMConfig
from src.detection.dataset import DownstreamFeatureDataset
from src.detection.evaluator import evaluate_detector
from src.detection.feature_pipeline import GTAEFeatureExtractor
from src.detection.models.ocsvm import OCSVMDetector

__all__ = [
    "GTAEFeatureExtractor",
    "DownstreamFeatureDataset",
    "OCSVMConfig",
    "OCSVMDetector",
    "evaluate_detector",
]

