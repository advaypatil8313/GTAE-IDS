"""Downstream feature dataset loader for GTAE-IDS anomaly detectors."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class DownstreamFeatureDataset:
    """
    Standardized, leakage-safe dataset loader for downstream anomaly detectors
    (e.g. OCSVM, Isolation Forest, HBOS, INNE).
    Provides access to 113-dimensional flow representations while ensuring
    ground-truth evaluation labels y are strictly isolated from feature matrices X.
    """

    def __init__(
        self,
        artifacts_dir: Path = Path("data/processed/artifacts"),
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.train_path = self.artifacts_dir / "gtae_features_train.pt"
        self.val_path = self.artifacts_dir / "gtae_features_val.pt"
        self.test_path = self.artifacts_dir / "gtae_features_test.pt"
        self.meta_path = self.artifacts_dir / "gtae_features_metadata.json"

        self._train_data: Optional[Dict[str, Any]] = None
        self._val_data: Optional[Dict[str, Any]] = None
        self._test_data: Optional[Dict[str, Any]] = None
        self._metadata: Optional[Dict[str, Any]] = None

    def _load_split(self, split: str) -> Dict[str, Any]:
        """Lazy loader for individual split tensor dictionaries."""
        if split == "train":
            if self._train_data is None:
                if not self.train_path.exists():
                    raise FileNotFoundError(f"Training features artifact not found: {self.train_path}")
                self._train_data = torch.load(self.train_path, weights_only=False, map_location="cpu")
            return self._train_data
        elif split == "val":
            if self._val_data is None:
                if not self.val_path.exists():
                    raise FileNotFoundError(f"Validation features artifact not found: {self.val_path}")
                self._val_data = torch.load(self.val_path, weights_only=False, map_location="cpu")
            return self._val_data
        elif split == "test":
            if self._test_data is None:
                if not self.test_path.exists():
                    raise FileNotFoundError(f"Test features artifact not found: {self.test_path}")
                self._test_data = torch.load(self.test_path, weights_only=False, map_location="cpu")
            return self._test_data
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Load and return pipeline metadata JSON."""
        if self._metadata is None:
            if not self.meta_path.exists():
                raise FileNotFoundError(f"Metadata JSON not found: {self.meta_path}")
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        return self._metadata

    def get_train_data(
        self, as_numpy: bool = True
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Return training feature matrix and labels.
        By design, the training split represents benign normal behavior (y == 0).
        """
        data = self._load_split("train")
        X = data["X"]
        y = data["y"]
        if as_numpy:
            return X.numpy(), y.numpy()
        return X, y

    def get_train_benign_data(
        self, as_numpy: bool = True
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Explicit getter for benign-only training feature matrix."""
        return self.get_train_data(as_numpy=as_numpy)

    def get_val_data(
        self, as_numpy: bool = True
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Return validation feature matrix and evaluation-only labels.
        Labels contain both benign (0) and malicious (1) flows.
        """
        data = self._load_split("val")
        X = data["X"]
        y = data["y"]
        if as_numpy:
            return X.numpy(), y.numpy()
        return X, y

    def get_test_data(
        self, as_numpy: bool = True
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Return test feature matrix and evaluation-only labels.
        Labels contain both benign (0) and malicious (1) flows.
        """
        data = self._load_split("test")
        X = data["X"]
        y = data["y"]
        if as_numpy:
            return X.numpy(), y.numpy()
        return X, y

    def get_metadata(self, split: str, as_numpy: bool = True) -> Dict[str, Any]:
        """
        Retrieve flow-level metadata for temporal and diagnostic analysis:
        - window_ids
        - edge_timestamps
        - edge_indices_in_window
        - scalar_recon_error
        - src_nodes, dst_nodes
        """
        data = self._load_split(split)
        meta_keys = [
            "window_ids",
            "edge_timestamps",
            "edge_indices_in_window",
            "scalar_recon_error",
            "src_nodes",
            "dst_nodes",
        ]
        out: Dict[str, Any] = {}
        for k in meta_keys:
            if k in data:
                val = data[k]
                out[k] = val.numpy() if (as_numpy and hasattr(val, "numpy")) else val
        return out

    def get_summary(self) -> Dict[str, Any]:
        """Return high-level summary of all splits."""
        train_d = self._load_split("train")
        val_d = self._load_split("val")
        test_d = self._load_split("test")

        return {
            "train": {
                "flows": train_d["num_flows"],
                "feature_dim": train_d["feature_dim"],
                "benign_flows": int((train_d["y"] == 0).sum().item()),
                "attack_flows": int((train_d["y"] == 1).sum().item()),
            },
            "val": {
                "flows": val_d["num_flows"],
                "feature_dim": val_d["feature_dim"],
                "benign_flows": int((val_d["y"] == 0).sum().item()),
                "attack_flows": int((val_d["y"] == 1).sum().item()),
            },
            "test": {
                "flows": test_d["num_flows"],
                "feature_dim": test_d["feature_dim"],
                "benign_flows": int((test_d["y"] == 0).sum().item()),
                "attack_flows": int((test_d["y"] == 1).sum().item()),
            },
        }
