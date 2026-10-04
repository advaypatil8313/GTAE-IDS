"""Leakage-safe feature normalization for GTAE-IDS graph nodes and edges."""

import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data


class GTAEFeatureScaler:
    """
    Fits feature scalers strictly on BENIGN training flows and nodes.
    Prevents data leakage across validation/test sets and across attack classes.
    """

    def __init__(self):
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.is_fitted = False
        self.num_edge_fit_rows = 0
        self.num_node_fit_rows = 0

    def fit(self, train_snapshots: List[Data]) -> "GTAEFeatureScaler":
        """
        Fit scalers strictly using benign training flows (y == 0) and benign-derived node features.
        Zero validation or test data is used. Zero attack flows contribute to either scaler.
        """
        from src.graph.feature_extraction import construct_node_features_from_edges

        benign_edges_list = []
        benign_nodes_list = []

        for snap in train_snapshots:
            if hasattr(snap, "y") and snap.y is not None:
                benign_mask = (snap.y == 0)
                if benign_mask.sum() == 0:
                    continue
                b_edge_index = snap.edge_index[:, benign_mask]
                b_edge_attr = snap.edge_attr[benign_mask]
                b_x = construct_node_features_from_edges(b_edge_index, b_edge_attr, snap.num_nodes)
                benign_edges_list.append(b_edge_attr.cpu().numpy())
                benign_nodes_list.append(b_x.cpu().numpy())
            else:
                benign_edges_list.append(snap.edge_attr.cpu().numpy())
                benign_nodes_list.append(snap.x.cpu().numpy())

        assert len(benign_edges_list) > 0, "No benign training edges found to fit scaler!"
        all_benign_edges = np.vstack(benign_edges_list)
        all_benign_nodes = np.vstack(benign_nodes_list)

        self.edge_scaler.fit(all_benign_edges)
        self.node_scaler.fit(all_benign_nodes)
        self.is_fitted = True
        self.num_edge_fit_rows = int(all_benign_edges.shape[0])
        self.num_node_fit_rows = int(all_benign_nodes.shape[0])

        return self

    def fit_from_benign_tensors(
        self, benign_nodes_list: List[np.ndarray], benign_edges_list: List[np.ndarray]
    ) -> "GTAEFeatureScaler":
        """Fit scalers directly from pre-computed benign node and edge numpy arrays."""
        assert len(benign_edges_list) > 0, "No benign edge arrays provided!"
        assert len(benign_nodes_list) > 0, "No benign node arrays provided!"

        all_benign_edges = np.vstack(benign_edges_list)
        all_benign_nodes = np.vstack(benign_nodes_list)

        self.edge_scaler.fit(all_benign_edges)
        self.node_scaler.fit(all_benign_nodes)
        self.is_fitted = True
        self.num_edge_fit_rows = int(all_benign_edges.shape[0])
        self.num_node_fit_rows = int(all_benign_nodes.shape[0])

        return self

    def transform_snapshot(self, snapshot: Data, clone: bool = True) -> Data:
        """
        Apply fitted scaling to node features x and edge attributes edge_attr.
        Preserves all metadata, targets, timestamps, and local indexing.
        """
        if not self.is_fitted:
            raise RuntimeError("GTAEFeatureScaler must be fitted before calling transform_snapshot.")

        snap = snapshot.clone() if clone else snapshot

        # Transform node features
        x_np = snap.x.cpu().numpy()
        x_scaled = self.node_scaler.transform(x_np).astype(np.float32)
        snap.x = torch.tensor(x_scaled, dtype=torch.float32, device=snap.x.device)

        # Transform edge attributes
        edge_np = snap.edge_attr.cpu().numpy()
        edge_scaled = self.edge_scaler.transform(edge_np).astype(np.float32)
        snap.edge_attr = torch.tensor(edge_scaled, dtype=torch.float32, device=snap.edge_attr.device)

        return snap

    def save(self, path: Path) -> None:
        """Serialize fitted scalers to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "node_scaler": self.node_scaler,
                    "edge_scaler": self.edge_scaler,
                    "is_fitted": self.is_fitted,
                    "num_edge_fit_rows": self.num_edge_fit_rows,
                    "num_node_fit_rows": self.num_node_fit_rows,
                },
                f,
                protocol=5,
            )

    @classmethod
    def load(cls, path: Path) -> "GTAEFeatureScaler":
        """Load serialized scalers from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found at: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        scaler = cls()
        scaler.node_scaler = data["node_scaler"]
        scaler.edge_scaler = data["edge_scaler"]
        scaler.is_fitted = data["is_fitted"]
        scaler.num_edge_fit_rows = data.get("num_edge_fit_rows", 0)
        scaler.num_node_fit_rows = data.get("num_node_fit_rows", 0)
        return scaler

