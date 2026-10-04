"""Validation routines for GTAE-IDS PyG temporal graph snapshots."""

import logging
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def validate_graph_snapshot(
    snapshot: Data,
    expected_node_dim: int = 16,
    expected_edge_dim: int = 81,
) -> None:
    """
    Validate structural, numerical, and label-isolation integrity of a temporal graph snapshot.
    Raises ValueError or AssertionError if any condition fails.
    """
    num_nodes = snapshot.num_nodes
    num_edges = snapshot.num_edges

    # 1. Edge index checks
    assert hasattr(snapshot, "edge_index"), "Missing edge_index attribute."
    assert snapshot.edge_index.dim() == 2, f"edge_index must be 2D, got shape {snapshot.edge_index.shape}"
    assert snapshot.edge_index.shape[0] == 2, f"edge_index must have 2 rows, got {snapshot.edge_index.shape[0]}"
    assert snapshot.edge_index.shape[1] == num_edges, (
        f"edge_index column count {snapshot.edge_index.shape[1]} != num_edges {num_edges}"
    )
    if num_edges > 0:
        min_idx = int(snapshot.edge_index.min())
        max_idx = int(snapshot.edge_index.max())
        assert min_idx >= 0, f"Negative node index found in edge_index: {min_idx}"
        assert max_idx < num_nodes, (
            f"Node index out of bounds: max_idx={max_idx} >= num_nodes={num_nodes}"
        )

    # 2. Node features checks
    assert hasattr(snapshot, "x"), "Missing node features tensor x."
    assert snapshot.x.shape == (num_nodes, expected_node_dim), (
        f"x shape {snapshot.x.shape} != expected ({num_nodes}, {expected_node_dim})"
    )
    assert not torch.isnan(snapshot.x).any(), "NaN found in node feature tensor x."
    assert not torch.isinf(snapshot.x).any(), "Inf found in node feature tensor x."

    # 3. Edge attributes checks
    assert hasattr(snapshot, "edge_attr"), "Missing edge attributes tensor edge_attr."
    assert snapshot.edge_attr.shape == (num_edges, expected_edge_dim), (
        f"edge_attr shape {snapshot.edge_attr.shape} != expected ({num_edges}, {expected_edge_dim})"
    )
    assert not torch.isnan(snapshot.edge_attr).any(), "NaN found in edge_attr tensor."
    assert not torch.isinf(snapshot.edge_attr).any(), "Inf found in edge_attr tensor."

    # 4. Target Label checks
    assert hasattr(snapshot, "y"), "Missing evaluation label tensor y."
    assert len(snapshot.y) == num_edges, f"y length {len(snapshot.y)} != num_edges {num_edges}"
    valid_labels = {0, 1}
    unique_labels = set(snapshot.y.cpu().numpy().tolist())
    assert unique_labels.issubset(valid_labels), f"Invalid label values found: {unique_labels}"

    # 5. Temporal ordering checks
    assert hasattr(snapshot, "edge_time"), "Missing edge_time tensor."
    assert len(snapshot.edge_time) == num_edges, (
        f"edge_time length {len(snapshot.edge_time)} != num_edges {num_edges}"
    )
    assert (snapshot.edge_time > 0).all(), "Zero or negative edge_time encountered."

    # Verify edge_time falls within the window boundaries
    if hasattr(snapshot, "window_start_ts") and hasattr(snapshot, "window_end_ts"):
        min_ts = int(snapshot.edge_time.min())
        max_ts = int(snapshot.edge_time.max())
        assert min_ts >= snapshot.window_start_ts, (
            f"Edge timestamp {min_ts} < window start {snapshot.window_start_ts}"
        )
        assert max_ts <= snapshot.window_end_ts, (
            f"Edge timestamp {max_ts} > window end {snapshot.window_end_ts}"
        )

    # 6. Node IP mapping checks
    assert hasattr(snapshot, "node_ip_map"), "Missing node_ip_map."
    assert len(snapshot.node_ip_map) == num_nodes, (
        f"node_ip_map size {len(snapshot.node_ip_map)} != num_nodes {num_nodes}"
    )
    assert len(set(snapshot.node_ip_map)) == num_nodes, "Duplicate IPs found in node_ip_map."


def validate_all_snapshots(
    snapshots: List[Data],
    total_expected_edges: int = 100_000,
    expected_node_dim: int = 16,
    expected_edge_dim: int = 81,
) -> dict:
    """
    Validate the entire sequence of temporal snapshots.
    Verifies that total edges match dataset count and all snapshots satisfy integrity rules.
    """
    total_edges = sum(s.num_edges for s in snapshots)
    total_nodes = sum(s.num_nodes for s in snapshots)
    attack_windows = sum(1 for s in snapshots if (s.y == 1).any())

    assert total_edges == total_expected_edges, (
        f"Total graph edges {total_edges:,} != expected dataset flows {total_expected_edges:,}"
    )

    for idx, s in enumerate(snapshots):
        try:
            validate_graph_snapshot(
                s,
                expected_node_dim=expected_node_dim,
                expected_edge_dim=expected_edge_dim,
            )
        except AssertionError as e:
            raise AssertionError(f"Snapshot index {idx} (window_id={getattr(s, 'window_id', idx)}) failed validation: {e}")

    return {
        "num_snapshots": len(snapshots),
        "total_edges": total_edges,
        "total_nodes_accumulated": total_nodes,
        "attack_windows_count": attack_windows,
        "attack_windows_percentage": round((attack_windows / len(snapshots)) * 100, 2),
    }
