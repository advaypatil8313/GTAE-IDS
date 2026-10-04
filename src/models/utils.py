"""Utility functions for GTAE model, dataset splitting, and metrics."""

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Data


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def get_chronological_splits(
    snapshots: List[Data],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Partition temporal graph snapshots strictly chronologically.
    NEVER shuffles windows to preserve temporal causality and prevent lookahead leakage.
    Default:
        Train: 70% (e.g. snapshots 0 to 208)
        Val:   15% (e.g. snapshots 209 to 253)
        Test:  15% (e.g. snapshots 254 to 297)
    """
    total = len(snapshots)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_snaps = snapshots[:n_train]
    val_snaps = snapshots[n_train : n_train + n_val]
    test_snaps = snapshots[n_train + n_val :]

    def _split_stats(snaps: List[Data], name: str) -> Dict[str, int]:
        edges = sum(s.num_edges for s in snaps)
        attacks = sum(int((s.y == 1).sum()) for s in snaps)
        benign = edges - attacks
        return {
            "name": name,
            "num_windows": len(snaps),
            "total_edges": edges,
            "benign_edges": benign,
            "attack_edges": attacks,
            "attack_percentage": round((attacks / max(1, edges)) * 100, 2),
        }

    return {
        "train_snapshots": train_snaps,
        "val_snapshots": val_snaps,
        "test_snapshots": test_snaps,
        "split_summary": {
            "train": _split_stats(train_snaps, "Train (70%)"),
            "val": _split_stats(val_snaps, "Validation (15%)"),
            "test": _split_stats(test_snaps, "Test (15%)"),
            "total_windows": total,
        },
    }

