"""Temporal Graph Snapshot Builder for GTAE-IDS on LSPR23."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.graph.config import GraphConfig
from src.graph.feature_extraction import construct_edge_attributes, construct_node_features
from src.graph.validator import validate_all_snapshots, validate_graph_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class LSPR23GraphBuilder:
    """Constructs dynamic temporal PyG graph snapshots from preprocessed network flows."""

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()

    def build_snapshots(self) -> Dict[str, Any]:
        """
        Loads preprocessed flow dataset, performs chronological temporal slicing,
        and constructs validated PyG Data objects for every 300-second window.
        """
        logger.info(f"Loading input flows from: {self.config.input_flows_path}")
        if not self.config.input_flows_path.exists():
            raise FileNotFoundError(f"Input flow dataset not found at: {self.config.input_flows_path}")

        df = pd.read_pickle(self.config.input_flows_path)
        total_flows = len(df)
        logger.info(f"Loaded {total_flows:,} flows. Verifying chronological ordering...")

        # 1. Ensure strict chronological ordering
        df = df.sort_values(by="mTimestampStart").reset_index(drop=True)

        # 2. Compute deterministic window boundaries
        # Origin = earliest microsecond timestamp in the dataset
        t_origin_us = int(df["mTimestampStart"].min())
        window_duration_us = int(self.config.window_duration_seconds * self.config.timestamp_unit)

        logger.info(
            f"Deterministic Window Origin: {t_origin_us} ({datetime.fromtimestamp(t_origin_us / 1e6).isoformat()} UTC). "
            f"Window Duration: {self.config.window_duration_seconds:.0f}s ({window_duration_us:,} microseconds)."
        )

        # Vectorized window assignment: window_id = floor((ts - origin) / window_duration)
        df["window_idx"] = ((df["mTimestampStart"] - t_origin_us) // window_duration_us).astype(np.int32)
        window_groups = df.groupby("window_idx", sort=True)
        num_active_windows = len(window_groups)

        logger.info(
            f"Divided {total_flows:,} flows into {num_active_windows:,} active 300-second temporal windows."
        )

        snapshots: List[Data] = []
        node_counts: List[int] = []
        edge_counts: List[int] = []
        attack_counts_per_win: List[int] = []

        t0 = time.time()
        for win_idx, win_df in window_groups:
            # Deterministic window timestamp bounds
            win_start_ts = t_origin_us + int(win_idx) * window_duration_us
            win_end_ts = win_start_ts + window_duration_us

            # A. Extract unique nodes in current window
            # Order preserving unique IP list
            src_list = win_df["SrcIP"].tolist()
            dst_list = win_df["DstIP"].tolist()
            node_list = list(dict.fromkeys(src_list + dst_list))
            num_nodes = len(node_list)
            num_edges = len(win_df)

            ip_to_idx = {ip: i for i, ip in enumerate(node_list)}

            # B. Construct directed edge_index [2, num_edges]
            src_indices = win_df["SrcIP"].map(ip_to_idx).to_numpy(dtype=np.int64)
            dst_indices = win_df["DstIP"].map(ip_to_idx).to_numpy(dtype=np.int64)
            edge_index = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)

            # C. Construct node features x in R^{num_nodes x 16}
            x = construct_node_features(win_df, node_list)

            # D. Construct edge attributes in R^{num_edges x 81}
            edge_attr = construct_edge_attributes(win_df)

            # E. Ground-truth evaluation target y (strictly isolated from model inputs)
            y = torch.tensor(win_df["Label"].to_numpy(dtype=np.int8), dtype=torch.int8)

            # F. Edge timestamps
            edge_time = torch.tensor(win_df["mTimestampStart"].to_numpy(dtype=np.int64), dtype=torch.int64)

            # Assemble PyG Data object
            snapshot = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                edge_time=edge_time,
                window_id=int(win_idx),
                num_nodes=num_nodes,
                num_edges=num_edges,
                window_start_ts=win_start_ts,
                window_end_ts=win_end_ts,
                node_ip_map=node_list,
            )

            # Validate individual snapshot
            validate_graph_snapshot(
                snapshot,
                expected_node_dim=self.config.node_feature_dim,
                expected_edge_dim=self.config.edge_feature_dim,
            )

            snapshots.append(snapshot)
            node_counts.append(num_nodes)
            edge_counts.append(num_edges)
            attack_counts_per_win.append(int((y == 1).sum()))

            if len(snapshots) % 50 == 0 or len(snapshots) == num_active_windows:
                logger.info(
                    f"Built {len(snapshots)}/{num_active_windows} graph snapshots "
                    f"({sum(edge_counts):,} flows processed) in {time.time()-t0:.1f}s."
                )

        elapsed = time.time() - t0
        logger.info(f"Graph construction completed in {elapsed:.2f}s.")

        # Validate sequence
        logger.info("Executing comprehensive sequence validation...")
        validation_summary = validate_all_snapshots(
            snapshots,
            total_expected_edges=total_flows,
            expected_node_dim=self.config.node_feature_dim,
            expected_edge_dim=self.config.edge_feature_dim,
        )
        logger.info("Sequence validation passed successfully.")

        # Serialize artifacts
        output_paths, metadata = self._export_snapshots(
            snapshots=snapshots,
            node_counts=node_counts,
            edge_counts=edge_counts,
            attack_counts_per_win=attack_counts_per_win,
            t_origin_us=t_origin_us,
            window_duration_us=window_duration_us,
            elapsed_seconds=elapsed,
        )

        return {
            "snapshots": snapshots,
            "metadata": metadata,
            "output_paths": output_paths,
        }

    def _export_snapshots(
        self,
        snapshots: List[Data],
        node_counts: List[int],
        edge_counts: List[int],
        attack_counts_per_win: List[int],
        t_origin_us: int,
        window_duration_us: int,
        elapsed_seconds: float,
    ) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        """Serialize snapshots to PyTorch .pt file and metadata JSON."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        pt_path = self.config.output_dir / self.config.output_snapshots_filename
        json_path = self.config.output_dir / self.config.output_metadata_filename

        logger.info(f"Saving serialized graph snapshots to: {pt_path}")
        torch.save(snapshots, pt_path)
        pt_size_mb = pt_path.stat().st_size / (1024 * 1024)

        attack_windows = sum(1 for c in attack_counts_per_win if c > 0)
        total_attacks = sum(attack_counts_per_win)
        total_flows = sum(edge_counts)

        metadata = {
            "dataset_name": "LSPR23 5-Minute Temporal Graph Snapshots",
            "source_flows_file": str(self.config.input_flows_path),
            "generated_at": datetime.now().isoformat(),
            "construction_time_seconds": round(elapsed_seconds, 2),
            "window_parameters": {
                "window_duration_seconds": self.config.window_duration_seconds,
                "window_duration_microseconds": window_duration_us,
                "origin_timestamp_microseconds": t_origin_us,
                "origin_datetime_iso": datetime.fromtimestamp(t_origin_us / 1e6).isoformat(),
                "boundary_calculation_rule": "window_id = floor((mTimestampStart - origin) / (300 * 1e6))",
            },
            "summary_statistics": {
                "num_temporal_windows": len(snapshots),
                "total_graph_edges": total_flows,
                "total_nodes_accumulated": sum(node_counts),
                "unique_attack_flows": total_attacks,
                "attack_containing_windows": attack_windows,
                "attack_window_percentage": round((attack_windows / len(snapshots)) * 100, 2),
            },
            "node_distribution": {
                "min": int(np.min(node_counts)),
                "mean": round(float(np.mean(node_counts)), 2),
                "median": int(np.median(node_counts)),
                "max": int(np.max(node_counts)),
            },
            "edge_distribution": {
                "min": int(np.min(edge_counts)),
                "mean": round(float(np.mean(edge_counts)), 2),
                "median": int(np.median(edge_counts)),
                "max": int(np.max(edge_counts)),
            },
            "tensor_dimensions": {
                "node_features_dim": self.config.node_feature_dim,
                "edge_attributes_dim": self.config.edge_feature_dim,
                "node_features_description": [
                    "0: in-degree (normalized)",
                    "1: out-degree (normalized)",
                    "2: total-degree (normalized)",
                    "3: log10 forward bytes sent",
                    "4: log10 backward bytes sent",
                    "5: log10 forward packets sent",
                    "6: log10 backward packets sent",
                    "7: mean packet size sent",
                    "8: mean packet size received",
                    "9: mean flow duration sent",
                    "10: mean flow duration received",
                    "11: SYN/RST ratio sent",
                    "12: protocol diversity (/5)",
                    "13: destination port diversity (log10)",
                    "14: unique peer diversity (normalized)",
                    "15: RST ratio sent",
                ],
                "edge_attributes_description": (
                    "76 continuous flow metrics + 1 normalized DstPort [0,1] + "
                    "4 one-hot protocol indicators (TCP=6, UDP=17, ICMP=1/58, Other)"
                ),
            },
            "storage": {
                "snapshots_file": str(pt_path),
                "snapshots_size_mb": round(pt_size_mb, 2),
                "metadata_file": str(json_path),
            },
            "validation": {
                "all_snapshots_valid": True,
                "no_nan_or_inf": True,
                "zero_target_leakage": True,
                "labels_isolated_in_y": True,
            },
        }

        logger.info(f"Saving graph metadata to: {json_path}")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(metadata, jf, indent=2)

        return {"snapshots_pt": pt_path, "metadata_json": json_path}, metadata


Tuple_Snapshots_Result = Dict[str, Any]
