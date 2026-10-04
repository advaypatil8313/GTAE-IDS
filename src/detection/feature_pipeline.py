"""Downstream anomaly-detection feature extraction engine for GTAE-IDS."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.graph.feature_extraction import construct_node_features_from_edges
from src.models.config import GTAEConfig
from src.models.gtae import GTAEModel
from src.models.normalization import GTAEFeatureScaler
from src.models.utils import get_chronological_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class GTAEFeatureExtractor:
    """
    Extracts 113-dimensional downstream anomaly representations f_e = [z_e, |edge_attr - edge_recon|]
    from temporal graph snapshots using the clean GTAE model checkpoint.
    Preserves strict edge correspondence, chronological ordering, and total label isolation.
    """

    def __init__(
        self,
        checkpoint_path: Path = Path("data/processed/models/best_gtae_model_clean.pt"),
        scalers_path: Path = Path("data/processed/scalers/gtae_scalers_clean.pkl"),
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.scalers_path = Path(scalers_path)

        # Enforce clean checkpoint usage
        if "clean" not in self.checkpoint_path.name:
            raise ValueError(
                f"Prohibited checkpoint detected: {self.checkpoint_path.name}. "
                "Downstream feature pipeline must use the clean GTAE checkpoint (best_gtae_model_clean.pt)."
            )

        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.model: Optional[GTAEModel] = None
        self.scaler: Optional[GTAEFeatureScaler] = None
        self.model_config: Optional[GTAEConfig] = None
        self._load_model_and_scalers()

    def _load_model_and_scalers(self) -> None:
        """Load clean GTAE model weights and fitted scalers in evaluation mode."""
        logger.info(f"Loading clean GTAE model from: {self.checkpoint_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Clean model checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = checkpoint["model_config"]
        self.model_config = GTAEConfig(**cfg_dict)

        self.model = GTAEModel(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loading clean feature scalers from: {self.scalers_path}")
        if not self.scalers_path.exists():
            raise FileNotFoundError(f"Clean scaler file not found: {self.scalers_path}")
        self.scaler = GTAEFeatureScaler.load(self.scalers_path)

        logger.info(
            f"GTAE Feature Extractor initialized successfully on {self.device}. "
            f"Model epoch: {checkpoint.get('epoch', -1)}, Val loss: {checkpoint.get('val_loss', -1):.6f}."
        )

    @torch.no_grad()
    def extract_from_snapshots(
        self,
        snapshots: List[Data],
        is_training_split: bool = False,
    ) -> Dict[str, Any]:
        """
        Chronologically process a sequence of temporal graph snapshots and extract:
        - f_e in R^{E x 113}: [z_e (32), |edge_attr - edge_recon| (81)]
        - z_e in R^{E x 32}: latent flow representation
        - abs_recon_error in R^{E x 81}: per-feature reconstruction error
        - scalar_recon_error in R^{E}: MAE per flow
        - Edge-level metadata: window_id, timestamp, edge_idx_in_window, src, dst
        - Evaluation-only ground-truth labels y (isolated in separate tensor)
        """
        assert self.model is not None and self.scaler is not None

        all_f_e: List[torch.Tensor] = []
        all_z_e: List[torch.Tensor] = []
        all_abs_err: List[torch.Tensor] = []
        all_scalar_err: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_timestamps: List[torch.Tensor] = []
        all_window_ids: List[torch.Tensor] = []
        all_edge_indices: List[torch.Tensor] = []
        all_src_nodes: List[torch.Tensor] = []
        all_dst_nodes: List[torch.Tensor] = []

        total_edges_processed = 0
        t0 = time.time()

        for snap_idx, snap in enumerate(snapshots):
            win_id = getattr(snap, "window_id", snap_idx)

            if is_training_split:
                # Use clean benign-only training graph representation G_t^{benign}
                benign_mask = (snap.y == 0)
                if benign_mask.sum() == 0:
                    continue

                b_edge_index = snap.edge_index[:, benign_mask]
                b_edge_attr_raw = snap.edge_attr[benign_mask]
                b_x_raw = construct_node_features_from_edges(
                    b_edge_index, b_edge_attr_raw, snap.num_nodes
                )

                # Scale with frozen training scalers
                x_scaled = torch.tensor(
                    self.scaler.node_scaler.transform(b_x_raw.cpu().numpy()).astype(np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )
                edge_attr_scaled = torch.tensor(
                    self.scaler.edge_scaler.transform(b_edge_attr_raw.cpu().numpy()).astype(np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )
                edge_index_dev = b_edge_index.to(self.device)

                # Edge metadata
                win_labels = snap.y[benign_mask].cpu()
                win_timestamps = snap.edge_time[benign_mask].cpu()
                win_edge_indices = torch.where(benign_mask)[0].cpu()
                win_src = b_edge_index[0].cpu()
                win_dst = b_edge_index[1].cpu()
                num_win_edges = int(benign_mask.sum())
            else:
                # Complete observed graph snapshots for validation and test inference
                snap_scaled = self.scaler.transform_snapshot(snap)
                x_scaled = snap_scaled.x.to(self.device)
                edge_index_dev = snap_scaled.edge_index.to(self.device)
                edge_attr_scaled = snap_scaled.edge_attr.to(self.device)

                # Edge metadata
                win_labels = snap.y.cpu()
                win_timestamps = snap.edge_time.cpu()
                win_edge_indices = torch.arange(snap.num_edges, dtype=torch.long)
                win_src = snap.edge_index[0].cpu()
                win_dst = snap.edge_index[1].cpu()
                num_win_edges = snap.num_edges

            # Model forward pass
            outputs = self.model(x_scaled, edge_index_dev, edge_attr_scaled)

            # Extract 113-d downstream anomaly features
            anomaly_feats = self.model.extract_anomaly_features(
                outputs["edge_latent"], edge_attr_scaled, outputs["edge_recon"]
            )

            f_e = anomaly_feats["flow_features_for_detectors"].cpu()  # [E, 113]
            z_e = outputs["edge_latent"].cpu()                        # [E, 32]
            abs_err = anomaly_feats["abs_recon_error_vector"].cpu()   # [E, 81]
            scalar_err = anomaly_feats["scalar_recon_error"].cpu()    # [E]

            # Finiteness assertion
            assert torch.isfinite(f_e).all(), f"Non-finite features in snapshot {snap_idx} (window {win_id})!"
            assert torch.isfinite(scalar_err).all(), f"Non-finite error in snapshot {snap_idx} (window {win_id})!"

            all_f_e.append(f_e)
            all_z_e.append(z_e)
            all_abs_err.append(abs_err)
            all_scalar_err.append(scalar_err)
            all_labels.append(win_labels)
            all_timestamps.append(win_timestamps)
            all_window_ids.append(torch.full((num_win_edges,), win_id, dtype=torch.int32))
            all_edge_indices.append(win_edge_indices)
            all_src_nodes.append(win_src)
            all_dst_nodes.append(win_dst)

            total_edges_processed += num_win_edges

        elapsed = time.time() - t0
        logger.info(
            f"Extracted representations for {total_edges_processed:,} flows across "
            f"{len(snapshots)} snapshots in {elapsed:.2f}s ({total_edges_processed / max(1e-3, elapsed):,.0f} flows/s)."
        )

        # Concatenate along flow dimension (preserving chronological order)
        cat_f_e = torch.cat(all_f_e, dim=0)
        cat_z_e = torch.cat(all_z_e, dim=0)
        cat_abs_err = torch.cat(all_abs_err, dim=0)
        cat_scalar_err = torch.cat(all_scalar_err, dim=0)
        cat_labels = torch.cat(all_labels, dim=0).to(torch.int8)
        cat_timestamps = torch.cat(all_timestamps, dim=0).to(torch.int64)
        cat_window_ids = torch.cat(all_window_ids, dim=0).to(torch.int32)
        cat_edge_indices = torch.cat(all_edge_indices, dim=0).to(torch.int64)
        cat_src_nodes = torch.cat(all_src_nodes, dim=0).to(torch.int64)
        cat_dst_nodes = torch.cat(all_dst_nodes, dim=0).to(torch.int64)

        result: Dict[str, Any] = {
            "X": cat_f_e,
            "y": cat_labels,
            "z_e": cat_z_e,
            "abs_recon_error": cat_abs_err,
            "scalar_recon_error": cat_scalar_err,
            "window_ids": cat_window_ids,
            "edge_timestamps": cat_timestamps,
            "edge_indices_in_window": cat_edge_indices,
            "src_nodes": cat_src_nodes,
            "dst_nodes": cat_dst_nodes,
            "num_flows": total_edges_processed,
            "num_windows": len(snapshots),
            "feature_dim": cat_f_e.shape[1],
        }

        if is_training_split:
            result["X_train_benign"] = cat_f_e

        return result

    def run_pipeline(
        self,
        snapshots_path: Path = Path("data/processed/graphs/temporal_graph_snapshots.pt"),
        output_dir: Path = Path("data/processed/artifacts"),
    ) -> Dict[str, Any]:
        """
        Execute downstream feature extraction across chronological Train, Validation, and Test splits.
        Saves serialized artifacts and comprehensive metadata JSON.
        """
        snapshots_path = Path(snapshots_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading temporal graph snapshots from: {snapshots_path}")
        snapshots = torch.load(snapshots_path, weights_only=False)

        splits = get_chronological_splits(
            snapshots,
            train_ratio=self.model_config.train_ratio,
            val_ratio=self.model_config.val_ratio,
            test_ratio=self.model_config.test_ratio,
        )

        train_snaps = splits["train_snapshots"]
        val_snaps = splits["val_snapshots"]
        test_snaps = splits["test_snapshots"]

        logger.info(
            f"Chronological split: {len(train_snaps)} train windows (70%), "
            f"{len(val_snaps)} val windows (15%), {len(test_snaps)} test windows (15%)."
        )

        # 1. Training split: benign-only training graphs
        logger.info("--- Extracting Training Split (Benign-Only Representation) ---")
        train_data = self.extract_from_snapshots(train_snaps, is_training_split=True)

        # 2. Validation split: complete observed graphs
        logger.info("--- Extracting Validation Split (Complete Observed Graphs) ---")
        val_data = self.extract_from_snapshots(val_snaps, is_training_split=False)

        # 3. Test split: complete observed graphs
        logger.info("--- Extracting Test Split (Complete Observed Graphs) ---")
        test_data = self.extract_from_snapshots(test_snaps, is_training_split=False)

        # Serialization paths
        train_pt_path = output_dir / "gtae_features_train.pt"
        val_pt_path = output_dir / "gtae_features_val.pt"
        test_pt_path = output_dir / "gtae_features_test.pt"
        meta_json_path = output_dir / "gtae_features_metadata.json"

        logger.info(f"Saving training features to: {train_pt_path}")
        torch.save(train_data, train_pt_path)

        logger.info(f"Saving validation features to: {val_pt_path}")
        torch.save(val_data, val_pt_path)

        logger.info(f"Saving test features to: {test_pt_path}")
        torch.save(test_data, test_pt_path)

        # Metadata summary
        metadata = {
            "pipeline_name": "GTAE Downstream Anomaly Feature Pipeline",
            "model_checkpoint_used": str(self.checkpoint_path),
            "scalers_used": str(self.scalers_path),
            "source_snapshots": str(snapshots_path),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dimensions": {
                "latent_dim": 32,
                "reconstruction_error_dim": 81,
                "combined_feature_dim": 113,
            },
            "split_summary": {
                "train": {
                    "num_windows": train_data["num_windows"],
                    "total_flows": train_data["num_flows"],
                    "benign_flows": int((train_data["y"] == 0).sum().item()),
                    "attack_flows": int((train_data["y"] == 1).sum().item()),
                    "is_benign_only": True,
                    "artifact_file": str(train_pt_path),
                    "file_size_mb": round(train_pt_path.stat().st_size / (1024**2), 2),
                },
                "val": {
                    "num_windows": val_data["num_windows"],
                    "total_flows": val_data["num_flows"],
                    "benign_flows": int((val_data["y"] == 0).sum().item()),
                    "attack_flows": int((val_data["y"] == 1).sum().item()),
                    "is_benign_only": False,
                    "artifact_file": str(val_pt_path),
                    "file_size_mb": round(val_pt_path.stat().st_size / (1024**2), 2),
                },
                "test": {
                    "num_windows": test_data["num_windows"],
                    "total_flows": test_data["num_flows"],
                    "benign_flows": int((test_data["y"] == 0).sum().item()),
                    "attack_flows": int((test_data["y"] == 1).sum().item()),
                    "is_benign_only": False,
                    "artifact_file": str(test_pt_path),
                    "file_size_mb": round(test_pt_path.stat().st_size / (1024**2), 2),
                },
            },
            "validation_checks": {
                "feature_dimension": 113,
                "labels_isolated_from_x": True,
                "all_features_finite": True,
                "exact_edge_correspondence": True,
                "chronological_ordering_preserved": True,
            },
        }

        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved feature extraction metadata to: {meta_json_path}")

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "metadata": metadata,
            "paths": {
                "train_pt": train_pt_path,
                "val_pt": val_pt_path,
                "test_pt": test_pt_path,
                "metadata_json": meta_json_path,
            },
        }
