"""Training engine for Graph Transformer-Based Autoencoder (GTAE)."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.graph.feature_extraction import construct_node_features_from_edges
from src.models.config import GTAEConfig
from src.models.gtae import GTAEModel
from src.models.normalization import GTAEFeatureScaler
from src.models.utils import count_parameters, get_chronological_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class GTAETrainingConfig:
    """Hyperparameters and file paths for GTAE training."""

    # Optimization
    max_epochs: int = 30
    early_stopping_patience: int = 5
    batch_size: int = 8             # Number of temporal graph snapshots per mini-batch
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    node_loss_weight: float = 0.5
    loss_type: str = "smooth_l1"

    # Paths & Filenames
    snapshots_path: Path = Path("data/processed/graphs/temporal_graph_snapshots.pt")
    checkpoint_dir: Path = Path("data/processed/models")
    scalers_dir: Path = Path("data/processed/scalers")
    artifacts_dir: Path = Path("data/processed/artifacts")
    checkpoint_filename: str = "best_gtae_model_clean.pt"
    scaler_filename: str = "gtae_scalers_clean.pkl"
    history_filename: str = "gtae_training_history_clean.json"
    config_filename: str = "gtae_training_config_clean.json"
    anomaly_artifacts_filename: str = "sample_anomaly_features_clean.pt"

    # Hardware
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.snapshots_path = Path(self.snapshots_path)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.scalers_dir = Path(self.scalers_dir)
        self.artifacts_dir = Path(self.artifacts_dir)



class GTAETrainer:
    """
    Orchestrates leakage-safe normalization, benign-only self-supervised training,
    chronological validation, early stopping, and artifact export.
    """

    def __init__(
        self,
        model: Optional[GTAEModel] = None,
        train_config: Optional[GTAETrainingConfig] = None,
        model_config: Optional[GTAEConfig] = None,
    ):
        self.train_config = train_config or GTAETrainingConfig()
        self.model_config = model_config or (model.config if model else GTAEConfig())
        self.device = torch.device(self.train_config.device)

        self.model = (model or GTAEModel(self.model_config)).to(self.device)
        self.scaler = GTAEFeatureScaler()

        # Training history audit
        self.history: List[Dict[str, Any]] = []
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = -1

    def prepare_data(
        self,
    ) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Load snapshots, perform chronological split, fit scaler on benign training data,
        and prepare normalized datasets.
        """
        logger.info(f"Loading temporal snapshots from: {self.train_config.snapshots_path}")
        snapshots = torch.load(self.train_config.snapshots_path, weights_only=False)

        # 1. Chronological split (70% train, 15% val, 15% test)
        splits = get_chronological_splits(
            snapshots,
            train_ratio=self.model_config.train_ratio,
            val_ratio=self.model_config.val_ratio,
            test_ratio=self.model_config.test_ratio,
        )
        raw_train = splits["train_snapshots"]
        raw_val = splits["val_snapshots"]
        raw_test = splits["test_snapshots"]

        # 2. Extract strictly benign edges and compute strictly benign-derived node features
        logger.info("Extracting benign training flows (y == 0) and recomputing benign-only node features...")
        raw_benign_train: List[Dict[str, Any]] = []
        for s in raw_train:
            benign_mask = (s.y == 0)
            if benign_mask.sum() == 0:
                continue
            b_edge_index = s.edge_index[:, benign_mask]
            b_edge_attr = s.edge_attr[benign_mask]  # Raw unscaled edge attributes
            b_x = construct_node_features_from_edges(b_edge_index, b_edge_attr, s.num_nodes)  # Raw unscaled node features
            raw_benign_train.append({
                "x": b_x,
                "edge_index": b_edge_index,
                "edge_attr": b_edge_attr,
                "num_nodes": s.num_nodes,
                "num_edges": int(benign_mask.sum()),
                "window_id": getattr(s, "window_id", -1),
            })

        # 3. Fit feature scaler ONLY on benign training flows/nodes
        logger.info("Fitting feature scalers strictly on BENIGN training flows and benign-derived node features...")
        benign_nodes_list = [d["x"].cpu().numpy() for d in raw_benign_train]
        benign_edges_list = [d["edge_attr"].cpu().numpy() for d in raw_benign_train]
        self.scaler.fit_from_benign_tensors(benign_nodes_list, benign_edges_list)
        scaler_save_path = self.train_config.scalers_dir / self.train_config.scaler_filename
        self.scaler.save(scaler_save_path)
        logger.info(
            f"Fitted scalers saved to: {scaler_save_path} "
            f"(Edge rows: {self.scaler.num_edge_fit_rows:,}, Node rows: {self.scaler.num_node_fit_rows:,})"
        )

        # 4. Transform BENIGN-ONLY training graphs using fitted scalers:
        train_benign_scaled: List[Data] = []
        for d in raw_benign_train:
            x_scaled = torch.tensor(
                self.scaler.node_scaler.transform(d["x"].cpu().numpy()).astype(np.float32),
                dtype=torch.float32,
            )
            edge_attr_scaled = torch.tensor(
                self.scaler.edge_scaler.transform(d["edge_attr"].cpu().numpy()).astype(np.float32),
                dtype=torch.float32,
            )
            train_snap = Data(
                x=x_scaled,
                edge_index=d["edge_index"],
                edge_attr=edge_attr_scaled,
                num_nodes=d["num_nodes"],
                num_edges=d["num_edges"],
                window_id=d["window_id"],
            )
            train_benign_scaled.append(train_snap)

        # 5. Transform validation and test snapshots (full complete graphs with frozen training scalers)
        # Target labels s.y and metadata are completely preserved for evaluation only
        val_scaled = [self.scaler.transform_snapshot(s) for s in raw_val]
        test_scaled = [self.scaler.transform_snapshot(s) for s in raw_test]

        logger.info(
            f"Data preparation complete:\n"
            f"  - Training windows (Benign-only): {len(train_benign_scaled)} graphs ({sum(s.num_edges for s in train_benign_scaled):,} flows)\n"
            f"  - Validation windows (Full):       {len(val_scaled)} graphs ({sum(s.num_edges for s in val_scaled):,} flows)\n"
            f"  - Test windows (Full):             {len(test_scaled)} graphs ({sum(s.num_edges for s in test_scaled):,} flows)"
        )

        return train_benign_scaled, val_scaled, test_scaled

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Execute one training epoch over benign training graphs."""
        self.model.train()
        total_loss_accum = 0.0
        edge_loss_accum = 0.0
        node_loss_accum = 0.0
        total_edges = 0

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            outputs = self.model(batch.x, batch.edge_index, batch.edge_attr)
            total_loss, edge_loss, node_loss = self.model.compute_loss(
                batch.x, batch.edge_attr, outputs
            )

            total_loss.backward()
            optimizer.step()

            batch_edges = int(batch.edge_index.shape[1])
            total_loss_accum += total_loss.item() * batch_edges
            edge_loss_accum += edge_loss.item() * batch_edges
            node_loss_accum += node_loss.item() * batch_edges
            total_edges += batch_edges

        return {
            "train_loss": total_loss_accum / max(1, total_edges),
            "train_edge_loss": edge_loss_accum / max(1, total_edges),
            "train_node_loss": node_loss_accum / max(1, total_edges),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate self-supervised reconstruction loss on validation graphs."""
        self.model.eval()
        total_loss_accum = 0.0
        edge_loss_accum = 0.0
        node_loss_accum = 0.0
        total_edges = 0

        for batch in loader:
            batch = batch.to(self.device)
            outputs = self.model(batch.x, batch.edge_index, batch.edge_attr)
            total_loss, edge_loss, node_loss = self.model.compute_loss(
                batch.x, batch.edge_attr, outputs
            )

            batch_edges = int(batch.edge_index.shape[1])
            total_loss_accum += total_loss.item() * batch_edges
            edge_loss_accum += edge_loss.item() * batch_edges
            node_loss_accum += node_loss.item() * batch_edges
            total_edges += batch_edges

        return {
            "val_loss": total_loss_accum / max(1, total_edges),
            "val_edge_loss": edge_loss_accum / max(1, total_edges),
            "val_node_loss": node_loss_accum / max(1, total_edges),
        }

    def train(self) -> Dict[str, Any]:
        """Execute the full GTAE training experiment with early stopping."""
        t_start_all = time.time()
        logger.info("=== Starting GTAE Benign-Only Self-Supervised Training ===")

        # Prepare data
        train_snaps, val_snaps, test_snaps = self.prepare_data()

        # Build PyG DataLoaders
        train_loader = DataLoader(
            train_snaps, batch_size=self.train_config.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            val_snaps, batch_size=self.train_config.batch_size, shuffle=False
        )

        # Setup Optimizer and Scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
        )

        patience_counter = 0
        best_checkpoint_path = self.train_config.checkpoint_dir / self.train_config.checkpoint_filename
        self.train_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        logger.info(
            f"Training for max {self.train_config.max_epochs} epochs on {self.device}. "
            f"Early stopping patience: {self.train_config.early_stopping_patience}."
        )

        # Epoch loop
        for epoch in range(1, self.train_config.max_epochs + 1):
            t_epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate(val_loader)
            epoch_time = time.time() - t_epoch_start

            scheduler.step(val_metrics["val_loss"])

            # Record metrics
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_metrics["train_loss"], 6),
                "train_edge_loss": round(train_metrics["train_edge_loss"], 6),
                "train_node_loss": round(train_metrics["train_node_loss"], 6),
                "val_loss": round(val_metrics["val_loss"], 6),
                "val_edge_loss": round(val_metrics["val_edge_loss"], 6),
                "val_node_loss": round(val_metrics["val_node_loss"], 6),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": round(epoch_time, 3),
            }
            self.history.append(epoch_record)

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_epoch = epoch
                patience_counter = 0
                train_cfg_dict = {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(self.train_config).items()
                }
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": self.best_val_loss,
                        "model_config": asdict(self.model_config),
                        "train_config": train_cfg_dict,
                    },
                    best_checkpoint_path,
                )
            else:
                patience_counter += 1

            status_star = "*" if is_best else " "
            logger.info(
                f"Epoch {epoch:2d}/{self.train_config.max_epochs:2d} {status_star} | "
                f"Train Loss: {train_metrics['train_loss']:.5f} (Edge: {train_metrics['train_edge_loss']:.5f}) | "
                f"Val Loss: {val_metrics['val_loss']:.5f} (Edge: {val_metrics['val_edge_loss']:.5f}) | "
                f"Time: {epoch_time:.2f}s"
            )

            # Early stopping check
            if patience_counter >= self.train_config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} (no validation improvement for {patience_counter} epochs)."
                )
                break

        total_train_time = time.time() - t_start_all

        # Save history and configuration
        history_path = self.train_config.checkpoint_dir / self.train_config.history_filename
        config_path = self.train_config.checkpoint_dir / self.train_config.config_filename

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_config": asdict(self.model_config),
                    "train_config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in asdict(self.train_config).items()
                    },
                    "total_parameters": count_parameters(self.model)["total_parameters"],
                    "best_epoch": self.best_epoch,
                    "best_val_loss": self.best_val_loss,
                    "total_training_time_seconds": round(total_train_time, 2),
                },
                f,
                indent=2,
            )

        # 5. Load best checkpoint back into model
        logger.info(f"Loading best checkpoint from epoch {self.best_epoch} for verification...")
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 6. Post-training sanity check & anomaly feature generation
        sanity_results, anomaly_artifacts_path = self._post_training_verification(
            train_snaps[0], val_snaps[0], test_snaps[0]
        )

        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.history),
            "early_stopping": patience_counter >= self.train_config.early_stopping_patience,
            "total_time_seconds": round(total_train_time, 2),
            "avg_epoch_time_seconds": round(total_train_time / len(self.history), 3),
            "best_checkpoint_path": best_checkpoint_path,
            "history_path": history_path,
            "config_path": config_path,
            "anomaly_artifacts_path": anomaly_artifacts_path,
            "sanity_results": sanity_results,
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            ),
            "process_ram_mb": psutil.Process().memory_info().rss / (1024**2),
        }

    @torch.no_grad()
    def _post_training_verification(
        self, train_snap: Data, val_snap: Data, test_snap: Data
    ) -> Tuple[Dict[str, Any], Path]:
        """
        Run inference on 1 train, 1 val, and 1 test snapshot using best checkpoint.
        Extract downstream-ready per-edge anomaly representation:
        f_e = [z_e, |edge_attr - edge_recon|] in R^{E x 113}.
        """
        logger.info("Executing post-training sanity check on Train, Val, and Test snapshots...")
        self.model.eval()
        sanity_results: Dict[str, Any] = {}
        diagnostic_artifacts: Dict[str, Any] = {}

        for split_name, snap in [
            ("train", train_snap),
            ("val", val_snap),
            ("test", test_snap),
        ]:
            snap_dev = snap.to(self.device)
            outputs = self.model(snap_dev.x, snap_dev.edge_index, snap_dev.edge_attr)
            total_loss, edge_loss, node_loss = self.model.compute_loss(
                snap_dev.x, snap_dev.edge_attr, outputs
            )

            # Extract downstream anomaly representations
            anomaly_feats = self.model.extract_anomaly_features(
                outputs["edge_latent"], snap_dev.edge_attr, outputs["edge_recon"]
            )

            f_e = anomaly_feats["flow_features_for_detectors"]  # [E, 113]
            scalar_err = anomaly_feats["scalar_recon_error"]     # [E]

            # Assertions
            assert torch.isfinite(f_e).all(), f"Non-finite anomaly features in {split_name}!"
            assert torch.isfinite(scalar_err).all(), f"Non-finite reconstruction errors in {split_name}!"
            assert f_e.shape == (snap.num_edges, 113), (
                f"Expected shape ({snap.num_edges}, 113), got {f_e.shape}"
            )
            assert outputs["edge_latent"].shape == (snap.num_edges, 32)

            sanity_results[split_name] = {
                "num_nodes": snap.num_nodes,
                "num_edges": snap.num_edges,
                "loss": round(total_loss.item(), 6),
                "edge_loss": round(edge_loss.item(), 6),
                "node_loss": round(node_loss.item(), 6),
                "mean_scalar_recon_error": round(scalar_err.mean().item(), 6),
                "max_scalar_recon_error": round(scalar_err.max().item(), 6),
                "f_e_shape": list(f_e.shape),
            }

            diagnostic_artifacts[split_name] = {
                "window_id": getattr(snap, "window_id", -1),
                "flow_features_113d": f_e.cpu(),
                "scalar_recon_error": scalar_err.cpu(),
                "labels": snap.y.cpu() if getattr(snap, "y", None) is not None else None,
                "edge_time": snap.edge_time.cpu() if getattr(snap, "edge_time", None) is not None else None,
            }

        # Save diagnostic artifact
        self.train_config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.train_config.artifacts_dir / self.train_config.anomaly_artifacts_filename
        torch.save(diagnostic_artifacts, artifact_path)
        logger.info(f"Saved diagnostic anomaly features (113-d) to: {artifact_path}")

        return sanity_results, artifact_path
