"""Flask web application for GTAE-IDS Network Intrusion Detection System.

Supports interactive selection and end-to-end evaluation of any real temporal
graph snapshot from the LSPR23 dataset using the verified clean GTAE model checkpoint.
"""

import datetime
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request

import networkx as nx
import numpy as np
import torch
import joblib

from src.detection.models.hbos import HBOSDetector
from src.detection.models.inne import INNEDetector
from src.detection.models.isolation_forest import IsolationForestDetector
from src.detection.models.ocsvm import OCSVMDetector
from src.models.config import GTAEConfig
from src.models.gtae import GTAEModel
from src.models.normalization import GTAEFeatureScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GTAE_IDS_WebApp")

app = Flask(__name__)

# Paths to verified artifacts
SNAPSHOTS_PATH = PROJECT_ROOT / "data/processed/graphs/temporal_graph_snapshots.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "data/processed/models/best_gtae_model_clean.pt"
SCALERS_PATH = PROJECT_ROOT / "data/processed/scalers/gtae_scalers_clean.pkl"
OCSVM_DASHBOARD_PATH = PROJECT_ROOT / "data/processed/models/ocsvm/ocsvm_dashboard_data.json"
OCSVM_MODEL_DIR = PROJECT_ROOT / "data/processed/models/ocsvm"
OCSVM_PCA_PATH = PROJECT_ROOT / "data/processed/models/ocsvm/ocsvm_pca2d.joblib"

IFOREST_DASHBOARD_PATH = PROJECT_ROOT / "data/processed/models/isolation_forest/iforest_dashboard_data.json"
IFOREST_MODEL_DIR = PROJECT_ROOT / "data/processed/models/isolation_forest"
IFOREST_PCA_PATH = PROJECT_ROOT / "data/processed/models/isolation_forest/iforest_pca2d.joblib"

HBOS_DASHBOARD_PATH = PROJECT_ROOT / "data/processed/models/hbos/hbos_dashboard_data.json"
HBOS_MODEL_DIR = PROJECT_ROOT / "data/processed/models/hbos"
HBOS_PCA_PATH = PROJECT_ROOT / "data/processed/models/hbos/hbos_pca2d.joblib"

INNE_DASHBOARD_PATH = PROJECT_ROOT / "data/processed/models/inne/inne_dashboard_data.json"
INNE_MODEL_DIR = PROJECT_ROOT / "data/processed/models/inne"
INNE_PCA_PATH = PROJECT_ROOT / "data/processed/models/inne/inne_pca2d.joblib"

DEFAULT_SNAPSHOT_INDEX = 46  # Window ID 1584

# Global cached memory state
_ALL_SNAPSHOTS = None
_CACHED_MODEL = None
_CACHED_SCALER = None
_CACHED_DEVICE = None
_CACHED_OCSVM_DATA = None
_CACHED_OCSVM_DETECTOR = None
_CACHED_OCSVM_PCA = None
_CACHED_IFOREST_DATA = None
_CACHED_IFOREST_DETECTOR = None
_CACHED_IFOREST_PCA = None
_CACHED_HBOS_DATA = None
_CACHED_HBOS_DETECTOR = None
_CACHED_HBOS_PCA = None
_CACHED_INNE_DATA = None
_CACHED_INNE_DETECTOR = None
_CACHED_INNE_PCA = None




def get_device():
    global _CACHED_DEVICE
    if _CACHED_DEVICE is None:
        _CACHED_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _CACHED_DEVICE


def load_all_snapshots():
    """Load all graph snapshots once into memory."""
    global _ALL_SNAPSHOTS
    if _ALL_SNAPSHOTS is None:
        logger.info(f"Loading real graph snapshots from {SNAPSHOTS_PATH}...")
        if not SNAPSHOTS_PATH.exists():
            raise FileNotFoundError(f"Snapshots file missing: {SNAPSHOTS_PATH}")
        _ALL_SNAPSHOTS = torch.load(SNAPSHOTS_PATH, weights_only=False)
        logger.info(f"Successfully loaded {len(_ALL_SNAPSHOTS)} real graph snapshots.")
    return _ALL_SNAPSHOTS


def get_snapshot(idx: int):
    """Retrieve a snapshot by index with bounds checking."""
    snapshots = load_all_snapshots()
    if idx < 0 or idx >= len(snapshots):
        raise IndexError(f"Snapshot index {idx} out of range [0, {len(snapshots) - 1}].")
    return snapshots[idx]


def load_model_and_scaler():
    """Load clean GTAE model checkpoint and clean feature scalers."""
    global _CACHED_MODEL, _CACHED_SCALER
    if _CACHED_MODEL is None or _CACHED_SCALER is None:
        device = get_device()
        logger.info(f"Loading clean GTAE model onto {device} from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model_cfg = GTAEConfig(**checkpoint["model_config"])
        model = GTAEModel(model_cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _CACHED_MODEL = model

        logger.info(f"Loading clean feature scalers from {SCALERS_PATH}...")
        _CACHED_SCALER = GTAEFeatureScaler.load(SCALERS_PATH)

    return _CACHED_MODEL, _CACHED_SCALER


def load_ocsvm_detector():
    """Load fitted OCSVM detector model."""
    global _CACHED_OCSVM_DETECTOR
    if _CACHED_OCSVM_DETECTOR is None:
        meta_file = OCSVM_MODEL_DIR / "ocsvm_metadata.json"
        if meta_file.exists():
            logger.info(f"Loading fitted OCSVM detector from {OCSVM_MODEL_DIR}...")
            _CACHED_OCSVM_DETECTOR = OCSVMDetector.load(OCSVM_MODEL_DIR)
        else:
            logger.warning(f"OCSVM model artifacts not found at {OCSVM_MODEL_DIR}.")
    return _CACHED_OCSVM_DETECTOR


def load_ocsvm_pca():
    """Load fitted 2D PCA projection model (fitted strictly on training benign flows)."""
    global _CACHED_OCSVM_PCA
    if _CACHED_OCSVM_PCA is None:
        if OCSVM_PCA_PATH.exists():
            logger.info(f"Loading fitted 2D PCA model from {OCSVM_PCA_PATH}...")
            _CACHED_OCSVM_PCA = joblib.load(OCSVM_PCA_PATH)
        else:
            logger.warning(f"OCSVM PCA model artifact not found at {OCSVM_PCA_PATH}.")
    return _CACHED_OCSVM_PCA


def load_iforest_detector():
    """Load fitted Isolation Forest detector model."""
    global _CACHED_IFOREST_DETECTOR
    if _CACHED_IFOREST_DETECTOR is None:
        meta_file = IFOREST_MODEL_DIR / "isolation_forest_metadata.json"
        if meta_file.exists():
            logger.info(f"Loading fitted IsolationForest detector from {IFOREST_MODEL_DIR}...")
            _CACHED_IFOREST_DETECTOR = IsolationForestDetector.load(IFOREST_MODEL_DIR)
        else:
            logger.warning(f"IsolationForest model artifacts not found at {IFOREST_MODEL_DIR}.")
    return _CACHED_IFOREST_DETECTOR


def load_iforest_pca():
    """Load fitted 2D PCA projection model for Isolation Forest (fitted strictly on raw training benign flows)."""
    global _CACHED_IFOREST_PCA
    if _CACHED_IFOREST_PCA is None:
        if IFOREST_PCA_PATH.exists():
            logger.info(f"Loading fitted IsolationForest 2D PCA model from {IFOREST_PCA_PATH}...")
            _CACHED_IFOREST_PCA = joblib.load(IFOREST_PCA_PATH)
        else:
            logger.warning(f"IsolationForest PCA model artifact not found at {IFOREST_PCA_PATH}.")
    return _CACHED_IFOREST_PCA


def load_hbos_detector():
    """Load fitted HBOS detector model."""
    global _CACHED_HBOS_DETECTOR
    if _CACHED_HBOS_DETECTOR is None:
        meta_file = HBOS_MODEL_DIR / "hbos_metadata.json"
        if meta_file.exists():
            logger.info(f"Loading fitted HBOS detector from {HBOS_MODEL_DIR}...")
            _CACHED_HBOS_DETECTOR = HBOSDetector.load(HBOS_MODEL_DIR)
        else:
            logger.warning(f"HBOS model artifacts not found at {HBOS_MODEL_DIR}.")
    return _CACHED_HBOS_DETECTOR


def load_hbos_pca():
    """Load fitted 2D PCA projection model for HBOS (fitted strictly on raw training benign flows)."""
    global _CACHED_HBOS_PCA
    if _CACHED_HBOS_PCA is None:
        if HBOS_PCA_PATH.exists():
            logger.info(f"Loading fitted HBOS 2D PCA model from {HBOS_PCA_PATH}...")
            _CACHED_HBOS_PCA = joblib.load(HBOS_PCA_PATH)
        else:
            logger.warning(f"HBOS PCA model artifact not found at {HBOS_PCA_PATH}.")
    return _CACHED_HBOS_PCA


def load_inne_detector():
    """Load fitted INNE detector model."""
    global _CACHED_INNE_DETECTOR
    if _CACHED_INNE_DETECTOR is None:
        meta_file = INNE_MODEL_DIR / "inne_metadata.json"
        if meta_file.exists():
            logger.info(f"Loading fitted INNE detector from {INNE_MODEL_DIR}...")
            _CACHED_INNE_DETECTOR = INNEDetector.load(INNE_MODEL_DIR)
        else:
            logger.warning(f"INNE model artifacts not found at {INNE_MODEL_DIR}.")
    return _CACHED_INNE_DETECTOR


def load_inne_pca():
    """Load fitted 2D PCA projection model for INNE (fitted on raw benign training flows)."""
    global _CACHED_INNE_PCA
    if _CACHED_INNE_PCA is None:
        if INNE_PCA_PATH.exists():
            logger.info(f"Loading fitted INNE 2D PCA model from {INNE_PCA_PATH}...")
            _CACHED_INNE_PCA = joblib.load(INNE_PCA_PATH)
        else:
            logger.warning(f"INNE PCA model artifact not found at {INNE_PCA_PATH}.")
    return _CACHED_INNE_PCA




def _marching_squares_contour(Z, x_vals, y_vals, isovalue):
    """
    Compute 2D contour line segments at `isovalue` on scalar grid Z of shape (ny, nx).
    Returns list of line segments: [ [[x1, y1], [x2, y2]], ... ]
    """
    ny, nx = Z.shape
    segments = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            z0 = Z[i, j]
            z1 = Z[i, j + 1]
            z2 = Z[i + 1, j + 1]
            z3 = Z[i + 1, j]

            b0 = 1 if z0 >= isovalue else 0
            b1 = 1 if z1 >= isovalue else 0
            b2 = 1 if z2 >= isovalue else 0
            b3 = 1 if z3 >= isovalue else 0

            case_idx = (b0) | (b1 << 1) | (b2 << 2) | (b3 << 3)
            if case_idx == 0 or case_idx == 15:
                continue

            x0, x1_c = x_vals[j], x_vals[j + 1]
            y0, y1_c = y_vals[i], y_vals[i + 1]

            def lerp(val_a, val_b, pos_a, pos_b):
                if abs(val_b - val_a) < 1e-12:
                    return (pos_a + pos_b) / 2.0
                t = (isovalue - val_a) / (val_b - val_a)
                t = max(0.0, min(1.0, t))
                return pos_a + t * (pos_b - pos_a)

            bottom = [round(float(lerp(z0, z1, x0, x1_c)), 3), round(float(y0), 3)]
            right  = [round(float(x1_c), 3), round(float(lerp(z1, z2, y0, y1_c)), 3)]
            top    = [round(float(lerp(z3, z2, x0, x1_c)), 3), round(float(y1_c), 3)]
            left   = [round(float(x0), 3), round(float(lerp(z0, z3, y0, y1_c)), 3)]

            if case_idx in (1, 14):
                segments.append([left, bottom])
            elif case_idx in (2, 13):
                segments.append([bottom, right])
            elif case_idx in (3, 12):
                segments.append([left, right])
            elif case_idx in (4, 11):
                segments.append([top, right])
            elif case_idx in (5, 10):
                segments.append([left, top])
                segments.append([bottom, right])
            elif case_idx in (6, 9):
                segments.append([bottom, top])
            elif case_idx in (7, 8):
                segments.append([left, top])
    return segments


def _extract_graph_data(snapshot, snap_idx: int):

    """Extract real topology, metadata, and deterministic layout for a snapshot."""
    window_id = getattr(snapshot, "window_id", snap_idx)
    num_nodes = snapshot.num_nodes
    num_edges = snapshot.num_edges

    # Format time range
    time_str = "N/A"
    if hasattr(snapshot, "window_start_ts") and hasattr(snapshot, "window_end_ts"):
        start_dt = datetime.datetime.fromtimestamp(
            snapshot.window_start_ts / 1e6, tz=datetime.timezone.utc
        )
        end_dt = datetime.datetime.fromtimestamp(
            snapshot.window_end_ts / 1e6, tz=datetime.timezone.utc
        )
        time_str = f"{start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC to {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"

    benign_count = int((snapshot.y == 0).sum().item()) if hasattr(snapshot, "y") else 0
    attack_count = int((snapshot.y == 1).sum().item()) if hasattr(snapshot, "y") else 0

    # Build NetworkX graph for layout coordinates
    G = nx.DiGraph()
    src_nodes = snapshot.edge_index[0].tolist()
    dst_nodes = snapshot.edge_index[1].tolist()
    labels = snapshot.y.tolist() if hasattr(snapshot, "y") else [0] * num_edges

    for u, v, lbl in zip(src_nodes, dst_nodes, labels):
        G.add_edge(u, v, label=lbl)

    # Deterministic 2D spring layout
    pos = nx.spring_layout(G, seed=42, k=0.45, iterations=60)
    degrees = dict(G.degree())

    nodes_data = []
    for n in range(num_nodes):
        x, y = pos.get(n, (0.0, 0.0))
        ip = snapshot.node_ip_map[n] if hasattr(snapshot, "node_ip_map") and n < len(snapshot.node_ip_map) else f"Host-{n}"
        nodes_data.append({
            "id": n,
            "ip": ip,
            "degree": degrees.get(n, 0),
            "x": float(x),
            "y": float(y),
        })

    edges_data = []
    for idx, (u, v, lbl) in enumerate(zip(src_nodes, dst_nodes, labels)):
        edges_data.append({
            "source": u,
            "target": v,
            "label": int(lbl),  # Ground truth label: 0=benign, 1=malicious
            "edge_idx": idx,
        })

    return {
        "status": "success",
        "snapshot_index": snap_idx,
        "window_id": int(window_id),
        "time_range": time_str,
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "node_feature_shape": list(snapshot.x.shape),
        "edge_feature_shape": list(snapshot.edge_attr.shape),
        "benign_count": benign_count,
        "attack_count": attack_count,
        "nodes": nodes_data,
        "edges": edges_data,
    }


def _execute_ocsvm_on_snapshot(snapshot, snap_idx: int, flow_features_113d, scalar_recon_error):
    """Execute real-time OCSVM prediction on the 113-D features of the selected snapshot."""
    detector = load_ocsvm_detector()
    if detector is None:
        return {
            "status": "unavailable",
            "message": "OCSVM model artifacts not found. Please train OCSVM first.",
        }

    # Determine split provenance for this snapshot
    if 0 <= snap_idx < 208:
        split_name = "Train Split"
        split_desc = "Chronological 0–70% (Benign Normality Baseline)"
    elif 208 <= snap_idx < 252:
        split_name = "Validation Split"
        split_desc = "Chronological 70–85% (Threshold Calibration Split)"
    else:
        split_name = "Test Split"
        split_desc = "Chronological 85–100% (Held-Out Final Evaluation Split)"

    preds, scores, infer_time = detector.predict(flow_features_113d.cpu())
    y = snapshot.y.cpu().numpy() if hasattr(snapshot, "y") else np.zeros(len(preds), dtype=int)
    num_edges = len(preds)

    benign_total = int((y == 0).sum())
    attack_total = int((y == 1).sum())
    detected_total = int((preds == 1).sum())
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    # Snapshot-specific metrics
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if attack_total == 0 else 0.0)
    recall = float(tp / attack_total) if attack_total > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / benign_total) if benign_total > 0 else 0.0

    # Score histogram for this snapshot
    b_scores = scores[y == 0] if benign_total > 0 else np.array([])
    a_scores = scores[y == 1] if attack_total > 0 else np.array([])
    all_min = float(min(scores.min(), detector.threshold - 0.2)) if len(scores) > 0 else -1.0
    all_max = float(max(scores.max(), detector.threshold + 0.2)) if len(scores) > 0 else 2.0
    bins = np.linspace(all_min, all_max, 21)
    b_counts, _ = np.histogram(b_scores, bins=bins) if len(b_scores) > 0 else (np.zeros(20, dtype=int), bins)
    a_counts, _ = np.histogram(a_scores, bins=bins) if len(a_scores) > 0 else (np.zeros(20, dtype=int), bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Flow details for this snapshot
    src_nodes = snapshot.edge_index[0].cpu().tolist()
    dst_nodes = snapshot.edge_index[1].cpu().tolist()
    edge_times = snapshot.edge_time.cpu().tolist() if hasattr(snapshot, "edge_time") else [0] * num_edges
    recon_err_list = scalar_recon_error.cpu().tolist()

    flows_table = []
    # Sort by anomaly score descending so top anomalies appear first
    sorted_order = np.argsort(-scores)
    for rank, edge_i in enumerate(sorted_order[:60]):  # top 60 in snapshot
        raw_ts = edge_times[edge_i]
        dt_str = "N/A"
        if raw_ts > 0:
            dt_str = datetime.datetime.fromtimestamp(raw_ts / 1e6, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        is_tp = bool((preds[edge_i] == 1) and (y[edge_i] == 1))
        is_fp = bool((preds[edge_i] == 1) and (y[edge_i] == 0))
        is_fn = bool((preds[edge_i] == 0) and (y[edge_i] == 1))
        is_tn = bool((preds[edge_i] == 0) and (y[edge_i] == 0))

        if is_tp:
            status_badge = "True Positive"
            badge_class = "badge-tp"
        elif is_fp:
            status_badge = "False Alarm"
            badge_class = "badge-fp"
        elif is_fn:
            status_badge = "Missed Attack"
            badge_class = "badge-fn"
        else:
            status_badge = "Benign Inlier"
            badge_class = "badge-tn"

        u = src_nodes[edge_i]
        v = dst_nodes[edge_i]
        src_label = snapshot.node_ip_map[u] if hasattr(snapshot, "node_ip_map") and u < len(snapshot.node_ip_map) else f"Host-{u}"
        dst_label = snapshot.node_ip_map[v] if hasattr(snapshot, "node_ip_map") and v < len(snapshot.node_ip_map) else f"Host-{v}"

        flows_table.append({
            "rank": rank + 1,
            "edge_idx": int(edge_i),
            "src": src_label,
            "dst": dst_label,
            "timestamp": dt_str,
            "recon_error": round(float(recon_err_list[edge_i]), 4),
            "anomaly_score": round(float(scores[edge_i]), 4),
            "predicted": int(preds[edge_i]),
            "ground_truth": int(y[edge_i]),
            "ground_truth_str": "Attack" if y[edge_i] == 1 else "Benign",
            "status_badge": status_badge,
            "badge_class": badge_class,
        })

    # 2D PCA Projection & Decision Boundary
    pca = load_ocsvm_pca()
    feats_np = flow_features_113d.cpu().numpy()
    feats_scaled = detector.scaler.transform(feats_np)
    pca_coords = pca.transform(feats_scaled) if pca is not None else np.zeros((num_edges, 2))

    # Compute bounding box with margin
    if len(pca_coords) > 0:
        c_min_x, c_max_x = float(pca_coords[:, 0].min()), float(pca_coords[:, 0].max())
        c_min_y, c_max_y = float(pca_coords[:, 1].min()), float(pca_coords[:, 1].max())
        span_x = max(c_max_x - c_min_x, 1.0)
        span_y = max(c_max_y - c_min_y, 1.0)
        pad_x = max(2.5, span_x * 0.18)
        pad_y = max(2.5, span_y * 0.18)
        x_min, x_max = c_min_x - pad_x, c_max_x + pad_x
        y_min, y_max = c_min_y - pad_y, c_max_y + pad_y
    else:
        x_min, x_max, y_min, y_max = -5.0, 5.0, -5.0, 5.0

    nx, ny = 40, 40
    x_vals = np.linspace(x_min, x_max, nx)
    y_vals = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    if pca is not None:
        grid_113d_scaled = pca.inverse_transform(grid_2d)
        df_grid = detector.model.decision_function(grid_113d_scaled)
        scores_grid = (-df_grid).reshape(ny, nx)
        boundary_segments = _marching_squares_contour(scores_grid, x_vals, y_vals, detector.threshold)
        region_mask = (scores_grid > detector.threshold).astype(int).tolist()
        pca_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    else:
        boundary_segments = []
        region_mask = [[0] * nx for _ in range(ny)]
        pca_var = [20.87, 8.97]

    flow_points = []
    for i in range(num_edges):
        u = src_nodes[i]
        v = dst_nodes[i]
        s_lbl = snapshot.node_ip_map[u] if hasattr(snapshot, "node_ip_map") and u < len(snapshot.node_ip_map) else f"Host-{u}"
        d_lbl = snapshot.node_ip_map[v] if hasattr(snapshot, "node_ip_map") and v < len(snapshot.node_ip_map) else f"Host-{v}"
        flow_points.append({
            "pc1": round(float(pca_coords[i, 0]), 3),
            "pc2": round(float(pca_coords[i, 1]), 3),
            "ground_truth": int(y[i]),
            "predicted": int(preds[i]),
            "anomaly_score": round(float(scores[i]), 4),
            "edge_idx": int(i),
            "src": s_lbl,
            "dst": d_lbl,
        })

    decision_boundary = {
        "x_range": [round(x_min, 3), round(x_max, 3)],
        "y_range": [round(y_min, 3), round(y_max, 3)],
        "grid_nx": nx,
        "grid_ny": ny,
        "region_mask": region_mask,
        "boundary_segments": boundary_segments,
        "pca_variance": pca_var,
        "frozen_threshold": round(float(detector.threshold), 6),
        "flow_points": flow_points,
    }

    return {
        "status": "success",
        "snapshot_index": snap_idx,
        "window_id": int(getattr(snapshot, "window_id", snap_idx)),
        "split_name": split_name,
        "split_desc": split_desc,
        "frozen_threshold": round(float(detector.threshold), 6),
        "inference_time_ms": round(infer_time * 1000, 2),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        },
        "counts": {
            "total_flows": num_edges,
            "benign_total": benign_total,
            "attack_total": attack_total,
            "detected_anomalies": detected_total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "decision_boundary": decision_boundary,
        "flows_table": flows_table,
    }


def _execute_iforest_on_snapshot(
    snapshot,
    snap_idx: int,
    flow_features_113d: torch.Tensor,
    scalar_recon_error: torch.Tensor,
):
    """
    Score snapshot flows using fitted Isolation Forest detector.
    Prepares 2D PCA projection points, 4 primary metrics, and detected flow table.
    Strictly adheres to protocol: No smooth geometric decision boundary is synthesized
    since Isolation Forest decisions are formed by tree ensembles.
    """
    detector = load_iforest_detector()
    if detector is None or not detector.is_fitted:
        return {"status": "not_loaded", "message": "Isolation Forest detector not fitted or loaded."}

    num_edges = snapshot.edge_index.size(1)
    if num_edges == 0:
        return {"status": "empty_snapshot", "message": "Snapshot contains no edges/flows."}

    # Split identification
    if snap_idx < 175:
        split_name = "Train Split"
        split_desc = "Chronological 0–70% (Benign Normality Baseline)"
    elif snap_idx < 233:
        split_name = "Validation Split"
        split_desc = "Chronological 70–80% (Calibration & Validation)"
    else:
        split_name = "Test Split"
        split_desc = "Chronological 80–100% (Strictly Held-Out Evaluation)"

    # Preprocessing decision: Tree-based orthogonal splits invariant to monotonic scaling
    # Raw 113-D features are scored directly
    feats_np = flow_features_113d.cpu().numpy()
    scores, infer_time = detector.score(feats_np)
    preds = (scores >= detector.threshold).astype(int)

    # Ground truth labels
    if hasattr(snapshot, "edge_y") and snapshot.edge_y is not None:
        y = snapshot.edge_y.cpu().numpy()
    elif hasattr(snapshot, "y") and snapshot.y is not None:
        y = snapshot.y.cpu().numpy()
    else:
        y = np.zeros(num_edges, dtype=int)

    # Primary 4 metrics
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    attack_total = tp + fn
    benign_total = tn + fp
    detected_total = tp + fp

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / attack_total) if attack_total > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / benign_total) if benign_total > 0 else 0.0

    # Flow table
    recon_err_list = scalar_recon_error.cpu().numpy().tolist()
    sorted_flow_indices = np.argsort(-scores)

    src_nodes = snapshot.edge_index[0].cpu().numpy().tolist()
    dst_nodes = snapshot.edge_index[1].cpu().numpy().tolist()

    edge_timestamps = None
    if hasattr(snapshot, "edge_timestamps") and snapshot.edge_timestamps is not None:
        edge_timestamps = snapshot.edge_timestamps.cpu().numpy().tolist()

    flows_table = []
    for rank_idx, edge_i in enumerate(sorted_flow_indices):
        u = src_nodes[edge_i]
        v = dst_nodes[edge_i]
        src_ip = snapshot.node_ip_map[u] if hasattr(snapshot, "node_ip_map") and u < len(snapshot.node_ip_map) else f"Host-{u}"
        dst_ip = snapshot.node_ip_map[v] if hasattr(snapshot, "node_ip_map") and v < len(snapshot.node_ip_map) else f"Host-{v}"

        if edge_timestamps and edge_i < len(edge_timestamps):
            ts_val = edge_timestamps[edge_i]
            dt_str = datetime.datetime.fromtimestamp(ts_val, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            dt_str = "N/A"

        if preds[edge_i] == 1 and y[edge_i] == 1:
            status_badge = "True Positive"
            badge_class = "badge-tp"
        elif preds[edge_i] == 1 and y[edge_i] == 0:
            status_badge = "False Alarm"
            badge_class = "badge-fp"
        elif preds[edge_i] == 0 and y[edge_i] == 1:
            status_badge = "Missed Attack"
            badge_class = "badge-fn"
        else:
            status_badge = "Benign Inlier"
            badge_class = "badge-tn"

        flows_table.append({
            "rank": rank_idx + 1,
            "edge_idx": int(edge_i),
            "src": src_ip,
            "dst": dst_ip,
            "timestamp": dt_str,
            "recon_error": round(float(recon_err_list[edge_i]), 4),
            "anomaly_score": round(float(scores[edge_i]), 4),
            "predicted": int(preds[edge_i]),
            "ground_truth": int(y[edge_i]),
            "ground_truth_str": "Attack" if y[edge_i] == 1 else "Benign",
            "status_badge": status_badge,
            "badge_class": badge_class,
        })

    # 2D PCA Projection for Isolation Forest (fitted strictly on raw benign training flows)
    pca = load_iforest_pca()
    pca_coords = pca.transform(feats_np) if pca is not None else np.zeros((num_edges, 2))

    if len(pca_coords) > 0:
        c_min_x, c_max_x = float(pca_coords[:, 0].min()), float(pca_coords[:, 0].max())
        c_min_y, c_max_y = float(pca_coords[:, 1].min()), float(pca_coords[:, 1].max())
        span_x = max(c_max_x - c_min_x, 1.0)
        span_y = max(c_max_y - c_min_y, 1.0)
        pad_x = max(2.5, span_x * 0.15)
        pad_y = max(2.5, span_y * 0.15)
        x_min, x_max = c_min_x - pad_x, c_max_x + pad_x
        y_min, y_max = c_min_y - pad_y, c_max_y + pad_y
    else:
        x_min, x_max, y_min, y_max = -5.0, 5.0, -5.0, 5.0

    pca_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_] if pca is not None else [21.67, 12.54]

    flow_points = []
    for i in range(num_edges):
        u = src_nodes[i]
        v = dst_nodes[i]
        s_lbl = snapshot.node_ip_map[u] if hasattr(snapshot, "node_ip_map") and u < len(snapshot.node_ip_map) else f"Host-{u}"
        d_lbl = snapshot.node_ip_map[v] if hasattr(snapshot, "node_ip_map") and v < len(snapshot.node_ip_map) else f"Host-{v}"
        flow_points.append({
            "pc1": round(float(pca_coords[i, 0]), 3),
            "pc2": round(float(pca_coords[i, 1]), 3),
            "ground_truth": int(y[i]),
            "predicted": int(preds[i]),
            "anomaly_score": round(float(scores[i]), 4),
            "edge_idx": int(i),
            "src": s_lbl,
            "dst": d_lbl,
        })

    pca_visualization = {
        "x_range": [round(x_min, 3), round(x_max, 3)],
        "y_range": [round(y_min, 3), round(y_max, 3)],
        "pca_variance": pca_var,
        "frozen_threshold": round(float(detector.threshold), 6),
        "flow_points": flow_points,
    }

    # Anomaly-score histogram calculation (benign vs malicious distributions + threshold line)
    thresh_val = float(detector.threshold)
    min_score = float(scores.min())
    max_score = float(scores.max())

    pad_left = max(2.0, (thresh_val - min_score) * 0.05) if thresh_val > min_score else 2.0
    pad_right = max(2.0, (max_score - thresh_val) * 0.05) if max_score > thresh_val else 5.0
    hist_min = min_score - pad_left
    hist_max = max(max_score, thresh_val) + pad_right

    num_bins = 28
    bin_edges = np.linspace(hist_min, hist_max, num_bins + 1)

    benign_scores = scores[y == 0]
    attack_scores = scores[y == 1]

    counts_benign, _ = np.histogram(benign_scores, bins=bin_edges)
    counts_attack, _ = np.histogram(attack_scores, bins=bin_edges)
    counts_pred_anom, _ = np.histogram(scores[preds == 1], bins=bin_edges)

    bins_data = []
    for bi in range(num_bins):
        b_start = round(float(bin_edges[bi]), 2)
        b_end = round(float(bin_edges[bi + 1]), 2)
        c_b = int(counts_benign[bi])
        c_a = int(counts_attack[bi])
        c_p = int(counts_pred_anom[bi])
        is_anom_bin = bool(bin_edges[bi + 1] >= thresh_val)
        bins_data.append({
            "bin_idx": bi,
            "range": [b_start, b_end],
            "benign_count": c_b,
            "attack_count": c_a,
            "total_count": c_b + c_a,
            "pred_anomaly_count": c_p,
            "is_anomaly_region": is_anom_bin,
        })

    score_histogram = {
        "frozen_threshold": round(thresh_val, 6),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "hist_min": round(hist_min, 4),
        "hist_max": round(hist_max, 4),
        "bin_edges": [round(float(e), 2) for e in bin_edges],
        "bins": bins_data,
        "max_bin_count": int(max([b["total_count"] for b in bins_data] + [1])),
        "benign_total": int(len(benign_scores)),
        "attack_total": int(len(attack_scores)),
        "predicted_anomalies_total": int(np.sum(preds == 1)),
    }


    return {
        "status": "success",
        "snapshot_index": snap_idx,
        "window_id": int(getattr(snapshot, "window_id", snap_idx)),
        "split_name": split_name,
        "split_desc": split_desc,
        "frozen_threshold": round(float(detector.threshold), 6),
        "inference_time_ms": round(infer_time * 1000, 2),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        },
        "counts": {
            "total_flows": num_edges,
            "benign_total": benign_total,
            "attack_total": attack_total,
            "detected_anomalies": detected_total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "score_histogram": score_histogram,
        "pca_visualization": pca_visualization,
        "flows_table": flows_table,
    }


def _execute_hbos_on_snapshot(
    snapshot,
    snap_idx: int,
    flow_features_113d: torch.Tensor,
    scalar_recon_error: torch.Tensor,
):
    """
    Score snapshot flows using fitted HBOS detector.
    Prepares 2D PCA projection points, 4 primary metrics, and detected flow table.
    """
    detector = load_hbos_detector()
    if detector is None or not detector.is_fitted:
        return {"status": "not_loaded", "message": "HBOS detector not fitted or loaded."}

    num_edges = snapshot.edge_index.size(1)
    if num_edges == 0:
        return {"status": "empty_snapshot", "message": "Snapshot contains no edges/flows."}

    # Split identification
    if snap_idx < 175:
        split_name = "Train Split"
        split_desc = "Chronological 0–70% (Benign Normality Baseline)"
    elif snap_idx < 233:
        split_name = "Validation Split"
        split_desc = "Chronological 70–80% (Calibration & Validation)"
    else:
        split_name = "Test Split"
        split_desc = "Chronological 80–100% (Strictly Held-Out Evaluation)"

    feats_np = flow_features_113d.cpu().numpy()
    scores, infer_time = detector.score(feats_np)
    preds = (scores >= detector.threshold).astype(int)

    # Ground truth labels
    if hasattr(snapshot, "edge_y") and snapshot.edge_y is not None:
        y = snapshot.edge_y.cpu().numpy()
    elif hasattr(snapshot, "y") and snapshot.y is not None:
        y = snapshot.y.cpu().numpy()
    else:
        y = np.zeros(num_edges, dtype=int)

    # Primary 4 metrics
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    attack_total = tp + fn
    benign_total = tn + fp
    detected_total = tp + fp

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / attack_total) if attack_total > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / benign_total) if benign_total > 0 else 0.0

    # Flow table
    recon_err_list = scalar_recon_error.cpu().numpy().tolist()
    sorted_flow_indices = np.argsort(-scores)

    src_nodes = snapshot.edge_index[0].cpu().numpy().tolist()
    dst_nodes = snapshot.edge_index[1].cpu().numpy().tolist()

    edge_timestamps = None
    if hasattr(snapshot, "edge_timestamps") and snapshot.edge_timestamps is not None:
        edge_timestamps = snapshot.edge_timestamps.cpu().numpy().tolist()

    flows_table = []
    for rank_idx, edge_i in enumerate(sorted_flow_indices):
        u = src_nodes[edge_i]
        v = dst_nodes[edge_i]
        src_ip = snapshot.node_ip_map[u] if hasattr(snapshot, "node_ip_map") and u < len(snapshot.node_ip_map) else f"Host-{u}"
        dst_ip = snapshot.node_ip_map[v] if hasattr(snapshot, "node_ip_map") and v < len(snapshot.node_ip_map) else f"Host-{v}"

        if edge_timestamps and edge_i < len(edge_timestamps):
            ts_val = edge_timestamps[edge_i]
            dt_str = datetime.datetime.fromtimestamp(ts_val, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            dt_str = "N/A"

        if preds[edge_i] == 1 and y[edge_i] == 1:
            status_badge = "True Positive"
            badge_class = "badge-tp"
        elif preds[edge_i] == 1 and y[edge_i] == 0:
            status_badge = "False Alarm"
            badge_class = "badge-fp"
        elif preds[edge_i] == 0 and y[edge_i] == 1:
            status_badge = "Missed Attack"
            badge_class = "badge-fn"
        else:
            status_badge = "Benign Inlier"
            badge_class = "badge-tn"

        flows_table.append({
            "rank": rank_idx + 1,
            "edge_idx": int(edge_i),
            "src": src_ip,
            "dst": dst_ip,
            "timestamp": dt_str,
            "recon_error": round(float(recon_err_list[edge_i]), 4),
            "anomaly_score": round(float(scores[edge_i]), 4),
            "predicted": int(preds[edge_i]),
            "ground_truth": int(y[edge_i]),
            "ground_truth_str": "Attack" if y[edge_i] == 1 else "Benign",
            "status_badge": status_badge,
            "badge_class": badge_class,
        })

    # 2D PCA Projection for HBOS (fitted strictly on raw benign training flows)
    pca = load_hbos_pca()
    pca_coords = pca.transform(feats_np) if pca is not None else np.zeros((num_edges, 2))

    if len(pca_coords) > 0:
        c_min_x, c_max_x = float(pca_coords[:, 0].min()), float(pca_coords[:, 0].max())
        c_min_y, c_max_y = float(pca_coords[:, 1].min()), float(pca_coords[:, 1].max())
        span_x = max(c_max_x - c_min_x, 1.0)
        span_y = max(c_max_y - c_min_y, 1.0)
        pad_x = max(2.5, span_x * 0.15)
        pad_y = max(2.5, span_y * 0.15)
        x_min, x_max = c_min_x - pad_x, c_max_x + pad_x
        y_min, y_max = c_min_y - pad_y, c_max_y + pad_y
    else:
        x_min, x_max, y_min, y_max = -5.0, 5.0, -5.0, 5.0

    pca_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_] if pca is not None else [21.67, 12.54]

    flow_points = []
    for i in range(num_edges):
        u = src_nodes[i]
        v = dst_nodes[i]
        s_lbl = snapshot.node_ip_map[u] if hasattr(snapshot, "node_ip_map") and u < len(snapshot.node_ip_map) else f"Host-{u}"
        d_lbl = snapshot.node_ip_map[v] if hasattr(snapshot, "node_ip_map") and v < len(snapshot.node_ip_map) else f"Host-{v}"
        flow_points.append({
            "pc1": round(float(pca_coords[i, 0]), 3),
            "pc2": round(float(pca_coords[i, 1]), 3),
            "ground_truth": int(y[i]),
            "predicted": int(preds[i]),
            "anomaly_score": round(float(scores[i]), 4),
            "edge_idx": int(i),
            "src": s_lbl,
            "dst": d_lbl,
        })

    pca_visualization = {
        "x_range": [round(x_min, 3), round(x_max, 3)],
        "y_range": [round(y_min, 3), round(y_max, 3)],
        "pca_variance": pca_var,
        "frozen_threshold": round(float(detector.threshold), 6),
        "flow_points": flow_points,
    }

    # Anomaly-score histogram calculation (benign vs malicious distributions + threshold line)
    thresh_val = float(detector.threshold)
    min_score = float(scores.min())
    max_score = float(scores.max())

    pad_left = max(2.0, (thresh_val - min_score) * 0.05) if thresh_val > min_score else 2.0
    pad_right = max(2.0, (max_score - thresh_val) * 0.05) if max_score > thresh_val else 5.0
    hist_min = min_score - pad_left
    hist_max = max(max_score, thresh_val) + pad_right

    num_bins = 28
    bin_edges = np.linspace(hist_min, hist_max, num_bins + 1)

    benign_scores = scores[y == 0]
    attack_scores = scores[y == 1]

    counts_benign, _ = np.histogram(benign_scores, bins=bin_edges)
    counts_attack, _ = np.histogram(attack_scores, bins=bin_edges)
    counts_pred_anom, _ = np.histogram(scores[preds == 1], bins=bin_edges)

    bins_data = []
    for bi in range(num_bins):
        b_start = round(float(bin_edges[bi]), 2)
        b_end = round(float(bin_edges[bi + 1]), 2)
        c_b = int(counts_benign[bi])
        c_a = int(counts_attack[bi])
        c_p = int(counts_pred_anom[bi])
        is_anom_bin = bool(bin_edges[bi + 1] >= thresh_val)
        bins_data.append({
            "bin_idx": bi,
            "range": [b_start, b_end],
            "benign_count": c_b,
            "attack_count": c_a,
            "total_count": c_b + c_a,
            "pred_anomaly_count": c_p,
            "is_anomaly_region": is_anom_bin,
        })

    score_histogram = {
        "frozen_threshold": round(thresh_val, 6),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "hist_min": round(hist_min, 4),
        "hist_max": round(hist_max, 4),
        "bin_edges": [round(float(e), 2) for e in bin_edges],
        "bins": bins_data,
        "max_bin_count": int(max([b["total_count"] for b in bins_data] + [1])),
        "benign_total": int(len(benign_scores)),
        "attack_total": int(len(attack_scores)),
        "predicted_anomalies_total": int(np.sum(preds == 1)),
    }


    # Anomaly-score histogram calculation (benign vs malicious distributions + threshold line)
    thresh_val = float(detector.threshold)
    min_score = float(scores.min())
    max_score = float(scores.max())

    pad_left = max(2.0, (thresh_val - min_score) * 0.05) if thresh_val > min_score else 2.0
    pad_right = max(2.0, (max_score - thresh_val) * 0.05) if max_score > thresh_val else 5.0
    hist_min = min_score - pad_left
    hist_max = max(max_score, thresh_val) + pad_right

    num_bins = 28
    bin_edges = np.linspace(hist_min, hist_max, num_bins + 1)

    benign_scores = scores[y == 0]
    attack_scores = scores[y == 1]

    counts_benign, _ = np.histogram(benign_scores, bins=bin_edges)
    counts_attack, _ = np.histogram(attack_scores, bins=bin_edges)
    counts_pred_anom, _ = np.histogram(scores[preds == 1], bins=bin_edges)

    bins_data = []
    for bi in range(num_bins):
        b_start = round(float(bin_edges[bi]), 2)
        b_end = round(float(bin_edges[bi + 1]), 2)
        c_b = int(counts_benign[bi])
        c_a = int(counts_attack[bi])
        c_p = int(counts_pred_anom[bi])
        is_anom_bin = bool(bin_edges[bi + 1] >= thresh_val)
        bins_data.append({
            "bin_idx": bi,
            "range": [b_start, b_end],
            "benign_count": c_b,
            "attack_count": c_a,
            "total_count": c_b + c_a,
            "pred_anomaly_count": c_p,
            "is_anomaly_region": is_anom_bin,
        })

    score_histogram = {
        "frozen_threshold": round(thresh_val, 6),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "hist_min": round(hist_min, 4),
        "hist_max": round(hist_max, 4),
        "bin_edges": [round(float(e), 2) for e in bin_edges],
        "bins": bins_data,
        "max_bin_count": int(max([b["total_count"] for b in bins_data] + [1])),
        "benign_total": int(len(benign_scores)),
        "attack_total": int(len(attack_scores)),
        "predicted_anomalies_total": int(np.sum(preds == 1)),
    }

    return {
        "status": "success",
        "snapshot_index": snap_idx,
        "window_id": int(getattr(snapshot, "window_id", snap_idx)),
        "split_name": split_name,
        "split_desc": split_desc,
        "frozen_threshold": round(float(detector.threshold), 6),
        "inference_time_ms": round(infer_time * 1000, 2),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        },
        "counts": {
            "total_flows": num_edges,
            "benign_total": benign_total,
            "attack_total": attack_total,
            "detected_anomalies": detected_total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "score_histogram": score_histogram,
        "pca_visualization": pca_visualization,
        "flows_table": flows_table,
    }


def _execute_inne_on_snapshot(
    snapshot,
    snap_idx: int,
    flow_features_113d: torch.Tensor,
    scalar_recon_error: torch.Tensor,
):
    """
    Score snapshot flows using fitted INNE detector.
    Prepares 2D PCA projection points, 4 primary metrics, and detected flow table.
    """
    detector = load_inne_detector()
    if detector is None or not detector.is_fitted:
        return {"status": "not_loaded", "message": "INNE detector not fitted or loaded."}

    num_edges = snapshot.edge_index.size(1)
    if num_edges == 0:
        return {"status": "empty_snapshot", "message": "Snapshot contains no edges/flows."}

    if snap_idx < 175:
        split_name = "Train Split"
        split_desc = "Chronological 0\u201370% (Benign Normality Baseline)"
    elif snap_idx < 233:
        split_name = "Validation Split"
        split_desc = "Chronological 70\u201380% (Calibration & Validation)"
    else:
        split_name = "Test Split"
        split_desc = "Chronological 80\u2013100% (Strictly Held-Out Evaluation)"

    feats_np = flow_features_113d.cpu().numpy()
    scores, infer_time = detector.score(feats_np)
    preds = (scores >= detector.threshold).astype(int)

    if hasattr(snapshot, "edge_y") and snapshot.edge_y is not None:
        y = snapshot.edge_y.cpu().numpy()
    elif hasattr(snapshot, "y") and snapshot.y is not None:
        y = snapshot.y.cpu().numpy()
    else:
        y = np.zeros(num_edges, dtype=int)

    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))

    attack_total = tp + fn
    benign_total = tn + fp
    detected_total = tp + fp

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / attack_total) if attack_total > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / benign_total) if benign_total > 0 else 0.0

    recon_err_list = scalar_recon_error.cpu().numpy().tolist()
    sorted_flow_indices = np.argsort(-scores)
    src_nodes = snapshot.edge_index[0].cpu().numpy().tolist()
    dst_nodes = snapshot.edge_index[1].cpu().numpy().tolist()

    edge_timestamps = None
    if hasattr(snapshot, "edge_timestamps") and snapshot.edge_timestamps is not None:
        edge_timestamps = snapshot.edge_timestamps.cpu().numpy().tolist()

    flows_table = []
    for rank_idx, edge_i in enumerate(sorted_flow_indices):
        u = src_nodes[edge_i]
        v = dst_nodes[edge_i]
        src_ip = snapshot.node_ip_map[u] if hasattr(snapshot, "node_ip_map") and u < len(snapshot.node_ip_map) else f"Host-{u}"
        dst_ip = snapshot.node_ip_map[v] if hasattr(snapshot, "node_ip_map") and v < len(snapshot.node_ip_map) else f"Host-{v}"

        if edge_timestamps and edge_i < len(edge_timestamps):
            ts_val = edge_timestamps[edge_i]
            dt_str = datetime.datetime.fromtimestamp(ts_val, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            dt_str = "N/A"

        if preds[edge_i] == 1 and y[edge_i] == 1:
            status_badge = "True Positive"
            badge_class = "badge-tp"
        elif preds[edge_i] == 1 and y[edge_i] == 0:
            status_badge = "False Alarm"
            badge_class = "badge-fp"
        elif preds[edge_i] == 0 and y[edge_i] == 1:
            status_badge = "Missed Attack"
            badge_class = "badge-fn"
        else:
            status_badge = "Benign Inlier"
            badge_class = "badge-tn"

        flows_table.append({
            "rank": rank_idx + 1,
            "edge_idx": int(edge_i),
            "src": src_ip,
            "dst": dst_ip,
            "timestamp": dt_str,
            "recon_error": round(float(recon_err_list[edge_i]), 4),
            "anomaly_score": round(float(scores[edge_i]), 4),
            "predicted": int(preds[edge_i]),
            "ground_truth": int(y[edge_i]),
            "ground_truth_str": "Attack" if y[edge_i] == 1 else "Benign",
            "status_badge": status_badge,
            "badge_class": badge_class,
        })

    # 2D PCA Projection
    pca = load_inne_pca()
    pca_coords = pca.transform(feats_np) if pca is not None else np.zeros((num_edges, 2))

    if len(pca_coords) > 0:
        c_min_x, c_max_x = float(pca_coords[:, 0].min()), float(pca_coords[:, 0].max())
        c_min_y, c_max_y = float(pca_coords[:, 1].min()), float(pca_coords[:, 1].max())
        span_x = max(c_max_x - c_min_x, 1.0)
        span_y = max(c_max_y - c_min_y, 1.0)
        pad_x = max(2.5, span_x * 0.15)
        pad_y = max(2.5, span_y * 0.15)
        x_min, x_max = c_min_x - pad_x, c_max_x + pad_x
        y_min, y_max = c_min_y - pad_y, c_max_y + pad_y
    else:
        x_min, x_max, y_min, y_max = -5.0, 5.0, -5.0, 5.0

    pca_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_] if pca is not None else [21.67, 12.54]

    flow_points = []
    for i in range(num_edges):
        u = src_nodes[i]
        v = dst_nodes[i]
        s_lbl = snapshot.node_ip_map[u] if hasattr(snapshot, "node_ip_map") and u < len(snapshot.node_ip_map) else f"Host-{u}"
        d_lbl = snapshot.node_ip_map[v] if hasattr(snapshot, "node_ip_map") and v < len(snapshot.node_ip_map) else f"Host-{v}"
        flow_points.append({
            "pc1": round(float(pca_coords[i, 0]), 3),
            "pc2": round(float(pca_coords[i, 1]), 3),
            "ground_truth": int(y[i]),
            "predicted": int(preds[i]),
            "anomaly_score": round(float(scores[i]), 4),
            "edge_idx": int(i),
            "src": s_lbl,
            "dst": d_lbl,
        })

    pca_visualization = {
        "x_range": [round(x_min, 3), round(x_max, 3)],
        "y_range": [round(y_min, 3), round(y_max, 3)],
        "pca_variance": pca_var,
        "frozen_threshold": round(float(detector.threshold), 6),
        "flow_points": flow_points,
    }

    # Anomaly-score histogram calculation (benign vs malicious distributions + threshold line)
    thresh_val = float(detector.threshold)
    min_score = float(scores.min())
    max_score = float(scores.max())

    pad_left = max(2.0, (thresh_val - min_score) * 0.05) if thresh_val > min_score else 2.0
    pad_right = max(2.0, (max_score - thresh_val) * 0.05) if max_score > thresh_val else 5.0
    hist_min = min_score - pad_left
    hist_max = max(max_score, thresh_val) + pad_right

    num_bins = 28
    bin_edges = np.linspace(hist_min, hist_max, num_bins + 1)

    benign_scores = scores[y == 0]
    attack_scores = scores[y == 1]

    counts_benign, _ = np.histogram(benign_scores, bins=bin_edges)
    counts_attack, _ = np.histogram(attack_scores, bins=bin_edges)
    counts_pred_anom, _ = np.histogram(scores[preds == 1], bins=bin_edges)

    bins_data = []
    for bi in range(num_bins):
        b_start = round(float(bin_edges[bi]), 2)
        b_end = round(float(bin_edges[bi + 1]), 2)
        c_b = int(counts_benign[bi])
        c_a = int(counts_attack[bi])
        c_p = int(counts_pred_anom[bi])
        is_anom_bin = bool(bin_edges[bi + 1] >= thresh_val)
        bins_data.append({
            "bin_idx": bi,
            "range": [b_start, b_end],
            "benign_count": c_b,
            "attack_count": c_a,
            "total_count": c_b + c_a,
            "pred_anomaly_count": c_p,
            "is_anomaly_region": is_anom_bin,
        })

    score_histogram = {
        "frozen_threshold": round(thresh_val, 6),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "hist_min": round(hist_min, 4),
        "hist_max": round(hist_max, 4),
        "bin_edges": [round(float(e), 2) for e in bin_edges],
        "bins": bins_data,
        "max_bin_count": int(max([b["total_count"] for b in bins_data] + [1])),
        "benign_total": int(len(benign_scores)),
        "attack_total": int(len(attack_scores)),
        "predicted_anomalies_total": int(np.sum(preds == 1)),
    }


    return {
        "status": "success",
        "snapshot_index": snap_idx,
        "window_id": int(getattr(snapshot, "window_id", snap_idx)),
        "split_name": split_name,
        "split_desc": split_desc,
        "frozen_threshold": round(float(detector.threshold), 6),
        "inference_time_ms": round(infer_time * 1000, 2),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        },
        "counts": {
            "total_flows": num_edges,
            "benign_total": benign_total,
            "attack_total": attack_total,
            "detected_anomalies": detected_total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "score_histogram": score_histogram,
        "pca_visualization": pca_visualization,
        "flows_table": flows_table,
    }


def _execute_gtae_forward(snapshot, snap_idx: int = DEFAULT_SNAPSHOT_INDEX):
    """Execute real forward pass of clean GTAE model on a snapshot and score with OCSVM."""
    model, scaler = load_model_and_scaler()
    device = get_device()

    # Scale features using clean training scaler
    snap_scaled = scaler.transform_snapshot(snapshot)

    x_dev = snap_scaled.x.to(device)
    edge_index_dev = snap_scaled.edge_index.to(device)
    edge_attr_dev = snap_scaled.edge_attr.to(device)

    with torch.no_grad():
        outputs = model(x_dev, edge_index_dev, edge_attr_dev)
        anomaly_feats = GTAEModel.extract_anomaly_features(
            outputs["edge_latent"], edge_attr_dev, outputs["edge_recon"]
        )

    node_latent = outputs["node_latent"]
    edge_latent = outputs["edge_latent"]
    node_recon = outputs["node_recon"]
    edge_recon = outputs["edge_recon"]
    scalar_recon_error = anomaly_feats["scalar_recon_error"]
    flow_features_113d = anomaly_feats["flow_features_for_detectors"]

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    param_count = sum(p.numel() for p in model.parameters())

    err_np = scalar_recon_error.cpu().numpy()
    mean_err = float(err_np.mean())
    median_err = float(np.median(err_np))
    max_err = float(err_np.max())

    hist_counts, bin_edges = np.histogram(err_np, bins=25)

    num_latent_sample = min(25, edge_latent.shape[0])
    latent_sample = np.round(edge_latent[:num_latent_sample].cpu().numpy(), 4).tolist()

    num_recon_sample = min(20, edge_attr_dev.shape[0])
    orig_recon_sample = np.round(edge_attr_dev[:num_recon_sample].cpu().numpy(), 4).tolist()
    pred_recon_sample = np.round(edge_recon[:num_recon_sample].cpu().numpy(), 4).tolist()

    flow_idx = 0
    single_113 = np.round(flow_features_113d[flow_idx].cpu().numpy(), 4).tolist()

    # Run OCSVM on this snapshot
    ocsvm_snapshot_data = _execute_ocsvm_on_snapshot(
        snapshot, snap_idx, flow_features_113d, scalar_recon_error
    )

    # Run Isolation Forest on this snapshot
    iforest_snapshot_data = _execute_iforest_on_snapshot(
        snapshot, snap_idx, flow_features_113d, scalar_recon_error
    )

    # Run HBOS on this snapshot
    hbos_snapshot_data = _execute_hbos_on_snapshot(
        snapshot, snap_idx, flow_features_113d, scalar_recon_error
    )

    # Run INNE on this snapshot
    inne_snapshot_data = _execute_inne_on_snapshot(
        snapshot, snap_idx, flow_features_113d, scalar_recon_error
    )

    return {
        "status": "success",
        "snapshot_index": snap_idx,
        "device": str(device),
        "gpu_name": gpu_name,
        "param_count": param_count,
        "architecture": {
            "in_node_dim": 16,
            "in_edge_dim": 81,
            "hidden_dim": 64,
            "latent_dim": 32,
        },
        "output_shapes": {
            "node_latent": list(node_latent.shape),
            "edge_latent": list(edge_latent.shape),
            "node_recon": list(node_recon.shape),
            "edge_recon": list(edge_recon.shape),
            "flow_features_113d": list(flow_features_113d.shape),
        },
        "latent_representation": {
            "num_flows": num_latent_sample,
            "latent_dim": 32,
            "data": latent_sample,
        },
        "reconstruction": {
            "num_flows": num_recon_sample,
            "feature_dim": 81,
            "original": orig_recon_sample,
            "reconstructed": pred_recon_sample,
        },
        "reconstruction_error": {
            "mean": mean_err,
            "median": median_err,
            "max": max_err,
            "histogram": {
                "counts": hist_counts.tolist(),
                "bin_edges": np.round(bin_edges, 4).tolist(),
            },
            "all_errors": np.round(err_np, 4).tolist(),
        },
        "feature_vector_113d": {
            "flow_index": flow_idx,
            "latent_dim": 32,
            "residual_dim": 81,
            "total_dim": 113,
            "latent_slice": single_113[:32],
            "residual_slice": single_113[32:],
            "full_113": single_113,
        },
        "ocsvm": ocsvm_snapshot_data,
        "iforest": iforest_snapshot_data,
        "hbos": hbos_snapshot_data,
        "inne": inne_snapshot_data,
    }



def _parse_index_param(default_val=DEFAULT_SNAPSHOT_INDEX):
    """Safely extract integer index parameter from GET query or POST JSON body."""
    raw_val = None
    if request.is_json and request.json and "index" in request.json:
        raw_val = request.json["index"]
    elif "index" in request.args:
        raw_val = request.args.get("index")

    if raw_val is None:
        return default_val

    try:
        idx = int(raw_val)
        return idx
    except (TypeError, ValueError):
        raise ValueError(f"Invalid snapshot index '{raw_val}'. Index must be an integer.")


@app.route("/")
def index():
    try:
        snapshots = load_all_snapshots()
        total_count = len(snapshots)
    except Exception as e:
        logger.error(f"Error reading snapshots list: {e}")
        total_count = 0

    return render_template(
        "review2_demo.html",
        available_snapshots=list(range(total_count)),
        default_snapshot_idx=DEFAULT_SNAPSHOT_INDEX,
        total_snapshots=total_count,
    )


@app.route("/api/snapshots", methods=["GET"])
def api_snapshots():
    """Return available snapshot indices and metadata summary."""
    try:
        snapshots = load_all_snapshots()
        return jsonify({
            "status": "success",
            "total_snapshots": len(snapshots),
            "default_index": DEFAULT_SNAPSHOT_INDEX,
            "indices": list(range(len(snapshots))),
        })
    except Exception as e:
        logger.error(f"Error in api_snapshots: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/graph", methods=["GET"])
def api_graph():
    """Return real snapshot graph structure with deterministic layout for the requested snapshot."""
    try:
        snap_idx = _parse_index_param()
        snapshot = get_snapshot(snap_idx)
        data = _extract_graph_data(snapshot, snap_idx)
        return jsonify(data)
    except (IndexError, ValueError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in api_graph: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/run_gtae", methods=["POST", "GET"])
def api_run_gtae():
    """Execute real forward pass of clean GTAE model on the requested snapshot and return OCSVM results."""
    try:
        snap_idx = _parse_index_param()
        snapshot = get_snapshot(snap_idx)
        data = _execute_gtae_forward(snapshot, snap_idx)
        data["snapshot_index"] = snap_idx
        return jsonify(data)
    except (IndexError, ValueError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in api_run_gtae: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/process_snapshot", methods=["POST", "GET"])
def api_process_snapshot():
    """Run full evaluation pipeline on the requested snapshot (graph extraction + GTAE + OCSVM detection)."""
    try:
        snap_idx = _parse_index_param()
        snapshot = get_snapshot(snap_idx)
        graph_data = _extract_graph_data(snapshot, snap_idx)
        gtae_data = _execute_gtae_forward(snapshot, snap_idx)
        gtae_data["snapshot_index"] = snap_idx
        ocsvm_data = gtae_data.get("ocsvm")
        iforest_data = gtae_data.get("iforest")
        hbos_data = gtae_data.get("hbos")
        inne_data = gtae_data.get("inne")

        return jsonify({
            "status": "success",
            "snapshot_index": snap_idx,
            "graph": graph_data,
            "gtae": gtae_data,
            "ocsvm": ocsvm_data,
            "iforest": iforest_data,
            "hbos": hbos_data,
            "inne": inne_data,
        })
    except (IndexError, ValueError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in api_process_snapshot: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route("/api/full_demo", methods=["POST", "GET"])
def api_full_demo():
    """Legacy alias for api_process_snapshot."""
    return api_process_snapshot()


@app.route("/api/ocsvm/dashboard_data", methods=["GET"])
def api_ocsvm_dashboard_data():
    """Return verified OCSVM evaluation results, metrics, and visualization curves."""
    global _CACHED_OCSVM_DATA
    try:
        if _CACHED_OCSVM_DATA is None:
            if not OCSVM_DASHBOARD_PATH.exists():
                return jsonify({
                    "status": "error",
                    "message": "OCSVM dashboard data not found. Please run scripts/run_ocsvm_pipeline.py."
                }), 404
            with open(OCSVM_DASHBOARD_PATH, "r", encoding="utf-8") as f:
                _CACHED_OCSVM_DATA = json.load(f)
        return jsonify(_CACHED_OCSVM_DATA)
    except Exception as e:
        logger.error(f"Error in api_ocsvm_dashboard_data: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/iforest/dashboard_data", methods=["GET"])
def api_iforest_dashboard_data():
    """Return verified Isolation Forest evaluation results and metrics."""
    global _CACHED_IFOREST_DATA
    try:
        if _CACHED_IFOREST_DATA is None:
            if not IFOREST_DASHBOARD_PATH.exists():
                return jsonify({
                    "status": "error",
                    "message": "Isolation Forest dashboard data not found. Please run scripts/train_isolation_forest.py."
                }), 404
            with open(IFOREST_DASHBOARD_PATH, "r", encoding="utf-8") as f:
                _CACHED_IFOREST_DATA = json.load(f)
        return jsonify(_CACHED_IFOREST_DATA)
    except Exception as e:
        logger.error(f"Error in api_iforest_dashboard_data: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/hbos/dashboard_data", methods=["GET"])
def api_hbos_dashboard_data():
    """Return verified HBOS evaluation results and metrics."""
    global _CACHED_HBOS_DATA
    try:
        if _CACHED_HBOS_DATA is None:
            if not HBOS_DASHBOARD_PATH.exists():
                return jsonify({
                    "status": "error",
                    "message": "HBOS dashboard data not found. Please run scripts/train_hbos.py."
                }), 404
            with open(HBOS_DASHBOARD_PATH, "r", encoding="utf-8") as f:
                _CACHED_HBOS_DATA = json.load(f)
        return jsonify(_CACHED_HBOS_DATA)
    except Exception as e:
        logger.error(f"Error in api_hbos_dashboard_data: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/inne/dashboard_data", methods=["GET"])
def api_inne_dashboard_data():
    """Return verified INNE evaluation results and metrics."""
    global _CACHED_INNE_DATA
    try:
        if _CACHED_INNE_DATA is None:
            if not INNE_DASHBOARD_PATH.exists():
                return jsonify({
                    "status": "error",
                    "message": "INNE dashboard data not found."
                }), 404
            with open(INNE_DASHBOARD_PATH, "r", encoding="utf-8") as f:
                _CACHED_INNE_DATA = json.load(f)
        return jsonify(_CACHED_INNE_DATA)
    except Exception as e:
        logger.error(f"Error in api_inne_dashboard_data: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == "__main__":

    logger.info("Starting GTAE-IDS Web Application on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
