"""Complete training, calibration, evaluation, and visualization pipeline for OCSVM."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.detection.config import OCSVMConfig
from src.detection.dataset import DownstreamFeatureDataset
from src.detection.evaluator import evaluate_detector
from src.detection.models.ocsvm import OCSVMDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ocsvm_pipeline(
    config: OCSVMConfig = OCSVMConfig(),
) -> Dict[str, Any]:
    """Execute complete end-to-end OCSVM workflow."""
    print("=" * 75)
    print("GTAE-IDS PHASE 4A: ONE-CLASS SVM (OCSVM) DOWNSTREAM PIPELINE")
    print("=" * 75)
    print(f"Artifacts Directory: {config.artifacts_dir}")
    print(f"Model Output Directory: {config.output_dir}")
    print(f"Kernel: {config.kernel}, Nu: {config.nu}, Gamma: {config.gamma}")
    print(f"Threshold Policy: {config.threshold_percentile}th percentile of benign validation flows")
    print("=" * 75)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits via standardized DownstreamFeatureDataset
    logger.info("Loading feature artifacts via DownstreamFeatureDataset...")
    dataset = DownstreamFeatureDataset(artifacts_dir=config.artifacts_dir)

    X_tr, y_tr = dataset.get_train_data(as_numpy=True)
    X_val, y_val = dataset.get_val_data(as_numpy=True)
    X_test, y_test = dataset.get_test_data(as_numpy=True)

    meta_val = dataset.get_metadata("val", as_numpy=True)
    meta_test = dataset.get_metadata("test", as_numpy=True)

    # Protocol Assertions
    assert X_tr.shape == (57305, 113), f"Expected (57305, 113), got {X_tr.shape}"
    assert (y_tr == 0).all(), "Training data contains attack flows! Must be strictly benign."
    assert X_val.shape[1] == 113 and X_test.shape[1] == 113, "Feature dimension must be 113."
    assert len(X_val) == 17331 and len(X_test) == 22462, "Unexpected split size."

    logger.info(
        f"Verified dataset integrity:\n"
        f"  Train: {len(X_tr):,} flows (100.0% benign, strictly y=0)\n"
        f"  Val:   {len(X_val):,} flows ({(y_val==0).sum():,} benign, {(y_val==1).sum():,} attack)\n"
        f"  Test:  {len(X_test):,} flows ({(y_test==0).sum():,} benign, {(y_test==1).sum():,} attack)"
    )

    # 2. Instantiate and Fit OCSVM (Train only)
    logger.info("--- Fitting OCSVM on Benign Training Flows (57,305 samples) ---")
    detector = OCSVMDetector(config=config)
    detector.fit(X_tr)

    # 3. Score Validation and Calibrate Frozen Decision Threshold
    logger.info("--- Scoring Validation Split & Calibrating Decision Threshold ---")
    val_scores, val_infer_time = detector.score(X_val)

    frozen_threshold = detector.calibrate_threshold(
        X_val, y_val, percentile=config.threshold_percentile
    )
    logger.info(f"Frozen Decision Threshold: {frozen_threshold:.6f}")

    # 4. Evaluate Validation Split
    logger.info("--- Evaluating Validation Split ---")
    val_results = evaluate_detector(
        y_true=y_val,
        scores=val_scores,
        threshold=frozen_threshold,
        inference_time=val_infer_time,
        training_time=detector.fit_time_seconds,
    )
    val_m = val_results["metrics"]
    logger.info(
        f"Validation Metrics: ROC-AUC={val_m['roc_auc']:.4f}, PR-AUC={val_m['pr_auc']:.4f}, "
        f"F1={val_m['f1_score']:.4f}, Prec={val_m['precision']:.4f}, Rec={val_m['recall']:.4f}, FPR={val_m['false_positive_rate']:.4f}"
    )

    # 5. Score and Evaluate Test Split (Using Frozen Threshold)
    logger.info("--- Evaluating Test Split with Frozen Threshold ---")
    test_scores, test_infer_time = detector.score(X_test)

    test_results = evaluate_detector(
        y_true=y_test,
        scores=test_scores,
        threshold=frozen_threshold,
        inference_time=test_infer_time,
        training_time=detector.fit_time_seconds,
    )
    test_m = test_results["metrics"]
    logger.info(
        f"Test Metrics: ROC-AUC={test_m['roc_auc']:.4f}, PR-AUC={test_m['pr_auc']:.4f}, "
        f"F1={test_m['f1_score']:.4f}, Prec={test_m['precision']:.4f}, Rec={test_m['recall']:.4f}, FPR={test_m['false_positive_rate']:.4f}"
    )

    # 6. Fit 2D PCA Projection on Training Data Only
    logger.info("--- Fitting 2D PCA Projection (Training Data Only) ---")
    pca = PCA(n_components=2, random_state=config.random_state)
    # Fit PCA strictly on scaled training data
    X_tr_scaled = detector.scaler.transform(X_tr)
    pca.fit(X_tr_scaled)
    explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    logger.info(f"PCA explained variance ratio (PC1, PC2): {explained_var}")

    # Project a representative stratified sample from Test Set for web visualization
    # (e.g. 600 benign, 600 attack to keep visualization responsive and clear)
    np.random.seed(config.random_state)
    benign_test_indices = np.where(y_test == 0)[0]
    attack_test_indices = np.where(y_test == 1)[0]

    sample_benign_idx = np.random.choice(benign_test_indices, size=min(600, len(benign_test_indices)), replace=False)
    sample_attack_idx = np.random.choice(attack_test_indices, size=min(600, len(attack_test_indices)), replace=False)
    selected_indices = np.concatenate([sample_benign_idx, sample_attack_idx])

    X_test_sample_scaled = detector.scaler.transform(X_test[selected_indices])
    pca_2d_coords = pca.transform(X_test_sample_scaled)

    pca_plot_data = []
    for idx, (coord, orig_idx) in enumerate(zip(pca_2d_coords, selected_indices)):
        pca_plot_data.append({
            "pc1": round(float(coord[0]), 3),
            "pc2": round(float(coord[1]), 3),
            "ground_truth": int(y_test[orig_idx]),
            "score": round(float(test_scores[orig_idx]), 4),
            "predicted_anomaly": int(test_scores[orig_idx] >= frozen_threshold),
            "flow_idx": int(orig_idx),
            "window_id": int(meta_test["window_ids"][orig_idx]),
        })

    # 7. Compute Score Distribution Histograms (Separate for Benign vs Attack)
    def compute_histograms(scores, y, n_bins=30):
        b_scores = scores[y == 0]
        a_scores = scores[y == 1]
        all_min = float(min(scores.min(), frozen_threshold - 0.2))
        all_max = float(max(scores.max(), frozen_threshold + 0.2))
        bins = np.linspace(all_min, all_max, n_bins + 1)
        b_counts, _ = np.histogram(b_scores, bins=bins)
        a_counts, _ = np.histogram(a_scores, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        return {
            "bin_centers": np.round(bin_centers, 4).tolist(),
            "benign_counts": b_counts.tolist(),
            "attack_counts": a_counts.tolist(),
            "threshold": round(float(frozen_threshold), 4),
        }

    val_hist = compute_histograms(val_scores, y_val)
    test_hist = compute_histograms(test_scores, y_test)

    # 8. Extract Table of Detected Anomalous Flows (From Test Set)
    test_preds = (test_scores >= frozen_threshold).astype(np.int8)
    detected_indices = np.where(test_preds == 1)[0]

    # Sort detected flows by anomaly score descending (highest severity first)
    sorted_det_indices = detected_indices[np.argsort(-test_scores[detected_indices])]

    import datetime
    detected_flows_table = []
    for rank, orig_idx in enumerate(sorted_det_indices[:100]):  # Top 100 detected flows
        raw_ts = int(meta_test["edge_timestamps"][orig_idx])
        dt = datetime.datetime.fromtimestamp(raw_ts / 1e6, tz=datetime.timezone.utc)
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        detected_flows_table.append({
            "rank": rank + 1,
            "flow_id": int(orig_idx),
            "window_id": int(meta_test["window_ids"][orig_idx]),
            "timestamp": ts_str,
            "raw_timestamp": raw_ts,
            "src_node": int(meta_test["src_nodes"][orig_idx]),
            "dst_node": int(meta_test["dst_nodes"][orig_idx]),
            "scalar_recon_error": round(float(meta_test["scalar_recon_error"][orig_idx]), 4),
            "anomaly_score": round(float(test_scores[orig_idx]), 4),
            "ground_truth": int(y_test[orig_idx]),
            "ground_truth_label": "Attack" if y_test[orig_idx] == 1 else "Benign (FP)",
            "is_true_positive": bool(y_test[orig_idx] == 1),
        })

    # 9. Save Artifacts & Models
    logger.info("--- Saving Outputs and Models ---")
    detector.save(config.output_dir)

    # Save PCA model
    pca_path = config.output_dir / "ocsvm_pca2d.joblib"
    joblib.dump(pca, pca_path)

    # Save full scores and predictions tensor file
    scores_preds_path = config.output_dir / "ocsvm_scores_predictions.pt"
    torch.save({
        "val_scores": torch.tensor(val_scores, dtype=torch.float32),
        "val_predictions": torch.tensor(test_preds, dtype=torch.int8),
        "val_y": torch.tensor(y_val, dtype=torch.int8),
        "test_scores": torch.tensor(test_scores, dtype=torch.float32),
        "test_predictions": torch.tensor(test_preds, dtype=torch.int8),
        "test_y": torch.tensor(y_test, dtype=torch.int8),
        "threshold": float(frozen_threshold),
    }, scores_preds_path)

    # Prepare comprehensive dashboard data JSON
    dashboard_data = {
        "status": "success",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": "One-Class Support Vector Machine (OCSVM)",
        "configuration": {
            "kernel": config.kernel,
            "nu": config.nu,
            "gamma": config.gamma,
            "cache_size_mb": config.cache_size,
            "threshold_percentile": config.threshold_percentile,
            "frozen_threshold": round(float(frozen_threshold), 6),
            "train_sample_count": len(X_tr),
            "feature_dim": 113,
            "scaler": "StandardScaler (fitted on train only)",
        },
        "training_stats": {
            "fit_time_seconds": round(detector.fit_time_seconds, 2),
            "num_support_vectors": detector.num_support_vectors,
            "support_vector_ratio": round(
                detector.num_support_vectors / len(X_tr) * 100, 2
            ),
        },
        "validation": val_results,
        "test": test_results,
        "pca_projection": {
            "explained_variance": explained_var,
            "points": pca_plot_data,
            "disclaimer": "2D PCA Projection fitted strictly on training data for visualization purposes only. Model evaluates in full 113-D space.",
        },
        "histograms": {
            "val": val_hist,
            "test": test_hist,
        },
        "detected_flows": detected_flows_table,
    }

    dash_json_path = config.output_dir / "ocsvm_dashboard_data.json"
    with open(dash_json_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2)

    logger.info(f"Saved dashboard data to: {dash_json_path}")
    logger.info(f"Saved scores/predictions to: {scores_preds_path}")

    print("\n" + "=" * 75)
    print("OCSVM PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Training Time:     {detector.fit_time_seconds:.2f}s ({detector.num_support_vectors:,} SVs)")
    print(f"Decision Threshold: {frozen_threshold:.6f} (99th pct benign val)")
    print(f"Validation F1:     {val_m['f1_score']:.4f} | ROC-AUC: {val_m['roc_auc']:.4f} | PR-AUC: {val_m['pr_auc']:.4f}")
    print(f"Test F1:           {test_m['f1_score']:.4f} | ROC-AUC: {test_m['roc_auc']:.4f} | PR-AUC: {test_m['pr_auc']:.4f}")
    print(f"Test Precision:    {test_m['precision']:.4f} | Recall: {test_m['recall']:.4f} | FPR: {test_m['false_positive_rate']:.4f}")
    print(f"Test Detected:     {test_results['counts']['detected_anomalies']:,} flows "
          f"({test_results['counts']['true_positives']:,} TP, {test_results['counts']['false_positives']:,} FP)")
    print("=" * 75)

    return dashboard_data


if __name__ == "__main__":
    run_ocsvm_pipeline()
