"""Training, calibration, and evaluation pipeline for Histogram-Based Outlier Score (HBOS) in GTAE-IDS."""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch

from src.detection.config import HBOSConfig
from src.detection.dataset import DownstreamFeatureDataset
from src.detection.evaluator import evaluate_detector
from src.detection.models.hbos import HBOSDetector, predict_hbos_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainHBOS")


def main():
    logger.info("=== Starting GTAE-IDS Phase 4C: HBOS Training & Evaluation ===")

    # 1. Load Feature Datasets
    logger.info("Loading downstream feature artifacts...")
    ds = DownstreamFeatureDataset()

    X_train, y_train = ds.get_train_data()
    X_val, y_val = ds.get_val_data()
    X_test, y_test = ds.get_test_data()

    # Protocol Verification Checks
    assert X_train.shape[0] == 57305, f"Expected 57,305 train samples, got {X_train.shape[0]}"
    assert X_train.shape[1] == 113, f"Expected 113 feature dimensions, got {X_train.shape[1]}"
    assert np.all(y_train == 0), "Training split must strictly contain only benign flows (y == 0)"
    assert X_val.shape[0] == 17331, f"Expected 17,331 val samples, got {X_val.shape[0]}"
    assert X_test.shape[0] == 22462, f"Expected 22,462 test samples, got {X_test.shape[0]}"

    logger.info(
        f"Verified feature tensors: Train={X_train.shape} (100% benign), "
        f"Val={X_val.shape} ({np.sum(y_val == 1):,} attacks, {np.sum(y_val == 0):,} benign), "
        f"Test={X_test.shape} ({np.sum(y_test == 1):,} attacks, {np.sum(y_test == 0):,} benign)"
    )

    # 2. Train HBOS Detector strictly on benign training flows
    output_dir = Path("data/processed/models/hbos")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = HBOSConfig(
        n_bins=10,
        alpha=0.1,
        tol=0.5,
        contamination=0.1,
        threshold_percentile=99.0,
        output_dir=output_dir,
    )
    detector = HBOSDetector(config=config)

    t0 = time.time()
    detector.fit(X_train)
    train_time = time.time() - t0
    logger.info(f"HBOS training completed in {train_time:.4f} seconds.")

    # 3. Calibrate Operational Decision Threshold on Benign Validation Flows
    logger.info("Calibrating operational decision threshold on benign validation flows...")
    frozen_threshold = detector.calibrate_threshold(X_val, y_val, percentile=99.0)
    logger.info(f"Frozen Operational Decision Threshold tau* = {frozen_threshold:.6f}")

    # 4. Evaluate Validation Calibration Split
    logger.info("Evaluating on Validation Calibration Split (17,331 flows)...")
    val_preds, val_scores, val_infer_time = detector.predict(X_val)
    val_eval = evaluate_detector(
        y_true=y_val,
        scores=val_scores,
        threshold=frozen_threshold,
        inference_time=val_infer_time,
        training_time=train_time,
    )

    # 5. Evaluate Held-Out Test Split (Using Frozen Threshold)
    logger.info("Evaluating on Held-Out Test Split (22,462 flows) using frozen threshold...")
    test_preds, test_scores, test_infer_time = detector.predict(X_test)
    test_eval = evaluate_detector(
        y_true=y_test,
        scores=test_scores,
        threshold=frozen_threshold,
        inference_time=test_infer_time,
        training_time=train_time,
    )

    # 6. Print Detailed Benchmark Results
    print("\n" + "=" * 80)
    print("GTAE-IDS PHASE 4C: HISTOGRAM-BASED OUTLIER SCORE (HBOS) BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model Configuration: n_bins={config.n_bins}, alpha={config.alpha}, tol={config.tol}, contamination={config.contamination}")
    print(f"Training Samples:    {len(X_train):,} Benign Flows (113 GTAE features)")
    print(f"Training Time:       {train_time:.4f} seconds")
    print(f"Frozen Threshold:    tau* = {frozen_threshold:.6f} (99th percentile benign val)")
    print("-" * 80)
    print("METRIC                        VALIDATION SET (17,331 flows)   TEST SET (22,462 flows)")
    print("-" * 80)
    print(f"Precision:                    {val_eval['metrics']['precision'] * 100:6.2f}%                       {test_eval['metrics']['precision'] * 100:6.2f}%")
    print(f"Recall (Detection Rate):      {val_eval['metrics']['recall'] * 100:6.2f}%                       {test_eval['metrics']['recall'] * 100:6.2f}%")
    print(f"F1-Score:                     {val_eval['metrics']['f1_score']:6.4f}                        {test_eval['metrics']['f1_score']:6.4f}")
    print(f"False Positive Rate (FPR):    {val_eval['metrics']['false_positive_rate'] * 100:6.2f}%                       {test_eval['metrics']['false_positive_rate'] * 100:6.2f}%")
    print(f"Accuracy:                     {val_eval['metrics']['accuracy'] * 100:6.2f}%                       {test_eval['metrics']['accuracy'] * 100:6.2f}%")
    print(f"ROC-AUC:                      {val_eval['metrics']['roc_auc']:6.4f}                        {test_eval['metrics']['roc_auc']:6.4f}")
    print(f"PR-AUC (Avg Precision):       {val_eval['metrics']['pr_auc']:6.4f}                        {test_eval['metrics']['pr_auc']:6.4f}")
    print(f"Inference Time:               {val_eval['timing']['inference_time_seconds']:6.4f} s                     {test_eval['timing']['inference_time_seconds']:6.4f} s")
    print(f"Throughput:                   {val_eval['timing']['flows_per_second']:6.1f} flows/s               {test_eval['timing']['flows_per_second']:6.1f} flows/s")
    print("-" * 80)
    print("DETECTION BREAKDOWN:")
    print(f"True Positives (Detected):    {val_eval['counts']['true_positives']:5d} / {val_eval['counts']['attack_total']} attacks ({val_eval['metrics']['recall'] * 100:.1f}%)        {test_eval['counts']['true_positives']:5d} / {test_eval['counts']['attack_total']} attacks ({test_eval['metrics']['recall'] * 100:.1f}%)")
    print(f"False Negatives (Missed):     {val_eval['counts']['false_negatives']:5d} / {val_eval['counts']['attack_total']} attacks             {test_eval['counts']['false_negatives']:5d} / {test_eval['counts']['attack_total']} attacks")
    print(f"False Positives (Alarms):     {val_eval['counts']['false_positives']:5d} / {val_eval['counts']['benign_total']} benign ({val_eval['metrics']['false_positive_rate'] * 100:.2f}%)        {test_eval['counts']['false_positives']:5d} / {test_eval['counts']['benign_total']} benign ({test_eval['metrics']['false_positive_rate'] * 100:.2f}%)")
    print(f"True Negatives (Correct):     {val_eval['counts']['true_negatives']:5d} / {val_eval['counts']['benign_total']} benign            {test_eval['counts']['true_negatives']:5d} / {test_eval['counts']['benign_total']} benign")
    print("=" * 80 + "\n")

    # 7. Save Model and Metadata
    detector.save(output_dir)

    metrics_payload = {
        "model_name": "HBOS",
        "library": "PyOD",
        "hyperparameters": {
            "n_bins": config.n_bins,
            "alpha": config.alpha,
            "tol": config.tol,
            "contamination": config.contamination,
        },
        "training": {
            "num_samples": len(X_train),
            "feature_dim": X_train.shape[1],
            "training_time_seconds": round(train_time, 6),
        },
        "threshold": {
            "value": round(float(frozen_threshold), 6),
            "percentile": config.threshold_percentile,
            "calibrated_on": "Benign validation flows (y_val == 0)",
        },
        "validation_results": val_eval,
        "test_results": test_eval,
    }

    metrics_path = output_dir / "hbos_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved full evaluation metrics to {metrics_path}")

    # 8. Save Tensor Predictions Artifact
    predictions_payload = {
        "val_scores": torch.from_numpy(val_scores),
        "val_preds": torch.from_numpy(val_preds),
        "val_y": torch.from_numpy(y_val),
        "test_scores": torch.from_numpy(test_scores),
        "test_preds": torch.from_numpy(test_preds),
        "test_y": torch.from_numpy(y_test),
        "frozen_threshold": float(frozen_threshold),
    }
    preds_path = output_dir / "hbos_predictions.pt"
    torch.save(predictions_payload, preds_path)
    logger.info(f"Saved prediction tensors to {preds_path}")

    # 9. Verify Reusable Snapshot Inference Function
    logger.info("Verifying reusable snapshot inference function on sample snapshots...")
    snapshots_path = PROJECT_ROOT / "data/processed/graphs/temporal_graph_snapshots.pt"
    if snapshots_path.exists():
        from src.models.gtae import GTAEModel
        from src.models.normalization import GTAEFeatureScaler

        snapshots = torch.load(snapshots_path, weights_only=False, map_location="cpu")
        scalers_obj = joblib.load(PROJECT_ROOT / "data/processed/scalers/gtae_scalers_clean.pkl")
        model_ckpt = torch.load(PROJECT_ROOT / "data/processed/models/best_gtae_model_clean.pt", weights_only=False, map_location="cpu")

        from src.models.config import GTAEConfig
        cfg = GTAEConfig(in_node_dim=16, in_edge_dim=81, hidden_dim=64, latent_dim=32)
        gtae = GTAEModel(cfg)
        gtae.load_state_dict(model_ckpt["model_state_dict"])
        gtae.eval()

        scaler = GTAEFeatureScaler.load(PROJECT_ROOT / "data/processed/scalers/gtae_scalers_clean.pkl")

        test_snap_indices = [46, 88, 208, 297]
        for s_idx in test_snap_indices:
            snap = snapshots[s_idx]
            snap_s = scaler.transform_snapshot(snap)
            with torch.no_grad():
                out = gtae(snap_s.x, snap_s.edge_index, snap_s.edge_attr)
                feats_dict = GTAEModel.extract_anomaly_features(out["edge_latent"], snap_s.edge_attr, out["edge_recon"])
                f_113d = feats_dict["flow_features_for_detectors"]

            snap_result = predict_hbos_snapshot(f_113d, model_dir=output_dir)
            total_edges = snap.edge_index.size(1)
            y_edges = snap.edge_y.cpu().numpy() if hasattr(snap, "edge_y") and snap.edge_y is not None else np.zeros(total_edges)
            attacks = int(np.sum(y_edges == 1))
            tp = int(np.sum((snap_result["predictions"] == 1) & (y_edges == 1)))
            fp = int(np.sum((snap_result["predictions"] == 1) & (y_edges == 0)))

            logger.info(
                f"Snapshot {s_idx:3d}: {total_edges:3d} flows ({attacks:2d} attacks) -> "
                f"HBOS detected {snap_result['detected_count']:2d} anomalies "
                f"({tp} TP, {fp} FP) in {snap_result['inference_time_seconds'] * 1000:.2f}ms"
            )

    logger.info("=== HBOS Training, Calibration, Evaluation & Verification Complete ===")


if __name__ == "__main__":
    main()
