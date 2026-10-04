"""Training and evaluation pipeline for Isolation Forest anomaly detector in GTAE-IDS."""

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
from sklearn.decomposition import PCA

from src.detection.config import IsolationForestConfig
from src.detection.dataset import DownstreamFeatureDataset
from src.detection.evaluator import evaluate_detector
from src.detection.models.isolation_forest import IsolationForestDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainIsolationForest")


def main():
    logger.info("=== Starting GTAE-IDS Phase 4B: Isolation Forest Training & Evaluation ===")

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

    # 2. Fit 2-Component PCA on Benign Training Features Only
    output_dir = Path("data/processed/models/isolation_forest")
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_path = output_dir / "iforest_pca2d.joblib"

    logger.info("Fitting 2-component PCA on benign training features for 2D visualization...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)

    var_ratio = pca.explained_variance_ratio_ * 100
    logger.info(
        f"Fitted 2D PCA successfully: PC1 explains {var_ratio[0]:.2f}% variance, "
        f"PC2 explains {var_ratio[1]:.2f}% variance (Total: {sum(var_ratio):.2f}%)."
    )
    joblib.dump(pca, pca_path)
    logger.info(f"Saved fitted PCA model to {pca_path}")

    # 3. Train Isolation Forest
    config = IsolationForestConfig(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
        output_dir=output_dir,
    )
    detector = IsolationForestDetector(config=config)

    t0 = time.time()
    detector.fit(X_train)
    train_time = time.time() - t0
    logger.info(f"Isolation Forest training completed in {train_time:.2f} seconds.")

    # 4. Calibrate Operational Threshold on Benign Validation Flows
    logger.info("Calibrating operational decision threshold on benign validation flows...")
    frozen_threshold = detector.calibrate_threshold(X_val, y_val, percentile=99.0)
    logger.info(f"Frozen Operational Decision Threshold tau* = {frozen_threshold:.6f}")

    # 5. Evaluate Validation Calibration Split
    logger.info("Evaluating on Validation Calibration Split (17,331 flows)...")
    val_preds, val_scores, val_infer_time = detector.predict(X_val)
    val_eval = evaluate_detector(
        y_true=y_val,
        scores=val_scores,
        threshold=frozen_threshold,
        inference_time=val_infer_time,
        training_time=train_time,
    )

    # 6. Evaluate Held-Out Test Split (Using Frozen Threshold)
    logger.info("Evaluating on Held-Out Test Split (22,462 flows) using frozen threshold...")
    test_preds, test_scores, test_infer_time = detector.predict(X_test)
    test_eval = evaluate_detector(
        y_true=y_test,
        scores=test_scores,
        threshold=frozen_threshold,
        inference_time=test_infer_time,
        training_time=train_time,
    )

    # 7. Print Detailed Benchmark Results
    print("\n" + "=" * 80)
    print("GTAE-IDS PHASE 4B: ISOLATION FOREST BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model Configuration: n_estimators={config.n_estimators}, contamination={config.contamination}, random_state={config.random_state}")
    print(f"Training Samples:    {len(X_train):,} Benign Flows (113 GTAE features, no scaler)")
    print(f"Training Time:       {train_time:.2f} seconds")
    print(f"Frozen Threshold:    tau* = {frozen_threshold:.6f} (99th percentile benign val)")
    print(f"PCA Variance (2D):   PC1 = {var_ratio[0]:.2f}%, PC2 = {var_ratio[1]:.2f}% (Total = {sum(var_ratio):.2f}%)")
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

    # 8. Save Model and Evaluation Artifacts
    detector.save(output_dir)

    metrics_payload = {
        "model_name": "IsolationForest",
        "hyperparameters": {
            "n_estimators": config.n_estimators,
            "max_samples": str(config.max_samples),
            "contamination": str(config.contamination),
            "random_state": config.random_state,
        },
        "training": {
            "num_samples": len(X_train),
            "feature_dim": X_train.shape[1],
            "training_time_seconds": round(train_time, 4),
        },
        "threshold": {
            "value": round(float(frozen_threshold), 6),
            "percentile": config.threshold_percentile,
            "calibrated_on": "Benign validation flows (y_val == 0)",
        },
        "pca_variance": {
            "pc1": round(float(var_ratio[0]), 2),
            "pc2": round(float(var_ratio[1]), 2),
            "total": round(float(sum(var_ratio)), 2),
        },
        "validation_results": val_eval,
        "test_results": test_eval,
    }

    metrics_path = output_dir / "iforest_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved full evaluation metrics to {metrics_path}")

    # 9. Save Dashboard Integration Payload
    dashboard_payload = {
        "model_name": "Isolation Forest",
        "frozen_threshold": round(float(frozen_threshold), 6),
        "pca_variance": [round(float(var_ratio[0]), 2), round(float(var_ratio[1]), 2)],
        "validation": {
            "metrics": val_eval["metrics"],
            "counts": val_eval["counts"],
            "confusion_matrix": val_eval["confusion_matrix"],
            "timing": val_eval["timing"],
        },
        "test": {
            "metrics": test_eval["metrics"],
            "counts": test_eval["counts"],
            "confusion_matrix": test_eval["confusion_matrix"],
            "timing": test_eval["timing"],
        },
    }
    dashboard_path = output_dir / "iforest_dashboard_data.json"
    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_payload, f, indent=2)
    logger.info(f"Saved dashboard integration data to {dashboard_path}")

    logger.info("=== Isolation Forest Training & Evaluation Completed Successfully ===")


if __name__ == "__main__":
    main()
