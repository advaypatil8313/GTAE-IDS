"""Standalone verification suite for GTAE-IDS Phase 4A: OCSVM Downstream Detector."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch
from app import app
from src.detection.dataset import DownstreamFeatureDataset
from src.detection.models.ocsvm import OCSVMDetector


def verify_ocsvm_implementation():
    print("=" * 75)
    print("GTAE-IDS PHASE 4A: OCSVM VERIFICATION SUITE")
    print("=" * 75)

    models_dir = PROJECT_ROOT / "data" / "processed" / "models" / "ocsvm"
    model_path = models_dir / "ocsvm_model.joblib"
    scaler_path = models_dir / "ocsvm_scaler.joblib"
    pca_path = models_dir / "ocsvm_pca2d.joblib"
    meta_path = models_dir / "ocsvm_metadata.json"
    dash_json_path = models_dir / "ocsvm_dashboard_data.json"
    scores_preds_path = models_dir / "ocsvm_scores_predictions.pt"

    # 1. Verify existence of saved artifacts
    for p in [model_path, scaler_path, pca_path, meta_path, dash_json_path, scores_preds_path]:
        assert p.exists(), f"Missing required OCSVM artifact: {p}"
        print(f"Artifact exists: {p.name} ({p.stat().st_size:,} bytes)")
    print("CHECK 1: All required OCSVM model artifacts and outputs exist on disk.")

    # 2. Verify model configuration and metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_type"] == "OneClassSVM"
    assert meta["kernel"] == "rbf"
    assert meta["nu"] == 0.01
    assert meta["gamma"] == "scale"
    assert meta["num_train_samples"] == 57305, f"Expected 57,305 train samples, got {meta['num_train_samples']}"
    assert meta["num_support_vectors"] == 881
    print(f"CHECK 2: Model configuration verified (RBF, nu=0.01, N=57,305, SVs=881, fit={meta['fit_time_seconds']}s).")

    # 3. Verify feature dimensions and strict label isolation
    dataset = DownstreamFeatureDataset(artifacts_dir=PROJECT_ROOT / "data" / "processed" / "artifacts")
    X_tr, y_tr = dataset.get_train_data(as_numpy=True)
    X_val, y_val = dataset.get_val_data(as_numpy=True)
    X_test, y_test = dataset.get_test_data(as_numpy=True)

    assert X_tr.shape == (57305, 113)
    assert X_val.shape == (17331, 113)
    assert X_test.shape == (22462, 113)
    assert (y_tr == 0).all(), "Training labels contain attack flows!"
    print("CHECK 3: 113-D input dimensionality and 100% benign training set (57,305 samples) confirmed.")

    # 4. Verify scaler was fit on train only
    scaler = joblib.load(scaler_path)
    assert scaler.n_features_in_ == 113
    assert len(scaler.mean_) == 113
    # Check that scaler mean matches train mean, not val or test mean
    assert np.allclose(scaler.mean_, X_tr.astype(np.float64).mean(axis=0), atol=1e-7), "Scaler was NOT fit on training data!"
    assert not np.allclose(scaler.mean_, X_val.astype(np.float64).mean(axis=0), atol=1e-3), "Scaler leaked validation data!"
    print("CHECK 4: StandardScaler was strictly fit on training data only (zero leak).")


    # 5. Verify threshold calibration and freeze
    frozen_thresh = meta["threshold"]
    val_benign_mask = (y_val == 0)

    scores_data = torch.load(scores_preds_path, weights_only=False)
    val_scores = scores_data["val_scores"].numpy()
    test_scores = scores_data["test_scores"].numpy()

    calibrated_val_thresh = float(np.percentile(val_scores[val_benign_mask], 99.0))
    assert abs(frozen_thresh - calibrated_val_thresh) < 1e-5, "Threshold mismatch!"
    assert abs(scores_data["threshold"] - frozen_thresh) < 1e-5
    print(f"CHECK 5: Decision threshold calibrated strictly on benign val (tau* = {frozen_thresh:.6f}).")

    # 6. Verify test evaluation uses the exact frozen threshold
    test_preds = scores_data["test_predictions"].numpy()
    expected_test_preds = (test_scores >= frozen_thresh).astype(np.int8)
    assert np.array_equal(test_preds, expected_test_preds), "Test predictions did not use frozen threshold!"
    print(f"CHECK 6: Test evaluation strictly uses frozen threshold tau* = {frozen_thresh:.6f}.")

    # 7. Verify dashboard JSON integrity
    with open(dash_json_path, "r", encoding="utf-8") as f:
        dash_data = json.load(f)

    assert dash_data["status"] == "success"
    assert "validation" in dash_data and "test" in dash_data
    assert "pca_projection" in dash_data
    assert len(dash_data["pca_projection"]["points"]) == 1200
    assert "disclaimer" in dash_data["pca_projection"]
    assert len(dash_data["detected_flows"]) == 100
    print("CHECK 7: Dashboard JSON structure and real visualization coordinates verified.")

    # 8. Verify Flask application endpoints (existing demo + new OCSVM route)
    client = app.test_client()

    # Existing GTAE demo routes
    res_index = client.get("/")
    assert res_index.status_code == 200, f"Index route failed with {res_index.status_code}"

    res_snaps = client.get("/api/snapshots")
    assert res_snaps.status_code == 200, f"Snapshots route failed with {res_snaps.status_code}"

    res_graph = client.get("/api/graph?index=46")
    assert res_graph.status_code == 200, f"Graph route failed with {res_graph.status_code}"

    # New OCSVM route
    res_ocsvm = client.get("/api/ocsvm/dashboard_data")
    assert res_ocsvm.status_code == 200, f"OCSVM API failed with {res_ocsvm.status_code}"
    ocsvm_payload = res_ocsvm.get_json()
    assert ocsvm_payload["status"] == "success"
    print("CHECK 8: Flask routes verified. Existing GTAE demo and new OCSVM route both fully functional (HTTP 200).")

    print("\n" + "=" * 75)
    print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 75)


if __name__ == "__main__":
    verify_ocsvm_implementation()
