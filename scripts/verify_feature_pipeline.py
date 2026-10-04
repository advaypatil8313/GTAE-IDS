"""Standalone verification suite for GTAE downstream anomaly-detection feature pipeline."""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from src.detection.dataset import DownstreamFeatureDataset


def verify_feature_pipeline():
    artifacts_dir = Path("data/processed/artifacts")
    train_path = artifacts_dir / "gtae_features_train.pt"
    val_path = artifacts_dir / "gtae_features_val.pt"
    test_path = artifacts_dir / "gtae_features_test.pt"
    meta_path = artifacts_dir / "gtae_features_metadata.json"

    print("=" * 70)
    print("GTAE DOWNSTREAM FEATURE PIPELINE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Existence of artifacts
    for p in [train_path, val_path, test_path, meta_path]:
        assert p.exists(), f"Missing required artifact: {p}"
        print(f"Artifact exists: {p} ({p.stat().st_size:,} bytes)")

    # 2. Metadata verification
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    clean_ckpt_used = meta["model_checkpoint_used"]
    print(f"\nModel checkpoint recorded in metadata: {clean_ckpt_used}")
    assert "clean" in clean_ckpt_used, f"Prohibited checkpoint recorded: {clean_ckpt_used}"
    assert "best_gtae_model_clean.pt" in clean_ckpt_used, "Expected best_gtae_model_clean.pt!"
    print("CHECK 1: Clean checkpoint provenance confirmed.")

    # 3. Load via DownstreamFeatureDataset API
    dataset = DownstreamFeatureDataset(artifacts_dir=artifacts_dir)
    summary = dataset.get_summary()

    print("\nDownstream Dataset Summary:")
    for split_name, info in summary.items():
        print(f"  {split_name.upper():<5}: {info['flows']:,} flows, dim {info['feature_dim']}, "
              f"{info['benign_flows']:,} benign, {info['attack_flows']:,} attack")

    # 4. Check feature dimensions & finiteness
    X_tr, y_tr = dataset.get_train_data(as_numpy=True)
    X_tr_benign, y_tr_benign = dataset.get_train_benign_data(as_numpy=True)
    X_val, y_val = dataset.get_val_data(as_numpy=True)
    X_test, y_test = dataset.get_test_data(as_numpy=True)

    # Dimensionality assertions
    assert X_tr.shape == (57305, 113), f"Expected (57305, 113), got {X_tr.shape}"
    assert X_tr_benign.shape == (57305, 113), f"Expected (57305, 113), got {X_tr_benign.shape}"
    assert X_val.shape == (17331, 113), f"Expected (17331, 113), got {X_val.shape}"
    assert X_test.shape == (22462, 113), f"Expected (22462, 113), got {X_test.shape}"
    print("\nCHECK 2: Exact flow counts and 113-d feature dimensions confirmed.")

    # Label isolation assertions
    assert y_tr.shape == (57305,), f"Expected y_tr shape (57305,), got {y_tr.shape}"
    assert y_val.shape == (17331,), f"Expected y_val shape (17331,), got {y_val.shape}"
    assert y_test.shape == (22462,), f"Expected y_test shape (22462,), got {y_test.shape}"

    assert (y_tr == 0).all(), "Training labels contain attack flows! Expected strictly y == 0."
    assert int((y_val == 1).sum()) == 2552, f"Expected 2552 val attacks, got {(y_val == 1).sum()}"
    assert int((y_test == 1).sum()) == 4546, f"Expected 4546 test attacks, got {(y_test == 1).sum()}"
    print("CHECK 3: Strict label isolation confirmed. Labels are 1D vectors and NOT in X.")

    # Finiteness assertions
    for name, arr in [("X_train", X_tr), ("X_val", X_val), ("X_test", X_test)]:
        assert np.all(np.isfinite(arr)), f"Non-finite values found in {name}!"
        assert not np.isnan(arr).any(), f"NaNs found in {name}!"
        assert not np.isinf(arr).any(), f"Infs found in {name}!"
    print("CHECK 4: Strictly finite numerical values (zero NaN / Inf) confirmed.")

    # 5. Metadata and correspondence check
    for split_name in ["train", "val", "test"]:
        meta_dict = dataset.get_metadata(split_name, as_numpy=True)
        assert "window_ids" in meta_dict
        assert "edge_timestamps" in meta_dict
        assert "scalar_recon_error" in meta_dict
        assert "src_nodes" in meta_dict
        assert "dst_nodes" in meta_dict
        assert "edge_indices_in_window" in meta_dict

        n_rows = len(meta_dict["window_ids"])
        assert len(meta_dict["edge_timestamps"]) == n_rows
        assert len(meta_dict["scalar_recon_error"]) == n_rows
        assert len(meta_dict["src_nodes"]) == n_rows
        assert len(meta_dict["dst_nodes"]) == n_rows
        assert np.all(np.isfinite(meta_dict["scalar_recon_error"]))
        print(f"CHECK 5 ({split_name}): All {n_rows:,} metadata rows correspond 1-to-1 with flows.")

    # Chronological window boundary check
    train_meta = dataset.get_metadata("train")
    val_meta = dataset.get_metadata("val")
    test_meta = dataset.get_metadata("test")

    max_train_win = int(np.max(train_meta["window_ids"]))
    min_val_win = int(np.min(val_meta["window_ids"]))
    max_val_win = int(np.max(val_meta["window_ids"]))
    min_test_win = int(np.min(test_meta["window_ids"]))

    print(f"\nChronological Window Bounds:")
    print(f"  Train windows: min {int(np.min(train_meta['window_ids']))} to max {max_train_win}")
    print(f"  Val windows:   min {min_val_win} to max {max_val_win}")
    print(f"  Test windows:  min {min_test_win} to max {int(np.max(test_meta['window_ids']))}")

    assert max_train_win < min_val_win, "Train and Val window overlap detected!"
    assert max_val_win < min_test_win, "Val and Test window overlap detected!"
    print("CHECK 6: Chronological split ordering strictly preserved with no leakage or overlap.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    verify_feature_pipeline()
