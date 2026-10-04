"""Verification of retrained clean GTAE model artifacts and downstream representations."""

import json
from pathlib import Path
import torch

def verify():
    clean_model_path = Path("data/processed/models/best_gtae_model_clean.pt")
    orig_model_path = Path("data/processed/models/best_gtae_model.pt")
    history_path = Path("data/processed/models/gtae_training_history_clean.json")
    config_path = Path("data/processed/models/gtae_training_config_clean.json")
    scalers_path = Path("data/processed/scalers/gtae_scalers_clean.pkl")
    artifacts_path = Path("data/processed/artifacts/sample_anomaly_features_clean.pt")

    print("=" * 65)
    print("AUDIT & VERIFICATION OF CLEAN RETRAINED GTAE")
    print("=" * 65)

    # 1. Existence and independence of checkpoints
    assert clean_model_path.exists(), "Clean checkpoint missing!"
    assert orig_model_path.exists(), "Original baseline checkpoint was deleted or overwritten!"
    clean_ckpt = torch.load(clean_model_path, map_location="cpu", weights_only=False)
    orig_ckpt = torch.load(orig_model_path, map_location="cpu", weights_only=False)
    
    clean_size = clean_model_path.stat().st_size
    orig_size = orig_model_path.stat().st_size
    print(f"Original Checkpoint: {orig_model_path} ({orig_size:,} bytes, Epoch {orig_ckpt['epoch']}, Val Loss: {orig_ckpt['val_loss']:.6f})")
    print(f"Clean Checkpoint:    {clean_model_path} ({clean_size:,} bytes, Epoch {clean_ckpt['epoch']}, Val Loss: {clean_ckpt['val_loss']:.6f})")

    # Verify weights are strictly finite
    for k, v in clean_ckpt["model_state_dict"].items():
        assert torch.isfinite(v).all(), f"Non-finite weight in {k}!"
    print("All clean model weights strictly finite: True")

    # 2. History & Metrics
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    print(f"\nTraining History ({len(history)} epochs):")
    print(f"  Initial Epoch 1:  Train Loss = {history[0]['train_loss']:.6f}, Val Loss = {history[0]['val_loss']:.6f}")
    print(f"  Final Epoch {len(history)}: Train Loss = {history[-1]['train_loss']:.6f}, Val Loss = {history[-1]['val_loss']:.6f}")
    print(f"  Best Epoch:       {clean_ckpt['epoch']} (Val Loss = {clean_ckpt['val_loss']:.6f})")

    # 3. Config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"\nModel Configuration:")
    print(f"  Parameters:        {cfg['total_parameters']:,}")
    print(f"  Training Time:     {cfg['total_training_time_seconds']}s")
    print(f"  Latent Dimension:  {cfg['model_config']['latent_dim']}")
    print(f"  Edge Input Dim:    {cfg['model_config']['in_edge_dim']}")
    print(f"  Node Input Dim:    {cfg['model_config']['in_node_dim']}")

    # 4. Diagnostic Downstream Anomaly Representations
    diag = torch.load(artifacts_path, map_location="cpu", weights_only=False)
    print(f"\nDiagnostic Anomaly Representations ({artifacts_path}):")
    for split_name in ["train", "val", "test"]:
        item = diag[split_name]
        f_e = item["flow_features_113d"]
        err = item["scalar_recon_error"]
        print(f"  {split_name.upper():<5}: shape {list(f_e.shape)}, mean err = {err.mean():.6f}, max err = {err.max():.6f}")
        assert f_e.shape[1] == 113, f"Expected 113 dims, got {f_e.shape[1]}"
        assert torch.isfinite(f_e).all(), f"Non-finite anomaly features in {split_name}!"
        assert torch.isfinite(err).all(), f"Non-finite errors in {split_name}!"

    print("\n" + "=" * 65)
    print("ALL POST-TRAINING VERIFICATION CHECKS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    verify()
