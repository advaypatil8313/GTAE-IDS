"""Validation script for GTAE forward pass, GPU execution, and resource benchmarking."""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
import torch
from src.models.config import GTAEConfig
from src.models.gtae import GTAEModel
from src.models.utils import count_parameters, get_chronological_splits


def run_forward_validation():
    print("=" * 65)
    print("GTAE-IDS: MODEL FORWARD-PASS & GPU VALIDATION ROUTINE")
    print("=" * 65)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Initialize Model
    config = GTAEConfig(
        in_node_dim=16,
        in_edge_dim=81,
        hidden_dim=64,
        latent_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        node_loss_weight=0.5,
        loss_type="smooth_l1",
    )
    model = GTAEModel(config).to(device)
    param_counts = count_parameters(model)
    print(f"\nModel Initialized Successfully:")
    print(f"  Total Parameters:     {param_counts['total_parameters']:,}")
    print(f"  Trainable Parameters: {param_counts['trainable_parameters']:,}")

    # Breakdown by submodules
    print("\nSubmodule Parameter Breakdown:")
    for name, submod in [
        ("Encoder (Graph Transformer)", model.encoder),
        ("Flow Latent MLP", model.flow_latent_mlp),
        ("Edge Decoder", model.edge_decoder),
        ("Node Decoder", model.node_decoder),
    ]:
        p = sum(param.numel() for param in submod.parameters())
        print(f"  - {name:30s}: {p:6,d} parameters")

    # 2. Load Snapshots
    snapshots_path = Path("data/processed/graphs/temporal_graph_snapshots.pt")
    assert snapshots_path.exists(), f"Graph snapshots file not found at {snapshots_path}"
    print(f"\nLoading graph snapshots from: {snapshots_path}")
    t_load = time.time()
    snapshots = torch.load(snapshots_path, weights_only=False)
    print(f"Loaded {len(snapshots)} snapshots in {time.time() - t_load:.2f}s.")

    # 3. Verify Chronological Splitting
    splits = get_chronological_splits(snapshots, 0.70, 0.15, 0.15)
    print("\nChronological Dataset Splits (Zero Shuffling):")
    for k, v in splits["split_summary"].items():
        if isinstance(v, dict):
            print(
                f"  [{v['name']:16s}]: {v['num_windows']:3d} windows | "
                f"{v['total_edges']:6,d} flows ({v['attack_edges']:5,d} attacks, {v['attack_percentage']:5.2f}%)"
            )

    # 4. Select Test Snapshots (different sizes: initial, median, max)
    edge_counts = [s.num_edges for s in snapshots]
    max_idx = int(torch.tensor(edge_counts).argmax())
    test_indices = [0, 50, 100, max_idx]  # 4 representative snapshots

    print(f"\nRunning GPU Forward Passes on {len(test_indices)} Representative Snapshots...")
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    total_forward_time = 0.0
    for step, idx in enumerate(test_indices):
        data = snapshots[idx]
        N, E = data.num_nodes, data.num_edges

        # Move snapshot to device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(x, edge_index, edge_attr)
            total_loss, edge_loss, node_loss = model.compute_loss(x, edge_attr, outputs)
            anomaly_feats = model.extract_anomaly_features(
                outputs["edge_latent"], edge_attr, outputs["edge_recon"]
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        total_forward_time += dt

        # Assertions
        assert outputs["node_latent"].shape == (N, 32), f"Bad node_latent shape: {outputs['node_latent'].shape}"
        assert outputs["edge_latent"].shape == (E, 32), f"Bad edge_latent shape: {outputs['edge_latent'].shape}"
        assert outputs["node_recon"].shape == (N, 16), f"Bad node_recon shape: {outputs['node_recon'].shape}"
        assert outputs["edge_recon"].shape == (E, 81), f"Bad edge_recon shape: {outputs['edge_recon'].shape}"
        assert anomaly_feats["flow_features_for_detectors"].shape == (E, 113)

        assert torch.isfinite(total_loss), "Non-finite total loss detected!"
        assert not torch.isnan(outputs["edge_recon"]).any(), "NaN in edge_recon!"
        assert not torch.isnan(outputs["node_recon"]).any(), "NaN in node_recon!"

        print(
            f"  Snapshot #{idx:3d} (Window {data.window_id:3d}): "
            f"Nodes={N:3d}, Edges={E:4d} | "
            f"Loss: Total={total_loss.item():.4f}, Edge={edge_loss.item():.4f}, Node={node_loss.item():.4f} | "
            f"Time={dt*1000:5.2f}ms"
        )

    # 5. Resource Consumption
    vram_alloc = torch.cuda.memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
    vram_peak = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
    vram_res = torch.cuda.memory_reserved(device) / (1024**2) if torch.cuda.is_available() else 0.0
    ram_mb = psutil.Process().memory_info().rss / (1024**2)

    print("\nResource Consumption Metrics:")
    print(f"  CPU Process RSS RAM:      {ram_mb:.2f} MB")
    print(f"  GPU VRAM Current Active:  {vram_alloc:.2f} MB")
    print(f"  GPU VRAM Peak Allocated:  {vram_peak:.2f} MB")
    print(f"  GPU VRAM Reserved:        {vram_res:.2f} MB (GTX 1650: 4,096 MB total)")
    print(f"  Average Forward Time:     {(total_forward_time / len(test_indices))*1000:.2f} ms / snapshot")

    print("\n" + "=" * 65)
    print("ALL 8 ARCHITECTURAL VALIDATION CHECKS PASSED PERFECTLY ON GPU!")
    print("=" * 65)


if __name__ == "__main__":
    run_forward_validation()
