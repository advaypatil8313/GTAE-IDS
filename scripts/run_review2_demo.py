"""Review-2 Real Demonstration Script for GTAE-IDS.

Visualizes the real components built and verified for Review 2:
- Real LSPR23 5-minute temporal graph snapshot
- Clean GTAE model loading and forward pass
- Real 32-D latent representation
- Real original vs reconstructed edge features
- Real reconstruction error distribution
- Real 113-D downstream feature vector construction
"""

import argparse
import datetime
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from src.models.config import GTAEConfig
from src.models.gtae import GTAEModel
from src.models.normalization import GTAEFeatureScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Review2Demo")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real Review-2 visualization demo for GTAE-IDS."
    )
    parser.add_argument(
        "--snapshots-path",
        type=str,
        default="data/processed/graphs/temporal_graph_snapshots.pt",
        help="Path to real temporal graph snapshots artifact.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="data/processed/models/best_gtae_model_clean.pt",
        help="Path to clean GTAE checkpoint.",
    )
    parser.add_argument(
        "--scalers-path",
        type=str,
        default="data/processed/scalers/gtae_scalers_clean.pkl",
        help="Path to clean feature scalers.",
    )
    parser.add_argument(
        "--snapshot-index",
        type=int,
        default=46,
        help="Index of representative 5-minute temporal graph snapshot to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/review2_demo",
        help="Directory to save generated matplotlib visualization artifacts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display interactive matplotlib figures on screen.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    secondary_output_dir = Path("data/processed/artifacts/review2_demo")
    secondary_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GTAE-IDS: TEMPORAL GRAPH & MODEL PIPELINE EXECUTION")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD ONE REAL GRAPH
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: LOAD ONE REAL GRAPH ---")
    snapshots_path = Path(args.snapshots_path)
    if not snapshots_path.exists():
        raise FileNotFoundError(f"Snapshots file not found at: {snapshots_path}")

    print(f"Loading real graph artifact from: {snapshots_path}")
    snapshots = torch.load(snapshots_path, weights_only=False)
    total_snapshots = len(snapshots)
    print(f"Total temporal graph snapshots available: {total_snapshots}")

    # Select representative 5-minute snapshot
    snap_idx = args.snapshot_index
    if snap_idx < 0 or snap_idx >= total_snapshots:
        raise IndexError(f"Snapshot index {snap_idx} out of range [0, {total_snapshots - 1}]")

    snapshot = snapshots[snap_idx]

    # Extract metadata and statistics
    window_id = getattr(snapshot, "window_id", snap_idx)
    num_nodes = snapshot.num_nodes
    num_edges = snapshot.num_edges
    node_feat_shape = list(snapshot.x.shape)
    edge_feat_shape = list(snapshot.edge_attr.shape)

    time_range_str = "N/A"
    if hasattr(snapshot, "window_start_ts") and hasattr(snapshot, "window_end_ts"):
        start_dt = datetime.datetime.fromtimestamp(
            snapshot.window_start_ts / 1e6, tz=datetime.timezone.utc
        )
        end_dt = datetime.datetime.fromtimestamp(
            snapshot.window_end_ts / 1e6, tz=datetime.timezone.utc
        )
        time_range_str = (
            f"{start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC to "
            f"{end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC "
            f"({snapshot.window_start_ts} to {snapshot.window_end_ts} us)"
        )

    benign_count = int((snapshot.y == 0).sum().item()) if hasattr(snapshot, "y") else 0
    attack_count = int((snapshot.y == 1).sum().item()) if hasattr(snapshot, "y") else 0

    print(f"Selected Snapshot Index:    {snap_idx}")
    print(f"Snapshot / Window ID:       {window_id}")
    print(f"Time Range:                 {time_range_str}")
    print(f"Number of Nodes:            {num_nodes}")
    print(f"Number of Edges:            {num_edges}")
    print(f"Node Feature Shape:         {node_feat_shape}")
    print(f"Edge Feature Shape:         {edge_feat_shape}")
    print(f"Normal Edge Count (Benign): {benign_count}")
    print(f"Malicious Edge Count:       {attack_count}")

    # -------------------------------------------------------------------------
    # STEP 2: VISUALIZE THE REAL GRAPH
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: VISUALIZE THE REAL GRAPH ---")
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=150)
    fig1.patch.set_facecolor("#fafafa")
    ax1.set_facecolor("#ffffff")

    G = nx.DiGraph()
    src_nodes = snapshot.edge_index[0].tolist()
    dst_nodes = snapshot.edge_index[1].tolist()
    labels = snapshot.y.tolist() if hasattr(snapshot, "y") else [0] * num_edges

    for u, v, lbl in zip(src_nodes, dst_nodes, labels):
        G.add_edge(u, v, label=lbl)

    # Spring layout for clear node dispersion
    pos = nx.spring_layout(G, seed=42, k=0.45, iterations=60)

    # Separate edges for color distinction
    benign_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label", 0) == 0]
    attack_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label", 0) == 1]

    # Draw nodes
    degrees = dict(G.degree())
    node_sizes = [max(80, min(600, 100 + 40 * degrees.get(n, 1))) for n in G.nodes()]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax1,
        node_size=node_sizes,
        node_color="#3498db",
        edgecolors="#1b4f72",
        linewidths=1.2,
        alpha=0.9,
    )

    # Draw directed edges (flows)
    if benign_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=benign_edges,
            ax=ax1,
            edge_color="#2980b9",
            alpha=0.5,
            arrows=True,
            arrowsize=10,
            width=1.0,
            connectionstyle="arc3,rad=0.08",
        )
    if attack_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=attack_edges,
            ax=ax1,
            edge_color="#e74c3c",
            alpha=0.85,
            arrows=True,
            arrowsize=14,
            width=2.0,
            connectionstyle="arc3,rad=0.12",
        )

    # Label top 6 most active nodes with their actual IP from node_ip_map
    if hasattr(snapshot, "node_ip_map") and snapshot.node_ip_map:
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:6]
        labels_dict = {n: f"IP:{snapshot.node_ip_map[n]}" for n in top_nodes if n < len(snapshot.node_ip_map)}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels_dict,
            font_size=7,
            font_weight="bold",
            font_color="#000000",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#aaaaaa", alpha=0.85),
            ax=ax1,
        )

    ax1.set_title(
        "Real LSPR23 5-Minute Temporal Graph",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    # Explanatory text box as specified
    explanation_text = (
        "Nodes = network hosts\n"
        "Edges = network flows\n"
        f"Window ID: {window_id} ({num_nodes} hosts, {num_edges} flows)\n"
        f"Benign flows: {benign_count} (blue) | Attack flows: {attack_count} (red)\n"
        "Full snapshot graph displayed directly from real LSPR23 artifacts"
    )
    ax1.text(
        0.02,
        0.98,
        explanation_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95),
    )
    ax1.axis("off")
    fig1.tight_layout()

    fig1_path = output_dir / "01_real_temporal_graph.png"
    fig1.savefig(fig1_path, bbox_inches="tight")
    fig1.savefig(secondary_output_dir / "01_real_temporal_graph.png", bbox_inches="tight")
    print(f"Saved graph visualization to: {fig1_path}")

    # -------------------------------------------------------------------------
    # STEP 3: LOAD THE CLEAN GTAE
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: LOAD THE CLEAN GTAE ---")
    checkpoint_path = Path(args.checkpoint_path)
    scalers_path = Path(args.scalers_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Clean GTAE checkpoint not found at: {checkpoint_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Clean scalers file not found at: {scalers_path}")

    # Enforce clean checkpoint usage
    if "clean" not in checkpoint_path.name:
        raise ValueError(
            f"Prohibited checkpoint: {checkpoint_path.name}. Must use clean model checkpoint."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device:                     {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name:                   {gpu_name}")
    else:
        print("GPU Name:                   N/A (Running on CPU)")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = GTAEConfig(**checkpoint["model_config"])
    model = GTAEModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameter Count:      {param_count:,}")
    print(f"Input Dimensions:           Node input = {model_cfg.in_node_dim}, Edge input = {model_cfg.in_edge_dim}")
    print(f"Architecture Specification: Hidden = {model_cfg.hidden_dim}, Latent = {model_cfg.latent_dim}")

    # Load fitted clean scaler
    scaler = GTAEFeatureScaler.load(scalers_path)

    # -------------------------------------------------------------------------
    # STEP 4: RUN THE REAL GTAE
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: RUN THE REAL GTAE ---")
    # Apply fitted training normalization to snapshot
    snap_scaled = scaler.transform_snapshot(snapshot)

    x_dev = snap_scaled.x.to(device)
    edge_index_dev = snap_scaled.edge_index.to(device)
    edge_attr_dev = snap_scaled.edge_attr.to(device)

    with torch.no_grad():
        outputs = model(x_dev, edge_index_dev, edge_attr_dev)

    node_latent = outputs["node_latent"]
    edge_latent = outputs["edge_latent"]
    node_recon = outputs["node_recon"]
    edge_recon = outputs["edge_recon"]

    # Verify strictly finite outputs
    assert torch.isfinite(node_latent).all(), "Non-finite node latent values!"
    assert torch.isfinite(edge_latent).all(), "Non-finite edge latent values!"
    assert torch.isfinite(node_recon).all(), "Non-finite node reconstruction values!"
    assert torch.isfinite(edge_recon).all(), "Non-finite edge reconstruction values!"

    print("Real GTAE forward pass completed successfully.")
    print(f"Node latent:         {list(node_latent.shape)}")
    print(f"Edge latent:         {list(edge_latent.shape)}")
    print(f"Node reconstruction: {list(node_recon.shape)}")
    print(f"Edge reconstruction: {list(edge_recon.shape)}")

    # -------------------------------------------------------------------------
    # STEP 5: VISUALIZE THE 32-D LATENT REPRESENTATION
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: VISUALIZE THE 32-D LATENT REPRESENTATION ---")
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=150)
    fig2.patch.set_facecolor("#fafafa")
    ax2.set_facecolor("#ffffff")

    # Take a readable sample of real edges (e.g. 25 flows)
    num_flows_sample = min(25, num_edges)
    edge_latent_sample = edge_latent[:num_flows_sample].cpu().numpy()

    cax2 = ax2.imshow(edge_latent_sample, aspect="auto", cmap="viridis")
    cbar2 = fig2.colorbar(cax2, ax=ax2, pad=0.02)
    cbar2.set_label("Latent Activation Value", fontsize=10)

    ax2.set_title("Actual GTAE 32-D Latent Representation", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("32 learned latent dimensions", fontsize=11, labelpad=8)
    ax2.set_ylabel(f"Network Flow Index (0 to {num_flows_sample - 1})", fontsize=11, labelpad=8)
    ax2.set_xticks(range(0, 32, 4))
    ax2.set_xticklabels([f"z_{i}" for i in range(0, 32, 4)])

    fig2.tight_layout()
    fig2_path = output_dir / "02_actual_gtae_latent_representation.png"
    fig2.savefig(fig2_path, bbox_inches="tight")
    fig2.savefig(secondary_output_dir / "02_actual_gtae_latent_representation.png", bbox_inches="tight")
    print(f"Saved 32-D latent visualization to: {fig2_path}")

    # -------------------------------------------------------------------------
    # STEP 6: VISUALIZE RECONSTRUCTION
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: VISUALIZE RECONSTRUCTION ---")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharey=True)
    fig3.patch.set_facecolor("#fafafa")

    num_recon_flows = min(20, num_edges)
    orig_edge_sample = edge_attr_dev[:num_recon_flows].cpu().numpy()
    recon_edge_sample = edge_recon[:num_recon_flows].cpu().numpy()

    # Shared color scale for direct visual comparison
    vmin = min(orig_edge_sample.min(), recon_edge_sample.min())
    vmax = max(orig_edge_sample.max(), recon_edge_sample.max())

    im3a = ax3a.imshow(orig_edge_sample, aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax)
    ax3a.set_title(f"Original Edge Features (81-D)\n(First {num_recon_flows} Real Flows)", fontsize=11, fontweight="bold")
    ax3a.set_xlabel("Feature Dimension (0 to 80)", fontsize=10)
    ax3a.set_ylabel(f"Flow Index (0 to {num_recon_flows - 1})", fontsize=10)

    im3b = ax3b.imshow(recon_edge_sample, aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax)
    ax3b.set_title(f"Reconstructed Edge Features (81-D)\n(GTAE Output)", fontsize=11, fontweight="bold")
    ax3b.set_xlabel("Feature Dimension (0 to 80)", fontsize=10)

    fig3.subplots_adjust(right=0.88, wspace=0.12)
    cbar_ax = fig3.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar3 = fig3.colorbar(im3b, cax=cbar_ax)
    cbar3.set_label("Standardized Feature Value", fontsize=10)

    fig3.suptitle("Original vs GTAE Reconstruction", fontsize=14, fontweight="bold", y=0.98)
    fig3.text(
        0.5,
        0.02,
        "The decoder attempts to reconstruct the original edge features.",
        ha="center",
        fontsize=10,
        style="italic",
    )

    fig3_path = output_dir / "03_original_vs_gtae_reconstruction.png"
    fig3.savefig(fig3_path, bbox_inches="tight")
    fig3.savefig(secondary_output_dir / "03_original_vs_gtae_reconstruction.png", bbox_inches="tight")
    print(f"Saved reconstruction comparison to: {fig3_path}")

    # -------------------------------------------------------------------------
    # STEP 7: VISUALIZE RECONSTRUCTION ERROR
    # -------------------------------------------------------------------------
    print("\n--- STEP 7: VISUALIZE RECONSTRUCTION ERROR ---")
    anomaly_feats = GTAEModel.extract_anomaly_features(
        edge_latent, edge_attr_dev, edge_recon
    )

    abs_recon_residual = anomaly_feats["abs_recon_error_vector"].cpu()  # [E, 81]
    scalar_recon_error = anomaly_feats["scalar_recon_error"].cpu()      # [E]

    mean_err = scalar_recon_error.mean().item()
    median_err = scalar_recon_error.median().item()
    max_err = scalar_recon_error.max().item()

    print(f"Mean Reconstruction Error:    {mean_err:.6f}")
    print(f"Median Reconstruction Error:  {median_err:.6f}")
    print(f"Maximum Reconstruction Error: {max_err:.6f}")
    print("Reconstruction error is an anomaly-related signal used by the downstream detection stage.")

    fig4, ax4 = plt.subplots(figsize=(9, 5), dpi=150)
    fig4.patch.set_facecolor("#fafafa")
    ax4.set_facecolor("#ffffff")

    errors_np = scalar_recon_error.numpy()
    n_bins = 30
    ax4.hist(errors_np, bins=n_bins, color="#3498db", edgecolor="#1b4f72", alpha=0.75, density=False)

    ax4.axvline(mean_err, color="#e74c3c", linestyle="--", linewidth=1.8, label=f"Mean: {mean_err:.4f}")
    ax4.axvline(median_err, color="#2ecc71", linestyle="-.", linewidth=1.8, label=f"Median: {median_err:.4f}")

    ax4.set_title("Distribution of Flow Reconstruction Errors", fontsize=13, fontweight="bold", pad=12)
    ax4.set_xlabel("Mean Absolute Reconstruction Error per Flow", fontsize=11, labelpad=8)
    ax4.set_ylabel("Number of Network Flows", fontsize=11, labelpad=8)
    ax4.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc")

    err_note = (
        "Reconstruction error is an anomaly-related signal\n"
        "used by the downstream detection stage."
    )
    ax4.text(
        0.48,
        0.75,
        err_note,
        transform=ax4.transAxes,
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef9e7", edgecolor="#f39c12", alpha=0.9),
    )

    fig4.tight_layout()
    fig4_path = output_dir / "04_reconstruction_error_distribution.png"
    fig4.savefig(fig4_path, bbox_inches="tight")
    fig4.savefig(secondary_output_dir / "04_reconstruction_error_distribution.png", bbox_inches="tight")
    print(f"Saved reconstruction error visualization to: {fig4_path}")

    # -------------------------------------------------------------------------
    # STEP 8: SHOW THE REAL 113-D FEATURE VECTOR
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: SHOW THE REAL 113-D FEATURE VECTOR ---")
    flow_features_113d = anomaly_feats["flow_features_for_detectors"].cpu()

    latent_dims = edge_latent.shape[1]
    residual_dims = abs_recon_residual.shape[1]
    final_dims = flow_features_113d.shape[1]

    print(f"Latent dimensions:          {latent_dims}")
    print(f"Residual dimensions:        {residual_dims}")
    print(f"Final feature dimensions:   {final_dims}")

    # Select ONE actual flow/edge from snapshot
    selected_flow_idx = 0
    single_flow_113 = flow_features_113d[selected_flow_idx].numpy()

    fig5, ax5 = plt.subplots(figsize=(12, 5), dpi=150)
    fig5.patch.set_facecolor("#fafafa")
    ax5.set_facecolor("#ffffff")

    # Bar-style visualization clearly divided into 1-32 and 33-113
    dims = np.arange(1, 114)
    ax5.bar(
        dims[:32],
        single_flow_113[:32],
        color="#2980b9",
        edgecolor="#1b4f72",
        linewidth=0.6,
        alpha=0.85,
        label="Dimensions 1–32: GTAE latent representation",
    )
    ax5.bar(
        dims[32:],
        single_flow_113[32:],
        color="#e67e22",
        edgecolor="#b95e00",
        linewidth=0.6,
        alpha=0.85,
        label="Dimensions 33–113: reconstruction residuals",
    )

    ax5.axvline(32.5, color="#2c3e50", linestyle="--", linewidth=1.5)

    ax5.set_title("Actual 113-D GTAE Feature Vector", fontsize=13, fontweight="bold", pad=12)
    ax5.set_xlabel("Downstream Feature Dimension Index (1 to 113)", fontsize=11, labelpad=8)
    ax5.set_ylabel("Feature Value", fontsize=11, labelpad=8)
    ax5.set_xlim(0, 114)
    y_max = float(np.max(single_flow_113))
    y_min = float(np.min(single_flow_113))
    ax5.set_ylim(bottom=y_min * 1.2 if y_min < 0 else -0.5, top=y_max * 1.32)
    ax5.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9)

    # Annotation tags on the plot
    ax5.text(
        16,
        ax5.get_ylim()[1] * 0.82,
        "Dimensions 1–32:\nGTAE latent representation",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="#1b4f72",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ebf5fb", edgecolor="#2980b9"),
    )
    ax5.text(
        55,
        ax5.get_ylim()[1] * 0.82,
        "Dimensions 33–113:\nreconstruction residuals",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="#7e3f00",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef5e7", edgecolor="#e67e22"),
    )

    fig5.tight_layout()
    fig5_path = output_dir / "05_actual_113d_feature_vector.png"
    fig5.savefig(fig5_path, bbox_inches="tight")
    fig5.savefig(secondary_output_dir / "05_actual_113d_feature_vector.png", bbox_inches="tight")
    print(f"Saved 113-D feature vector visualization to: {fig5_path}")

    # Optionally show figures if requested
    if args.show:
        print("\nOpening interactive matplotlib display windows...")
        plt.show()

    # -------------------------------------------------------------------------
    # FINAL TERMINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("GTAE-IDS PIPELINE EXECUTION COMPLETE")
    print("=" * 40 + "\n")
    print("[PASS] Real LSPR23 graph loaded")
    print("[PASS] Real 5-minute graph visualized")
    print("[PASS] Clean GTAE checkpoint loaded")
    print("[PASS] Real GTAE forward pass completed")
    print("[PASS] 32-D latent representation generated")
    print("[PASS] Reconstruction generated")
    print("[PASS] Reconstruction error calculated")
    print("[PASS] 113-D feature vector generated\n")
    print("Current project stage:")
    print("LSPR23 -> Graph -> GTAE -> Reconstruction -> 113-D Features\n")
    print("Next stage:")
    print("OCSVM + Isolation Forest + HBOS + INNE + Ensemble\n")
    print("The next stage is NOT implemented by this task.")


if __name__ == "__main__":
    main()
