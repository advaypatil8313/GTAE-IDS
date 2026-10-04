"""CLI executable script to run temporal graph construction on LSPR23."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.config import GraphConfig
from src.graph.graph_builder import LSPR23GraphBuilder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LSPR23 temporal graph snapshot construction (5-minute windows) for GTAE-IDS."
    )
    parser.add_argument(
        "--input-flows",
        type=str,
        default="data/processed/lspr23_dev_100k.pkl",
        help="Path to preprocessed flows pickle file (default: data/processed/lspr23_dev_100k.pkl).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/graphs",
        help="Directory to store graph snapshots and metadata (default: data/processed/graphs).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=300.0,
        help="Temporal window duration in seconds (default: 300.0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = GraphConfig(
        input_flows_path=Path(args.input_flows),
        output_dir=Path(args.output_dir),
        window_duration_seconds=args.window_seconds,
    )

    builder = LSPR23GraphBuilder(config)
    results = builder.build_snapshots()

    snapshots = results["snapshots"]
    meta = results["metadata"]
    paths = results["output_paths"]

    print("\n" + "=" * 65)
    print("SUCCESS: Temporal Graph Construction Complete & Verified!")
    print(f"Total Snapshots:            {len(snapshots):,}")
    print(f"Total Graph Edges:          {meta['summary_statistics']['total_graph_edges']:,}")
    print(f"Total Graph Nodes (Sum):    {meta['summary_statistics']['total_nodes_accumulated']:,}")
    print(f"Attack-Containing Windows:  {meta['summary_statistics']['attack_containing_windows']:,} ({meta['summary_statistics']['attack_window_percentage']}%)")
    print(f"Nodes per Window (Mean/Max): {meta['node_distribution']['mean']} / {meta['node_distribution']['max']}")
    print(f"Edges per Window (Mean/Max): {meta['edge_distribution']['mean']} / {meta['edge_distribution']['max']}")
    print(f"Node Features Tensor Shape: [num_nodes, {meta['tensor_dimensions']['node_features_dim']}]")
    print(f"Edge Attributes Tensor Shape:[num_edges, {meta['tensor_dimensions']['edge_attributes_dim']}]")
    print(f"Snapshots PyTorch .pt File: {paths['snapshots_pt']} ({meta['storage']['snapshots_size_mb']} MB)")
    print(f"Metadata JSON File:         {paths['metadata_json']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
