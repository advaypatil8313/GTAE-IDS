"""CLI runner for GTAE downstream anomaly-detection feature extraction."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.detection.feature_pipeline import GTAEFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 113-dimensional GTAE anomaly representations for downstream detectors."
    )
    parser.add_argument(
        "--snapshots-path",
        type=str,
        default="data/processed/graphs/temporal_graph_snapshots.pt",
        help="Path to temporal graph snapshots .pt file.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="data/processed/models/best_gtae_model_clean.pt",
        help="Path to clean GTAE model checkpoint.",
    )
    parser.add_argument(
        "--scalers-path",
        type=str,
        default="data/processed/scalers/gtae_scalers_clean.pkl",
        help="Path to clean GTAE feature scalers.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/artifacts",
        help="Directory to save extracted downstream feature artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: cuda:0 if available, else cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("GTAE DOWNSTREAM ANOMALY-DETECTION FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Snapshots Source:   {args.snapshots_path}")
    print(f"Clean Checkpoint:   {args.checkpoint_path}")
    print(f"Clean Scalers:      {args.scalers_path}")
    print(f"Output Artifacts:   {args.output_dir}")
    print("=" * 70)

    extractor = GTAEFeatureExtractor(
        checkpoint_path=Path(args.checkpoint_path),
        scalers_path=Path(args.scalers_path),
        device=args.device,
    )

    results = extractor.run_pipeline(
        snapshots_path=Path(args.snapshots_path),
        output_dir=Path(args.output_dir),
    )

    meta = results["metadata"]
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Train Split: {meta['split_summary']['train']['total_flows']:,} flows "
          f"({meta['split_summary']['train']['benign_flows']:,} benign, {meta['split_summary']['train']['attack_flows']} attack) "
          f"-> {results['paths']['train_pt']}")
    print(f"Val Split:   {meta['split_summary']['val']['total_flows']:,} flows "
          f"({meta['split_summary']['val']['benign_flows']:,} benign, {meta['split_summary']['val']['attack_flows']:,} attack) "
          f"-> {results['paths']['val_pt']}")
    print(f"Test Split:  {meta['split_summary']['test']['total_flows']:,} flows "
          f"({meta['split_summary']['test']['benign_flows']:,} benign, {meta['split_summary']['test']['attack_flows']:,} attack) "
          f"-> {results['paths']['test_pt']}")
    print(f"Metadata:    {results['paths']['metadata_json']}")
    print(f"Feature Dim: {meta['dimensions']['combined_feature_dim']} (32 latent + 81 recon error)")
    print("=" * 70)


if __name__ == "__main__":
    main()
