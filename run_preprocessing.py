"""CLI executable script to run the LSPR23 preprocessing pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PreprocessingConfig
from src.preprocessing.pipeline import LSPR23Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LSPR23 chunked streaming, sampling, cleaning, and preprocessing pipeline."
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=50_000,
        help="Number of flow records to sample for the development subset (default: 50,000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per chunk when streaming from ZIP (default: 100,000).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=30,
        help="Maximum chunks to scan from raw archive (default: 30 chunks = ~3M rows; 0 or -1 scans all).",
    )
    parser.add_argument(
        "--malicious-ratio",
        type=float,
        default=None,
        help="Optional target malicious ratio (e.g., 0.10 for 10%% attacks). None preserves natural observed ratio.",
    )
    parser.add_argument(
        "--raw-zip",
        type=str,
        default="data/raw/ls23pr_flows.zip",
        help="Path to raw dataset ZIP archive (default: data/raw/ls23pr_flows.zip).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed artifacts (default: data/processed).",
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        default="lspr23_dev_subset",
        help="Base name for output files (default: lspr23_dev_subset).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = PreprocessingConfig(
        raw_zip_path=Path(args.raw_zip),
        processed_dir=Path(args.output_dir),
        output_basename=args.output_basename,
        subset_size=args.subset_size,
        chunk_size=args.chunk_size,
        random_seed=args.seed,
        max_chunks_to_scan=args.max_chunks,
        target_malicious_ratio=args.malicious_ratio,
    )

    preprocessor = LSPR23Preprocessor(config)
    results = preprocessor.run()

    df = results["df"]
    paths = results["output_paths"]
    print("\n" + "=" * 60)
    print("SUCCESS: LSPR23 Preprocessing Complete!")
    print(f"Total Rows:        {len(df):,}")
    print(f"Benign Flows (0):  {(df['Label'] == 0).sum():,} ({(df['Label'] == 0).mean()*100:.2f}%)")
    print(f"Attack Flows (1):  {(df['Label'] == 1).sum():,} ({(df['Label'] == 1).mean()*100:.2f}%)")
    print(f"Pickle Path:       {paths['pickle']}")
    print(f"Gzip CSV Path:     {paths['csv_gz']}")
    print(f"Metadata Path:     {paths['metadata_json']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
