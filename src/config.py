"""Configuration settings for GTAE-IDS pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessingConfig:
    """Configuration for dataset ingestion, sampling, cleaning, and export."""

    # Dataset paths
    raw_zip_path: Path = Path("data/raw/ls23pr_flows.zip")
    raw_inner_csv: str = "ls23pr_v1.csv"
    processed_dir: Path = Path("data/processed")
    output_basename: str = "lspr23_dev_subset"

    # Ingestion & sampling parameters
    subset_size: int = 50_000
    chunk_size: int = 100_000
    random_seed: int = 42
    max_chunks_to_scan: int = 30  # Scans ~3,000,000 flows (30 chunks). 0 or -1 scans full dataset.
    target_malicious_ratio: Optional[float] = None  # None = preserve natural empirical ratio

    def __post_init__(self):
        self.raw_zip_path = Path(self.raw_zip_path)
        self.processed_dir = Path(self.processed_dir)
