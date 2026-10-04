"""Configuration settings for temporal graph construction in GTAE-IDS."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GraphConfig:
    """Configuration for temporal graph snapshot builder."""

    # Input dataset
    input_flows_path: Path = Path("data/processed/lspr23_dev_100k.pkl")

    # Output directory
    output_dir: Path = Path("data/processed/graphs")
    output_snapshots_filename: str = "temporal_graph_snapshots.pt"
    output_metadata_filename: str = "graph_metadata.json"

    # Temporal window parameters
    window_duration_seconds: float = 300.0  # 5 minutes (300 seconds)
    timestamp_unit: float = 1e6             # Timestamps are microsecond epochs

    # Feature dimensions
    node_feature_dim: int = 16
    edge_feature_dim: int = 81              # 76 numeric flow metrics + 1 norm port + 4 proto one-hot

    def __post_init__(self):
        self.input_flows_path = Path(self.input_flows_path)
        self.output_dir = Path(self.output_dir)
