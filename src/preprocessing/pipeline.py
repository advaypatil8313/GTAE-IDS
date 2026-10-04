"""Data preprocessing pipeline for LSPR23 dataset."""

import json
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PreprocessingConfig
from src.preprocessing.schema import (
    DTYPE_MAPPING,
    EXCLUDED_LEAKAGE_COLUMNS,
    FLOW_FEATURE_COLUMNS,
    GRAPH_ENDPOINT_COLUMNS,
    TARGET_COLUMN,
    TEMPORAL_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class LSPR23Preprocessor:
    """Preprocessor for streaming, sampling, cleaning, and exporting LSPR23 flow data."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def stream_and_sample(self) -> pd.DataFrame:
        """Stream chunks directly from raw ZIP and sample a representative development subset."""
        logger.info(f"Opening raw dataset archive: {self.config.raw_zip_path}")
        if not self.config.raw_zip_path.exists():
            raise FileNotFoundError(f"Raw ZIP file not found at: {self.config.raw_zip_path}")

        sampled_chunks: List[pd.DataFrame] = []
        total_rows_scanned = 0
        chunks_scanned = 0

        # Columns to load: graph endpoints + temporal + target + flow features
        # Note: we explicitly load only required columns, bypassing unnecessary high-cardinality metadata
        required_cols = list(
            dict.fromkeys(
                GRAPH_ENDPOINT_COLUMNS
                + [c for c in TEMPORAL_COLUMNS if c != "Flow Duration"]
                + [TARGET_COLUMN]
                + FLOW_FEATURE_COLUMNS
            )
        )

        max_chunks = self.config.max_chunks_to_scan
        if max_chunks <= 0:
            max_chunks = None  # Scan full dataset

        logger.info(
            f"Streaming in chunks of {self.config.chunk_size:,} rows. "
            f"Max chunks to scan: {max_chunks if max_chunks else 'All (Full Dataset)'}. "
            f"Target subset size: {self.config.subset_size:,}."
        )

        t0 = time.time()
        with zipfile.ZipFile(self.config.raw_zip_path, "r") as z:
            with z.open(self.config.raw_inner_csv) as f:
                reader = pd.read_csv(
                    f,
                    chunksize=self.config.chunk_size,
                    usecols=lambda col: col in required_cols,
                    low_memory=False,
                )

                for chunk in reader:
                    chunks_scanned += 1
                    chunk_rows = len(chunk)
                    total_rows_scanned += chunk_rows

                    # Target sampling per chunk
                    if self.config.target_malicious_ratio is not None:
                        # Controlled stratified quota
                        target_ratio = float(self.config.target_malicious_ratio)
                        target_mal = int(self.config.subset_size * target_ratio)
                        target_ben = self.config.subset_size - target_mal

                        ben_df = chunk[chunk[TARGET_COLUMN] == 0]
                        mal_df = chunk[chunk[TARGET_COLUMN] == 1]

                        # For benign, sample proportionally across all scanned chunks for temporal coverage
                        n_chunks_est = max_chunks if max_chunks else 164
                        n_ben = max(1, int(np.ceil((target_ben * 1.3) / n_chunks_est)))
                        s_ben = ben_df.sample(n=min(len(ben_df), n_ben), random_state=self.rng.bit_generator)

                        # For malicious, attacks are clustered in specific exercise injection windows.
                        # Collect available candidate attacks across chunks up to target_mal pool headroom.
                        if len(mal_df) > 0:
                            chunk_mal_quota = max(1000, target_mal // 2)
                            s_mal = mal_df.sample(n=min(len(mal_df), chunk_mal_quota), random_state=self.rng.bit_generator)
                            chunk_sample = pd.concat([s_ben, s_mal], ignore_index=True)
                        else:
                            chunk_sample = s_ben
                    else:
                        # Natural uniform sampling across the scanned chunks
                        n_chunks_est = max_chunks if max_chunks else 164
                        sample_n = int(np.ceil(self.config.subset_size / n_chunks_est))
                        sample_n = min(len(chunk), sample_n)
                        chunk_sample = chunk.sample(n=sample_n, random_state=self.rng.bit_generator)

                    sampled_chunks.append(chunk_sample)

                    if chunks_scanned % 5 == 0 or (max_chunks and chunks_scanned == max_chunks):
                        logger.info(
                            f"Scanned {chunks_scanned} chunks ({total_rows_scanned:,} rows) in {time.time()-t0:.1f}s. "
                            f"Accumulated sample rows: {sum(len(c) for c in sampled_chunks):,}."
                        )

                    if max_chunks and chunks_scanned >= max_chunks:
                        break

        # Combine sampled chunks
        combined_df = pd.concat(sampled_chunks, ignore_index=True)
        logger.info(
            f"Streaming pass complete. Total scanned: {total_rows_scanned:,} rows across {chunks_scanned} chunks. "
            f"Pre-trim sample size: {len(combined_df):,}."
        )

        # Final trim/resample to exact subset_size
        if self.config.target_malicious_ratio is not None:
            # Stratified final trim according to target ratio
            target_mal = int(self.config.subset_size * float(self.config.target_malicious_ratio))
            
            ben_df = combined_df[combined_df[TARGET_COLUMN] == 0]
            mal_df = combined_df[combined_df[TARGET_COLUMN] == 1]

            actual_mal = min(len(mal_df), target_mal)
            needed_ben = self.config.subset_size - actual_mal

            s_ben = ben_df.sample(n=min(len(ben_df), needed_ben), random_state=self.rng.bit_generator)
            s_mal = mal_df.sample(n=actual_mal, random_state=self.rng.bit_generator)
            final_df = pd.concat([s_ben, s_mal], ignore_index=True)
            final_df = final_df.sample(frac=1.0, random_state=self.rng.bit_generator).reset_index(drop=True)
        elif len(combined_df) > self.config.subset_size:
            final_df = combined_df.sample(
                n=self.config.subset_size,
                random_state=self.rng.bit_generator,
            ).reset_index(drop=True)
        else:
            final_df = combined_df.reset_index(drop=True)

        return final_df

    def clean_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean numerical features strictly adhering to project guidelines:
        1. Replace +/-Inf with NaN.
        2. Impute NaN using documented deterministic domain strategy (0.0 for 0-duration/idle).
        3. Preserve legitimate extreme values (NO outlier clipping).
        4. Downcast data types to memory-efficient representations.
        """
        logger.info("Cleaning features and handling NaN/Inf values...")
        cleaning_stats: Dict[str, Any] = {
            "inf_values_handled": {},
            "nan_values_handled": {},
            "total_inf_replaced": 0,
            "total_nan_imputed": 0,
        }

        # 1. Clean numerical flow features
        for col in FLOW_FEATURE_COLUMNS:
            if col not in df.columns:
                continue

            # Coerce to numeric in case of mixed string types
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Count infinite values
            pos_inf = int(np.isposinf(df[col]).sum())
            neg_inf = int(np.isneginf(df[col]).sum())
            total_inf = pos_inf + neg_inf
            if total_inf > 0:
                cleaning_stats["inf_values_handled"][col] = {
                    "pos_inf": pos_inf,
                    "neg_inf": neg_inf,
                    "total": total_inf,
                }
                cleaning_stats["total_inf_replaced"] += total_inf
                # Replace Inf with NaN first
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # Count NaN values (including those converted from Inf)
            nan_count = int(df[col].isna().sum())
            if nan_count > 0:
                cleaning_stats["nan_values_handled"][col] = nan_count
                cleaning_stats["total_nan_imputed"] += nan_count
                # Deterministic imputation:
                # Rates (bytes/s, pkts/s) with 0 duration or intervals with 1 packet are physically 0
                df[col] = df[col].fillna(0.0)

            # Downcast flow continuous metric to float32
            df[col] = df[col].astype(np.float32)

        # 2. Downcast endpoint columns
        for col, dtype in DTYPE_MAPPING.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)

        # Ensure SrcIP and DstIP are clean strings
        df["SrcIP"] = df["SrcIP"].astype(str)
        df["DstIP"] = df["DstIP"].astype(str)

        # 3. Sort chronologically by mTimestampStart for proper temporal ordering
        if "mTimestampStart" in df.columns:
            df = df.sort_values(by="mTimestampStart").reset_index(drop=True)

        logger.info(
            f"Cleaning completed: Replaced {cleaning_stats['total_inf_replaced']} Inf values and "
            f"imputed {cleaning_stats['total_nan_imputed']} NaN values with 0.0. "
            f"Zero outliers were clipped (extreme values preserved)."
        )
        return df, cleaning_stats

    def verify_integrity(self, df: pd.DataFrame) -> None:
        """Validate all integrity checks before persisting."""
        logger.info("Verifying data integrity...")

        # A. Check feature matrix NaN / Inf
        nan_cols = [c for c in FLOW_FEATURE_COLUMNS if df[c].isna().any()]
        if nan_cols:
            raise ValueError(f"Integrity check failed: NaNs found in features: {nan_cols}")

        inf_cols = [c for c in FLOW_FEATURE_COLUMNS if np.isinf(df[c]).any()]
        if inf_cols:
            raise ValueError(f"Integrity check failed: Infs found in features: {inf_cols}")

        # B. Check target leakage
        forbidden_present = [c for c in EXCLUDED_LEAKAGE_COLUMNS if c in df.columns]
        if forbidden_present:
            raise ValueError(f"Target leakage violation: Forbidden columns present in dataset: {forbidden_present}")

        # C. Check Label classes
        classes = df[TARGET_COLUMN].unique()
        if len(classes) < 2:
            raise ValueError(f"Class imbalance error: Dataset does not contain both classes. Unique: {classes}")

        # D. Check endpoints and timestamps
        assert (df["mTimestampStart"] > 0).all(), "Invalid negative/zero timestamps found."
        assert (df["SrcIP"].str.len() > 0).all(), "Empty SrcIP values found."
        assert (df["DstIP"].str.len() > 0).all(), "Empty DstIP values found."

        logger.info("Data integrity verified successfully.")

    def export_processed(self, df: pd.DataFrame, cleaning_stats: Dict[str, Any]) -> Dict[str, Path]:
        """Save processed development subset as .pkl, .csv.gz, and metadata.json."""
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        base_path = self.config.processed_dir / self.config.output_basename

        pkl_path = base_path.with_suffix(".pkl")
        csv_gz_path = self.config.processed_dir / f"{self.config.output_basename}.csv.gz"
        json_path = self.config.processed_dir / f"{self.config.output_basename}_metadata.json"

        # 1. Export pickle (fast native loading preserving float32/uint16 types)
        logger.info(f"Saving pickle to: {pkl_path}")
        df.to_pickle(pkl_path, protocol=5)

        # 2. Export compressed CSV (portable and inspectable)
        logger.info(f"Saving compressed CSV to: {csv_gz_path}")
        df.to_csv(csv_gz_path, compression="gzip", index=False)

        # 3. Generate and export metadata JSON
        label_counts = df[TARGET_COLUMN].value_counts().to_dict()
        benign_count = int(label_counts.get(0, 0))
        malicious_count = int(label_counts.get(1, 0))
        total_rows = len(df)

        min_ts = int(df["mTimestampStart"].min())
        max_ts = int(df["mTimestampStart"].max())
        min_dt = datetime.fromtimestamp(min_ts / 1e6).isoformat()
        max_dt = datetime.fromtimestamp(max_ts / 1e6).isoformat()
        span_secs = (max_ts - min_ts) / 1e6

        metadata = {
            "dataset_name": "LSPR23 Development Flow Subset",
            "source_archive": str(self.config.raw_zip_path),
            "generated_at": datetime.now().isoformat(),
            "total_rows": total_rows,
            "benign_count": benign_count,
            "benign_percentage": round((benign_count / total_rows) * 100, 4),
            "malicious_count": malicious_count,
            "malicious_percentage": round((malicious_count / total_rows) * 100, 4),
            "random_seed": self.config.random_seed,
            "configured_sampling_ratio": self.config.target_malicious_ratio,
            "temporal_range": {
                "min_timestamp_microseconds": min_ts,
                "max_timestamp_microseconds": max_ts,
                "min_datetime_iso": min_dt,
                "max_datetime_iso": max_dt,
                "time_span_seconds": round(span_secs, 2),
                "time_span_hours": round(span_secs / 3600, 2),
            },
            "columns": {
                "graph_endpoint_columns": GRAPH_ENDPOINT_COLUMNS,
                "temporal_columns": TEMPORAL_COLUMNS,
                "target_column": TARGET_COLUMN,
                "flow_feature_columns_count": len(FLOW_FEATURE_COLUMNS),
                "flow_feature_columns": FLOW_FEATURE_COLUMNS,
                "excluded_leakage_columns": EXCLUDED_LEAKAGE_COLUMNS,
            },
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_and_infinite_handling": {
                "strategy": "Replace +/-Inf with NaN, impute NaN with 0.0 (physically justified for 0-duration/rate metrics). No clipping of extreme values.",
                "total_inf_replaced": cleaning_stats["total_inf_replaced"],
                "total_nan_imputed": cleaning_stats["total_nan_imputed"],
                "inf_details_by_column": cleaning_stats["inf_values_handled"],
                "nan_details_by_column": cleaning_stats["nan_values_handled"],
            },
            "memory_usage": {
                "total_bytes": int(df.memory_usage(deep=True).sum()),
                "total_megabytes": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            },
            "output_files": {
                "pickle": str(pkl_path),
                "csv_gz": str(csv_gz_path),
                "metadata_json": str(json_path),
            },
        }

        logger.info(f"Saving metadata to: {json_path}")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(metadata, jf, indent=2)

        return {
            "pickle": pkl_path,
            "csv_gz": csv_gz_path,
            "metadata_json": json_path,
        }

    def run(self) -> Dict[str, Any]:
        """Execute full preprocessing pipeline."""
        t_start = time.time()
        logger.info("=== Starting LSPR23 Preprocessing Pipeline ===")

        # 1. Stream and sample
        df = self.stream_and_sample()

        # 2. Clean features
        df, cleaning_stats = self.clean_features(df)

        # 3. Verify integrity
        self.verify_integrity(df)

        # 4. Export artifacts
        output_paths = self.export_processed(df, cleaning_stats)

        elapsed = time.time() - t_start
        logger.info(f"=== Pipeline completed successfully in {elapsed:.2f}s ===")

        return {
            "df": df,
            "cleaning_stats": cleaning_stats,
            "output_paths": output_paths,
            "elapsed_seconds": elapsed,
        }
