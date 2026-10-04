"""CLI executable script to run GTAE benign-only self-supervised training."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.config import GTAEConfig
from src.models.training import GTAETrainer, GTAETrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benign-only self-supervised training for GTAE on LSPR23."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of graph snapshots per mini-batch (default: 8).",
    )
    parser.add_argument(
        "--snapshots-path",
        type=str,
        default="data/processed/graphs/temporal_graph_snapshots.pt",
        help="Path to input graph snapshots (default: data/processed/graphs/temporal_graph_snapshots.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/models",
        help="Directory to save model checkpoints (default: data/processed/models).",
    )
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default="best_gtae_model_clean.pt",
        help="Filename for best model checkpoint (default: best_gtae_model_clean.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_cfg = GTAETrainingConfig(
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        snapshots_path=Path(args.snapshots_path),
        checkpoint_dir=Path(args.output_dir),
        checkpoint_filename=args.checkpoint_filename,
    )

    model_cfg = GTAEConfig(
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

    trainer = GTAETrainer(train_config=train_cfg, model_config=model_cfg)
    results = trainer.train()

    print("\n" + "=" * 65)
    print("SUCCESS: GTAE Training Experiment Complete!")
    print(f"Total Epochs Run:           {results['total_epochs']}")
    print(f"Best Checkpoint Epoch:      {results['best_epoch']}")
    print(f"Best Validation Loss:       {results['best_val_loss']:.6f}")
    print(f"Early Stopping Occurred:    {results['early_stopping']}")
    print(f"Total Training Time:        {results['total_time_seconds']}s (avg {results['avg_epoch_time_seconds']}s/epoch)")
    print(f"Peak GPU VRAM Allocated:    {results['peak_vram_mb']:.2f} MB")
    print(f"Process RAM:                {results['process_ram_mb']:.2f} MB")
    print(f"Best Checkpoint File:       {results['best_checkpoint_path']}")
    print(f"History File:               {results['history_path']}")
    print(f"Diagnostic Artifact File:   {results['anomaly_artifacts_path']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
