"""Configuration dataclass for Graph Transformer-Based Autoencoder (GTAE)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GTAEConfig:
    """Architectural and training hyperparameters for GTAE."""

    # Dimensionality
    in_node_dim: int = 16
    in_edge_dim: int = 81
    hidden_dim: int = 64
    latent_dim: int = 32

    # Graph Transformer parameters
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    # Loss parameters
    node_loss_weight: float = 0.5  # Weight alpha for node reconstruction loss
    loss_type: str = "smooth_l1"   # "smooth_l1" (Huber) or "mse"

    # Optimization parameters (for future training)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Chronological dataset split fractions
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
