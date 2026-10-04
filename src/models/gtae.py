"""Graph Transformer-Based Autoencoder (GTAE) for GTAE-IDS."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from src.models.config import GTAEConfig


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer Encoder utilizing PyG TransformerConv layers with edge attribute support.
    Produces node latent embeddings h_v in R^{N x latent_dim}.
    """

    def __init__(self, config: GTAEConfig):
        super().__init__()
        self.config = config

        in_node = config.in_node_dim
        in_edge = config.in_edge_dim
        hidden = config.hidden_dim
        latent = config.latent_dim
        heads = config.num_heads
        head_dim = hidden // heads  # 64 // 4 = 16

        # Layer 1: in_node (16) -> hidden (64) via 4 heads x 16
        self.conv1 = TransformerConv(
            in_channels=in_node,
            out_channels=head_dim,
            heads=heads,
            edge_dim=in_edge,
            concat=True,
            dropout=config.dropout,
        )
        self.norm1 = nn.LayerNorm(hidden)

        # Layer 2: hidden (64) -> hidden (64) via 4 heads x 16
        self.conv2 = TransformerConv(
            in_channels=hidden,
            out_channels=head_dim,
            heads=heads,
            edge_dim=in_edge,
            concat=True,
            dropout=config.dropout,
        )
        self.norm2 = nn.LayerNorm(hidden)

        # Optional Layer 3 if configured
        self.use_layer3 = config.num_layers >= 3
        if self.use_layer3:
            self.conv3 = TransformerConv(
                in_channels=hidden,
                out_channels=head_dim,
                heads=heads,
                edge_dim=in_edge,
                concat=True,
                dropout=config.dropout,
            )
            self.norm3 = nn.LayerNorm(hidden)

        # Node latent projection: hidden (64) -> latent (32)
        self.node_proj = nn.Linear(hidden, latent)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # Layer 1
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = self.norm1(h1)
        h1 = F.gelu(h1)

        # Layer 2 with residual connection
        h2 = self.conv2(h1, edge_index, edge_attr)
        h2 = self.norm2(h2)
        h2 = F.gelu(h2 + h1)

        # Optional Layer 3 with residual connection
        if self.use_layer3:
            h3 = self.conv3(h2, edge_index, edge_attr)
            h3 = self.norm3(h3)
            h2 = F.gelu(h3 + h2)

        # Project to node latent space [N, latent_dim]
        h_v = self.node_proj(h2)
        return h_v


class FlowLatentMLP(nn.Module):
    """
    Constructs per-edge latent representations:
    z_e = MLP([h_src, h_dst, edge_attr]) in R^{E x latent_dim}.
    """

    def __init__(self, config: GTAEConfig):
        super().__init__()
        # Input: h_src (32) + h_dst (32) + edge_attr (81) = 145
        in_dim = config.latent_dim * 2 + config.in_edge_dim
        hidden = config.hidden_dim
        latent = config.latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, latent),
        )

    def forward(
        self, h_src: torch.Tensor, h_dst: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        z_e = self.mlp(combined)
        return z_e


class EdgeDecoder(nn.Module):
    """
    Reconstructs original 81-dimensional flow features from latent flow representation z_e.
    """

    def __init__(self, config: GTAEConfig):
        super().__init__()
        latent = config.latent_dim
        hidden = config.hidden_dim
        out_edge = config.in_edge_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, out_edge),
        )

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_e)


class NodeDecoder(nn.Module):
    """
    Reconstructs original 16-dimensional node behavioral features from node latent representation h_v.
    """

    def __init__(self, config: GTAEConfig):
        super().__init__()
        latent = config.latent_dim
        hidden = config.hidden_dim
        out_node = config.in_node_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden // 2),  # 32 -> 32
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden // 2, out_node),
        )

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        return self.decoder(h_v)


class GTAEModel(nn.Module):
    """
    Graph Transformer-Based Autoencoder for Network Intrusion Detection (GTAE-IDS).

    Architecture:
    - Graph Transformer Encoder: x [N, 16] + edge_attr [E, 81] -> h_v [N, 32]
    - Flow Latent MLP: [h_src, h_dst, edge_attr] -> z_e [E, 32]
    - Edge Decoder: z_e [E, 32] -> edge_recon [E, 81]
    - Node Decoder: h_v [N, 32] -> node_recon [N, 16]
    """

    def __init__(self, config: Optional[GTAEConfig] = None):
        super().__init__()
        self.config = config or GTAEConfig()

        self.encoder = GraphTransformerEncoder(self.config)
        self.flow_latent_mlp = FlowLatentMLP(self.config)
        self.edge_decoder = EdgeDecoder(self.config)
        self.node_decoder = NodeDecoder(self.config)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GTAE.
        Returns:
            dict containing:
                'node_latent': [N, 32]
                'edge_latent': [E, 32]
                'node_recon':  [N, 16]
                'edge_recon':  [E, 81]
        """
        # 1. Encode nodes via Graph Transformer message passing
        h_v = self.encoder(x, edge_index, edge_attr)

        # 2. Extract endpoints for every edge
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        h_src = h_v[src_idx]
        h_dst = h_v[dst_idx]

        # 3. Produce latent edge/flow representations z_e
        z_e = self.flow_latent_mlp(h_src, h_dst, edge_attr)

        # 4. Decode edges and nodes
        edge_recon = self.edge_decoder(z_e)
        node_recon = self.node_decoder(h_v)

        return {
            "node_latent": h_v,
            "edge_latent": z_e,
            "node_recon": node_recon,
            "edge_recon": edge_recon,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute self-supervised reconstruction loss (zero ground-truth labels used).
        Loss = Edge_Loss + alpha * Node_Loss.
        """
        node_recon = outputs["node_recon"]
        edge_recon = outputs["edge_recon"]

        if self.config.loss_type == "mse":
            edge_loss = F.mse_loss(edge_recon, edge_attr)
            node_loss = F.mse_loss(node_recon, x)
        else:  # smooth_l1 / Huber
            edge_loss = F.smooth_l1_loss(edge_recon, edge_attr, beta=1.0)
            node_loss = F.smooth_l1_loss(node_recon, x, beta=1.0)

        total_loss = edge_loss + self.config.node_loss_weight * node_loss
        return total_loss, edge_loss, node_loss

    @staticmethod
    def extract_anomaly_features(
        edge_latent: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_recon: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract flow representations and reconstruction errors for downstream anomaly detectors:
        f_e = [z_e, |edge_attr - edge_recon|] in R^{E x 113}.
        Also returns scalar reconstruction error: Mean Absolute Error per flow.
        """
        abs_recon_error = torch.abs(edge_attr - edge_recon)  # [E, 81]
        scalar_recon_error = abs_recon_error.mean(dim=-1)     # [E]

        combined_features = torch.cat([edge_latent, abs_recon_error], dim=-1)  # [E, 113]

        return {
            "flow_features_for_detectors": combined_features,
            "scalar_recon_error": scalar_recon_error,
            "abs_recon_error_vector": abs_recon_error,
        }
