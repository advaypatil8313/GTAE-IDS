"""Feature extraction functions for GTAE-IDS graph nodes and edges."""

from typing import List, Tuple
import numpy as np
import pandas as pd
import torch

from src.preprocessing.schema import FLOW_FEATURE_COLUMNS


def encode_protocols(protocol_series: pd.Series) -> np.ndarray:
    """
    Deterministic one-hot encoding for transport layer protocols:
    - Index 0: TCP (Protocol 6)
    - Index 1: UDP (Protocol 17)
    - Index 2: ICMP / ICMPv6 (Protocol 1 or 58)
    - Index 3: Other protocols (e.g. SCTP 132, IGMP 2, OSPF 89, etc.)
    """
    proto = protocol_series.to_numpy(dtype=np.int32)
    n = len(proto)
    one_hot = np.zeros((n, 4), dtype=np.float32)

    one_hot[proto == 6, 0] = 1.0
    one_hot[proto == 17, 1] = 1.0
    one_hot[(proto == 1) | (proto == 58), 2] = 1.0
    one_hot[(proto != 6) & (proto != 17) & (proto != 1) & (proto != 58), 3] = 1.0

    return one_hot


def construct_edge_attributes(window_df: pd.DataFrame) -> torch.Tensor:
    """
    Construct 81-dimensional edge attributes from:
    - 76 approved continuous flow metrics (float32)
    - 1 normalized destination port: DstPort / 65535.0 (float32 in [0, 1])
    - 4 deterministic protocol one-hot indicators (float32)
    """
    # 1. 76 continuous flow metrics
    flow_features = window_df[FLOW_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # 2. 1 normalized destination port
    dst_ports = (window_df["DstPort"].to_numpy(dtype=np.float32) / 65535.0).reshape(-1, 1)

    # 3. 4 protocol one-hot indicators
    proto_one_hot = encode_protocols(window_df["Protocol"])

    # Concatenate: shape [num_edges, 81]
    edge_attr_np = np.hstack([flow_features, dst_ports, proto_one_hot])

    # Final safety check for NaN / Inf
    if not np.all(np.isfinite(edge_attr_np)):
        edge_attr_np = np.nan_to_num(edge_attr_np, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.tensor(edge_attr_np, dtype=torch.float32)


def construct_node_features(window_df: pd.DataFrame, node_list: List[str]) -> torch.Tensor:
    """
    Construct 16-dimensional node feature matrix X in R^{|V_t| x 16}.
    Every feature is strictly derived from incident flow behavior and window topology.
    ZERO ground-truth attack or label annotations are used.

    Feature Definitions:
    0: Normalized In-Degree (incoming connection count / max partners)
    1: Normalized Out-Degree (outgoing connection count / max partners)
    2: Normalized Total Degree (total flow involvement / 2*max partners)
    3: Log10 Total Forward Bytes Sent: log10(1 + sum(Total Length of Fwd Packet on Out))
    4: Log10 Total Backward Bytes Sent: log10(1 + sum(Total Length of Bwd Packet on Out))
    5: Log10 Total Forward Packets Sent: log10(1 + sum(Tot Fwd Pkts on Out))
    6: Log10 Total Backward Packets Sent: log10(1 + sum(Tot Bwd Pkts on Out))
    7: Mean Packet Size Sent: mean of Average Packet Size on Out (0 if none)
    8: Mean Packet Size Received: mean of Average Packet Size on In (0 if none)
    9: Mean Flow Duration Sent: mean of Flow Duration on Out (0 if none)
    10: Mean Flow Duration Received: mean of Flow Duration on In (0 if none)
    11: SYN/RST Ratio Sent: (SYN sent) / (1 + SYN sent + RST sent)
    12: Protocol Diversity: (unique protocols used) / 5.0
    13: Destination Port Diversity: log10(1 + unique DstPorts contacted)
    14: Unique Peer Diversity: (unique communication partners) / max partners
    15: RST Ratio Sent: (RST sent) / (1 + forward packets sent)
    """
    num_nodes = len(node_list)
    max_partners = max(1, num_nodes - 1)

    # Create IP to local index mapping
    ip_to_idx = {ip: idx for idx, ip in enumerate(node_list)}
    src_indices = window_df["SrcIP"].map(ip_to_idx).to_numpy()
    dst_indices = window_df["DstIP"].map(ip_to_idx).to_numpy()

    # Pre-extract numpy columns for speed
    fwd_bytes = window_df["Total Length of Fwd Packet"].to_numpy(dtype=np.float32)
    bwd_bytes = window_df["Total Length of Bwd Packet"].to_numpy(dtype=np.float32)
    fwd_pkts = window_df["Tot Fwd Pkts"].to_numpy(dtype=np.float32)
    bwd_pkts = window_df["Tot Bwd Pkts"].to_numpy(dtype=np.float32)
    avg_pkt_size = window_df["Average Packet Size"].to_numpy(dtype=np.float32)
    duration = window_df["Flow Duration"].to_numpy(dtype=np.float32)
    syn_cnt = window_df["SYN Flag Cnt"].to_numpy(dtype=np.float32)
    rst_cnt = window_df["RST Flag Cnt"].to_numpy(dtype=np.float32)
    dst_ports = window_df["DstPort"].to_numpy(dtype=np.int32)
    protocols = window_df["Protocol"].to_numpy(dtype=np.int32)

    # Initialize node feature arrays
    in_deg = np.zeros(num_nodes, dtype=np.float32)
    out_deg = np.zeros(num_nodes, dtype=np.float32)
    sum_fwd_bytes = np.zeros(num_nodes, dtype=np.float32)
    sum_bwd_bytes = np.zeros(num_nodes, dtype=np.float32)
    sum_fwd_pkts = np.zeros(num_nodes, dtype=np.float32)
    sum_bwd_pkts = np.zeros(num_nodes, dtype=np.float32)
    sum_pkt_size_sent = np.zeros(num_nodes, dtype=np.float32)
    sum_pkt_size_recv = np.zeros(num_nodes, dtype=np.float32)
    sum_dur_sent = np.zeros(num_nodes, dtype=np.float32)
    sum_dur_recv = np.zeros(num_nodes, dtype=np.float32)
    sum_syn_sent = np.zeros(num_nodes, dtype=np.float32)
    sum_rst_sent = np.zeros(num_nodes, dtype=np.float32)

    # Accumulate using np.add.at (fast vectorized unbuffered accumulation)
    np.add.at(out_deg, src_indices, 1.0)
    np.add.at(in_deg, dst_indices, 1.0)

    np.add.at(sum_fwd_bytes, src_indices, fwd_bytes)
    np.add.at(sum_bwd_bytes, src_indices, bwd_bytes)
    np.add.at(sum_fwd_pkts, src_indices, fwd_pkts)
    np.add.at(sum_bwd_pkts, src_indices, bwd_pkts)

    np.add.at(sum_pkt_size_sent, src_indices, avg_pkt_size)
    np.add.at(sum_pkt_size_recv, dst_indices, avg_pkt_size)

    np.add.at(sum_dur_sent, src_indices, duration)
    np.add.at(sum_dur_recv, dst_indices, duration)

    np.add.at(sum_syn_sent, src_indices, syn_cnt)
    np.add.at(sum_rst_sent, src_indices, rst_cnt)

    # Diversity metrics per node (protocol, destination ports, unique peers)
    proto_div = np.zeros(num_nodes, dtype=np.float32)
    port_div = np.zeros(num_nodes, dtype=np.float32)
    peer_div = np.zeros(num_nodes, dtype=np.float32)

    # Groupby for distinct set sizes
    out_ports_by_src = window_df.groupby("SrcIP")["DstPort"].nunique()
    for ip, cnt in out_ports_by_src.items():
        if ip in ip_to_idx:
            port_div[ip_to_idx[ip]] = float(cnt)

    # Protocol diversity across both src and dst roles
    src_protos = window_df.groupby("SrcIP")["Protocol"].apply(set)
    dst_protos = window_df.groupby("DstIP")["Protocol"].apply(set)
    for ip in node_list:
        p_set = set()
        if ip in src_protos:
            p_set.update(src_protos[ip])
        if ip in dst_protos:
            p_set.update(dst_protos[ip])
        proto_div[ip_to_idx[ip]] = len(p_set)

    # Peer diversity (unique communication partners)
    out_peers = window_df.groupby("SrcIP")["DstIP"].apply(set)
    in_peers = window_df.groupby("DstIP")["SrcIP"].apply(set)
    for ip in node_list:
        peers = set()
        if ip in out_peers:
            peers.update(out_peers[ip])
        if ip in in_peers:
            peers.update(in_peers[ip])
        peer_div[ip_to_idx[ip]] = len(peers)

    # Compute derived features with safe division
    safe_out_deg = np.maximum(1.0, out_deg)
    safe_in_deg = np.maximum(1.0, in_deg)

    mean_pkt_sent = np.where(out_deg > 0, sum_pkt_size_sent / safe_out_deg, 0.0)
    mean_pkt_recv = np.where(in_deg > 0, sum_pkt_size_recv / safe_in_deg, 0.0)
    mean_dur_sent = np.where(out_deg > 0, sum_dur_sent / safe_out_deg, 0.0)
    mean_dur_recv = np.where(in_deg > 0, sum_dur_recv / safe_in_deg, 0.0)

    # Build matrix
    X = np.column_stack([
        in_deg / max_partners,                                           # 0: in-degree
        out_deg / max_partners,                                          # 1: out-degree
        (in_deg + out_deg) / (2.0 * max_partners),                       # 2: total degree
        np.log10(1.0 + np.maximum(0.0, sum_fwd_bytes)),                   # 3: fwd bytes
        np.log10(1.0 + np.maximum(0.0, sum_bwd_bytes)),                   # 4: bwd bytes
        np.log10(1.0 + np.maximum(0.0, sum_fwd_pkts)),                    # 5: fwd pkts
        np.log10(1.0 + np.maximum(0.0, sum_bwd_pkts)),                    # 6: bwd pkts
        mean_pkt_sent,                                                   # 7: mean pkt size sent
        mean_pkt_recv,                                                   # 8: mean pkt size recv
        mean_dur_sent,                                                   # 9: mean duration sent
        mean_dur_recv,                                                   # 10: mean duration recv
        sum_syn_sent / (1.0 + sum_syn_sent + sum_rst_sent),              # 11: SYN/RST ratio
        proto_div / 5.0,                                                 # 12: protocol diversity
        np.log10(1.0 + port_div),                                        # 13: port diversity
        peer_div / max_partners,                                         # 14: peer diversity
        sum_rst_sent / (1.0 + sum_fwd_pkts),                             # 15: RST ratio
    ]).astype(np.float32)

    # Ensure strictly finite
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.tensor(X, dtype=torch.float32)


def construct_node_features_from_edges(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Construct 16-dimensional node feature matrix X in R^{num_nodes x 16} directly from
    raw (unscaled) edge_index and edge_attr tensors.

    Semantics strictly match construct_node_features():
    0: Normalized In-Degree (in_deg / max_partners)
    1: Normalized Out-Degree (out_deg / max_partners)
    2: Normalized Total Degree ((in_deg + out_deg) / (2 * max_partners))
    3: Log10 Total Forward Bytes Sent: log10(1 + sum(Total Length of Fwd Packet on Out))
    4: Log10 Total Backward Bytes Sent: log10(1 + sum(Total Length of Bwd Packet on Out))
    5: Log10 Total Forward Packets Sent: log10(1 + sum(Tot Fwd Pkts on Out))
    6: Log10 Total Backward Packets Sent: log10(1 + sum(Tot Bwd Pkts on Out))
    7: Mean Packet Size Sent: mean of Average Packet Size on Out (0 if out_deg == 0)
    8: Mean Packet Size Received: mean of Average Packet Size on In (0 if in_deg == 0)
    9: Mean Flow Duration Sent: mean of Flow Duration on Out (0 if out_deg == 0)
    10: Mean Flow Duration Received: mean of Flow Duration on In (0 if in_deg == 0)
    11: SYN/RST Ratio Sent: (SYN sent) / (1 + SYN sent + RST sent)
    12: Protocol Diversity: (unique protocols used) / 5.0
    13: Destination Port Diversity: log10(1 + unique DstPorts contacted on Out)
    14: Unique Peer Diversity: (unique communication partners) / max_partners
    15: RST Ratio Sent: (RST sent) / (1 + forward packets sent)

    Nodes with no incident edges in edge_index (e.g. isolated nodes after filtering)
    deterministically receive all 0.0 values, preserving snapshot node indexing.
    """
    max_partners = max(1, num_nodes - 1)
    num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0

    if num_edges == 0:
        return torch.zeros((num_nodes, 16), dtype=torch.float32)

    src_indices = edge_index[0].cpu().numpy()
    dst_indices = edge_index[1].cpu().numpy()
    edge_np = edge_attr.cpu().numpy()

    # Raw column indices according to FLOW_FEATURE_COLUMNS and construct_edge_attributes:
    # 0: Flow Duration
    # 3: Tot Fwd Pkts
    # 4: Tot Bwd Pkts
    # 5: Total Length of Fwd Packet
    # 6: Total Length of Bwd Packet
    # 43: SYN Flag Cnt
    # 44: RST Flag Cnt
    # 51: Average Packet Size
    # 76: DstPort / 65535.0 (recovers integer port via round(val * 65535.0))
    # 77:81: Protocol one-hot (TCP=77, UDP=78, ICMP=79, Other=80)
    duration = edge_np[:, 0]
    fwd_pkts = edge_np[:, 3]
    bwd_pkts = edge_np[:, 4]
    fwd_bytes = edge_np[:, 5]
    bwd_bytes = edge_np[:, 6]
    syn_cnt = edge_np[:, 43]
    rst_cnt = edge_np[:, 44]
    avg_pkt_size = edge_np[:, 51]
    norm_dst_ports = edge_np[:, 76]
    proto_one_hot = edge_np[:, 77:81]

    # Initialize node feature arrays
    in_deg = np.zeros(num_nodes, dtype=np.float32)
    out_deg = np.zeros(num_nodes, dtype=np.float32)
    sum_fwd_bytes = np.zeros(num_nodes, dtype=np.float32)
    sum_bwd_bytes = np.zeros(num_nodes, dtype=np.float32)
    sum_fwd_pkts = np.zeros(num_nodes, dtype=np.float32)
    sum_bwd_pkts = np.zeros(num_nodes, dtype=np.float32)
    sum_pkt_size_sent = np.zeros(num_nodes, dtype=np.float32)
    sum_pkt_size_recv = np.zeros(num_nodes, dtype=np.float32)
    sum_dur_sent = np.zeros(num_nodes, dtype=np.float32)
    sum_dur_recv = np.zeros(num_nodes, dtype=np.float32)
    sum_syn_sent = np.zeros(num_nodes, dtype=np.float32)
    sum_rst_sent = np.zeros(num_nodes, dtype=np.float32)

    # Accumulate features using unbuffered vectorized additions
    np.add.at(out_deg, src_indices, 1.0)
    np.add.at(in_deg, dst_indices, 1.0)

    np.add.at(sum_fwd_bytes, src_indices, fwd_bytes)
    np.add.at(sum_bwd_bytes, src_indices, bwd_bytes)
    np.add.at(sum_fwd_pkts, src_indices, fwd_pkts)
    np.add.at(sum_bwd_pkts, src_indices, bwd_pkts)

    np.add.at(sum_pkt_size_sent, src_indices, avg_pkt_size)
    np.add.at(sum_pkt_size_recv, dst_indices, avg_pkt_size)

    np.add.at(sum_dur_sent, src_indices, duration)
    np.add.at(sum_dur_recv, dst_indices, duration)

    np.add.at(sum_syn_sent, src_indices, syn_cnt)
    np.add.at(sum_rst_sent, src_indices, rst_cnt)

    # Destination Port Diversity (unique DstPorts contacted by node as source)
    dst_port_ints = np.round(norm_dst_ports * 65535.0).astype(np.int32)
    port_div = np.zeros(num_nodes, dtype=np.float32)
    ports_by_src: dict = {}
    for s, p in zip(src_indices, dst_port_ints):
        if s not in ports_by_src:
            ports_by_src[s] = set()
        ports_by_src[s].add(p)
    for s, pset in ports_by_src.items():
        port_div[s] = float(len(pset))

    # Protocol Diversity (unique protocols used by node as source or destination)
    proto_ids = np.argmax(proto_one_hot, axis=1)
    proto_div = np.zeros(num_nodes, dtype=np.float32)
    protos_by_node = [set() for _ in range(num_nodes)]
    for s, d, p in zip(src_indices, dst_indices, proto_ids):
        protos_by_node[s].add(p)
        protos_by_node[d].add(p)
    for i in range(num_nodes):
        proto_div[i] = float(len(protos_by_node[i]))

    # Peer Diversity (unique communication partners as source or destination)
    peers_by_node = [set() for _ in range(num_nodes)]
    for s, d in zip(src_indices, dst_indices):
        peers_by_node[s].add(d)
        peers_by_node[d].add(s)
    peer_div = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        peer_div[i] = float(len(peers_by_node[i]))

    # Safe division for mean statistics
    safe_out_deg = np.maximum(1.0, out_deg)
    safe_in_deg = np.maximum(1.0, in_deg)

    mean_pkt_sent = np.where(out_deg > 0, sum_pkt_size_sent / safe_out_deg, 0.0)
    mean_pkt_recv = np.where(in_deg > 0, sum_pkt_size_recv / safe_in_deg, 0.0)
    mean_dur_sent = np.where(out_deg > 0, sum_dur_sent / safe_out_deg, 0.0)
    mean_dur_recv = np.where(in_deg > 0, sum_dur_recv / safe_in_deg, 0.0)

    # Assemble 16-dimensional matrix
    X = np.column_stack([
        in_deg / max_partners,                                           # 0: in-degree
        out_deg / max_partners,                                          # 1: out-degree
        (in_deg + out_deg) / (2.0 * max_partners),                       # 2: total degree
        np.log10(1.0 + np.maximum(0.0, sum_fwd_bytes)),                   # 3: fwd bytes
        np.log10(1.0 + np.maximum(0.0, sum_bwd_bytes)),                   # 4: bwd bytes
        np.log10(1.0 + np.maximum(0.0, sum_fwd_pkts)),                    # 5: fwd pkts
        np.log10(1.0 + np.maximum(0.0, sum_bwd_pkts)),                    # 6: bwd pkts
        mean_pkt_sent,                                                   # 7: mean pkt size sent
        mean_pkt_recv,                                                   # 8: mean pkt size recv
        mean_dur_sent,                                                   # 9: mean duration sent
        mean_dur_recv,                                                   # 10: mean duration recv
        sum_syn_sent / (1.0 + sum_syn_sent + sum_rst_sent),              # 11: SYN/RST ratio
        proto_div / 5.0,                                                 # 12: protocol diversity
        np.log10(1.0 + port_div),                                        # 13: port diversity
        peer_div / max_partners,                                         # 14: peer diversity
        sum_rst_sent / (1.0 + sum_fwd_pkts),                             # 15: RST ratio
    ]).astype(np.float32)

    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.tensor(X, dtype=torch.float32)

