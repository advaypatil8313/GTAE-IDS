"""Dataset schema definitions and column classifications for LSPR23."""

import numpy as np

# 1. Graph Topology / Network Endpoint Columns (retained for graph nodes/edges, excluded from ML feature matrix X)
GRAPH_ENDPOINT_COLUMNS = [
    "SrcIP",
    "DstIP",
    "SrcPort",
    "DstPort",
    "Protocol",
]

# 2. Temporal Columns (retained for dynamic windowing and temporal snapshot construction)
TEMPORAL_COLUMNS = [
    "mTimestampStart",
    "mTimestampLast",
    "Flow Duration",
]

# 3. Ground Truth Target (binary target: 0 = Benign, 1 = Malicious)
TARGET_COLUMN = "Label"

# 4. Target Leakage & Metadata Exclusions (strictly forbidden from entering the ML feature matrix X)
EXCLUDED_LEAKAGE_COLUMNS = [
    "Flow ID",          # Arbitrary string identifier
    "SigID revision",   # IDS alert signature metadata
    "Category",         # Multi-class ground-truth attack category
    "Severity",         # Ground-truth attack severity level
    "Anomaly_event",    # Exercise alert indicator
    "L3/L4 Protocol",   # Redundant/range protocol classification
    "Int/Ext Dst IP",   # Range subnet boundary indicator
    "Conn_state",       # Zeek connection state string
    "Service",          # Zeek service application string
    "Label_src",        # Source host exercise role (Red/Blue/Green team)
    "Label_dst",        # Destination host exercise role
    "External_src",     # Range boundary crossing indicator
    "External_dst",     # Range boundary crossing indicator
    "Segment_src",      # Subnet segment identifier
    "Segment_dst",      # Subnet segment identifier
    "Expoid_src",       # Exposure identifier
    "Expoid_dst",       # Exposure identifier
]

# 5. Core Numerical Flow Features (75 features used as model inputs X)
FLOW_FEATURE_COLUMNS = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Fwd Packet Length Min",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Min",
    "Flow IAT Max",
    "Flow IAT Stddev",
    "Fwd IAT Min",
    "Fwd IAT Max",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Tot",
    "Bwd IAT Min",
    "Bwd IAT Max",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Tot",
    "Fwd PSH flags",
    "Bwd PSH flags",
    "Fwd URG flags",
    "Bwd URG flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWR Flag Cnt",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Average Packet Size",
    "Fwd Segment Size Avg",
    "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Min",
    "Active Mean",
    "Active Max",
    "Active Std",
    "Idle Min",
    "Idle Mean",
    "Idle Max",
    "Idle Std",
]

# 6. Dtype Downcasting Specifications (ensuring memory efficiency)
DTYPE_MAPPING = {
    # Endpoints
    "SrcPort": np.uint16,
    "DstPort": np.uint16,
    "Protocol": np.uint8,
    # Timestamps (microseconds since epoch)
    "mTimestampStart": np.int64,
    "mTimestampLast": np.int64,
    # Target
    "Label": np.int8,
}
