"""
Feature Encoders for the Temporal Graph Model.

Handles temporal encoding for edge features and categorical embedding definitions.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

class EdgeEncoder(nn.Module):
    """
    Encodes raw edge features into a 22-dimensional vector.
    """
    def __init__(self):
        super().__init__()
        # 3 banks -> 9 possible pairs, embedding size 4
        self.bank_pair_emb = nn.Embedding(9, 4)
        # 5 transaction types -> embedding size 4
        self.tx_type_emb = nn.Embedding(5, 4)
        # Normalize numeric edge features (1 + 8 + 1 + 1 + 1 + 1 + 1 = 14 dims)
        self.numeric_norm = nn.LayerNorm(14)
        self.output_dim = 22

    def forward(
        self,
        log_amount,
        ts_encodings,
        bank_pairs,
        tx_types,
        country_pair_risks,
        time_since_prevs,
        time_gap_between_edges,
        rolling_tx_count_7d,
        rolling_tx_count_30d,
    ):
        """
        Args:
            log_amount: (E, 1) float tensor
            ts_encodings: (E, 8) float tensor
            bank_pairs: (E,) long tensor (0-8)
            tx_types: (E,) long tensor (0-4)
            country_pair_risks: (E, 1) float tensor
            time_since_prevs: (E, 1) float tensor
            time_gap_between_edges: (E, 1) float tensor
            rolling_tx_count_7d: (E, 1) float tensor
            rolling_tx_count_30d: (E, 1) float tensor
        Returns:
            (E, 22) float tensor
        """
        bank_emb = self.bank_pair_emb(bank_pairs)  # (E, 4)
        tx_emb = self.tx_type_emb(tx_types)        # (E, 4)

        numeric = torch.cat([
            log_amount,             # 1
            ts_encodings,           # 8
            country_pair_risks,     # 1
            time_since_prevs,       # 1
            time_gap_between_edges, # 1
            rolling_tx_count_7d,    # 1
            rolling_tx_count_30d,   # 1
        ], dim=-1)                  # Total = 14
        numeric = self.numeric_norm(numeric)

        return torch.cat([
            numeric,                # 14
            bank_emb,               # 4
            tx_emb                  # 4
        ], dim=-1)                  # Total = 22


def encode_timestamp(unix_ts: float, window_start: float, window_end: float, log_time_since_prev: float = 0.0) -> np.ndarray:
    """
    Encode a Unix timestamp into an 8-dimensional temporal feature vector.
    """
    dt = datetime.fromtimestamp(unix_ts)
    
    # Time-of-day (sinusoidal, 2d)
    hour_frac = (dt.hour + dt.minute / 60.0) / 24.0
    tod_sin = math.sin(2 * math.pi * hour_frac)
    tod_cos = math.cos(2 * math.pi * hour_frac)
    
    # Day-of-week (sinusoidal, 2d)
    dow_frac = dt.weekday() / 7.0
    dow_sin = math.sin(2 * math.pi * dow_frac)
    dow_cos = math.cos(2 * math.pi * dow_frac)
    
    # Day-of-month (sinusoidal, 2d)
    dom_frac = dt.day / 31.0
    dom_sin = math.sin(2 * math.pi * dom_frac)
    dom_cos = math.cos(2 * math.pi * dom_frac)
    
    # Position within window (1d)
    window_pos = (unix_ts - window_start) / (window_end - window_start + 1e-8)
    window_pos = max(0.0, min(1.0, window_pos))
    
    return np.array([
        tod_sin, tod_cos,          # 2d
        dow_sin, dow_cos,          # 2d
        dom_sin, dom_cos,          # 2d
        window_pos,                # 1d
        log_time_since_prev        # 1d
    ], dtype=np.float32)
