"""
Temporal GraphSAGE model for detecting laundering patterns.

v2: Multi-task model with:
  - Node binary classifier (suspicious/clean)
  - 7-class pattern detection head (chain, fan_in, fan_out, burst, structuring, round_trip, mule)
  - Edge risk scoring head
  - trust_score = 1 - sigmoid(node_logit)
  - MC Dropout for confidence estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from config import NUM_PATTERNS, PATTERN_NAMES


class EdgeAwareSAGEConv(MessagePassing):
    """
    Custom GraphSAGE layer that incorporates multi-dimensional edge features.
    """
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int):
        super().__init__(aggr='mean')

        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, out], dim=-1)
        out = self.update_mlp(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class TemporalGraphSAGE(nn.Module):
    def __init__(self, node_dim: int = 130, edge_dim: int = 27, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        # Input normalization
        self.input_norm = nn.LayerNorm(node_dim)

        # Input projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # GNN layers
        self.conv1 = EdgeAwareSAGEConv(hidden_dim, hidden_dim, edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = EdgeAwareSAGEConv(hidden_dim, out_dim, edge_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.res2 = nn.Linear(hidden_dim, out_dim)

        # --- Task 1: Binary node classifier (suspicious / clean) ---
        self.node_classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim // 2, 1)
        )

        # --- Task 2: Pattern classification (7 patterns, multi-label) ---
        self.pattern_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, NUM_PATTERNS)  # 7 outputs
        )

        # --- Task 3: Edge risk scorer ---
        # Edge embedding = concat(src_emb, dst_emb, edge_attr)
        self.edge_risk_head = nn.Sequential(
            nn.Linear(out_dim * 2 + edge_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )

    def _encode(self, x, edge_index, edge_attr):
        """Shared encoder — produces node embeddings."""
        x = self.input_norm(x)
        x = self.node_proj(x)
        x = F.relu(x)

        # Layer 1
        h1 = self.conv1(x, edge_index, edge_attr)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        x = x + h1

        # Layer 2
        h2 = self.conv2(x, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        x = self.res2(x) + h2
        return x

    def forward(self, x, edge_index, edge_attr, return_embeddings: bool = False):
        """
        Standard forward pass — returns binary node logits only (for backward compatibility).
        """
        emb = self._encode(x, edge_index, edge_attr)
        logits = self.node_classifier(emb).squeeze(-1)
        if return_embeddings:
            return logits, emb
        return logits

    def forward_multitask(self, x, edge_index, edge_attr):
        """
        Multi-task forward — returns all heads.

        Returns:
            node_logits:    [N] — binary suspicious logit
            pattern_logits: [N, 7] — pattern logits (multi-label)
            edge_risk:      [E] — per-edge risk logit
        """
        emb = self._encode(x, edge_index, edge_attr)

        # Task 1
        node_logits = self.node_classifier(emb).squeeze(-1)

        # Task 2: pattern probabilities per node
        pattern_logits = self.pattern_head(emb)

        # Task 3: edge risk = concat(src_emb, dst_emb, edge_feat)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        src_emb = emb[src_idx]
        dst_emb = emb[dst_idx]
        edge_combined = torch.cat([src_emb, dst_emb, edge_attr], dim=-1)
        edge_risk = self.edge_risk_head(edge_combined).squeeze(-1)

        return node_logits, pattern_logits, edge_risk

    def predict_full(self, x, edge_index, edge_attr):
        """
        Full inference output: trust scores, risk scores, pattern probs, edge risks.

        Returns dict with all tensors on same device.
        """
        self.eval()
        with torch.no_grad():
            node_logits, pattern_logits, edge_risk = self.forward_multitask(x, edge_index, edge_attr)

        risk_score = torch.sigmoid(node_logits)          # [N]
        trust_score = 1.0 - risk_score                   # [N]
        pattern_probs = torch.sigmoid(pattern_logits)    # [N, 7]
        tx_risk_score = torch.sigmoid(edge_risk)         # [E]

        return {
            "risk_score": risk_score,
            "trust_score": trust_score,
            "pattern_probs": pattern_probs,
            "tx_risk_score": tx_risk_score,
            "pattern_names": PATTERN_NAMES,
        }

    def predict_with_confidence(self, x, edge_index, edge_attr, num_passes: int = 10):
        """
        Monte Carlo Dropout for uncertainty estimation.
        Returns mean_prob and confidence (1 - std).
        """
        self.train()  # Force dropout active

        predictions = []
        with torch.no_grad():
            for _ in range(num_passes):
                logits = self.forward(x, edge_index, edge_attr)
                probs = torch.sigmoid(logits)
                predictions.append(probs)

        self.eval()

        preds = torch.stack(predictions, dim=0)
        mean_prob = preds.mean(dim=0)
        std_prob = preds.std(dim=0)
        confidence = 1.0 - std_prob

        return mean_prob, confidence
