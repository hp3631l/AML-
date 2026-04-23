"""
Training loop for Temporal GraphSAGE model.

Includes:
    - data diagnostics
    - focal-loss multitask learning (node + edge)
    - threshold tuning
    - Platt scaling calibration
    - tabular baseline ablation
"""

import json
import math
import os
import sqlite3
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Adam

try:
    from torch_geometric.loader import NeighborLoader
except ImportError:
    print("Warning: torch_geometric is not installed. Please install it to train the model.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    BANK_DB_PATHS,
    BATCH_SIZE,
    CLIP_GRAD_NORM,
    DROPOUT,
    GRAPHSAGE_HIDDEN_DIM,
    GRAPHSAGE_NUM_NEIGHBORS,
    GRAPHSAGE_OUT_DIM,
    LEARNING_RATE,
    MC_DROPOUT_PASSES,
    MODEL_CHECKPOINT_DIR,
    WEIGHT_DECAY,
)
from model.data_prep import build_pyg_graph  # noqa: E402
from model.encoder import EdgeEncoder  # noqa: E402
from model.gnn import TemporalGraphSAGE  # noqa: E402


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * torch.pow((1 - p_t).clamp(min=1e-6), self.gamma) * bce
        return focal.mean()


class AMLModel(nn.Module):
    """Wrapper that combines EdgeEncoder + TemporalGraphSAGE + edge head."""

    def __init__(self, node_dim: int):
        super().__init__()
        self.edge_encoder = EdgeEncoder()
        self.gnn = TemporalGraphSAGE(
            node_dim=node_dim,
            edge_dim=self.edge_encoder.output_dim,
            hidden_dim=GRAPHSAGE_HIDDEN_DIM,
            out_dim=GRAPHSAGE_OUT_DIM,
            dropout=DROPOUT,
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear((GRAPHSAGE_OUT_DIM * 2) + self.edge_encoder.output_dim, GRAPHSAGE_OUT_DIM),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(GRAPHSAGE_OUT_DIM, 1),
        )
        self.node_classifier = nn.Sequential(
            nn.Linear(GRAPHSAGE_OUT_DIM + 8, GRAPHSAGE_OUT_DIM),
            nn.BatchNorm1d(GRAPHSAGE_OUT_DIM),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(GRAPHSAGE_OUT_DIM, 1),
        )

    def _aggregate_edge_outputs(
        self,
        edge_probs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_unix_ts: torch.Tensor,
        num_nodes: int,
        global_edge_probs: torch.Tensor = None,
        global_edge_index: torch.Tensor = None,
        global_edge_unix_ts: torch.Tensor = None,
    ) -> torch.Tensor:
        """Aggregate edge-level risk signals per node.
        
        If global_edge_probs/index/ts are provided, aggregate over the FULL graph
        edge history first, then override with batch-local values where available.
        This ensures every node sees its complete 90-day edge risk history.
        """
        # Use global graph if provided, otherwise fall back to batch subgraph
        if global_edge_probs is not None:
            src = global_edge_index[0]
            dst = global_edge_index[1]
            probs = torch.cat([global_edge_probs, global_edge_probs], dim=0)
            nodes = torch.cat([src, dst], dim=0)
            ts = torch.cat([global_edge_unix_ts, global_edge_unix_ts], dim=0).float()
        else:
            src = edge_index[0]
            dst = edge_index[1]
            nodes = torch.cat([src, dst], dim=0)
            probs = torch.cat([edge_probs, edge_probs], dim=0)
            ts = torch.cat([edge_unix_ts, edge_unix_ts], dim=0).float()

        if probs.numel() == 0:
            return torch.zeros((num_nodes, 8), device=edge_index.device, dtype=torch.float32)

        now_ts = ts.max()
        age_sec = (now_ts - ts).clamp(min=0.0)
        decay = torch.exp(-age_sec / (30.0 * 86400.0))

        ones = torch.ones_like(probs)
        cnt = torch.zeros(num_nodes, device=probs.device, dtype=probs.dtype)
        cnt.index_add_(0, nodes, ones)

        sum_prob = torch.zeros_like(cnt)
        sum_prob.index_add_(0, nodes, probs)
        mean_prob = sum_prob / (cnt + 1e-6)

        max_prob = torch.full((num_nodes,), -1e9, device=probs.device, dtype=probs.dtype)
        max_prob.scatter_reduce_(0, nodes, probs, reduce="amax", include_self=True)
        max_prob = torch.where(max_prob < -1e8, torch.zeros_like(max_prob), max_prob)

        high_mask = (probs >= 0.70).float()
        count_high = torch.zeros_like(cnt)
        count_high.index_add_(0, nodes, high_mask)
        count_high = torch.log1p(count_high)

        weighted_sum = torch.zeros_like(cnt)
        weighted_sum.index_add_(0, nodes, probs * decay)
        weight_sum = torch.zeros_like(cnt)
        weight_sum.index_add_(0, nodes, decay)
        decay_weighted = weighted_sum / (weight_sum + 1e-6)

        last30_mask = (age_sec <= (30.0 * 86400.0)).float()
        suspicious_edge_count_last_30d = torch.zeros_like(cnt)
        suspicious_edge_count_last_30d.index_add_(0, nodes, ((probs >= 0.70).float() * last30_mask))
        suspicious_edge_count_last_30d = torch.log1p(suspicious_edge_count_last_30d)

        max_edge_risk_last_30d_vals = torch.where(last30_mask > 0.5, probs, torch.full_like(probs, -1e9))
        max_edge_risk_last_30d = torch.full((num_nodes,), -1e9, device=probs.device, dtype=probs.dtype)
        max_edge_risk_last_30d.scatter_reduce_(0, nodes, max_edge_risk_last_30d_vals, reduce="amax", include_self=True)
        max_edge_risk_last_30d = torch.where(
            max_edge_risk_last_30d < -1e8, torch.zeros_like(max_edge_risk_last_30d), max_edge_risk_last_30d
        )

        sum_edge_risk_last_30d = torch.zeros_like(cnt)
        sum_edge_risk_last_30d.index_add_(0, nodes, probs * last30_mask)
        cnt_edge_last_30d = torch.zeros_like(cnt)
        cnt_edge_last_30d.index_add_(0, nodes, last30_mask)
        avg_edge_risk_last_30d = sum_edge_risk_last_30d / (cnt_edge_last_30d + 1e-6)

        age_days = age_sec / 86400.0
        min_age_days = torch.full((num_nodes,), 9999.0, device=probs.device, dtype=probs.dtype)
        suspicious_mask = probs >= 0.70
        if suspicious_mask.any():
            min_age_days.scatter_reduce_(
                0,
                nodes[suspicious_mask],
                age_days[suspicious_mask],
                reduce="amin",
                include_self=True,
            )
        time_since_last_suspicious_edge = torch.log1p(min_age_days.clamp(max=90.0)) / math.log1p(90.0)

        agg = torch.stack([
            mean_prob,
            max_prob,
            count_high,
            decay_weighted,
            suspicious_edge_count_last_30d,
            max_edge_risk_last_30d,
            avg_edge_risk_last_30d,
            time_since_last_suspicious_edge,
        ], dim=-1)
        return agg

    def forward(
        self,
        x,
        edge_index,
        edge_log_amount,
        edge_ts_encodings,
        edge_bank_pairs,
        edge_tx_types,
        edge_country_risks,
        edge_time_since_prevs,
        edge_time_gap_between_edges,
        edge_rolling_tx_count_7d,
        edge_rolling_tx_count_30d,
        edge_unix_ts,
        global_edge_probs=None,
        global_edge_index=None,
        global_edge_unix_ts=None,
        full_num_nodes=None,
    ):
        edge_attr = self.edge_encoder(
            edge_log_amount,
            edge_ts_encodings,
            edge_bank_pairs,
            edge_tx_types,
            edge_country_risks,
            edge_time_since_prevs,
            edge_time_gap_between_edges,
            edge_rolling_tx_count_7d,
            edge_rolling_tx_count_30d,
        )
        node_logits, node_emb = self.gnn(x, edge_index, edge_attr, return_embeddings=True)
        src = edge_index[0]
        dst = edge_index[1]
        edge_input = torch.cat([node_emb[src], node_emb[dst], edge_attr], dim=-1)
        edge_logits = self.edge_classifier(edge_input).squeeze(-1)
        edge_probs_batch = torch.sigmoid(edge_logits)

        if global_edge_probs is not None and full_num_nodes is not None:
            # Aggregate over full graph, result shape [full_num_nodes, 8]
            edge_agg_full = self._aggregate_edge_outputs(
                edge_probs_batch.detach(),
                edge_index,
                edge_unix_ts,
                full_num_nodes,
                global_edge_probs=global_edge_probs,
                global_edge_index=global_edge_index,
                global_edge_unix_ts=global_edge_unix_ts,
            )
            # Extract rows for batch nodes using their global IDs (stored in batch.n_id)
            # n_id is set by NeighborLoader — maps batch local idx -> global idx
            # We store it as self._batch_n_id before calling forward via the loader
            if hasattr(self, '_batch_n_id') and self._batch_n_id is not None:
                edge_agg = edge_agg_full[self._batch_n_id]  # [batch_size, 8]
            else:
                # Fallback: use local batch aggregation
                edge_agg = self._aggregate_edge_outputs(
                    edge_probs_batch.detach(), edge_index, edge_unix_ts, x.size(0)
                )
        else:
            edge_agg = self._aggregate_edge_outputs(
                edge_probs_batch.detach(), edge_index, edge_unix_ts, x.size(0)
            )

        node_input = torch.cat([node_emb, edge_agg], dim=-1)
        node_logits = self.node_classifier(node_input).squeeze(-1)
        return node_logits, edge_logits

    def predict_with_confidence(self, *args, num_passes=MC_DROPOUT_PASSES):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(num_passes):
                node_logits, _ = self.forward(*args)
                preds.append(torch.sigmoid(node_logits))
        self.eval()
        stacked = torch.stack(preds)
        return stacked.mean(dim=0), 1.0 - stacked.std(dim=0)


def _safe_auc(labels, preds):
    try:
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, preds))
    except Exception:
        return float("nan")


def _safe_pr_auc(labels, preds):
    try:
        precision, recall, _ = precision_recall_curve(labels, preds)
        return float(auc(recall, precision))
    except Exception:
        return float("nan")


def _best_threshold(labels: np.ndarray, probs: np.ndarray, min_precision: float = 0.45) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(thresholds) == 0:
        return 0.5
    p = precision[:-1]
    r = recall[:-1]
    f1_scores = (2 * p * r) / (p + r + 1e-8)
    constrained = np.where((p >= min_precision) & (r >= 0.10))[0]
    if constrained.size > 0:
        best_local = constrained[int(np.argmax(f1_scores[constrained]))]
        return float(thresholds[best_local])
    return float(thresholds[int(np.argmax(f1_scores))])


def _metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (probs >= threshold).astype(int)
    return {
        "auc": _safe_auc(labels, probs),
        "pr_auc": _safe_pr_auc(labels, probs),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "brier": float(brier_score_loss(labels, probs)),
        "threshold": float(threshold),
    }


def _hist_compare(name: str, normal_vals: np.ndarray, suspicious_vals: np.ndarray, bins: int = 12):
    all_vals = np.concatenate([normal_vals, suspicious_vals]) if normal_vals.size and suspicious_vals.size else (
        normal_vals if normal_vals.size else suspicious_vals
    )
    if all_vals.size == 0:
        return {
            "normal_hist": [],
            "suspicious_hist": [],
            "intersection": float("nan"),
            "range_overlap": float("nan"),
            "flag": "no-data",
        }
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    bins_arr = np.linspace(vmin, vmax, bins + 1)
    hn, _ = np.histogram(normal_vals, bins=bins_arr)
    hs, _ = np.histogram(suspicious_vals, bins=bins_arr)
    hn_n = hn / max(1, hn.sum())
    hs_n = hs / max(1, hs.sum())
    intersection = float(np.minimum(hn_n, hs_n).sum())

    n_min = float(np.min(normal_vals)) if len(normal_vals) > 0 else 0.0
    n_max = float(np.max(normal_vals)) if len(normal_vals) > 0 else 0.0
    s_min = float(np.min(suspicious_vals)) if len(suspicious_vals) > 0 else 0.0
    s_max = float(np.max(suspicious_vals)) if len(suspicious_vals) > 0 else 0.0
    overlap = max(0.0, min(n_max, s_max) - max(n_min, s_min))
    union = max(n_max, s_max) - min(n_min, s_min) + 1e-8
    range_overlap = float(overlap / union)

    if range_overlap < 0.05:
        flag = "FLAG: fully-separable"
    elif intersection > 0.98:
        flag = "FLAG: nearly-identical"
    else:
        flag = "ok"

    print(f"[HIST] {name}")
    print(f"  normal:     {hn.tolist()}")
    print(f"  suspicious: {hs.tolist()}")
    print(f"  intersection={intersection:.4f} range_overlap={range_overlap:.4f} -> {flag}")

    return {
        "normal_hist": hn.tolist(),
        "suspicious_hist": hs.tolist(),
        "intersection": intersection,
        "range_overlap": range_overlap,
        "flag": flag,
    }


def run_data_diagnostics(data) -> Dict:
    print("\n=== STEP 1: DATA DIAGNOSTICS ===")
    node_y = data.y.cpu().numpy().astype(int)
    edge_y = data.edge_y.cpu().numpy().astype(int)
    num_nodes = int(data.num_nodes)
    num_edges = int(data.num_edges)
    node_pos = int(node_y.sum())
    node_neg = int(num_nodes - node_pos)
    edge_pos = int(edge_y.sum())
    edge_neg = int(num_edges - edge_pos)
    print(f"Accounts: normal={node_neg} suspicious={node_pos} ({node_pos / max(1, num_nodes):.2%} suspicious)")
    print(f"Edges:    normal={edge_neg} suspicious={edge_pos} ({edge_pos / max(1, num_edges):.2%} suspicious)")

    motif_counts = {}
    for _, db_path in BANK_DB_PATHS.items():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT motif_type, COUNT(*) as c FROM labels WHERE is_suspicious=1 GROUP BY motif_type"
        ).fetchall()
        conn.close()
        for r in rows:
            m = r["motif_type"] or "unknown"
            motif_counts[m] = motif_counts.get(m, 0) + int(r["c"])
    print("Motif counts (suspicious accounts):")
    for motif, c in sorted(motif_counts.items(), key=lambda x: -x[1]):
        print(f"  {motif:24s} {c}")

    tx_freq = data.node_tx_frequency.cpu().numpy()
    tx_freq_normal = tx_freq[node_y == 0]
    tx_freq_susp = tx_freq[node_y == 1]

    delay_hours = np.expm1(data.edge_time_since_prevs_raw.cpu().numpy().reshape(-1)) / 3600.0
    delay_normal = delay_hours[edge_y == 0]
    delay_susp = delay_hours[edge_y == 1]

    cross_country = data.edge_cross_country.cpu().numpy().reshape(-1)
    cc_normal = cross_country[edge_y == 0]
    cc_susp = cross_country[edge_y == 1]
    cc_rate_normal = float(cc_normal.mean()) if cc_normal.size else 0.0
    cc_rate_susp = float(cc_susp.mean()) if cc_susp.size else 0.0
    print(f"Country transitions: normal={cc_rate_normal:.4f} suspicious={cc_rate_susp:.4f}")

    hist_tx_freq = _hist_compare("tx_frequency_per_account_per_month", tx_freq_normal, tx_freq_susp)
    hist_delay = _hist_compare("delay_between_transactions_hours", delay_normal, delay_susp)
    hist_cross = _hist_compare("country_transition_indicator", cc_normal, cc_susp, bins=2)

    return {
        "class_distribution": {
            "accounts": {"normal": node_neg, "suspicious": node_pos},
            "edges": {"normal": edge_neg, "suspicious": edge_pos},
        },
        "motif_counts": motif_counts,
        "country_transition_rate": {"normal": cc_rate_normal, "suspicious": cc_rate_susp},
        "histograms": {
            "tx_frequency": hist_tx_freq,
            "delay_between_transactions": hist_delay,
            "country_transition": hist_cross,
        },
    }


def _build_balanced_input_nodes(train_idx: np.ndarray, labels: np.ndarray, batch_size: int) -> np.ndarray:
    pos = train_idx[labels[train_idx] == 1]
    neg = train_idx[labels[train_idx] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return train_idx

    pos_per = max(1, int(batch_size * 0.25))
    neg_per = max(1, batch_size - pos_per)
    n_batches = max(1, int(math.ceil(len(train_idx) / batch_size)))
    pos_draw = np.random.choice(pos, size=n_batches * pos_per, replace=(len(pos) < n_batches * pos_per))
    neg_draw = np.random.choice(neg, size=n_batches * neg_per, replace=(len(neg) < n_batches * neg_per))

    ordered = []
    for b in range(n_batches):
        chunk = np.concatenate([
            pos_draw[b * pos_per:(b + 1) * pos_per],
            neg_draw[b * neg_per:(b + 1) * neg_per],
        ])
        np.random.shuffle(chunk)
        ordered.extend(chunk.tolist())
    return np.array(ordered, dtype=np.int64)


def _batch_args(batch):
    return (
        batch.x,
        batch.edge_index,
        batch.edge_log_amount,
        batch.edge_ts_encodings,
        batch.edge_bank_pairs,
        batch.edge_tx_types,
        batch.edge_country_risks,
        batch.edge_time_since_prevs,
        batch.edge_time_gap_between_edges,
        batch.edge_rolling_tx_count_7d,
        batch.edge_rolling_tx_count_30d,
        batch.edge_unix_ts,
    )


def _compute_global_edge_probs(model, data, device):
    """Run edge classifier on full graph (no gradient) to get complete edge risk probs."""
    model.eval()
    with torch.no_grad():
        full = data.to(device)
        _, e_logits = model(*_batch_args(full))
        return torch.sigmoid(e_logits).detach()


def _collect_outputs(loader, model, device, global_edge_cache=None):
    """Collect node predictions. If global_edge_cache provided, inject full-graph
    edge probs so every node sees its complete 90-day edge risk history."""
    node_probs = []
    node_logits = []
    node_labels = []
    edge_probs = []
    edge_logits = []
    edge_labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if global_edge_cache is not None:
                # Store n_id (global node IDs for this batch) on the model
                # so forward() can index the full aggregation correctly
                model._batch_n_id = batch.n_id.to(device)
                gep = global_edge_cache["probs"]
                gei = global_edge_cache["index"]
                gets = global_edge_cache["unix_ts"]
                n_logits, e_logits = model(
                    *_batch_args(batch),
                    global_edge_probs=gep,
                    global_edge_index=gei,
                    global_edge_unix_ts=gets,
                    full_num_nodes=global_edge_cache["num_nodes"],
                )
                model._batch_n_id = None
            else:
                n_logits, e_logits = model(*_batch_args(batch))

            seed = int(batch.batch_size)
            n_logits_seed = n_logits[:seed]
            n_probs_seed = torch.sigmoid(n_logits_seed)
            node_probs.append(n_probs_seed.detach().cpu().numpy())
            node_logits.append(n_logits_seed.detach().cpu().numpy())
            node_labels.append(batch.y[:seed].detach().cpu().numpy())

            if hasattr(batch, "edge_y"):
                e_probs = torch.sigmoid(e_logits)
                edge_probs.append(e_probs.detach().cpu().numpy())
                edge_logits.append(e_logits.detach().cpu().numpy())
                edge_labels.append(batch.edge_y.detach().cpu().numpy())

    def _cat(chunks):
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)

    return {
        "node_probs": _cat(node_probs),
        "node_logits": _cat(node_logits),
        "node_labels": _cat(node_labels),
        "edge_probs": _cat(edge_probs),
        "edge_logits": _cat(edge_logits),
        "edge_labels": _cat(edge_labels),
    }


def train_model():
    os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = build_pyg_graph()
    diagnostics = run_data_diagnostics(data)
    with open(os.path.join(MODEL_CHECKPOINT_DIR, "data_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    labels_np = data.y.cpu().numpy().astype(int)
    idx = np.arange(data.num_nodes)
    node_ts = data.node_last_ts.cpu().numpy()

    # ── STRATIFIED TEMPORAL SPLIT ─────────────────────────────────────────────
    # Problem: plain chronological split creates train=38% / test=15% positive
    # rate because suspicious accounts concentrate in the first 70% of the
    # timeline.  Fix: split each class independently by timestamp, then merge.
    # This gives ≤2% class-ratio variance across splits while preserving order.
    def stratified_temporal_split(idx, labels, node_ts, ratios=(0.70, 0.15, 0.15)):
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
        pos_idx = idx[labels[idx] == 1]
        neg_idx = idx[labels[idx] == 0]

        def _split_class(class_idx):
            order = class_idx[np.argsort(node_ts[class_idx])]
            n = len(order)
            c1 = int(ratios[0] * n)
            c2 = c1 + int(ratios[1] * n)
            return order[:c1], order[c1:c2], order[c2:]

        pos_tr, pos_val, pos_te = _split_class(pos_idx)
        neg_tr, neg_val, neg_te = _split_class(neg_idx)

        tr  = np.concatenate([pos_tr,  neg_tr])
        val = np.concatenate([pos_val, neg_val])
        te  = np.concatenate([pos_te,  neg_te])

        # Shuffle within each split so suspicious/normal aren't all at the front,
        # but DO NOT sort by timestamp again (would break stratum ordering).
        rng = np.random.default_rng(42)
        tr  = rng.permutation(tr)
        val = rng.permutation(val)
        te  = rng.permutation(te)
        return tr, val, te

    train_idx, val_idx, test_idx = stratified_temporal_split(idx, labels_np, node_ts)

    def _rate(split): return labels_np[split].mean() * 100
    print(
        f"Stratified temporal split: "
        f"train={len(train_idx)} ({_rate(train_idx):.1f}% pos) | "
        f"val={len(val_idx)} ({_rate(val_idx):.1f}% pos) | "
        f"test={len(test_idx)} ({_rate(test_idx):.1f}% pos)"
    )
    max_rate = max(_rate(train_idx), _rate(val_idx), _rate(test_idx))
    min_rate = min(_rate(train_idx), _rate(val_idx), _rate(test_idx))
    if (max_rate - min_rate) > 5.0:
        print(f"  [WARN] Class ratio variance {max_rate-min_rate:.1f}% exceeds 5% target")

    balanced_train_nodes = _build_balanced_input_nodes(train_idx, labels_np, BATCH_SIZE)
    per_batch_pos = []
    for i in range(0, len(balanced_train_nodes), BATCH_SIZE):
        chunk = balanced_train_nodes[i:i + BATCH_SIZE]
        if chunk.size == 0:
            continue
        per_batch_pos.append(int(labels_np[chunk].sum()))
    if per_batch_pos:
        print(
            f"Balanced batch suspicious seeds: min={min(per_batch_pos)} "
            f"mean={np.mean(per_batch_pos):.2f} max={max(per_batch_pos)}"
        )

    train_loader = NeighborLoader(
        data,
        num_neighbors=GRAPHSAGE_NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=torch.tensor(balanced_train_nodes, dtype=torch.long),
        shuffle=False,
        directed=False,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=GRAPHSAGE_NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=torch.tensor(val_idx, dtype=torch.long),
        shuffle=False,
        directed=False,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=GRAPHSAGE_NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=torch.tensor(test_idx, dtype=torch.long),
        shuffle=False,
        directed=False,
    )

    model = AMLModel(node_dim=int(data.x.shape[1])).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    node_pos_ratio = float(labels_np.mean())
    edge_pos_ratio = float(data.edge_y.cpu().numpy().mean())
    node_alpha = float(np.clip(1.0 - node_pos_ratio, 0.25, 0.90))
    edge_alpha = float(np.clip(1.0 - edge_pos_ratio, 0.25, 0.90))
    node_loss_fn = FocalLossWithLogits(alpha=node_alpha, gamma=2.0)
    edge_loss_fn = FocalLossWithLogits(alpha=edge_alpha, gamma=2.0)
    print(f"Using focal loss: node_alpha={node_alpha:.3f}, edge_alpha={edge_alpha:.3f}")

    epochs = 60
    best_pr_auc = -1.0
    patience = 12
    patience_counter = 0

    best_model_path = os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), best_model_path)

    print("\n=== STEP 6: TRAINING ===")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_node_loss = 0.0
        total_edge_loss = 0.0
        total_seed = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            node_logits, edge_logits = model(*_batch_args(batch))
            seed = int(batch.batch_size)
            seed_labels = batch.y[:seed]

            n_loss = node_loss_fn(node_logits[:seed], seed_labels)
            if hasattr(batch, "edge_y") and batch.edge_y.numel() > 0:
                e_loss = edge_loss_fn(edge_logits, batch.edge_y)
            else:
                e_loss = torch.tensor(0.0, device=device)

            loss = n_loss + (0.5 * e_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
            optimizer.step()

            total_loss += float(loss.item()) * max(1, seed)
            total_node_loss += float(n_loss.item()) * max(1, seed)
            total_edge_loss += float(e_loss.item()) * max(1, seed)
            total_seed += max(1, seed)

        train_loss = total_loss / max(1, total_seed)
        train_node_loss = total_node_loss / max(1, total_seed)
        train_edge_loss = total_edge_loss / max(1, total_seed)

        val_out = _collect_outputs(val_loader, model, device)
        val_thr = _best_threshold(val_out["node_labels"], val_out["node_probs"])
        val_metrics = _metrics_at_threshold(val_out["node_labels"], val_out["node_probs"], val_thr)
        scheduler.step(val_metrics["pr_auc"] if not math.isnan(val_metrics["pr_auc"]) else 0.0)

        print(
            f"Epoch {epoch:02d} | Loss {train_loss:.4f} (node {train_node_loss:.4f}, edge {train_edge_loss:.4f}) | "
            f"Val AUC {val_metrics['auc']:.4f} | Val PR-AUC {val_metrics['pr_auc']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | P {val_metrics['precision']:.4f} | R {val_metrics['recall']:.4f} | "
            f"Brier {val_metrics['brier']:.4f} | Thr {val_metrics['threshold']:.4f}"
        )

        if not math.isnan(val_metrics["pr_auc"]) and val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Val PR-AUC={best_pr_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no PR-AUC improvement for {patience} epochs)")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    # Recompute global cache with best model weights
    try:
        g_edge_probs_final = _compute_global_edge_probs(model, data, device)
        final_cache = {
            "probs": g_edge_probs_final,
            "index": data.edge_index.to(device),
            "unix_ts": data.edge_unix_ts.to(device),
            "num_nodes": data.num_nodes,
        }
    except Exception as _e:
        print(f"[WARN] Final global cache failed: {_e}")
        final_cache = None

    val_out = _collect_outputs(val_loader, model, device, global_edge_cache=final_cache)
    test_out = _collect_outputs(test_loader, model, device, global_edge_cache=final_cache)

    # STEP 7 — Platt scaling calibration
    platt = None
    val_probs_cal = val_out["node_probs"].copy()
    test_probs_cal = test_out["node_probs"].copy()
    try:
        if len(np.unique(val_out["node_labels"])) >= 2:
            platt = LogisticRegression(solver="lbfgs", max_iter=1000)
            platt.fit(val_out["node_logits"].reshape(-1, 1), val_out["node_labels"])
            val_probs_cal = platt.predict_proba(val_out["node_logits"].reshape(-1, 1))[:, 1]
            test_probs_cal = platt.predict_proba(test_out["node_logits"].reshape(-1, 1))[:, 1]
            np.savez(
                os.path.join(MODEL_CHECKPOINT_DIR, "platt_scaler.npz"),
                coef=platt.coef_,
                intercept=platt.intercept_,
            )
    except Exception as exc:
        print(f"[WARN] Platt scaling failed: {exc}")

    best_threshold = _best_threshold(val_out["node_labels"], val_probs_cal)
    np.save(os.path.join(MODEL_CHECKPOINT_DIR, "best_threshold.npy"), np.array([best_threshold], dtype=np.float32))

    graph_metrics_raw = _metrics_at_threshold(test_out["node_labels"], test_out["node_probs"], _best_threshold(val_out["node_labels"], val_out["node_probs"]))
    graph_metrics_cal = _metrics_at_threshold(test_out["node_labels"], test_probs_cal, best_threshold)

    cal_frac_pos, cal_mean_pred = calibration_curve(test_out["node_labels"], test_probs_cal, n_bins=10, strategy="quantile")
    calibration_json = {
        "mean_predicted_value": [float(x) for x in cal_mean_pred.tolist()],
        "fraction_of_positives": [float(x) for x in cal_frac_pos.tolist()],
    }
    with open(os.path.join(MODEL_CHECKPOINT_DIR, "calibration_curve.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_json, f, indent=2)

    # STEP 5 — Edge classification evaluation on full graph (chronological split)
    with torch.no_grad():
        full_data = data.to(device)
        _, full_edge_logits = model(*_batch_args(full_data))
        full_edge_probs = torch.sigmoid(full_edge_logits).detach().cpu().numpy()
    edge_labels = data.edge_y.cpu().numpy().astype(int)
    edge_ts = data.edge_unix_ts.cpu().numpy()
    edge_order = np.argsort(edge_ts)
    e_n = len(edge_order)
    edge_train_idx = edge_order[: int(0.70 * e_n)]
    edge_val_idx = edge_order[int(0.70 * e_n): int(0.85 * e_n)]
    edge_test_idx = edge_order[int(0.85 * e_n):]
    edge_thr = _best_threshold(edge_labels[edge_val_idx], full_edge_probs[edge_val_idx])
    edge_metrics = _metrics_at_threshold(edge_labels[edge_test_idx], full_edge_probs[edge_test_idx], edge_thr)

    # STEP 8 — Tabular baseline ablation
    X = data.x.cpu().numpy()
    y = labels_np
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    baseline.fit(X[train_idx], y[train_idx])
    b_val_probs = baseline.predict_proba(X[val_idx])[:, 1]
    b_test_probs = baseline.predict_proba(X[test_idx])[:, 1]
    b_thr = _best_threshold(y[val_idx], b_val_probs)
    baseline_metrics = _metrics_at_threshold(y[test_idx], b_test_probs, b_thr)

    delta_f1 = graph_metrics_cal["f1"] - baseline_metrics["f1"]
    delta_pr = graph_metrics_cal["pr_auc"] - baseline_metrics["pr_auc"]
    graph_weak = (delta_f1 < 0.02) or (delta_pr < 0.02)

    print("\n=== FINAL METRICS ===")
    print(
        f"Graph (raw):       AUC={graph_metrics_raw['auc']:.4f} PR-AUC={graph_metrics_raw['pr_auc']:.4f} "
        f"F1={graph_metrics_raw['f1']:.4f} P={graph_metrics_raw['precision']:.4f} "
        f"R={graph_metrics_raw['recall']:.4f} Brier={graph_metrics_raw['brier']:.4f}"
    )
    print(
        f"Graph (calibrated):AUC={graph_metrics_cal['auc']:.4f} PR-AUC={graph_metrics_cal['pr_auc']:.4f} "
        f"F1={graph_metrics_cal['f1']:.4f} P={graph_metrics_cal['precision']:.4f} "
        f"R={graph_metrics_cal['recall']:.4f} Brier={graph_metrics_cal['brier']:.4f} "
        f"Thr={graph_metrics_cal['threshold']:.4f}"
    )
    print(
        f"Edge classifier:   AUC={edge_metrics['auc']:.4f} PR-AUC={edge_metrics['pr_auc']:.4f} "
        f"F1={edge_metrics['f1']:.4f} P={edge_metrics['precision']:.4f} "
        f"R={edge_metrics['recall']:.4f} Brier={edge_metrics['brier']:.4f} Thr={edge_metrics['threshold']:.4f}"
    )
    print(
        f"Tabular baseline:  AUC={baseline_metrics['auc']:.4f} PR-AUC={baseline_metrics['pr_auc']:.4f} "
        f"F1={baseline_metrics['f1']:.4f} P={baseline_metrics['precision']:.4f} "
        f"R={baseline_metrics['recall']:.4f} Brier={baseline_metrics['brier']:.4f} Thr={baseline_metrics['threshold']:.4f}"
    )
    print(
        f"Ablation delta:    dF1={delta_f1:+.4f} dPR-AUC={delta_pr:+.4f} "
        f"{'FLAG: graph weak' if graph_weak else 'graph improvement retained'}"
    )

    metrics_bundle = {
        "graph_raw": graph_metrics_raw,
        "graph_calibrated": graph_metrics_cal,
        "edge_classifier": edge_metrics,
        "tabular_baseline": baseline_metrics,
        "ablation": {
            "delta_f1": float(delta_f1),
            "delta_pr_auc": float(delta_pr),
            "graph_weak": bool(graph_weak),
        },
    }
    with open(os.path.join(MODEL_CHECKPOINT_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_bundle, f, indent=2)


if __name__ == "__main__":
    train_model()
