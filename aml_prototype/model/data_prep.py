"""
Data preparation for Temporal GraphSAGE training.

Builds a PyG graph with:
    - temporal edge filtering
    - bounded neighborhood density
    - node + edge temporal features
    - edge labels for auxiliary edge classification
"""

import os
import sys
import sqlite3
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    BANK_DB_PATHS,
    CENTRAL_GRAPH_DB,
    GRAPH_MAX_EDGES,
    GRAPH_MAX_NEIGHBORS_PER_NODE,
    GRAPH_MIN_TX_AMOUNT,
    GRAPH_WINDOW_DAYS,
    NODE_FEATURE_DIM,
)
from bank_node.embedding_generator import generate_embedding  # noqa: E402
from bank_node.kyc_codebook import get_country_risk  # noqa: E402
from model.encoder import encode_timestamp  # noqa: E402


def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std()) if arr.size else 1.0
    if std < 1e-8:
        std = 1.0
    return (arr - mean) / std


def _is_suspicious_memo(memo: str, src_is_suspicious: bool) -> int:
    if not src_is_suspicious:
        return 0
        
    m = (memo or "").strip().lower()
    suspicious_keywords = [
        "fan_out",
        "fan_in",
        "chain",
        "low_and_slow",
        "scatter_gather",
        "peel_off",
        "recursive_loop",
        "agentic_bot",
        "burst"
    ]
    for kw in suspicious_keywords:
        if kw in m:
            return 1
    return 0


def _limit_neighbors_per_source(sorted_txs: List[Dict], max_neighbors: int) -> List[Dict]:
    kept_by_src: Dict[str, deque] = defaultdict(deque)
    for tx in sorted_txs:
        dq = kept_by_src[tx["src"]]
        dq.append(tx)
        while len(dq) > max_neighbors:
            dq.popleft()
    limited = []
    for dq in kept_by_src.values():
        limited.extend(list(dq))
    limited.sort(key=lambda x: x["timestamp"])
    return limited


def _collect_transactions(conns: Dict[str, sqlite3.Connection], node_id_to_idx: Dict[str, int]) -> List[Dict]:
    all_txs: List[Dict] = []
    min_ts = float("inf")
    max_ts = 0.0

    for conn in conns.values():
        txs = conn.execute("SELECT * FROM transactions").fetchall()
        for tx in txs:
            src = tx["src_account_id"]
            dst = tx["dst_account_id"]
            if src not in node_id_to_idx or dst not in node_id_to_idx:
                continue
            amount = float(tx["amount"] or 0.0)
            if amount < GRAPH_MIN_TX_AMOUNT:
                continue
            ts = datetime.fromisoformat(tx["timestamp"]).timestamp()
            min_ts = min(min_ts, ts)
            max_ts = max(max_ts, ts)
            all_txs.append({
                "src": src,
                "dst": dst,
                "amount": amount,
                "tx_type": tx["tx_type"],
                "timestamp": ts,
                "src_bank_id": tx["src_bank_id"],
                "dst_bank_id": tx["dst_bank_id"],
                "src_country": tx["src_country"],
                "dst_country": tx["dst_country"],
                "memo": tx["memo"] if "memo" in tx.keys() else "",
            })

    if not all_txs:
        return []

    cutoff_ts = max_ts - (GRAPH_WINDOW_DAYS * 86400.0)
    all_txs = [t for t in all_txs if t["timestamp"] >= cutoff_ts]
    all_txs.sort(key=lambda x: x["timestamp"])

    all_txs = _limit_neighbors_per_source(all_txs, GRAPH_MAX_NEIGHBORS_PER_NODE)
    if len(all_txs) > GRAPH_MAX_EDGES:
        all_txs = all_txs[-GRAPH_MAX_EDGES:]
    return all_txs


def build_pyg_graph():
    """Build the global transaction graph from local bank databases."""
    # Pass 1 — index all accounts across all banks before any transaction collection
    print("Gathering nodes and embeddings...")
    node_id_to_idx: Dict[str, int] = {}
    node_features: List[np.ndarray] = []
    node_labels: List[int] = []

    conns: Dict[str, sqlite3.Connection] = {}
    for bank_id, db_path in BANK_DB_PATHS.items():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conns[bank_id] = conn

    central_conn = sqlite3.connect(CENTRAL_GRAPH_DB, check_same_thread=False)
    central_conn.row_factory = sqlite3.Row

    # Create historical_vectors table if not exists
    central_conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_vectors (
            account_id TEXT PRIMARY KEY,
            vector BLOB,
            last_compressed TEXT
        )
    """)
    central_conn.commit()

    # Compression pass
    print("Running memory compression on expired transactions...")
    compressed_count = 0
    from memory.compression import compress_expired_transactions
    for bank_id, conn in conns.items():
        accounts = conn.execute("SELECT account_id FROM kyc").fetchall()
        for row in accounts:
            acc_id = row["account_id"]
            try:
                compress_expired_transactions(acc_id, conn, central_conn)
                compressed_count += 1
            except Exception as e:
                print(f"Failed to compress {acc_id}: {e}")
                continue
    print(f"Compression complete. Processed {compressed_count} accounts.")

    idx = 0
    for bank_id, conn in conns.items():
        accounts = conn.execute("SELECT account_id FROM kyc").fetchall()
        for row in accounts:
            acc_id = row["account_id"]
            emb = generate_embedding(acc_id, conn, bank_id)
            if not emb:
                continue

            profile_vec = np.array(emb["profile_vector"], dtype=np.float32)  # 34d
            padded_vec = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
            padded_vec[:34] = profile_vec

            hist_row = central_conn.execute(
                "SELECT vector FROM historical_vectors WHERE account_id=?",
                (acc_id,),
            ).fetchone()
            if hist_row and hist_row["vector"]:
                hist_vec = np.frombuffer(hist_row["vector"], dtype=np.float32)
                padded_vec[34:98] = hist_vec[:64]

            label_row = conn.execute(
                "SELECT is_suspicious FROM labels WHERE account_id=?",
                (acc_id,),
            ).fetchone()
            is_suspicious = int(label_row["is_suspicious"]) if label_row else 0

            node_id_to_idx[acc_id] = idx
            node_features.append(padded_vec)
            node_labels.append(is_suspicious)
            idx += 1

    central_conn.close()
    print(f"Total nodes extracted: {len(node_features)}")

    # Pass 2 — collect transactions now that all accounts are indexed
    print("Gathering edges...")
    all_txs = _collect_transactions(conns, node_id_to_idx)
    print(
        f"Total edges extracted after filters: {len(all_txs)} "
        f"(window={GRAPH_WINDOW_DAYS}d, min_amount={GRAPH_MIN_TX_AMOUNT}, "
        f"max_neighbors={GRAPH_MAX_NEIGHBORS_PER_NODE}, max_edges={GRAPH_MAX_EDGES})"
    )

    edge_index = [[], []]
    edge_log_amount = []
    edge_ts_encodings = []
    edge_bank_pairs = []
    edge_tx_types = []
    edge_country_risks = []
    edge_time_since_prevs = []
    edge_time_gap_between_edges = []
    edge_rolling_tx_count_7d = []
    edge_rolling_tx_count_30d = []
    edge_cross_country = []
    edge_labels = []
    edge_unix_ts = []

    tx_type_map = {"wire": 0, "ach": 1, "cash_deposit": 2, "cash_withdrawal": 3, "internal": 4}
    bank_idx_map = {"bank_a": 0, "bank_b": 1, "bank_c": 2}

    if all_txs:
        min_ts = all_txs[0]["timestamp"]
        max_ts = all_txs[-1]["timestamp"]
    else:
        min_ts = 0.0
        max_ts = 1.0

    last_seen_src_ts: Dict[str, float] = {}
    last_global_ts = None
    rolling7: Dict[str, deque] = defaultdict(deque)
    rolling30: Dict[str, deque] = defaultdict(deque)
    src_ts_map: Dict[str, List[float]] = defaultdict(list)
    src_counterparties: Dict[str, set] = defaultdict(set)
    src_amounts: Dict[str, List[float]] = defaultdict(list)
    src_cross_country_count: Dict[str, int] = defaultdict(int)
    src_total_count: Dict[str, int] = defaultdict(int)
    node_edge_risk_events: Dict[str, List[tuple]] = defaultdict(list)

    suspicious_edge_count = 0  # To verify the fix worked

    for tx in all_txs:
        src = tx["src"]
        dst = tx["dst"]
        ts = tx["timestamp"]
        amount = tx["amount"]

        src_ts_map[src].append(ts)
        src_counterparties[src].add(dst)
        src_amounts[src].append(amount)
        src_total_count[src] += 1
        if tx["src_country"] != tx["dst_country"]:
            src_cross_country_count[src] += 1

        dq7 = rolling7[src]
        while dq7 and (ts - dq7[0]) > (7 * 86400.0):
            dq7.popleft()
        c7 = len(dq7)

        dq30 = rolling30[src]
        while dq30 and (ts - dq30[0]) > (30 * 86400.0):
            dq30.popleft()
        c30 = len(dq30)

        prev_src_ts = last_seen_src_ts.get(src, ts)
        time_since_prev = max(0.0, ts - prev_src_ts)
        last_seen_src_ts[src] = ts

        time_gap_global = 0.0 if last_global_ts is None else max(0.0, ts - last_global_ts)
        last_global_ts = ts

        dq7.append(ts)
        dq30.append(ts)

        u = node_id_to_idx[src]
        v = node_id_to_idx[dst]
        edge_index[0].append(u)
        edge_index[1].append(v)
        src_is_suspicious = bool(node_labels[u])

        edge_log_amount.append(np.log1p(amount))
        edge_ts_encodings.append(encode_timestamp(ts, min_ts, max_ts, log_time_since_prev=np.log1p(time_since_prev)))
        s_bank = bank_idx_map.get(tx["src_bank_id"], 0)
        d_bank = bank_idx_map.get(tx["dst_bank_id"], 0)
        edge_bank_pairs.append(s_bank * 3 + d_bank)
        edge_tx_types.append(tx_type_map.get(tx["tx_type"], 0))
        s_risk = get_country_risk(tx["src_country"])
        d_risk = get_country_risk(tx["dst_country"])
        edge_country_risks.append(max(s_risk, d_risk))
        edge_time_since_prevs.append(np.log1p(time_since_prev))
        edge_time_gap_between_edges.append(np.log1p(time_gap_global))
        edge_rolling_tx_count_7d.append(float(c7))
        edge_rolling_tx_count_30d.append(float(c30))
        edge_cross_country.append(1.0 if tx["src_country"] != tx["dst_country"] else 0.0)
        
        edge_risk_label = float(_is_suspicious_memo(tx.get("memo", ""), src_is_suspicious))
        if edge_risk_label == 1.0:
            suspicious_edge_count += 1
            
        edge_labels.append(edge_risk_label)
        edge_unix_ts.append(ts)
        node_edge_risk_events[src].append((ts, edge_risk_label))
        node_edge_risk_events[dst].append((ts, edge_risk_label))

    for conn in conns.values():
        conn.close()

    print(f"VERIFICATION: Total edges={len(edge_labels)}, Suspicious edges={suspicious_edge_count}")

    # Node temporal features
    node_avg_interval = np.zeros(len(node_features), dtype=np.float32)
    node_var_interval = np.zeros(len(node_features), dtype=np.float32)
    node_last_activity = np.zeros(len(node_features), dtype=np.float32)
    node_tx_frequency = np.zeros(len(node_features), dtype=np.float32)
    node_unique_counterparties = np.zeros(len(node_features), dtype=np.float32)
    node_mean_amount = np.zeros(len(node_features), dtype=np.float32)
    node_cross_country_ratio = np.zeros(len(node_features), dtype=np.float32)
    node_suspicious_edge_count_last_30d = np.zeros(len(node_features), dtype=np.float32)
    node_max_edge_risk_last_30d = np.zeros(len(node_features), dtype=np.float32)
    node_avg_edge_risk_last_30d = np.zeros(len(node_features), dtype=np.float32)
    node_time_since_last_suspicious_edge = np.zeros(len(node_features), dtype=np.float32)
    months = max(1e-6, GRAPH_WINDOW_DAYS / 30.0)
    edge_30d_cutoff = max_ts - (30.0 * 86400.0)

    for acc_id, i in node_id_to_idx.items():
        ts_list = sorted(src_ts_map.get(acc_id, []))
        node_tx_frequency[i] = float(len(ts_list) / months)
        if len(ts_list) >= 2:
            intervals_h = np.diff(np.array(ts_list, dtype=np.float64)) / 3600.0
            node_avg_interval[i] = float(intervals_h.mean())
            node_var_interval[i] = float(intervals_h.var())
        elif len(ts_list) == 1:
            node_avg_interval[i] = float(GRAPH_WINDOW_DAYS * 24.0)
            node_var_interval[i] = 0.0
        else:
            node_avg_interval[i] = float(GRAPH_WINDOW_DAYS * 24.0)
            node_var_interval[i] = 0.0

        if ts_list:
            node_last_activity[i] = float((ts_list[-1] - min_ts) / (max(max_ts - min_ts, 1e-6)))
        else:
            node_last_activity[i] = 0.0
        node_unique_counterparties[i] = float(len(src_counterparties.get(acc_id, set())))
        amounts = src_amounts.get(acc_id, [])
        node_mean_amount[i] = float(np.mean(amounts)) if amounts else 0.0
        total_c = max(1, src_total_count.get(acc_id, 0))
        node_cross_country_ratio[i] = float(src_cross_country_count.get(acc_id, 0) / total_c)

        edge_events = node_edge_risk_events.get(acc_id, [])
        if edge_events:
            event_ts = np.array([e[0] for e in edge_events], dtype=np.float64)
            event_risk = np.array([e[1] for e in edge_events], dtype=np.float64)
            last30_mask = event_ts >= edge_30d_cutoff
            if last30_mask.any():
                last30_risk = event_risk[last30_mask]
                node_suspicious_edge_count_last_30d[i] = float(np.sum(last30_risk >= 0.7))
                node_max_edge_risk_last_30d[i] = float(np.max(last30_risk))
                node_avg_edge_risk_last_30d[i] = float(np.mean(last30_risk))
            suspicious_mask = event_risk >= 0.7
            if suspicious_mask.any():
                last_susp_ts = float(np.max(event_ts[suspicious_mask]))
                node_time_since_last_suspicious_edge[i] = float(max(0.0, (max_ts - last_susp_ts) / 86400.0))
            else:
                node_time_since_last_suspicious_edge[i] = float(GRAPH_WINDOW_DAYS)
        else:
            node_time_since_last_suspicious_edge[i] = float(GRAPH_WINDOW_DAYS)

    node_avg_interval_z = _zscore(node_avg_interval.astype(np.float64)).astype(np.float32)
    node_var_interval_z = _zscore(node_var_interval.astype(np.float64)).astype(np.float32)
    node_last_activity_z = _zscore(node_last_activity.astype(np.float64)).astype(np.float32)
    node_tx_freq_z = _zscore(node_tx_frequency.astype(np.float64)).astype(np.float32)
    node_unique_counterparties_z = _zscore(node_unique_counterparties.astype(np.float64)).astype(np.float32)
    node_mean_amount_z = _zscore(node_mean_amount.astype(np.float64)).astype(np.float32)
    node_cross_country_ratio_z = _zscore(node_cross_country_ratio.astype(np.float64)).astype(np.float32)
    node_suspicious_edge_count_last_30d_z = _zscore(
        np.log1p(node_suspicious_edge_count_last_30d.astype(np.float64))
    ).astype(np.float32)
    node_max_edge_risk_last_30d_z = _zscore(node_max_edge_risk_last_30d.astype(np.float64)).astype(np.float32)
    node_avg_edge_risk_last_30d_z = _zscore(node_avg_edge_risk_last_30d.astype(np.float64)).astype(np.float32)
    node_time_since_last_suspicious_edge_z = _zscore(
        np.log1p(node_time_since_last_suspicious_edge.astype(np.float64))
    ).astype(np.float32)

    for i in range(len(node_features)):
        node_features[i][98] = node_avg_interval_z[i]
        node_features[i][99] = node_var_interval_z[i]
        node_features[i][100] = node_last_activity_z[i]
        node_features[i][101] = node_suspicious_edge_count_last_30d_z[i]
        node_features[i][102] = node_max_edge_risk_last_30d_z[i]
        node_features[i][103] = node_avg_edge_risk_last_30d_z[i]
        node_features[i][104] = node_time_since_last_suspicious_edge_z[i]

    # Normalize new edge numeric features
    edge_log_amount_arr = np.array(edge_log_amount, dtype=np.float32)
    edge_country_risks_arr = np.array(edge_country_risks, dtype=np.float32)
    edge_time_since_arr_raw = np.array(edge_time_since_prevs, dtype=np.float32)
    edge_time_gap_arr_raw = np.array(edge_time_gap_between_edges, dtype=np.float32)
    edge_roll7_arr_raw = np.array(edge_rolling_tx_count_7d, dtype=np.float32)
    edge_roll30_arr_raw = np.array(edge_rolling_tx_count_30d, dtype=np.float32)
    edge_cross_country_arr = np.array(edge_cross_country, dtype=np.float32)
    edge_labels_arr = np.array(edge_labels, dtype=np.float32)

    edge_time_since_arr = _zscore(edge_time_since_arr_raw.astype(np.float64)).astype(np.float32)
    edge_time_gap_arr = _zscore(edge_time_gap_arr_raw.astype(np.float64)).astype(np.float32)
    edge_roll7_arr = _zscore(edge_roll7_arr_raw.astype(np.float64)).astype(np.float32)
    edge_roll30_arr = _zscore(edge_roll30_arr_raw.astype(np.float64)).astype(np.float32)

    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    y = torch.tensor(np.array(node_labels), dtype=torch.float32)
    edge_index_t = torch.tensor(np.array(edge_index), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index_t, y=y)
    data.edge_log_amount = torch.tensor(edge_log_amount_arr.reshape(-1, 1), dtype=torch.float32)
    data.edge_ts_encodings = torch.tensor(np.array(edge_ts_encodings), dtype=torch.float32)
    data.edge_bank_pairs = torch.tensor(np.array(edge_bank_pairs), dtype=torch.long)
    data.edge_tx_types = torch.tensor(np.array(edge_tx_types), dtype=torch.long)
    data.edge_country_risks = torch.tensor(edge_country_risks_arr.reshape(-1, 1), dtype=torch.float32)
    data.edge_time_since_prevs = torch.tensor(edge_time_since_arr.reshape(-1, 1), dtype=torch.float32)
    data.edge_time_gap_between_edges = torch.tensor(edge_time_gap_arr.reshape(-1, 1), dtype=torch.float32)
    data.edge_rolling_tx_count_7d = torch.tensor(edge_roll7_arr.reshape(-1, 1), dtype=torch.float32)
    data.edge_rolling_tx_count_30d = torch.tensor(edge_roll30_arr.reshape(-1, 1), dtype=torch.float32)
    data.edge_cross_country = torch.tensor(edge_cross_country_arr.reshape(-1, 1), dtype=torch.float32)
    data.edge_y = torch.tensor(edge_labels_arr, dtype=torch.float32)
    data.edge_unix_ts = torch.tensor(np.array(edge_unix_ts, dtype=np.float64), dtype=torch.float64)

    # Raw fields for diagnostics
    data.edge_time_since_prevs_raw = torch.tensor(edge_time_since_arr_raw.reshape(-1, 1), dtype=torch.float32)
    data.edge_time_gap_between_edges_raw = torch.tensor(edge_time_gap_arr_raw.reshape(-1, 1), dtype=torch.float32)
    data.edge_rolling_tx_count_7d_raw = torch.tensor(edge_roll7_arr_raw.reshape(-1, 1), dtype=torch.float32)
    data.edge_rolling_tx_count_30d_raw = torch.tensor(edge_roll30_arr_raw.reshape(-1, 1), dtype=torch.float32)
    data.node_tx_frequency = torch.tensor(node_tx_frequency, dtype=torch.float32)
    data.node_avg_interval_raw = torch.tensor(node_avg_interval, dtype=torch.float32)
    data.node_interval_var_raw = torch.tensor(node_var_interval, dtype=torch.float32)
    data.node_last_activity_raw = torch.tensor(node_last_activity, dtype=torch.float32)
    data.node_suspicious_edge_count_last_30d_raw = torch.tensor(
        node_suspicious_edge_count_last_30d, dtype=torch.float32
    )
    data.node_max_edge_risk_last_30d_raw = torch.tensor(node_max_edge_risk_last_30d, dtype=torch.float32)
    data.node_avg_edge_risk_last_30d_raw = torch.tensor(node_avg_edge_risk_last_30d, dtype=torch.float32)
    data.node_time_since_last_suspicious_edge_raw = torch.tensor(
        node_time_since_last_suspicious_edge, dtype=torch.float32
    )

    # Chronological node split helper
    node_last_ts_arr = np.full(len(node_features), min_ts if all_txs else 0.0, dtype=np.float64)
    for acc_id, ts in last_seen_src_ts.items():
        if acc_id in node_id_to_idx:
            node_last_ts_arr[node_id_to_idx[acc_id]] = ts
    data.node_last_ts = torch.tensor(node_last_ts_arr, dtype=torch.float64)

    return data


if __name__ == "__main__":
    g = build_pyg_graph()
    print("Graph construction complete")
    print(f"Nodes: {g.num_nodes}, Edges: {g.num_edges}")
    print(f"Node feature matrix: {tuple(g.x.shape)}")
    
    node_positive_rate = float(g.y.sum()) / max(1, g.num_nodes)
    edge_positive_rate = float(g.edge_y.sum()) / max(1, g.num_edges)
    print(f"Node y=1 rate: {node_positive_rate:.2%}")
    print(f"Edge y=1 rate: {edge_positive_rate:.2%}")
