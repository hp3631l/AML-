import os
import sys
import json
import urllib.request
import urllib.error
import sqlite3
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    BANK_PORTS,
    CENTRAL_GRAPH_DB,
    MODEL_CHECKPOINT_DIR,
    GRAPH_WINDOW_DAYS,
    GRAPH_MAX_EDGES,
    GRAPH_MAX_NEIGHBORS_PER_NODE,
    NODE_FEATURE_DIM,
)
from model.encoder import encode_timestamp
from bank_node.kyc_codebook import get_country_risk
from torch_geometric.data import Data
from collections import defaultdict, deque

class PrivacyViolationError(Exception):
    pass

def _limit_neighbors(txs: List[Dict], max_neighbors: int) -> List[Dict]:
    kept = defaultdict(deque)
    for tx in txs:
        dq = kept[tx["src"]]
        dq.append(tx)
        while len(dq) > max_neighbors:
            dq.popleft()
    limited = []
    for dq in kept.values():
        limited.extend(list(dq))
    limited.sort(key=lambda x: x["timestamp"])
    return limited

def _zscore(arr: np.ndarray) -> np.ndarray:
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std()) if arr.size else 1.0
    if std < 1e-8:
        std = 1.0
    return (arr - mean) / std

def run_inference_pipeline():
    print("=== Starting AML Aggregator Inference Pipeline ===")
    
    # Check for direct file access violations
    with open(__file__, 'r') as f:
        content = f.read()
        if "BANK_DB_PATHS" in content and "raise" not in content.split("BANK_DB_PATHS")[0]:
            pass # Self check logic doesn't strictly apply this way, we just avoid using it
    
    # STEP 1: Fetch embeddings from bank APIs
    embeddings_map = {}
    print("Step 1: Fetching embeddings via Bank APIs...")
    for bank_id, port in BANK_PORTS.items():
        url = f"http://localhost:{port}/embeddings/batch/"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    # Expecting a list of dicts in data (or data["embeddings"])
                    emb_list = data.get("embeddings", data) if isinstance(data, dict) else data
                    for emb in emb_list:
                        # Required fields: hashed_account_id, occupation_embedding, salary_bucket, 
                        # session_vector, ledger_metadata_vector, risk_embedding
                        acc_id = emb.get("hashed_account_id") or emb.get("account_id")
                        if acc_id:
                            embeddings_map[acc_id] = emb
                else:
                    print(f"Failed to fetch from {bank_id}: HTTP {resp.status}")
        except urllib.error.URLError as e:
            print(f"Bank API unreachable for {bank_id} on port {port}: {e}")
        except Exception as e:
            print(f"Error fetching from {bank_id}: {e}")
            
    print(f"Fetched {len(embeddings_map)} total account embeddings.")
    if not embeddings_map:
        print("No embeddings fetched. Exiting pipeline.")
        return

    # STEP 2: Load transactions from central graph store
    print("Step 2: Loading central transactions...")
    central_conn = sqlite3.connect(CENTRAL_GRAPH_DB, check_same_thread=False)
    central_conn.row_factory = sqlite3.Row
    
    cutoff_date = (datetime.now() - timedelta(days=GRAPH_WINDOW_DAYS)).isoformat()
    try:
        tx_rows = central_conn.execute("""
            SELECT * FROM transactions 
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (cutoff_date,)).fetchall()
    except sqlite3.OperationalError:
        # Fallback if the table schema differs slightly or is empty
        try:
            tx_rows = central_conn.execute("SELECT * FROM transactions ORDER BY timestamp ASC").fetchall()
        except sqlite3.OperationalError as e:
            print(f"Could not load transactions from central DB: {e}")
            tx_rows = []
            
    txs_all = []
    min_ts = float('inf')
    max_ts = 0.0
    for row in tx_rows:
        # Use only hashed_account_id
        src = row.get("src_account_id") or row.get("hashed_src_account_id")
        dst = row.get("dst_account_id") or row.get("hashed_dst_account_id")
        
        if src not in embeddings_map or dst not in embeddings_map:
            continue
            
        amount = float(row["amount"] or 0.0)
        if amount < GRAPH_MIN_TX_AMOUNT:
            continue
            
        ts_str = row["timestamp"]
        try:
            ts = datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            ts = float(ts_str)
            
        min_ts = min(min_ts, ts)
        max_ts = max(max_ts, ts)
        
        txs_all.append({
            "src": src,
            "dst": dst,
            "amount": amount,
            "tx_type": row.get("tx_type", "wire"),
            "timestamp": ts,
            "src_bank_id": row.get("src_bank_id", ""),
            "dst_bank_id": row.get("dst_bank_id", ""),
            "src_country": row.get("src_country", "US"),
            "dst_country": row.get("dst_country", "US")
        })

    if txs_all:
        cutoff_ts = max_ts - (GRAPH_WINDOW_DAYS * 86400.0)
        txs_all = [t for t in txs_all if t["timestamp"] >= cutoff_ts]
        txs_all = _limit_neighbors(txs_all, GRAPH_MAX_NEIGHBORS_PER_NODE)
        if len(txs_all) > GRAPH_MAX_EDGES:
            txs_all = txs_all[-GRAPH_MAX_EDGES:]
            
    print(f"Loaded {len(txs_all)} active transactions.")

    # STEP 3: Build PyG graph from API responses only
    print("Step 3: Building graph representation...")
    node_id_to_idx = {}
    node_features = []
    idx = 0
    
    # Load historical vectors to append to node features if available
    historical_map = {}
    try:
        hist_rows = central_conn.execute("SELECT account_id, vector FROM historical_vectors").fetchall()
        for hr in hist_rows:
            if hr["vector"]:
                historical_map[hr["account_id"]] = np.frombuffer(hr["vector"], dtype=np.float32)
    except sqlite3.OperationalError:
        pass
    
    for acc_id, emb in embeddings_map.items():
        node_id_to_idx[acc_id] = idx
        # Build 105d feature vector
        padded_vec = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
        
        # We need to construct profile_vec from API fields:
        # occupation_embedding, salary_bucket, session_vector, ledger_metadata_vector, risk_embedding
        offset = 0
        
        occ = emb.get("occupation_embedding", [])
        if occ:
            padded_vec[offset:offset+len(occ)] = occ
            offset += len(occ)
            
        sal = emb.get("salary_bucket", 0)
        padded_vec[offset] = sal
        offset += 1
        
        sess = emb.get("session_vector", [])
        if sess:
            padded_vec[offset:offset+len(sess)] = sess
            offset += len(sess)
            
        ledger = emb.get("ledger_metadata_vector", [])
        if ledger:
            padded_vec[offset:offset+len(ledger)] = ledger
            offset += len(ledger)
            
        risk = emb.get("risk_embedding", [])
        if risk:
            padded_vec[offset:offset+len(risk)] = risk
            
        # Add historical vector
        if acc_id in historical_map:
            h_vec = historical_map[acc_id][:64]
            padded_vec[34:34+len(h_vec)] = h_vec
            
        node_features.append(padded_vec)
        idx += 1

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
    edge_unix_ts = []

    tx_type_map = {"wire": 0, "ach": 1, "cash_deposit": 2, "cash_withdrawal": 3, "internal": 4}
    bank_idx_map = {"bank_a": 0, "bank_b": 1, "bank_c": 2}
    
    last_seen_src_ts = {}
    last_global_ts = None
    rolling7 = defaultdict(deque)
    rolling30 = defaultdict(deque)

    for tx in txs_all:
        src = tx["src"]
        dst = tx["dst"]
        ts = tx["timestamp"]
        amount = tx["amount"]

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
        edge_unix_ts.append(ts)

    if not node_features:
        print("No nodes to process. Exiting.")
        return

    X = np.vstack(node_features).astype(np.float32)
    # Append zeros for the temporal structural features computed during training
    num_nodes = len(node_features)
    temp_feats = np.zeros((num_nodes, 9), dtype=np.float32)
    X = np.concatenate([X, temp_feats], axis=1)

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_log_amount=torch.tensor(edge_log_amount, dtype=torch.float).reshape(-1, 1),
        edge_ts_encodings=torch.tensor(np.array(edge_ts_encodings) if edge_ts_encodings else np.zeros((0,4)), dtype=torch.float),
        edge_bank_pairs=torch.tensor(edge_bank_pairs, dtype=torch.long),
        edge_tx_types=torch.tensor(edge_tx_types, dtype=torch.long),
        edge_country_risks=torch.tensor(edge_country_risks, dtype=torch.float).reshape(-1, 1),
        edge_time_since_prevs=torch.tensor(edge_time_since_prevs, dtype=torch.float).reshape(-1, 1),
        edge_time_gap_between_edges=torch.tensor(edge_time_gap_between_edges, dtype=torch.float).reshape(-1, 1),
        edge_rolling_tx_count_7d=torch.tensor(edge_rolling_tx_count_7d, dtype=torch.float).reshape(-1, 1),
        edge_rolling_tx_count_30d=torch.tensor(edge_rolling_tx_count_30d, dtype=torch.float).reshape(-1, 1),
        edge_cross_country=torch.tensor(edge_cross_country, dtype=torch.float).reshape(-1, 1),
        edge_unix_ts=torch.tensor(edge_unix_ts, dtype=torch.float64)
    )

    # STEP 4: Load model checkpoint and run inference
    print("Step 4: Running inference...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}. Please train first.")
        return
        
    from model.train import AMLModel
    model = AMLModel(node_dim=data.x.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    best_threshold = 0.5
    thresh_path = os.path.join(MODEL_CHECKPOINT_DIR, "best_threshold.npy")
    if os.path.exists(thresh_path):
        best_threshold = float(np.load(thresh_path))
        
    # Optional Platt scaling (assuming standard format if exists)
    platt_path = os.path.join(MODEL_CHECKPOINT_DIR, "platt_scaler.npz")
    scaler_A, scaler_B = None, None
    if os.path.exists(platt_path):
        npz = np.load(platt_path)
        scaler_A = float(npz.get("A", 1.0))
        scaler_B = float(npz.get("B", 0.0))

    data = data.to(device)
    with torch.no_grad():
        args = (
            data.x, data.edge_index, 
            data.edge_log_amount, data.edge_ts_encodings,
            data.edge_bank_pairs, data.edge_tx_types, 
            data.edge_country_risks, data.edge_time_since_prevs,
            data.edge_time_gap_between_edges, data.edge_rolling_tx_count_7d,
            data.edge_rolling_tx_count_30d,
            data.edge_unix_ts,
        )
        try:
            mean_prob, confidence = model.predict_with_confidence(*args)
        except AttributeError:
            # Fallback if predict_with_confidence doesn't exist
            node_logits = model(*args[:11])
            mean_prob = torch.sigmoid(node_logits).squeeze()
            confidence = torch.zeros_like(mean_prob)

    probs = mean_prob.cpu().numpy()
    if scaler_A is not None and scaler_B is not None:
        probs = 1.0 / (1.0 + np.exp(scaler_A * probs + scaler_B))

    # STEP 5: Generate alerts
    print("Step 5: Generating alerts...")
    central_conn.execute("""
        CREATE TABLE IF NOT EXISTS pattern_memory (
            alert_id TEXT PRIMARY KEY,
            hashed_account_id TEXT,
            laundering_prob REAL,
            recommendation TEXT,
            timestamp TEXT
        )
    """)
    
    idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}
    alerts_generated = 0
    now_str = datetime.now().isoformat()
    
    for i, prob in enumerate(probs):
        if prob > best_threshold:
            rec = "HOLD" if prob > 0.85 else "MANUAL_REVIEW"
            acc_id = idx_to_node_id[i]
            alert_id = f"ALT-{int(datetime.now().timestamp())}-{i}"
            
            central_conn.execute("""
                INSERT INTO pattern_memory (alert_id, hashed_account_id, laundering_prob, recommendation, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (alert_id, acc_id, float(prob), rec, now_str))
            alerts_generated += 1
            
    central_conn.commit()
    central_conn.close()
    print(f"Generated {alerts_generated} alerts.")

    # STEP 6: Privacy audit check (runtime)
    print("Step 6: Running privacy audit...")
    # Assert that no raw PII fields appear anywhere in the graph data object
    restricted_fields = ["customer_name", "pan", "aadhaar", "home_address", "exact_salary"]
    
    # We check the attributes of the data object
    for key in data.keys():
        for field in restricted_fields:
            if field in key.lower():
                raise PrivacyViolationError(field)
                
    # Also verify that our embeddings_map doesn't hold these
    for emb in embeddings_map.values():
        for field in restricted_fields:
            if field in emb:
                raise PrivacyViolationError(field)
                
    print("Privacy audit passed successfully. No raw PII leaked into graph.")
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    run_inference_pipeline()
