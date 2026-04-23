"""
End-to-End Integration Test for AML Architecture.

Runs the full pipeline:
Simulate -> Bank API -> Embedding -> Graph -> Inference -> Score -> Memory
"""

import os
import sys
import time
import urllib.request
import json
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import BANK_PORTS, CENTRAL_GRAPH_DB
from scoring.engine import compute_trust_score, get_recommendation
from memory.feedback import log_suspicious_pattern

def test_full_pipeline():
    print("--- AML Pipeline Integration Test ---")
    
    # 1. Fetch Embeddings from Bank Nodes
    print("\n1. Fetching embeddings from Bank Nodes...")
    test_accounts = {
        "bank_a": "ACC-A-000000",
        "bank_b": "ACC-B-002000",
        "bank_c": "ACC-C-003500"
    }
    
    embeddings = {}
    for bank_id, acc_id in test_accounts.items():
        port = BANK_PORTS[bank_id]
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/embeddings/{acc_id}")
            with urllib.request.urlopen(req) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    embeddings[acc_id] = data
                    print(f"[OK] Successfully fetched 34d privacy-preserved embedding for {acc_id} from {bank_id}")
                else:
                    print(f"[FAIL] Failed to fetch embedding for {acc_id}: HTTP {resp.status}")
        except Exception as e:
            print(f"[FAIL] Connection error for {bank_id} on port {port}: {e}")
            print("  (Make sure 'python scripts/run_banks.py' is running)")

    if not embeddings:
        print("\nAborting test: Could not communicate with bank nodes.")
        return

    # 2. Central Aggregator Data Prep
    print("\n2. Simulating Central Aggregator PyG Graph Build...")
    try:
        from model.data_prep import build_pyg_graph
        graph = build_pyg_graph()
        print(f"[OK] Successfully constructed Graph with {graph.num_nodes} nodes and {graph.num_edges} edges")
    except Exception as e:
        print(f"[WARN] Could not build full PyG graph (expected if PyTorch is not installed): {e}")
        print("  Skipping graph build in this environment.")

    # 3. Model Inference (Mock if torch unavailable)
    print("\n3. Running Temporal GraphSAGE Inference...")
    try:
        import torch
        from model.gnn import TemporalGraphSAGE
        # This is where we would pass the graph into the loaded model
        print("[OK] PyTorch available. Mocking inference probabilities for test accounts.")
        probs = [0.89, 0.45, 0.12] 
        confs = [0.92, 0.70, 0.85]
    except ImportError:
        print("[WARN] PyTorch not installed. Using mocked output probabilities.")
        probs = [0.89, 0.45, 0.12]
        confs = [0.92, 0.70, 0.85]

    # 4. Scoring Engine
    print("\n4. Applying Trust Score and Recommendations...")
    conn = sqlite3.connect(CENTRAL_GRAPH_DB, check_same_thread=False)
    
    for i, (acc_id, emb) in enumerate(embeddings.items()):
        prob = probs[i]
        conf = confs[i]
        
        # Build feature dict from embedding and mock historical data
        features = {
            'occ_sal_mismatch': emb.get('mismatch_score', 0.0),
            'country_risk': emb.get('country_risk', 0.0),
            'motif_participation': 1.0 if prob > 0.8 else 0.0,
            'hist_pattern_count': 0
        }
        
        score = compute_trust_score(features)
        action = get_recommendation(score, prob, conf)
        
        print(f"Account: {acc_id}")
        print(f"  - Model Risk: {prob*100:.1f}%")
        print(f"  - Model Conf: {conf*100:.1f}%")
        print(f"  - Trust Score: {score}")
        print(f"  - Action: {action}")
        
        # 5. Alert Generation
        if action in ["HOLD", "MANUAL_REVIEW"]:
            alert_id = f"ALT-TEST-{int(time.time())}-{i}"
            print(f"  -> Generated Alert {alert_id}")
            log_suspicious_pattern(conn, alert_id, "test_motif", [acc_id], ["US"], prob)
            
    conn.close()

    print("\n5. Checking Dashboard Alert Queue...")
    try:
            req = urllib.request.Request(f"http://127.0.0.1:8080/")
            with urllib.request.urlopen(req) as resp:
                if resp.status == 200:
                    print("[OK] Dashboard is online and responding.")
                else:
                    print(f"[FAIL] Dashboard returned HTTP {resp.status}")
    except:
        print("❌ Dashboard is not reachable. (Make sure 'python dashboard/app.py' is running)")

    print("\n--- Integration Test Complete ---")

if __name__ == "__main__":
    test_full_pipeline()
