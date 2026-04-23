"""
Alert Generator.

Takes the output of the GNN, computes trust scores, and pushes alerts into the central database.
"""

import sqlite3
import json
import time
import uuid
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import CENTRAL_GRAPH_DB
from scoring.engine import compute_trust_score, get_recommendation
from memory.feedback import log_suspicious_pattern

def generate_alerts(probs, confidences, data):
    """
    Generate alerts based on model predictions.
    
    probs: Tensor of shape [N] containing laundering probabilities.
    confidences: Tensor of shape [N] containing model confidence.
    data: The PyG Data object containing idx_to_node_id and embeddings_map.
    """
    import numpy as np
    print("Evaluating predictions and generating alerts...")
    conn = sqlite3.connect(CENTRAL_GRAPH_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Load calibrated threshold if available; fall back to config value
    threshold_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "model", "checkpoints", "best_threshold.npy"
    )
    if os.path.exists(threshold_path):
        review_threshold = float(np.load(threshold_path)[0])
        hold_threshold = min(review_threshold + 0.20, 0.95)
    else:
        from config import LAUNDERING_PROB_HOLD, LAUNDERING_PROB_REVIEW
        review_threshold = LAUNDERING_PROB_REVIEW
        hold_threshold = LAUNDERING_PROB_HOLD

    from config import CONFIDENCE_THRESHOLD

    alert_count = 0
    for idx, prob in enumerate(probs.tolist()):
        conf = confidences[idx].item()

        if prob < review_threshold:
            continue

        account_id = data.idx_to_node_id[idx]
        emb = data.embeddings_map.get(account_id, {})

        # Fetch historical features
        hist_row = conn.execute(
            "SELECT vector FROM historical_vectors WHERE account_id=?", (account_id,)
        ).fetchone()

        hist_vec = None
        hist_pattern_count = 0
        hold_count = 0
        motif_participation = 0.0  # CHECK 2 FIX: derived from history, NOT from model output

        if hist_row and hist_row['vector']:
            hist_vec = np.frombuffer(hist_row['vector'], dtype=np.float32)
            # Dims 0-7: motif participation counts (genuine historical signal)
            motif_participation = float(np.clip(hist_vec[0:8].sum(), 0.0, 1.0))
            hist_pattern_count = int(hist_vec[20]) if len(hist_vec) > 20 else 0
            hold_count = int(hist_vec[16]) if len(hist_vec) > 16 else 0

        # Motif count: how many distinct motif types have been detected historically
        motif_count = int((hist_vec[0:8] > 0).sum()) if hist_vec is not None else 0

        features = {
            'occ_sal_mismatch': emb.get('mismatch_score', 0.0),
            'sal_tx_mismatch': 0.0,
            'country_risk': emb.get('country_risk', 0.5) if 'country_risk' in emb else 0.5,
            'session_anomaly': 0.0,
            'motif_participation': motif_participation,  # CHECK 2 FIX: historical, not model-circular
            'hist_pattern_count': hist_pattern_count,
            'hold_count': hold_count,
            'cross_country_chain': 1 if hold_count > 0 else 0,
        }

        score = compute_trust_score(features)

        # CHECK 12 FIX: HOLD requires high prob + high confidence + ≥2 distinct motifs
        # Prevents mass false holds when model recall is high
        can_hold = (
            prob >= hold_threshold and
            conf >= CONFIDENCE_THRESHOLD and
            motif_count >= 2
        )

        if can_hold:
            action = "HOLD"
        elif prob >= review_threshold:
            action = "MANUAL_REVIEW"
        elif score < 40.0:
            action = "MANUAL_REVIEW"
        else:
            action = "ALLOW"

        if action in ["HOLD", "MANUAL_REVIEW"]:
            alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"

            motif_type = "unknown_suspicious_pattern"
            if prob > 0.90 and motif_count >= 2:
                motif_type = "high_confidence_network"
            elif prob > 0.75:
                motif_type = "model_flagged_pattern"

            log_suspicious_pattern(
                db=conn,
                alert_id=alert_id,
                motif_type=motif_type,
                account_ids=[account_id],
                countries=[emb.get("country_code", "XX")],
                laundering_prob=prob
            )
            alert_count += 1

    conn.close()
    print(f"Generated {alert_count} alerts.")
    return alert_count
