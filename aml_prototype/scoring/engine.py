"""
Trust Score Engine and Action Recommendation.

Implements the heuristic trust score and hold recommendation engine based on
the outputs of the Temporal GraphSAGE model and historical features.
"""

import math

def compute_trust_score(features: dict) -> float:
    """
    Computes a heuristic Trust Score (0-100) based on multiple risk signals.
    Note: This is a heuristic for prioritization and UI color-coding,
    not a calibrated statistical probability.
    
    Expected features:
        - occ_sal_mismatch (float [0, 1])
        - sal_tx_mismatch (float [0, 1])
        - country_risk (float [0, 1])
        - session_anomaly (float [0, 1])
        - motif_participation (float [0, 1])
        - hist_pattern_count (int)
        - hold_count (int)
        - cross_country_chain (int)
    """
    w1 = -0.15   # occupation-salary mismatch
    w2 = -0.15   # salary-transaction mismatch
    w3 = -0.10   # country risk
    w4 = -0.10   # session anomaly
    w5 = -0.20   # current motif participation
    w6 = -0.10   # historical confirmed pattern count
    w7 = -0.05   # previous hold count
    w8 = -0.10   # cross-country chain involvement
    bias = 2.0   # baseline bias
    
    raw = (
        bias +
        w1 * features.get('occ_sal_mismatch', 0.0) +
        w2 * features.get('sal_tx_mismatch', 0.0) +
        w3 * features.get('country_risk', 0.0) +
        w4 * features.get('session_anomaly', 0.0) +
        w5 * features.get('motif_participation', 0.0) +
        w6 * min(features.get('hist_pattern_count', 0) / 5.0, 1.0) +
        w7 * min(features.get('hold_count', 0) / 3.0, 1.0) +
        w8 * min(features.get('cross_country_chain', 0) / 5.0, 1.0)
    )
    
    trust_score = 100.0 * (1.0 / (1.0 + math.exp(-raw)))
    return round(trust_score, 2)


def get_recommendation(trust_score: float, laundering_prob: float, confidence: float) -> str:
    """
    Determine the recommended action based on thresholds defined in config.py.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import LAUNDERING_PROB_HOLD, LAUNDERING_PROB_REVIEW, CONFIDENCE_THRESHOLD

    # The GraphSAGE probability is the primary driver for holds
    if laundering_prob > LAUNDERING_PROB_HOLD and confidence >= CONFIDENCE_THRESHOLD:
        return "HOLD"
    elif laundering_prob > LAUNDERING_PROB_REVIEW:
        return "MANUAL_REVIEW"
        
    # Trust score acts as a secondary heuristic for review queues
    if trust_score < 40.0:
        return "MANUAL_REVIEW"
        
    return "ALLOW"
