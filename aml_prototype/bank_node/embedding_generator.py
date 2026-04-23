"""
Local Embedding Generator for Bank Nodes.

Converts raw, unshareable KYC/ledger/session data into a 34-dimensional
privacy-preserving profile vector. Raw data NEVER leaves this module.
"""

import math
import numpy as np
from datetime import datetime
import sqlite3
from typing import Dict, List, Optional

from bank_node.kyc_codebook import EXPECTED_SALARY_BANDS, get_country_risk

# Deterministic 8d occupation embedding table (0-30 occupations)
np.random.seed(42)
OCCUPATION_EMBEDDINGS = {
    i: np.random.randn(8).astype(np.float32) for i in range(31)
}


def _infer_band_from_ledger(avg_tx_amount_30d: float) -> int:
    """Infer rough salary band from average transaction amount."""
    if avg_tx_amount_30d < 50: return 1
    if avg_tx_amount_30d < 200: return 2
    if avg_tx_amount_30d < 500: return 3
    if avg_tx_amount_30d < 1500: return 4
    if avg_tx_amount_30d < 3000: return 5
    if avg_tx_amount_30d < 5000: return 6
    if avg_tx_amount_30d < 10000: return 7
    if avg_tx_amount_30d < 25000: return 8
    if avg_tx_amount_30d < 50000: return 9
    return 10


def compute_mismatch_score(occupation_code: int, country_code: str, actual_salary_band: int, avg_tx_amount_30d: float) -> float:
    """Calculate occupation-salary-transaction mismatch."""
    expected_min, expected_max = EXPECTED_SALARY_BANDS.get(
        (occupation_code, country_code), (1, 10)
    )
    expected_midpoint = (expected_min + expected_max) / 2.0
    band_range = float(expected_max - expected_min + 1)
    
    # Salary band mismatch
    salary_mismatch = abs(actual_salary_band - expected_midpoint) / band_range
    
    # Transaction-implied band
    tx_implied_band = _infer_band_from_ledger(avg_tx_amount_30d)
    
    # Transaction vs salary mismatch
    tx_salary_mismatch = abs(tx_implied_band - actual_salary_band) / 10.0
    
    # Combined score (capped at 1.0)
    return min(1.0, 0.6 * salary_mismatch + 0.4 * tx_salary_mismatch)


def compute_session_vector(sessions: List[sqlite3.Row]) -> np.ndarray:
    """
    Compute 8d session feature vector from recent sessions.
    [mean_duration, std_duration, login_freq, unique_devices,
     frac_api, mean_actions, std_interval, frac_unusual_hours]
    """
    if not sessions:
        return np.zeros(8, dtype=np.float32)

    durations = [s['session_duration_seconds'] for s in sessions]
    devices = {s['device_fingerprint_hash'] for s in sessions if s['device_fingerprint_hash']}
    login_methods = [s['login_method'] for s in sessions]
    actions = [s['actions_count'] for s in sessions]

    mean_dur = np.mean(durations)
    std_dur = np.std(durations) if len(durations) > 1 else 0.0
    login_freq = len(sessions) / 4.0  # Approx per week (max 30 days)
    uniq_dev = len(devices)
    frac_api = login_methods.count('api') / len(sessions)
    mean_act = np.mean(actions)

    # Intervals
    login_times = [datetime.fromisoformat(s['login_timestamp']) for s in sessions]
    login_times.sort()
    intervals = [(login_times[i] - login_times[i-1]).total_seconds() for i in range(1, len(login_times))]
    std_interval = np.std(intervals) if len(intervals) > 1 else 0.0

    # Unusual hours (midnight to 6am)
    unusual = sum(1 for t in login_times if 0 <= t.hour < 6)
    frac_unusual = unusual / len(sessions)

    # Normalize roughly
    vec = np.array([
        np.log1p(mean_dur),
        np.log1p(std_dur),
        login_freq,
        uniq_dev,
        frac_api,
        np.log1p(mean_act),
        np.log1p(std_interval) / 10.0,
        frac_unusual
    ], dtype=np.float32)
    return vec


def compute_ledger_vector(ledger: Optional[sqlite3.Row]) -> np.ndarray:
    """
    Compute 8d ledger feature vector.
    """
    if not ledger:
        return np.zeros(8, dtype=np.float32)

    avg_30 = ledger['avg_tx_amount_30d'] or 0.0
    cnt_30 = ledger['tx_count_30d'] or 0
    avg_90 = ledger['avg_tx_amount_90d'] or 0.0
    cnt_90 = ledger['tx_count_90d'] or 0
    uniq_cp = ledger['unique_counterparties_30d'] or 0
    uniq_cc = ledger['unique_countries_30d'] or 0
    max_30 = ledger['max_single_tx_30d'] or 0.0

    velocity_ratio = cnt_30 / ((cnt_90 / 3.0) + 1.0)

    vec = np.array([
        np.log1p(avg_30),
        np.log1p(cnt_30),
        np.log1p(avg_90),
        np.log1p(cnt_90),
        uniq_cp / 100.0,
        uniq_cc / 10.0,
        np.log1p(max_30),
        velocity_ratio
    ], dtype=np.float32)
    return vec


def generate_embedding(account_id: str, db: sqlite3.Connection, bank_id: str) -> Optional[Dict]:
    """Generate the full 34d embedding for a single account."""
    kyc = db.execute("SELECT * FROM kyc WHERE account_id=?", (account_id,)).fetchone()
    if not kyc:
        return None

    sessions = db.execute(
        "SELECT * FROM sessions WHERE account_id=? ORDER BY login_timestamp DESC LIMIT 30",
        (account_id,)
    ).fetchall()
    ledger = db.execute("SELECT * FROM ledger_summary WHERE account_id=?", (account_id,)).fetchone()

    # 1. Occupation (8d)
    occ_code = kyc['occupation_code'] or 0
    occupation_emb = OCCUPATION_EMBEDDINGS.get(occ_code, OCCUPATION_EMBEDDINGS[0])

    # 2. Salary band (1d)
    salary_band = kyc['salary_band'] or 1
    salary_feat = np.array([salary_band / 10.0], dtype=np.float32)

    # 3. Country risk (1d)
    country_code = kyc['country_code'] or "US"
    country_risk = get_country_risk(country_code)
    country_feat = np.array([country_risk], dtype=np.float32)

    # 4. Session vector (8d)
    session_vec = compute_session_vector(sessions)
    
    # 5. Ledger vector (8d)
    ledger_vec = compute_ledger_vector(ledger)
    
    # Calculate mismatch score
    avg_30d = (ledger['avg_tx_amount_30d'] if ledger and ledger['avg_tx_amount_30d'] is not None else 0.0)
    mismatch = compute_mismatch_score(occ_code, country_code, salary_band, avg_30d)
    
    # 6. Trust score history & historical patterns (8d total)
    # Populate with lightweight behavioral signals until full history is stored.
    trust_and_history = np.zeros(8, dtype=np.float32)
    if ledger:
        unique_countries = ledger['unique_countries_30d'] or 0
        trust_and_history[1] = min(1.0, unique_countries / 5.0)
    trust_and_history[0] = mismatch

    # Combine all (8 + 1 + 1 + 8 + 8 + 8 = 34d)
    profile_vec = np.concatenate([
        occupation_emb,
        salary_feat,
        country_feat,
        session_vec,
        ledger_vec,
        trust_and_history
    ])

    return {
        "account_id": account_id,
        "bank_id": bank_id,
        "profile_vector": profile_vec.tolist(),
        "occupation_code": occ_code,
        "salary_band": salary_band,
        "country_code": country_code,
        "country_risk": float(country_risk),
        "mismatch_score": float(mismatch)
    }
