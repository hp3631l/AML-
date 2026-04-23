"""
Global configuration for the AML Prototype.

Hardware target: i7-12650HX, 16GB RAM, 6GB VRAM.
Security: TPM 2.0 + HashiCorp Vault (no SGX/TDX).
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

BANK_DB_PATHS = {
    "bank_a": os.path.join(DATA_DIR, "bank_a", "bank_a.db"),
    "bank_b": os.path.join(DATA_DIR, "bank_b", "bank_b.db"),
    "bank_c": os.path.join(DATA_DIR, "bank_c", "bank_c.db"),
    "bank_e": os.path.join(DATA_DIR, "bank_e", "bank_e.db"),
    "bank_f": os.path.join(DATA_DIR, "bank_f", "bank_f.db"),
    "bank_g": os.path.join(DATA_DIR, "bank_g", "bank_g.db"),
    "bank_h": os.path.join(DATA_DIR, "bank_h", "bank_h.db"),
}

CENTRAL_GRAPH_DB = os.path.join(DATA_DIR, "central", "graph_store.db")
CENTRAL_HISTORY_DB = os.path.join(DATA_DIR, "central", "historical_vectors.db")
MODEL_CHECKPOINT_DIR = os.path.join(DATA_DIR, "model", "checkpoints")
TRAINING_LOG_DIR = os.path.join(DATA_DIR, "model", "training_logs")

# ─── Bank Configuration ─────────────────────────────────────────────────────
BANK_PORTS = {
    "bank_a": 8001, "bank_b": 8002, "bank_c": 8003,
    "bank_e": 8005, "bank_f": 8006, "bank_g": 8007, "bank_h": 8008,
}
AGGREGATOR_PORT = 8000
DASHBOARD_PORT = 8080

ACCOUNTS_PER_BANK = {
    "bank_a": 1500,
    "bank_b": 1000,
    "bank_c": 1000,
    "bank_e": 750,
    "bank_f": 750,
    "bank_g": 750,
    "bank_h": 750,
}
TOTAL_ACCOUNTS = sum(ACCOUNTS_PER_BANK.values())  # 6500

# ─── Simulator Configuration ────────────────────────────────────────────────
NUM_SCENARIOS = 350
MIN_MOTIF_INSTANCES = 2
MAX_TOTAL_TRANSACTIONS = 200000
MAX_AVG_TX_PER_ACCOUNT_PER_MONTH = 30.0
MIN_CROSS_BANK_SUSPICIOUS_PCT = 30.0

# Scenario type distribution targets (categories overlap)
SCENARIO_DISTRIBUTION = {
    "low_and_slow": 0.50,   # ≥ 2,500 scenarios, spread over 14+ days
    "cross_country": 0.30,  # ≥ 1,500 scenarios, at least 2 countries
    "hybrid": 0.20,         # ≥ 1,000 scenarios, 2+ motif types combined
}

# Normal traffic ratio: ~80% of total transaction volume is benign
NORMAL_TRAFFIC_RATIO = 0.80

# Transaction date range for simulation
SIMULATION_DAYS = 365  # 1 year of history
ACTIVE_WINDOW_DAYS = 90

# ─── Feature Dimensions ─────────────────────────────────────────────────────
OCCUPATION_EMBEDDING_DIM = 8
OCCUPATION_VOCAB_SIZE = 30
SESSION_VECTOR_DIM = 8
LEDGER_VECTOR_DIM = 8
TRUST_HISTORY_DIM = 4
HISTORICAL_PATTERNS_DIM = 4
PROFILE_VECTOR_DIM = 34  # 8 + 1 + 1 + 8 + 8 + 4 + 4

# Pattern detection dimensions (new in v2)
PATTERN_NAMES = ["chain", "fan_in", "fan_out", "burst", "structuring", "round_trip", "mule_coordination"]
NUM_PATTERNS = len(PATTERN_NAMES)

NODE_FEATURE_DIM = 130   # profile(34) + historical(64) + temporal(9) + pattern_feats(9) + structural(14)
EDGE_FEATURE_DIM = 27    # original(22) + new(5: time_gap, repeated_pair, same_amount, near_threshold, short_gap)
COMPRESSED_MEMORY_DIM = 64

# Threshold for structuring detection (near-CTR threshold)
CTR_THRESHOLD = 10000.0
STRUCTURING_MARGIN = 0.10  # within 10% of threshold = 9000–10000

# Graph preprocessing for model training/inference
GRAPH_WINDOW_DAYS = 90
GRAPH_MAX_EDGES = 100000
GRAPH_MAX_NEIGHBORS_PER_NODE = 15
GRAPH_MIN_TX_AMOUNT = 50.0

# ─── Model Hyperparameters ───────────────────────────────────────────────────
GRAPHSAGE_HIDDEN_DIM = 128
GRAPHSAGE_OUT_DIM = 64
GRAPHSAGE_NUM_NEIGHBORS = [15, 10]  # Layer 1: 15, Layer 2: 10
BATCH_SIZE = 256
LEARNING_RATE = 0.001
DROPOUT = 0.3
POS_WEIGHT = 15.0  # Overridden dynamically in train.py based on actual class ratio
MC_DROPOUT_PASSES = 10
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 2.0

# ─── Temporal Decay ──────────────────────────────────────────────────────────
DECAY_LAMBDA = 0.01  # half-life ≈ 69 days

# ─── Trust Score Weights (Heuristic — see Section 7) ────────────────────────
TRUST_WEIGHTS = {
    "occ_sal_mismatch": -0.15,
    "sal_tx_mismatch": -0.15,
    "country_risk": -0.10,
    "session_anomaly": -0.10,
    "motif_participation": -0.20,
    "hist_pattern_count": -0.10,
    "hold_count": -0.05,
    "cross_country_chain": -0.10,
}
TRUST_BIAS = 2.0

# ─── Action Thresholds ───────────────────────────────────────────────────────
TRUST_THRESHOLD_LOW = 40    # < 40 = high risk
TRUST_THRESHOLD_MED = 70    # 40-70 = medium risk

LAUNDERING_PROB_HOLD = 0.85       # > 0.85 → recommend hold
LAUNDERING_PROB_REVIEW = 0.60     # 0.60-0.85 → manual review
CONFIDENCE_THRESHOLD = 0.70       # minimum for hold justification

# ─── Pattern Memory ──────────────────────────────────────────────────────────
CONFIRMATION_BOOST = 2.0

# ─── Security ────────────────────────────────────────────────────────────────
# Key names managed in HashiCorp Vault (backed by TPM 2.0)
VAULT_KEY_NAMES = [
    "bank_a_key", "bank_b_key", "bank_c_key",
    "bank_e_key", "bank_f_key", "bank_g_key", "bank_h_key",
    "aml_cache_key",
    "backup_key",
]

# Encryption: AES-256-GCM
# Signing: HMAC-SHA256
# Account IDs: SHA-256 hashed
ENCRYPTION_ALGORITHM = "AES-256-GCM"
SIGNING_ALGORITHM = "HMAC-SHA256"
HASH_ALGORITHM = "SHA-256"
