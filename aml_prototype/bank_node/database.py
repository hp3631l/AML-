"""
SQLite database schema and connection management for bank nodes.

Each bank has its own SQLite database containing:
    - KYC data (LOCAL ONLY — NEVER SHARED)
    - Transaction history
    - Session metadata
    - Ledger summaries
    - Ground truth labels (for simulator)
"""

import logging
import sqlite3
import os

logger = logging.getLogger(__name__)


def create_bank_schema(db_path: str) -> sqlite3.Connection:
    """
    Create all tables for a bank node database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection with WAL mode enabled.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    cursor = conn.cursor()

    # ─── KYC Store (LOCAL ONLY — NEVER SHARED) ───────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS kyc (
        account_id TEXT PRIMARY KEY,
        customer_name TEXT,           -- NEVER leaves bank
        pan TEXT,                     -- NEVER leaves bank
        aadhaar TEXT,                 -- NEVER leaves bank
        home_address TEXT,            -- NEVER leaves bank
        exact_salary REAL,            -- NEVER leaves bank
        raw_occupation TEXT,          -- NEVER leaves bank
        occupation_code INTEGER,      -- derived, shareable as embedding
        salary_band INTEGER,          -- derived, shareable
        country_code TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ─── Transaction History ─────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        tx_id TEXT PRIMARY KEY,
        src_account_id TEXT,
        dst_account_id TEXT,
        amount REAL,
        currency TEXT DEFAULT 'USD',
        tx_type TEXT,          -- 'wire', 'ach', 'cash_deposit', 'cash_withdrawal', 'internal'
        timestamp TIMESTAMP,
        src_bank_id TEXT,
        dst_bank_id TEXT,
        src_country TEXT,
        dst_country TEXT,
        memo TEXT               -- NEVER leaves bank
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_src ON transactions(src_account_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_dst ON transactions(dst_account_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_time ON transactions(timestamp)")

    # ─── Session Metadata ────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        account_id TEXT,
        login_timestamp TIMESTAMP,
        logout_timestamp TIMESTAMP,
        session_duration_seconds REAL,
        device_fingerprint_hash TEXT,
        ip_country TEXT,
        login_method TEXT,      -- 'web', 'mobile', 'api'
        actions_count INTEGER
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sess_acct ON sessions(account_id)")

    # ─── Ledger Summary ──────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ledger_summary (
        account_id TEXT PRIMARY KEY,
        avg_tx_amount_30d REAL DEFAULT 0,
        tx_count_30d INTEGER DEFAULT 0,
        avg_tx_amount_90d REAL DEFAULT 0,
        tx_count_90d INTEGER DEFAULT 0,
        unique_counterparties_30d INTEGER DEFAULT 0,
        unique_countries_30d INTEGER DEFAULT 0,
        max_single_tx_30d REAL DEFAULT 0,
        last_updated TIMESTAMP
    )
    """)

    # ─── Ground Truth Labels (for simulator evaluation) ──────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS labels (
        account_id TEXT PRIMARY KEY,
        is_suspicious INTEGER DEFAULT 0,
        motif_type TEXT,
        scenario_id INTEGER,
        role TEXT,              -- 'source', 'intermediary', 'destination', 'benign'
        confidence REAL DEFAULT 1.0
    )
    """)

    conn.commit()
    logger.info("Created bank schema at: %s", db_path)
    return conn


def create_central_schema(db_path: str) -> sqlite3.Connection:
    """Create schema for the central aggregator database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    cursor = conn.cursor()

    # ─── Active Transactions (aggregated, 90-day window) ─────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS active_transactions (
        tx_id TEXT PRIMARY KEY,
        src_account_id TEXT,
        dst_account_id TEXT,
        amount REAL,
        tx_type TEXT,
        timestamp TIMESTAMP,
        src_bank_id TEXT,
        dst_bank_id TEXT,
        src_country TEXT,
        dst_country TEXT
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_atx_time ON active_transactions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_atx_src ON active_transactions(src_account_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_atx_dst ON active_transactions(dst_account_id)")

    # ─── Compressed Historical Vectors ───────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_vectors (
        account_id TEXT PRIMARY KEY,
        vector BLOB,               -- 64d float32 vector (256 bytes)
        last_compressed TIMESTAMP
    )
    """)

    # ─── Pattern Memory ──────────────────────────────────────────────────
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pattern_memory (
        alert_id TEXT PRIMARY KEY,
        scenario_timestamp TIMESTAMP,
        motif_type TEXT,
        account_ids TEXT,           -- JSON array
        countries TEXT,             -- JSON array
        confidence_score REAL,
        laundering_probability REAL,
        agent_decision TEXT DEFAULT 'pending',
        agent_decision_timestamp TIMESTAMP,
        agent_notes TEXT,
        model_version TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pat_motif ON pattern_memory(motif_type)")

    conn.commit()
    logger.info("Created central schema at: %s", db_path)
    return conn
