"""
Long-Term Memory System and Temporal Decay.

Handles the compression of transactions older than the 90-day active window
into a 64-dimensional historical risk vector.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime

class HistoricalVector:
    """
    Manages the 64-dimensional compressed historical memory for an account.
    """
    def __init__(self, account_id: str, vector_bytes: bytes = None):
        self.account_id = account_id
        if vector_bytes:
            self.vec = np.frombuffer(vector_bytes, dtype=np.float32).copy()
        else:
            self.vec = np.zeros(64, dtype=np.float32)

    def to_bytes(self) -> bytes:
        return self.vec.tobytes()

    def update_motif_participation(self, expired_txs: list):
        """Dims 0-7: Motif participation counts (decayed over time)."""
        motif_names = [
            "fan_out", "fan_in", "chain", "low_and_slow_chain",
            "scatter_gather", "peel_off", "recursive_loop", "agentic_bot"
        ]
        for tx in expired_txs:
            # Convert row to dict for safe access
            tx_dict = dict(tx)
            memo = tx_dict.get('memo', '')
            if not memo:
                continue
            for i, motif in enumerate(motif_names):
                if motif in memo:
                    self.vec[i] += 1.0
                    break

    def update_country_diversity(self, expired_txs: list):
        """Dim 14: Count distinct countries in history."""
        countries = set()
        for tx in expired_txs:
            tx_dict = dict(tx)
            if tx_dict.get('src_country'):
                countries.add(tx_dict['src_country'])
            if tx_dict.get('dst_country'):
                countries.add(tx_dict['dst_country'])
        self.vec[14] += len(countries)

    def update_bank_diversity(self, expired_txs: list):
        """Dim 15: Count distinct banks."""
        banks = set()
        for tx in expired_txs:
            tx_dict = dict(tx)
            if tx_dict.get('src_bank_id'):
                banks.add(tx_dict['src_bank_id'])
            if tx_dict.get('dst_bank_id'):
                banks.add(tx_dict['dst_bank_id'])
        self.vec[15] += len(banks)

    def update_hold_count(self, count: int):
        """Dim 16: Number of holds."""
        self.vec[16] += count

    def apply_temporal_decay(self, days_elapsed: float, lambda_decay: float = 0.01):
        """
        Apply exponential decay to historical features.
        weight = exp(-lambda * days)

        CHECK 8 FIX: Maximum-type and hold-count dimensions are protected
        from full decay so the system permanently remembers prior holds
        and never forgets a worst-case risk signal.

        Protected dims:
            16: hold_count — decays slowly (50% per cycle, floor=1 if ever held)
            17: max_amount_ever — never decayed (worst-case signal)
            18: max_risk_score_ever — never decayed (worst-case signal)
        """
        decay_factor = np.exp(-lambda_decay * days_elapsed)

        # Decay all frequency/count dims except the protected ones
        protected = {16, 17, 18}
        mask = np.ones(len(self.vec), dtype=np.float32)
        for p in protected:
            if p < len(mask):
                mask[p] = 0.0
        self.vec *= (mask * decay_factor + (1.0 - mask))

        # Slow decay for hold_count: halve per compression cycle, floor at 1
        if self.vec[16] > 0:
            self.vec[16] = max(self.vec[16] * 0.5, 1.0)
        # Dims 17, 18 (max_amount_ever, max_risk_ever): never decayed — already excluded above


def compress_expired_transactions(account_id: str, bank_db: sqlite3.Connection, central_db: sqlite3.Connection):
    """
    Compress transactions older than 90 days into historical vector.
    """
    # FIX: Ensure row_factory is set so column-name access works
    bank_db.row_factory = sqlite3.Row
    central_db.row_factory = sqlite3.Row

    expired = bank_db.execute("""
        SELECT * FROM transactions
        WHERE (src_account_id = ? OR dst_account_id = ?)
        AND timestamp < datetime('now', '-90 days')
    """, (account_id, account_id)).fetchall()

    if not expired:
        return

    row = central_db.execute(
        "SELECT vector, last_compressed FROM historical_vectors WHERE account_id=?",
        (account_id,)
    ).fetchone()

    if row and row['vector']:
        hist = HistoricalVector(account_id, row['vector'])

        # Apply decay based on time since last compression
        if row['last_compressed']:
            last_date = datetime.fromisoformat(row['last_compressed'])
            days_elapsed = (datetime.now() - last_date).total_seconds() / 86400.0
            hist.apply_temporal_decay(days_elapsed)
    else:
        hist = HistoricalVector(account_id)

    # Update logic
    hist.update_motif_participation(expired)
    hist.update_country_diversity(expired)
    hist.update_bank_diversity(expired)

    # Save back
    central_db.execute("""
        INSERT INTO historical_vectors (account_id, vector, last_compressed)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(account_id) DO UPDATE SET
        vector=excluded.vector, last_compressed=excluded.last_compressed
    """, (account_id, hist.to_bytes()))
    central_db.commit()

    # Delete compressed transactions from active window
    bank_db.execute("""
        DELETE FROM transactions
        WHERE (src_account_id = ? OR dst_account_id = ?)
        AND timestamp < datetime('now', '-90 days')
    """, (account_id, account_id))
    bank_db.commit()
