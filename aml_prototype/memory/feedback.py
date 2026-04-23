"""
Pattern Memory Store and Agent Feedback Loop.

Handles storing identified suspicious patterns and updating them
based on manual review by bank compliance officers.
"""

import sqlite3
import json

def log_suspicious_pattern(db: sqlite3.Connection, alert_id: str, motif_type: str, account_ids: list, countries: list, laundering_prob: float):
    """
    Log a newly detected suspicious pattern into the central pattern memory.
    """
    db.execute("""
        INSERT INTO pattern_memory 
        (alert_id, scenario_timestamp, motif_type, account_ids, countries, laundering_probability, agent_decision)
        VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, 'pending')
    """, (alert_id, motif_type, json.dumps(account_ids), json.dumps(countries), laundering_prob))
    db.commit()


def process_agent_feedback(db: sqlite3.Connection, alert_id: str, decision: str, notes: str):
    """
    Process an agent's decision ('confirm' or 'reject') on an alert.
    If confirmed, this will eventually trigger an update to the historical 
    vectors of the involved accounts, applying the CONFIRMATION_BOOST.
    """
    db.execute("""
        UPDATE pattern_memory 
        SET agent_decision = ?, agent_notes = ?, agent_decision_timestamp = CURRENT_TIMESTAMP
        WHERE alert_id = ?
    """, (decision, notes, alert_id))
    
    if decision == 'confirm':
        # Retrieve accounts involved to boost their historical risk
        row = db.execute("SELECT account_ids FROM pattern_memory WHERE alert_id=?", (alert_id,)).fetchone()
        if row:
            accounts = json.loads(row['account_ids'])
            _boost_historical_vectors(db, accounts)
            
    db.commit()


def _boost_historical_vectors(db: sqlite3.Connection, account_ids: list):
    """
    Apply CONFIRMATION_BOOST to the historical confirmed count (dim 20)
    for all accounts involved in a confirmed laundering pattern.
    """
    from config import CONFIRMATION_BOOST
    import numpy as np
    from memory.compression import HistoricalVector
    
    for acc in account_ids:
        row = db.execute("SELECT vector FROM historical_vectors WHERE account_id=?", (acc,)).fetchone()
        if row and row['vector']:
            hist = HistoricalVector(acc, row['vector'])
            hist.vec[20] += CONFIRMATION_BOOST  # Boost confirmed suspicious dimension
            
            db.execute("""
                UPDATE historical_vectors SET vector = ? WHERE account_id = ?
            """, (hist.to_bytes(), acc))
