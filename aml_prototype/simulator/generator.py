"""
Main simulator: orchestrates account generation, scenario creation,
normal traffic, session metadata, labeling, and DB population.

Generates:
    - 5,000 accounts across 3 banks (2000 + 1500 + 1500)
    - 5,000 unique laundering scenarios
    - Normal background traffic (~80% of volume)
    - Session metadata for all accounts
    - Ground truth labels

All data is stored in per-bank SQLite databases.
"""

import os
import sys
import time
import sqlite3
from datetime import datetime
from typing import Dict, List

import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    BANK_DB_PATHS, TOTAL_ACCOUNTS, NUM_SCENARIOS,
    ACCOUNTS_PER_BANK, SIMULATION_DAYS,
    MAX_TOTAL_TRANSACTIONS, MAX_AVG_TX_PER_ACCOUNT_PER_MONTH,
    MIN_MOTIF_INSTANCES, MIN_CROSS_BANK_SUSPICIOUS_PCT,
)
from bank_node.database import create_bank_schema
from simulator.accounts import generate_accounts, insert_accounts_to_db
from simulator.scenarios import generate_all_scenarios
from simulator.normal_traffic import generate_normal_transactions, generate_sessions
from simulator.labels import label_scenarios, update_labels_in_db


SIMULATION_START = datetime(2024, 1, 1)


def build_accounts_map(accounts_by_bank: Dict[str, List]) -> Dict:
    """Build a flat lookup: account_id → {bank_id, country_code}."""
    amap = {}
    for bank_id, accounts in accounts_by_bank.items():
        for acc in accounts:
            amap[acc.account_id] = {
                "bank_id": bank_id,
                "country_code": acc.country_code,
                "occupation_code": acc.occupation_code,
                "salary_band": acc.salary_band,
            }
    return amap


def insert_transactions(conn, transactions: List[Dict]) -> int:
    """Insert transactions into a bank's database. Returns count."""
    cursor = conn.cursor()
    count = 0
    for tx in transactions:
        try:
            cursor.execute("""
            INSERT OR IGNORE INTO transactions
                (tx_id, src_account_id, dst_account_id, amount, currency,
                 tx_type, timestamp, src_bank_id, dst_bank_id,
                 src_country, dst_country, memo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx["tx_id"], tx["src_account_id"], tx["dst_account_id"],
                tx["amount"], tx["currency"], tx["tx_type"],
                tx["timestamp"], tx["src_bank_id"], tx["dst_bank_id"],
                tx["src_country"], tx["dst_country"], tx["memo"],
            ))
            count += 1
        except sqlite3.IntegrityError:
            pass  # duplicate tx_id
    conn.commit()
    return count


def insert_sessions(conn, sessions: List[Dict]) -> int:
    """Insert sessions into a bank's database. Returns count."""
    cursor = conn.cursor()
    count = 0
    for sess in sessions:
        try:
            cursor.execute("""
            INSERT OR IGNORE INTO sessions
                (session_id, account_id, login_timestamp, logout_timestamp,
                 session_duration_seconds, device_fingerprint_hash,
                 ip_country, login_method, actions_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sess["session_id"], sess["account_id"],
                sess["login_timestamp"], sess["logout_timestamp"],
                sess["session_duration_seconds"],
                sess["device_fingerprint_hash"],
                sess["ip_country"], sess["login_method"],
                sess["actions_count"],
            ))
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return count


def update_ledger_summaries(conn, sim_end_date: str = "2024-12-31") -> None:
    """
    Compute and update ledger summaries from transactions.
    Uses sim_end_date as the reference 'now' since all data is historical.
    """
    cursor = conn.cursor()

    cursor.execute(f"""
    INSERT OR REPLACE INTO ledger_summary
        (account_id, avg_tx_amount_30d, tx_count_30d,
         avg_tx_amount_90d, tx_count_90d,
         unique_counterparties_30d, unique_countries_30d,
         max_single_tx_30d, last_updated)
    SELECT
        src_account_id,
        AVG(CASE WHEN timestamp >= datetime('{sim_end_date}', '-30 days') THEN amount END),
        COUNT(CASE WHEN timestamp >= datetime('{sim_end_date}', '-30 days') THEN 1 END),
        AVG(CASE WHEN timestamp >= datetime('{sim_end_date}', '-90 days') THEN amount END),
        COUNT(CASE WHEN timestamp >= datetime('{sim_end_date}', '-90 days') THEN 1 END),
        COUNT(DISTINCT CASE WHEN timestamp >= datetime('{sim_end_date}', '-30 days')
              THEN dst_account_id END),
        COUNT(DISTINCT CASE WHEN timestamp >= datetime('{sim_end_date}', '-30 days')
              THEN dst_country END),
        MAX(CASE WHEN timestamp >= datetime('{sim_end_date}', '-30 days') THEN amount END),
        datetime('now')
    FROM transactions
    GROUP BY src_account_id
    """)
    conn.commit()



def run_simulation(seed: int = 42) -> Dict:
    """
    Run the full Phase 1 simulation.

    Returns:
        Summary statistics dict.
    """
    print("=" * 70)
    print("AML PROTOTYPE — PHASE 1: TRANSACTION SIMULATOR")
    print("=" * 70)
    print()

    t_start = time.time()

    # ── Step 1: Generate accounts ────────────────────────────────────────
    print("[1/6] Generating accounts...")
    accounts_by_bank = generate_accounts(seed=seed)
    accounts_map = build_accounts_map(accounts_by_bank)
    all_account_ids = list(accounts_map.keys())
    all_accounts_flat = []
    for bank_accounts in accounts_by_bank.values():
        all_accounts_flat.extend(bank_accounts)

    # ── Step 2: Create databases and insert accounts ─────────────────────
    print("[2/6] Creating databases and inserting accounts...")
    db_connections = {}
    for bank_id, db_path in BANK_DB_PATHS.items():
        # Remove existing DB for fresh start
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = create_bank_schema(db_path)
        insert_accounts_to_db(conn, accounts_by_bank[bank_id])
        db_connections[bank_id] = conn
        print(f"  {bank_id}: {len(accounts_by_bank[bank_id])} accounts -> {db_path}")

    # ── Step 3: Generate laundering scenarios ────────────────────────────
    scenario_target = max(5000, NUM_SCENARIOS)
    print(f"[3/6] Generating {scenario_target} laundering scenarios...")
    scenarios = generate_all_scenarios(
        all_account_ids=all_account_ids,
        accounts_map=accounts_map,
        num_scenarios=scenario_target,
        simulation_start=SIMULATION_START,
        simulation_days=SIMULATION_DAYS,
        seed=seed,
    )
    print(f"  Generated {len(scenarios)} scenarios.")

    # Insert scenario transactions into appropriate bank DBs
    scenario_tx_count = 0
    hard_positive_scenarios = 0
    suspicious_cross_bank_scenarios = 0
    suspicious_by_bank = {"bank_a": [], "bank_b": [], "bank_c": []}
    suspicious_sessions_by_bank = {"bank_a": [], "bank_b": [], "bank_c": []}

    for scenario in scenarios:
        if scenario.is_hard_positive:
            hard_positive_scenarios += 1
        if scenario.has_cross_bank:
            suspicious_cross_bank_scenarios += 1

        for tx in scenario.transactions:
            src_bank = tx.get("src_bank_id", "bank_a")
            if src_bank in suspicious_by_bank:
                suspicious_by_bank[src_bank].append(tx)

        # Insert bot sessions
        for sess in scenario.sessions:
            acc_info = accounts_map.get(sess["account_id"], {})
            bank = acc_info.get("bank_id", "bank_a")
            if bank in suspicious_sessions_by_bank:
                suspicious_sessions_by_bank[bank].append(sess)

    planned_suspicious_tx = sum(len(v) for v in suspicious_by_bank.values())
    if planned_suspicious_tx > MAX_TOTAL_TRANSACTIONS:
        print(f"  Capping suspicious transactions to {MAX_TOTAL_TRANSACTIONS} from {planned_suspicious_tx}...")
        for bank_id in ["bank_a", "bank_b", "bank_c"]:
            share = len(suspicious_by_bank[bank_id]) / max(1, planned_suspicious_tx)
            keep_n = int(MAX_TOTAL_TRANSACTIONS * share)
            suspicious_by_bank[bank_id] = suspicious_by_bank[bank_id][:keep_n]

    for bank_id in ["bank_a", "bank_b", "bank_c"]:
        scenario_tx_count += insert_transactions(db_connections[bank_id], suspicious_by_bank[bank_id])
        insert_sessions(db_connections[bank_id], suspicious_sessions_by_bank[bank_id])

    print(f"  Inserted {scenario_tx_count} suspicious transactions.")

    # ── Step 4: Generate normal background traffic ───────────────────────
    print("[4/6] Generating normal background traffic...")
    normal_tx_total = 0
    hard_negative_accounts_total = 0
    hard_negative_tx_total = 0
    remaining_budget = max(0, int(MAX_TOTAL_TRANSACTIONS - scenario_tx_count))
    accounts_total = sum(len(v) for v in accounts_by_bank.values())
    budget_used = 0

    for bank_id, accounts in accounts_by_bank.items():
        proportion = len(accounts) / max(1, accounts_total)
        bank_budget = int(remaining_budget * proportion)
        # Put any rounding remainder into the final bank.
        if bank_id == "bank_c":
            bank_budget = max(0, remaining_budget - budget_used)

        normal_txs, normal_meta = generate_normal_transactions(
            accounts=accounts,
            accounts_map=accounts_map,
            all_account_ids=all_account_ids,
            start_date=SIMULATION_START,
            num_days=SIMULATION_DAYS,
            seed=seed + hash(bank_id) % 1000,
            tx_budget=bank_budget,
        )
        count = insert_transactions(db_connections[bank_id], normal_txs)
        normal_tx_total += count
        budget_used += count
        hard_negative_accounts_total += int(normal_meta["hard_negative_accounts"])
        hard_negative_tx_total += int(normal_meta["hard_negative_transactions"])
        print(f"  {bank_id}: {count} normal transactions")

    # ── Step 5: Generate and insert sessions ─────────────────────────────
    print("[5/6] Generating session metadata...")
    sessions_by_bank = generate_sessions(
        accounts=all_accounts_flat,
        accounts_map=accounts_map,
        start_date=SIMULATION_START,
        num_days=SIMULATION_DAYS,
        seed=seed,
    )
    session_total = 0
    for bank_id, sessions in sessions_by_bank.items():
        count = insert_sessions(db_connections[bank_id], sessions)
        session_total += count
        print(f"  {bank_id}: {count} sessions")

    # ── Step 6: Apply labels and compute ledger summaries ────────────────
    print("[6/6] Applying labels and computing ledger summaries...")
    total_suspicious = 0
    for bank_id, conn in db_connections.items():
        # Get scenarios involving this bank's accounts
        bank_account_ids = {a.account_id for a in accounts_by_bank[bank_id]}
        bank_scenarios = [
            s for s in scenarios
            if any(a in bank_account_ids for a in s.accounts)
        ]
        labels = label_scenarios(bank_scenarios, conn)
        # Filter to this bank's accounts
        bank_labels = {
            k: v for k, v in labels.items() if k in bank_account_ids
        }
        count = update_labels_in_db(conn, bank_labels)
        total_suspicious += count
        print(f"  {bank_id}: {count} suspicious accounts labeled")

        # Update ledger summaries
        update_ledger_summaries(conn)

    # ── Close connections ────────────────────────────────────────────────
    for conn in db_connections.values():
        conn.close()

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    # Distribution validation
    low_and_slow = sum(1 for s in scenarios if s.is_low_and_slow)
    cross_country = sum(1 for s in scenarios if s.is_cross_country)
    hybrid = sum(1 for s in scenarios if s.is_hybrid)

    # Motif type distribution
    motif_counts = {}
    for s in scenarios:
        motif_counts[s.motif_type] = motif_counts.get(s.motif_type, 0) + 1

    summary = {
        "total_accounts": TOTAL_ACCOUNTS,
        "total_scenarios": len(scenarios),
        "total_suspicious_transactions": scenario_tx_count,
        "total_normal_transactions": normal_tx_total,
        "total_transactions": scenario_tx_count + normal_tx_total,
        "total_sessions": session_total,
        "total_suspicious_accounts": total_suspicious,
        "hard_negative_accounts": hard_negative_accounts_total,
        "hard_negative_transactions": hard_negative_tx_total,
        "hard_positive_scenarios": hard_positive_scenarios,
        "suspicious_cross_bank_scenarios": suspicious_cross_bank_scenarios,
        "suspicious_cross_bank_pct": (suspicious_cross_bank_scenarios / max(1, len(scenarios))) * 100.0,
        "low_and_slow_count": low_and_slow,
        "low_and_slow_pct": low_and_slow / len(scenarios) * 100,
        "cross_country_count": cross_country,
        "cross_country_pct": cross_country / len(scenarios) * 100,
        "hybrid_count": hybrid,
        "hybrid_pct": hybrid / len(scenarios) * 100,
        "motif_distribution": motif_counts,
        "avg_tx_per_account_per_month": (
            (scenario_tx_count + normal_tx_total) /
            max(1.0, TOTAL_ACCOUNTS * (SIMULATION_DAYS / 30.0))
        ),
        "elapsed_seconds": round(elapsed, 2),
    }

    summary["constraint_checks"] = {
        "total_transactions_le_200k": summary["total_transactions"] <= MAX_TOTAL_TRANSACTIONS,
        "avg_tx_per_account_month_le_30": (
            summary["avg_tx_per_account_per_month"] <= MAX_AVG_TX_PER_ACCOUNT_PER_MONTH
        ),
        "scenarios_ge_5000": summary["total_scenarios"] >= 5000,
        "cross_bank_suspicious_ge_30pct": (
            summary["suspicious_cross_bank_pct"] >= MIN_CROSS_BANK_SUSPICIOUS_PCT
        ),
        "each_motif_ge_100": all(
            count >= MIN_MOTIF_INSTANCES for count in summary["motif_distribution"].values()
        ),
    }

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Accounts:           {summary['total_accounts']}")
    print(f"  Scenarios:          {summary['total_scenarios']}")
    print(f"  Suspicious TX:      {summary['total_suspicious_transactions']}")
    print(f"  Normal TX:          {summary['total_normal_transactions']}")
    print(f"  Total TX:           {summary['total_transactions']}")
    print(f"  Avg TX/acct/month:  {summary['avg_tx_per_account_per_month']:.2f}")
    print(f"  Sessions:           {summary['total_sessions']}")
    print(f"  Suspicious Accts:   {summary['total_suspicious_accounts']}")
    print(f"  Hard Neg Accts:     {summary['hard_negative_accounts']}")
    print(f"  Hard Neg TX:        {summary['hard_negative_transactions']}")
    print(f"  Hard Pos Scenarios: {summary['hard_positive_scenarios']}")
    print(f"  Cross-bank Susp:    {summary['suspicious_cross_bank_scenarios']} "
          f"({summary['suspicious_cross_bank_pct']:.1f}%)")
    print()
    print("  Distribution Validation:")
    print(f"    Low-and-slow:   {summary['low_and_slow_count']} "
          f"({summary['low_and_slow_pct']:.1f}%) -- target >=50%")
    print(f"    Cross-country:  {summary['cross_country_count']} "
          f"({summary['cross_country_pct']:.1f}%) -- target >=30%")
    print(f"    Hybrid:         {summary['hybrid_count']} "
          f"({summary['hybrid_pct']:.1f}%) -- target >=20%")
    print()
    print("  Motif Type Distribution:")
    for motif, count in sorted(motif_counts.items(), key=lambda x: -x[1]):
        print(f"    {motif:30s} {count:5d}")
    print()
    print("  Constraint Checks:")
    for k, v in summary["constraint_checks"].items():
        print(f"    {k:35s} {'PASS' if v else 'FAIL'}")
    print()
    print(f"  Time elapsed: {summary['elapsed_seconds']}s")
    print()

    return summary
