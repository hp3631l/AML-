"""
Run the Phase 1 Transaction Simulator.

Usage:
    python scripts/run_simulator.py

Generates 5,000 accounts, 5,000 laundering scenarios,
and stores everything in per-bank SQLite databases.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.generator import run_simulation
from config import (
    BANK_DB_PATHS,
    MAX_TOTAL_TRANSACTIONS,
    MAX_AVG_TX_PER_ACCOUNT_PER_MONTH,
    MIN_MOTIF_INSTANCES,
    MIN_CROSS_BANK_SUSPICIOUS_PCT,
)


if __name__ == "__main__":
    summary = run_simulation(seed=42)

    # Validate distribution targets
    errors = []
    if summary["low_and_slow_pct"] < 50:
        errors.append(f"Low-and-slow: {summary['low_and_slow_pct']:.1f}% < 50%")
    if summary["cross_country_pct"] < 30:
        errors.append(f"Cross-country: {summary['cross_country_pct']:.1f}% < 30%")
    if summary["hybrid_pct"] < 20:
        errors.append(f"Hybrid: {summary['hybrid_pct']:.1f}% < 20%")

    if errors:
        print("[WARN] DISTRIBUTION VALIDATION WARNINGS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[OK] All distribution targets met.")

    # Verify each motif type has >=100 scenarios
    motif_warnings = []
    for motif, count in summary["motif_distribution"].items():
        if count < MIN_MOTIF_INSTANCES:
            motif_warnings.append(f"{motif}: {count} < 100")

    if motif_warnings:
        print("[WARN] MOTIF COUNT WARNINGS:")
        for w in motif_warnings:
            print(f"  - {w}")
    else:
        print("[OK] All motif types have >=100 scenarios.")

    # Hard constraints
    constraint_errors = []
    if summary["total_transactions"] > MAX_TOTAL_TRANSACTIONS:
        constraint_errors.append(
            f"total_transactions={summary['total_transactions']} > {MAX_TOTAL_TRANSACTIONS}"
        )
    if summary["avg_tx_per_account_per_month"] > MAX_AVG_TX_PER_ACCOUNT_PER_MONTH:
        constraint_errors.append(
            f"avg_tx/account/month={summary['avg_tx_per_account_per_month']:.2f} > "
            f"{MAX_AVG_TX_PER_ACCOUNT_PER_MONTH}"
        )
    if summary["total_scenarios"] < 5000:
        constraint_errors.append(f"scenarios={summary['total_scenarios']} < 5000")
    if summary["suspicious_cross_bank_pct"] < MIN_CROSS_BANK_SUSPICIOUS_PCT:
        constraint_errors.append(
            f"cross-bank suspicious={summary['suspicious_cross_bank_pct']:.1f}% < "
            f"{MIN_CROSS_BANK_SUSPICIOUS_PCT}%"
        )

    if constraint_errors:
        print("[WARN] CONSTRAINT VALIDATION WARNINGS:")
        for e in constraint_errors:
            print(f"  - {e}")
    else:
        print("[OK] All hard constraints satisfied.")

    print()
    print("Database files created:")
    for bank_id, path in BANK_DB_PATHS.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {bank_id}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {bank_id}: {path} (NOT FOUND)")
