"""
Ground truth labeling logic.

Every account involved in a scenario receives a label.
Accounts in multiple scenarios receive the highest-risk label.
"""

from typing import List, Dict


def label_scenarios(scenarios, conn) -> Dict[str, Dict]:
    """
    Apply ground truth labels from scenarios to the database.

    Args:
        scenarios: List of LaunderingScenario objects.
        conn: SQLite connection for the bank.

    Returns:
        Dict[account_id → label dict] for all labeled accounts.
    """
    labels = {}

    for scenario in scenarios:
        for account_id in scenario.accounts:
            role = scenario.roles.get(account_id, "participant")

            # If account already labeled, keep highest risk
            if account_id in labels and labels[account_id]["is_suspicious"]:
                existing = labels[account_id]
                # Keep if existing has higher scenario_id (more complex)
                if existing.get("scenario_id", -1) > scenario.scenario_id:
                    continue

            labels[account_id] = {
                "account_id": account_id,
                "is_suspicious": True,
                "motif_type": scenario.motif_type,
                "scenario_id": scenario.scenario_id,
                "role": role,
                "confidence": 1.0,
            }

    return labels


def update_labels_in_db(conn, labels: Dict[str, Dict]) -> int:
    """
    Update label records in the database.

    Returns:
        Number of accounts labeled as suspicious.
    """
    cursor = conn.cursor()
    count = 0

    for account_id, label in labels.items():
        cursor.execute("""
        INSERT OR REPLACE INTO labels
            (account_id, is_suspicious, motif_type, scenario_id, role, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            label["account_id"],
            1 if label["is_suspicious"] else 0,
            label["motif_type"],
            label["scenario_id"],
            label["role"],
            label["confidence"],
        ))
        if label["is_suspicious"]:
            count += 1

    conn.commit()
    return count
