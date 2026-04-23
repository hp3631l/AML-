"""
Laundering motif generators.

8 base motif types from Section 4 of the implementation plan:
    1. Recursive Loop
    2. Peel-Off
    3. Scatter-Gather
    4. Fan-In
    5. Fan-Out
    6. Burst
    7. Chain
    8. Agentic Bot Signature

Each generator returns a list of transaction dicts ready for DB insertion.
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np

from simulator.distributions import sample_amount, sample_delay_hours, sample_retention_rate


def _tx(src: str, dst: str, amount: float, timestamp: datetime,
        src_bank: str, dst_bank: str, src_country: str, dst_country: str,
        tx_type: str = "wire") -> Dict:
    """Create a transaction dict."""
    return {
        "tx_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
        "src_account_id": src,
        "dst_account_id": dst,
        "amount": round(amount, 2),
        "currency": "USD",
        "tx_type": tx_type,
        "timestamp": timestamp.isoformat(),
        "src_bank_id": src_bank,
        "dst_bank_id": dst_bank,
        "src_country": src_country,
        "dst_country": dst_country,
        "memo": f"Transfer {uuid.uuid4().hex[:8]}",
    }


def _get_info(account, accounts_map):
    """Get bank_id and country for an account."""
    info = accounts_map.get(account)
    if info:
        return info["bank_id"], info["country_code"]
    return "bank_a", "US"  # fallback


# ─── 1. Recursive Loop ──────────────────────────────────────────────────────

def generate_recursive_loop(
    accounts: List[str],
    accounts_map: Dict,
    start_date: datetime,
    period_days: int = 14,
    num_cycles: int = 3,
    amount_range: Tuple[float, float] = (500, 9500),
    amount_dist: str = "uniform",
    delay_dist: str = "exponential",
) -> List[Dict]:
    """
    Generate a recursive loop: A → B → C → ... → A repeated.

    Args:
        accounts: List of ≥3 account IDs forming the cycle.
        accounts_map: Dict[account_id → {bank_id, country_code}].
        start_date: Base timestamp.
        period_days: Days between cycle repetitions (7–30).
        num_cycles: Number of cycle repetitions (≥2).
        amount_range: (min, max) transaction amount.

    Returns:
        List of transaction dicts.
    """
    assert len(accounts) >= 3, "Recursive loop needs ≥3 accounts"
    transactions = []

    for cycle in range(num_cycles):
        base_time = start_date + timedelta(days=cycle * period_days)
        current_time = base_time

        for i in range(len(accounts)):
            src = accounts[i]
            dst = accounts[(i + 1) % len(accounts)]
            src_bank, src_country = _get_info(src, accounts_map)
            dst_bank, dst_country = _get_info(dst, accounts_map)

            delay = sample_delay_hours(delay_dist, 2, 24)
            current_time += timedelta(hours=delay)

            amount = sample_amount(amount_dist, amount_range)
            amount *= (0.95 ** i)  # slight decay per hop

            transactions.append(_tx(
                src, dst, amount, current_time,
                src_bank, dst_bank, src_country, dst_country
            ))

    return transactions


# ─── 2. Peel-Off ─────────────────────────────────────────────────────────────

def generate_peel_off(
    source: str,
    intermediaries: List[str],
    accounts_map: Dict,
    start_date: datetime,
    initial_amount: float = 25000,
    amount_dist: str = "uniform",
    delay_dist: str = "exponential",
) -> List[Dict]:
    """
    Generate a peel-off chain: source → intermediaries with retention.

    Each intermediary keeps 5-30% and forwards the rest.
    """
    transactions = []
    current_amount = initial_amount
    current_src = source
    current_time = start_date

    for intermediary in intermediaries:
        src_bank, src_country = _get_info(current_src, accounts_map)
        dst_bank, dst_country = _get_info(intermediary, accounts_map)

        delay = sample_delay_hours(delay_dist, 6, 48)
        current_time += timedelta(hours=delay)

        transactions.append(_tx(
            current_src, intermediary, current_amount, current_time,
            src_bank, dst_bank, src_country, dst_country
        ))

        retention = sample_retention_rate()
        current_amount *= retention
        current_src = intermediary

    return transactions


# ─── 3. Scatter-Gather ───────────────────────────────────────────────────────

def generate_scatter_gather(
    source: str,
    intermediaries: List[str],
    destination: str,
    accounts_map: Dict,
    start_date: datetime,
    total_amount: float = 50000,
    delay_days: int = 5,
    amount_dist: str = "uniform",
) -> List[Dict]:
    """
    Generate scatter-gather: source fans out, then intermediaries gather.

    Fan-out window: 0-48 hours. Fan-in after delay_days.
    """
    n = len(intermediaries)
    assert n >= 5, "Scatter-gather needs ≥5 intermediaries"

    # Random split (Dirichlet)
    splits = np.random.dirichlet(np.ones(n)) * total_amount
    transactions = []

    # Fan-out
    fan_out_time = start_date
    src_bank, src_country = _get_info(source, accounts_map)

    for i, inter in enumerate(intermediaries):
        dst_bank, dst_country = _get_info(inter, accounts_map)
        t = fan_out_time + timedelta(hours=np.random.uniform(0, 48))
        transactions.append(_tx(
            source, inter, splits[i], t,
            src_bank, dst_bank, src_country, dst_country
        ))

    # Fan-in
    fan_in_time = start_date + timedelta(days=delay_days)
    dst_bank, dst_country = _get_info(destination, accounts_map)

    for i, inter in enumerate(intermediaries):
        src_bank2, src_country2 = _get_info(inter, accounts_map)
        t = fan_in_time + timedelta(hours=np.random.uniform(0, 48))
        fee = splits[i] * 0.02  # 2% fee
        transactions.append(_tx(
            inter, destination, splits[i] - fee, t,
            src_bank2, dst_bank, src_country2, dst_country
        ))

    return transactions


# ─── 4. Fan-In ───────────────────────────────────────────────────────────────

def generate_fan_in(
    sources: List[str],
    destination: str,
    accounts_map: Dict,
    start_date: datetime,
    amount_range: Tuple[float, float] = (1000, 9000),
    span_days: int = 7,
    amount_dist: str = "uniform",
) -> List[Dict]:
    """
    Multiple unrelated sources send to a single destination.

    No preceding fan-out. Amounts are similar (low CV).
    """
    assert len(sources) >= 3
    transactions = []
    dst_bank, dst_country = _get_info(destination, accounts_map)

    # Base amount with low variance
    base_amount = np.random.uniform(*amount_range)

    for src in sources:
        src_bank, src_country = _get_info(src, accounts_map)
        amount = base_amount * np.random.uniform(0.85, 1.15)
        t = start_date + timedelta(
            hours=np.random.uniform(0, span_days * 24)
        )
        transactions.append(_tx(
            src, destination, amount, t,
            src_bank, dst_bank, src_country, dst_country
        ))

    return transactions


# ─── 5. Fan-Out ──────────────────────────────────────────────────────────────

def generate_fan_out(
    source: str,
    destinations: List[str],
    accounts_map: Dict,
    start_date: datetime,
    amount_range: Tuple[float, float] = (1000, 9500),
    span_days: int = 3,
    amount_dist: str = "just_below_threshold",
) -> List[Dict]:
    """
    Single source sends to multiple destinations.

    Amounts structured below reporting thresholds.
    """
    assert len(destinations) >= 3
    transactions = []
    src_bank, src_country = _get_info(source, accounts_map)

    for dst in destinations:
        dst_bank, dst_country = _get_info(dst, accounts_map)
        amount = sample_amount(amount_dist, amount_range)
        t = start_date + timedelta(
            hours=np.random.uniform(0, span_days * 24)
        )
        transactions.append(_tx(
            source, dst, amount, t,
            src_bank, dst_bank, src_country, dst_country
        ))

    return transactions


# ─── 6. Burst ────────────────────────────────────────────────────────────────

def generate_burst(
    account: str,
    counterparties: List[str],
    accounts_map: Dict,
    start_date: datetime,
    window_hours: float = 12,
    num_transactions: int = 15,
    amount_range: Tuple[float, float] = (200, 9500),
    amount_dist: str = "uniform",
) -> List[Dict]:
    """
    Burst: many transactions from one account in a short window.

    ≥10 transactions in ≤24 hours.
    """
    assert num_transactions >= 10
    transactions = []
    src_bank, src_country = _get_info(account, accounts_map)

    for i in range(num_transactions):
        dst = random.choice(counterparties)
        dst_bank, dst_country = _get_info(dst, accounts_map)
        amount = sample_amount(amount_dist, amount_range)
        t = start_date + timedelta(
            hours=np.random.uniform(0, window_hours)
        )
        transactions.append(_tx(
            account, dst, amount, t,
            src_bank, dst_bank, src_country, dst_country
        ))

    return transactions


# ─── 7. Chain ────────────────────────────────────────────────────────────────

def generate_chain(
    accounts: List[str],
    accounts_map: Dict,
    start_date: datetime,
    amount: float = 10000,
    delay_dist: str = "exponential",
    delay_scale_hours: float = 48,
) -> List[Dict]:
    """
    Linear chain: A → B → C → ... → Z.

    ≥4 hops (5 accounts). Amount decreases by small fee per hop.
    """
    assert len(accounts) >= 5, "Chain needs ≥5 accounts"
    transactions = []
    current_amount = amount
    current_time = start_date

    for i in range(len(accounts) - 1):
        src = accounts[i]
        dst = accounts[i + 1]
        src_bank, src_country = _get_info(src, accounts_map)
        dst_bank, dst_country = _get_info(dst, accounts_map)

        delay = sample_delay_hours(delay_dist, 6, delay_scale_hours * 2)
        current_time += timedelta(hours=delay)

        fee = current_amount * np.random.uniform(0.01, 0.05)
        current_amount -= fee

        transactions.append(_tx(
            src, dst, current_amount, current_time,
            src_bank, dst_bank, src_country, dst_country
        ))

    return transactions


# ─── 8. Agentic Bot Signature ────────────────────────────────────────────────

def generate_agentic_bot(
    bot_account: str,
    targets: List[str],
    accounts_map: Dict,
    start_date: datetime,
    num_sessions: int = 20,
    amount_range: Tuple[float, float] = (500, 9000),
    amount_dist: str = "uniform",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Agentic bot: automated transactions with bot-like session signatures.

    Returns:
        (transactions, sessions) — both as lists of dicts.

    Bot signature:
        - inter_tx_interval_cv < 0.15
        - session_duration_cv < 0.10
        - login_method = 'api'
        - actions_per_session = 1
        - uniform time distribution
    """
    transactions = []
    sessions = []
    src_bank, src_country = _get_info(bot_account, accounts_map)

    # Bot timing: very regular intervals
    base_interval_hours = np.random.uniform(4, 24)

    for i in range(num_sessions):
        dst = random.choice(targets)
        dst_bank, dst_country = _get_info(dst, accounts_map)

        # Regular interval with very low variance (bot-like)
        interval = base_interval_hours * np.random.uniform(0.92, 1.08)
        t = start_date + timedelta(hours=interval * i)

        amount = sample_amount(amount_dist, amount_range)
        transactions.append(_tx(
            bot_account, dst, amount, t,
            src_bank, dst_bank, src_country, dst_country,
            tx_type="wire"
        ))

        # Bot session: very consistent duration, always API
        session_duration = np.random.uniform(3, 7)  # seconds (very fast)
        sessions.append({
            "session_id": f"SESS-{uuid.uuid4().hex[:12].upper()}",
            "account_id": bot_account,
            "login_timestamp": t.isoformat(),
            "logout_timestamp": (t + timedelta(seconds=session_duration)).isoformat(),
            "session_duration_seconds": round(session_duration, 2),
            "device_fingerprint_hash": hashlib.sha256(
                f"bot_{bot_account}".encode()
            ).hexdigest()[:16] if False else f"BOTDEV-{bot_account[-6:]}",
            "ip_country": src_country,
            "login_method": "api",
            "actions_count": 1,
        })

    return transactions, sessions


# ─── 9. Structuring (Smurfing) ───────────────────────────────────────────────

def generate_structuring(
    src_account: str,
    dst_accounts: List[str],
    accounts_map: Dict,
    start_date: datetime,
    threshold: float = 10000.0,
    margin: float = 0.10,
    num_transactions: int = 8,
    span_hours: int = 48,
) -> List[Dict]:
    """
    Generate structuring: repeated amounts just below reporting threshold.
    e.g. 9000-9900 USD (within margin% of threshold), spread across a short window.
    """
    transactions = []
    src_bank, src_country = _get_info(src_account, accounts_map)
    lo = threshold * (1 - margin)
    hi = threshold - 1.0  # stay strictly below

    for i in range(num_transactions):
        dst = random.choice(dst_accounts)
        dst_bank, dst_country = _get_info(dst, accounts_map)
        amount = round(random.uniform(lo, hi), 2)
        t = start_date + timedelta(hours=(span_hours / num_transactions) * i + random.uniform(-0.5, 0.5))
        tx = _tx(src_account, dst, amount, t, src_bank, dst_bank, src_country, dst_country, tx_type="cash_deposit")
        tx["memo"] = f"structuring_split_{i}"
        transactions.append(tx)

    return transactions


# ─── 10. Round-Tripping ──────────────────────────────────────────────────────

def generate_round_trip(
    accounts: List[str],
    accounts_map: Dict,
    start_date: datetime,
    loop_delay_days: int = 7,
    disguise_factor: float = 0.92,
    num_hops: int = 4,
) -> List[Dict]:
    """
    Generate a round-trip: A → B → C → ... → A with temporal delays and partial
    amount disguise at each hop. The money returns to origin but smaller.
    """
    transactions = []
    if len(accounts) < 3:
        return transactions

    accs = accounts[:num_hops] if len(accounts) >= num_hops else accounts
    t = start_date

    for hop in range(len(accs)):
        src = accs[hop]
        dst = accs[(hop + 1) % len(accs)]
        src_bank, src_country = _get_info(src, accounts_map)
        dst_bank, dst_country = _get_info(dst, accounts_map)

        base_amount = 5000.0 * (disguise_factor ** hop)
        amount = round(base_amount * random.uniform(0.95, 1.05), 2)
        delay = timedelta(days=loop_delay_days) + timedelta(hours=random.uniform(-12, 12))
        t = t + delay
        tx = _tx(src, dst, amount, t, src_bank, dst_bank, src_country, dst_country, tx_type="wire")
        tx["memo"] = f"round_trip_hop{hop}"
        transactions.append(tx)

    return transactions


# ─── 11. Mule Coordination ───────────────────────────────────────────────────

def generate_mule_coordination(
    coordinator: str,
    mules: List[str],
    collector: str,
    accounts_map: Dict,
    start_date: datetime,
    amount_range: Tuple[float, float] = (800, 3500),
    sync_window_minutes: int = 30,
) -> List[Dict]:
    """
    Generate mule coordination: coordinator sends similar amounts to multiple
    mules within a tight time window; all mules then forward to a single collector.
    """
    transactions = []
    coord_bank, coord_country = _get_info(coordinator, accounts_map)
    coll_bank, coll_country = _get_info(collector, accounts_map)

    base_amount = random.uniform(*amount_range)

    # Phase 1: coordinator → mules (tight window)
    for i, mule in enumerate(mules):
        mule_bank, mule_country = _get_info(mule, accounts_map)
        jitter = timedelta(minutes=random.uniform(0, sync_window_minutes))
        t = start_date + jitter
        amount = round(base_amount * random.uniform(0.97, 1.03), 2)
        tx = _tx(coordinator, mule, amount, t, coord_bank, mule_bank, coord_country, mule_country, tx_type="ach")
        tx["memo"] = f"mule_coordination_distribute_{i}"
        transactions.append(tx)

    # Phase 2: mules → collector (small time delay after receiving)
    for i, mule in enumerate(mules):
        mule_bank, mule_country = _get_info(mule, accounts_map)
        forward_delay = timedelta(hours=random.uniform(2, 24))
        t2 = start_date + forward_delay + timedelta(minutes=i * 5)
        fwd_amount = round(base_amount * random.uniform(0.88, 0.95), 2)  # fees extracted
        tx = _tx(mule, collector, fwd_amount, t2, mule_bank, coll_bank, mule_country, coll_country, tx_type="wire")
        tx["memo"] = f"mule_coordination_collect_{i}"
        transactions.append(tx)

    return transactions


# ─── Motif registry ──────────────────────────────────────────────────────────

import hashlib

MOTIF_TYPES = [
    "recursive_loop",
    "peel_off",
    "scatter_gather",
    "fan_in",
    "fan_out",
    "burst",
    "chain",
    "agentic_bot",
    "structuring",
    "round_trip",
    "mule_coordination",
]

MOTIF_TYPE_TO_INDEX = {m: i for i, m in enumerate(MOTIF_TYPES)}
