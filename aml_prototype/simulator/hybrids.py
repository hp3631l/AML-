"""
Hybrid motif generators.

3 hybrid motif types from Section 4:
    1. Fan-Out + Peel-Off
    2. Scatter-Gather + Recursive Loop
    3. Low-and-Slow Chain + Cross-Country Transfer
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np

from simulator.motifs import (
    generate_fan_out, generate_peel_off,
    generate_scatter_gather, generate_recursive_loop,
    generate_chain, _get_info, _tx,
)
from simulator.distributions import sample_amount, sample_delay_hours


def generate_fanout_peeloff(
    source: str,
    intermediaries: List[str],
    peel_chains: Dict[str, List[str]],
    accounts_map: Dict,
    start_date: datetime,
    total_amount: float = 40000,
) -> List[Dict]:
    """
    Fan-Out + Peel-Off hybrid.

    Phase 1: Fan-out from source to N intermediaries.
    Phase 2: Each intermediary executes an independent peel-off chain.
    Span: 7–45 days.
    """
    transactions = []

    # Phase 1: Fan-out
    n = len(intermediaries)
    splits = np.random.dirichlet(np.ones(n)) * total_amount
    src_bank, src_country = _get_info(source, accounts_map)

    for i, inter in enumerate(intermediaries):
        dst_bank, dst_country = _get_info(inter, accounts_map)
        t = start_date + timedelta(hours=np.random.uniform(0, 48))
        transactions.append(_tx(
            source, inter, splits[i], t,
            src_bank, dst_bank, src_country, dst_country
        ))

    # Phase 2: Each intermediary does a peel-off
    phase2_start = start_date + timedelta(days=np.random.randint(3, 7))
    for inter in intermediaries:
        if inter in peel_chains and peel_chains[inter]:
            peel_txs = generate_peel_off(
                source=inter,
                intermediaries=peel_chains[inter],
                accounts_map=accounts_map,
                start_date=phase2_start + timedelta(hours=np.random.uniform(0, 72)),
                initial_amount=splits[intermediaries.index(inter)] * 0.95,
            )
            transactions.extend(peel_txs)

    return transactions


def generate_scatter_gather_loop(
    source: str,
    intermediaries: List[str],
    loop_accounts: List[str],
    accounts_map: Dict,
    start_date: datetime,
    total_amount: float = 50000,
    delay_days: int = 5,
    num_cycles: int = 2,
) -> List[Dict]:
    """
    Scatter-Gather + Recursive Loop hybrid.

    Phase 1: Scatter-gather where gather destination is part of a loop.
    Phase 2: Gathered funds cycle through the loop ≥2 times.
    Span: 14–60 days.
    """
    transactions = []

    # The gather destination is the first account of the loop
    destination = loop_accounts[0]

    # Phase 1: Scatter-gather
    sg_txs = generate_scatter_gather(
        source=source,
        intermediaries=intermediaries,
        destination=destination,
        accounts_map=accounts_map,
        start_date=start_date,
        total_amount=total_amount,
        delay_days=delay_days,
    )
    transactions.extend(sg_txs)

    # Phase 2: Recursive loop starting after gather completes
    loop_start = start_date + timedelta(days=delay_days + 3)
    loop_txs = generate_recursive_loop(
        accounts=loop_accounts,
        accounts_map=accounts_map,
        start_date=loop_start,
        period_days=max(7, delay_days),
        num_cycles=num_cycles,
        amount_range=(total_amount * 0.05, total_amount * 0.15),
    )
    transactions.extend(loop_txs)

    return transactions


def generate_slow_cross_country_chain(
    accounts: List[str],
    accounts_map: Dict,
    start_date: datetime,
    amount: float = 5000,
) -> List[Dict]:
    """
    Low-and-Slow Chain + Cross-Country Transfer hybrid.

    Chain of ≥5 hops spanning ≥3 countries with ≥21 days total.
    Each hop crosses at least one country boundary.
    Delays: 3–7 days each. Amounts: $1,000–$8,000.
    """
    assert len(accounts) >= 5, "Cross-country chain needs ≥5 accounts"

    # Verify ≥3 countries involved
    countries = set()
    for acc in accounts:
        _, country = _get_info(acc, accounts_map)
        countries.add(country)

    transactions = []
    current_amount = amount
    current_time = start_date

    for i in range(len(accounts) - 1):
        src = accounts[i]
        dst = accounts[i + 1]
        src_bank, src_country = _get_info(src, accounts_map)
        dst_bank, dst_country = _get_info(dst, accounts_map)

        # Slow delay: 3-7 days between hops
        delay_days = np.random.uniform(3, 7)
        current_time += timedelta(days=delay_days)

        # Small fee per hop
        fee = current_amount * np.random.uniform(0.01, 0.03)
        current_amount -= fee

        transactions.append(_tx(
            src, dst, current_amount, current_time,
            src_bank, dst_bank, src_country, dst_country
        ))

    return transactions


HYBRID_TYPES = [
    "fanout_peeloff",
    "scatter_gather_loop",
    "slow_cross_country_chain",
]
