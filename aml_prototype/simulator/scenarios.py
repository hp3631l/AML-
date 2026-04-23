"""
Scenario parameter sampling and orchestration.

Generates laundering scenarios with:
    - >=5,000 suspicious scenarios
    - motif floor coverage
    - hard positives (low activity, delayed hidden loops)
    - noisy/partial motif completion
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from simulator.motifs import (
    _get_info,
    _tx,
    generate_agentic_bot,
    generate_burst,
    generate_chain,
    generate_fan_in,
    generate_fan_out,
    generate_peel_off,
    generate_recursive_loop,
    generate_scatter_gather,
)
from simulator.hybrids import (
    generate_fanout_peeloff,
    generate_scatter_gather_loop,
    generate_slow_cross_country_chain,
)
from config import MIN_MOTIF_INSTANCES


@dataclass
class LaunderingScenario:
    """Fully specified laundering scenario."""
    scenario_id: int
    motif_type: str
    accounts: List[str]
    roles: Dict[str, str]
    is_low_and_slow: bool
    is_cross_country: bool
    is_hybrid: bool
    is_hard_positive: bool
    has_cross_bank: bool
    transactions: List[Dict]
    sessions: List[Dict]
    start_date: datetime
    span_days: int


def _pick_accounts(
    available_accounts: List[str],
    accounts_map: Dict,
    n: int,
    require_multi_country: bool = False,
    require_multi_bank: bool = False,
) -> List[str]:
    candidates = list(available_accounts)
    random.shuffle(candidates)
    if not candidates:
        return []
    if not require_multi_country and not require_multi_bank:
        return candidates[:n]

    selected = [candidates[0]]
    for acc in candidates[1:]:
        if len(selected) >= n:
            break

        should_add = False
        if require_multi_country:
            existing_countries = {accounts_map[a]["country_code"] for a in selected}
            new_country = accounts_map[acc]["country_code"]
            if new_country not in existing_countries or len(existing_countries) < 2:
                should_add = True
        if require_multi_bank:
            existing_banks = {accounts_map[a]["bank_id"] for a in selected}
            new_bank = accounts_map[acc]["bank_id"]
            if new_bank not in existing_banks or len(existing_banks) < 2:
                should_add = True

        if should_add or random.random() < 0.35:
            selected.append(acc)

    if len(selected) < n:
        remaining = [a for a in candidates if a not in selected]
        selected.extend(remaining[: max(0, n - len(selected))])
    return selected[:n]


def _assign_roles(accounts: List[str]) -> Dict[str, str]:
    roles = {}
    if len(accounts) >= 3:
        roles[accounts[0]] = "source"
        roles[accounts[-1]] = "destination"
        for acc in accounts[1:-1]:
            roles[acc] = "intermediary"
    else:
        for acc in accounts:
            roles[acc] = "participant"
    return roles


def _apply_noise(transactions: List[Dict]) -> List[Dict]:
    """Inject timestamp/amount noise + partial motif completion."""
    if not transactions:
        return transactions

    noisy = []
    for tx in transactions:
        t = datetime.fromisoformat(tx["timestamp"])
        # Random, non-fixed interval noise
        t = t + timedelta(hours=float(np.random.normal(loc=0.0, scale=9.0)))
        amount = max(10.0, float(tx["amount"]) * float(np.random.uniform(0.75, 1.25)))
        tx2 = dict(tx)
        tx2["timestamp"] = t.isoformat()
        tx2["amount"] = round(amount, 2)
        noisy.append(tx2)

    noisy.sort(key=lambda x: x["timestamp"])

    # Partial motif completion in subset of scenarios
    if len(noisy) >= 4 and random.random() < 0.35:
        min_keep = 3
        max_drop = max(1, int(len(noisy) * 0.35))
        drop_count = random.randint(1, max_drop)
        candidates = list(range(1, len(noisy) - 1))
        if candidates:
            drop_idx = set(random.sample(candidates, k=min(drop_count, len(candidates))))
            noisy = [tx for idx, tx in enumerate(noisy) if idx not in drop_idx]
            if len(noisy) < min_keep:
                noisy = noisy[:min_keep]
    return noisy


def _build_quota(num_scenarios: int) -> Dict[str, int]:
    categories = [
        "hybrid_slow_cross",
        "hybrid_fanout_peel",
        "hybrid_sg_loop",
        "chain",
        "recursive_loop",
        "scatter_gather",
        "peel_off",
        "fan_in",
        "fan_out",
        "burst",
        "agentic_bot",
    ]
    quota = {c: MIN_MOTIF_INSTANCES for c in categories + ["hard_positive_hidden_loop"]}

    assigned = sum(quota.values())
    if assigned > num_scenarios:
        raise ValueError(
            f"Cannot satisfy motif floors for {num_scenarios} scenarios. "
            f"Need at least {assigned}."
        )

    weights = {
        "chain": 0.25,
        "scatter_gather": 0.15,
        "peel_off": 0.10,
        "hybrid_slow_cross": 0.10,
        "hybrid_fanout_peel": 0.05,
        "hybrid_sg_loop": 0.05,
        "recursive_loop": 0.05,
        "hard_positive_hidden_loop": 0.05,
        "fan_in": 0.05,
        "fan_out": 0.05,
        "burst": 0.05,
        "agentic_bot": 0.05,
    }

    remaining = num_scenarios - assigned
    ordered = list(weights.keys())
    w = np.array([weights[k] for k in ordered], dtype=np.float64)
    w = w / w.sum()
    alloc = np.floor(w * remaining).astype(int)

    for k, v in zip(ordered, alloc):
        quota[k] += int(v)

    leftover = int(remaining - int(alloc.sum()))
    for i in range(leftover):
        quota[ordered[i % len(ordered)]] += 1

    return quota


def generate_all_scenarios(
    all_account_ids: List[str],
    accounts_map: Dict,
    num_scenarios: int = 5000,
    simulation_start: datetime = datetime(2024, 1, 1),
    simulation_days: int = 365,
    seed: int = 42,
) -> List[LaunderingScenario]:
    random.seed(seed)
    np.random.seed(seed)

    scenarios: List[LaunderingScenario] = []
    used_accounts = set()
    
    # Cap suspicious accounts at 7% (around 350 for 5000 total)
    suspicious_pool_size = max(250, int(len(all_account_ids) * 0.07))
    suspicious_account_pool = random.sample(all_account_ids, suspicious_pool_size)
    quota = _build_quota(num_scenarios)

    scenario_id = 0
    for category, count in quota.items():
        for _ in range(count):
            start_offset = random.randint(0, max(1, simulation_days - 120))
            start_date = simulation_start + timedelta(days=start_offset)
            scenario = _generate_single_scenario(
                scenario_id=scenario_id,
                category=category,
                all_account_ids=suspicious_account_pool,
                accounts_map=accounts_map,
                used_accounts=used_accounts,
                start_date=start_date,
            )
            if scenario:
                scenarios.append(scenario)
                for acc in scenario.accounts[:2]:
                    used_accounts.add(acc)
                scenario_id += 1

    if len(scenarios) > num_scenarios:
        scenarios = scenarios[:num_scenarios]
    elif len(scenarios) < num_scenarios:
        while len(scenarios) < num_scenarios:
            sid = len(scenarios)
            start_offset = random.randint(0, max(1, simulation_days - 120))
            scenario = _generate_single_scenario(
                scenario_id=sid,
                category="chain",
                all_account_ids=suspicious_account_pool,
                accounts_map=accounts_map,
                used_accounts=used_accounts,
                start_date=simulation_start + timedelta(days=start_offset),
            )
            if scenario:
                scenarios.append(scenario)

    random.shuffle(scenarios)
    return scenarios


def _generate_single_scenario(
    scenario_id: int,
    category: str,
    all_account_ids: List[str],
    accounts_map: Dict,
    used_accounts: set,
    start_date: datetime,
) -> Optional[LaunderingScenario]:
    available = [a for a in all_account_ids if a not in used_accounts]
    if len(available) < 25:
        available = list(all_account_ids)

    try:
        if category == "hybrid_slow_cross":
            return _gen_slow_cross_country(scenario_id, available, accounts_map, start_date)
        if category == "hybrid_fanout_peel":
            return _gen_fanout_peeloff(scenario_id, available, accounts_map, start_date)
        if category == "hybrid_sg_loop":
            return _gen_sg_loop(scenario_id, available, accounts_map, start_date)
        if category == "chain":
            return _gen_chain(scenario_id, available, accounts_map, start_date)
        if category == "recursive_loop":
            return _gen_recursive_loop(scenario_id, available, accounts_map, start_date)
        if category == "hard_positive_hidden_loop":
            return _gen_hard_positive_hidden_loop(scenario_id, available, accounts_map, start_date)
        if category == "scatter_gather":
            return _gen_scatter_gather(scenario_id, available, accounts_map, start_date)
        if category == "peel_off":
            return _gen_peel_off(scenario_id, available, accounts_map, start_date)
        if category == "fan_in":
            return _gen_fan_in(scenario_id, available, accounts_map, start_date)
        if category == "fan_out":
            return _gen_fan_out(scenario_id, available, accounts_map, start_date)
        if category == "burst":
            return _gen_burst(scenario_id, available, accounts_map, start_date)
        if category == "agentic_bot":
            return _gen_agentic_bot(scenario_id, available, accounts_map, start_date)
    except Exception as exc:
        print(f"Warning: Failed to generate scenario {scenario_id} ({category}): {exc}")
        return None
    return None


def _finalize_scenario(
    scenario_id: int,
    motif_type: str,
    accounts: List[str],
    accounts_map: Dict,
    transactions: List[Dict],
    sessions: List[Dict],
    start_date: datetime,
    is_hybrid: bool,
    is_hard_positive: bool,
) -> LaunderingScenario:
    txs = _apply_noise(transactions)
    if not txs:
        txs = transactions
    txs.sort(key=lambda x: x["timestamp"])

    countries = set()
    for acc in accounts:
        if acc in accounts_map:
            countries.add(accounts_map[acc]["country_code"])

    has_cross_bank = any(tx["src_bank_id"] != tx["dst_bank_id"] for tx in txs)
    span_days = 1
    if len(txs) >= 2:
        t0 = datetime.fromisoformat(txs[0]["timestamp"])
        t1 = datetime.fromisoformat(txs[-1]["timestamp"])
        span_days = max(1, int((t1 - t0).total_seconds() / 86400.0))

    return LaunderingScenario(
        scenario_id=scenario_id,
        motif_type=motif_type,
        accounts=list(dict.fromkeys(accounts)),
        roles=_assign_roles(list(dict.fromkeys(accounts))),
        is_low_and_slow=span_days >= 14,
        is_cross_country=len(countries) >= 2,
        is_hybrid=is_hybrid,
        is_hard_positive=is_hard_positive,
        has_cross_bank=has_cross_bank,
        transactions=txs,
        sessions=sessions,
        start_date=start_date,
        span_days=span_days,
    )


def _gen_slow_cross_country(sid, available, amap, start):
    n = random.randint(5, 7)
    accounts = _pick_accounts(available, amap, n, require_multi_country=True, require_multi_bank=True)
    txs = generate_slow_cross_country_chain(
        accounts,
        amap,
        start,
        amount=float(np.random.uniform(700, 3000)),
    )
    return _finalize_scenario(sid, "slow_cross_country_chain", accounts, amap, txs, [], start, True, False)


def _gen_fanout_peeloff(sid, available, amap, start):
    source = random.choice(available)
    n_inter = random.randint(3, 4)
    intermediaries = _pick_accounts(
        [a for a in available if a != source],
        amap,
        n_inter,
        require_multi_country=True,
        require_multi_bank=True,
    )
    peel_chains = {}
    remaining = [a for a in available if a != source and a not in intermediaries]
    for inter in intermediaries:
        chain_len = random.randint(1, 2)
        peel_chains[inter] = remaining[:chain_len]
        remaining = remaining[chain_len:]

    txs = generate_fanout_peeloff(
        source,
        intermediaries,
        peel_chains,
        amap,
        start,
        total_amount=float(np.random.uniform(6000, 24000)),
    )
    accounts = [source] + intermediaries + [x for chain in peel_chains.values() for x in chain]
    return _finalize_scenario(sid, "fanout_peeloff", accounts, amap, txs, [], start, True, False)


def _gen_sg_loop(sid, available, amap, start):
    source = random.choice(available)
    n_inter = random.randint(5, 6)
    intermediaries = _pick_accounts(
        [a for a in available if a != source],
        amap,
        n_inter,
        require_multi_country=True,
        require_multi_bank=True,
    )
    remaining = [a for a in available if a != source and a not in intermediaries]
    loop_accounts = _pick_accounts(remaining, amap, 3, require_multi_bank=True)
    txs = generate_scatter_gather_loop(
        source,
        intermediaries,
        loop_accounts,
        amap,
        start,
        total_amount=float(np.random.uniform(9000, 28000)),
        delay_days=random.randint(5, 12),
        num_cycles=2,
    )
    accounts = [source] + intermediaries + loop_accounts
    return _finalize_scenario(sid, "scatter_gather_loop", accounts, amap, txs, [], start, True, False)


def _gen_chain(sid, available, amap, start):
    slow = random.random() < 0.90
    n = random.randint(5, 7) if slow else random.randint(5, 6)
    accounts = _pick_accounts(
        available,
        amap,
        n,
        require_multi_country=slow,
        require_multi_bank=True,
    )
    txs = generate_chain(
        accounts,
        amap,
        start,
        amount=float(np.random.uniform(800, 5500)),
        delay_dist="human_mimicking" if slow else "exponential",
        delay_scale_hours=96 if slow else 30,
    )
    return _finalize_scenario(sid, "chain", accounts, amap, txs, [], start, False, False)


def _gen_recursive_loop(sid, available, amap, start):
    n = random.randint(3, 4)
    accounts = _pick_accounts(available, amap, n, require_multi_country=True, require_multi_bank=True)
    txs = generate_recursive_loop(
        accounts,
        amap,
        start,
        period_days=random.randint(8, 24),
        num_cycles=random.randint(2, 3),
        amount_range=(300, 3000),
        amount_dist="uniform",
        delay_dist="human_mimicking",
    )
    return _finalize_scenario(sid, "recursive_loop", accounts, amap, txs, [], start, False, False)


def _gen_hard_positive_hidden_loop(sid, available, amap, start):
    accounts = _pick_accounts(
        available,
        amap,
        3,
        require_multi_country=False,
        require_multi_bank=True,
    )
    if len(accounts) < 3:
        return None

    a, b, c = accounts[0], accounts[1], accounts[2]
    txs: List[Dict] = []
    t = start + timedelta(hours=float(np.random.uniform(2, 20)))
    amount = float(np.random.uniform(180, 850))

    # Cycle 1 (full hidden loop over long delays)
    for src, dst in [(a, b), (b, c), (c, a)]:
        sb, sc = _get_info(src, amap)
        db, dc = _get_info(dst, amap)
        t = t + timedelta(days=float(np.random.uniform(8, 24)), hours=float(np.random.uniform(1, 14)))
        amount = max(60.0, amount * float(np.random.uniform(0.90, 0.99)))
        txs.append(_tx(src, dst, amount, t, sb, db, sc, dc, tx_type="internal"))

    # Cycle 2 (possibly partial completion)
    second_edges = [(a, b), (b, c), (c, a)]
    if random.random() < 0.70:
        second_edges = second_edges[:-1]
    for src, dst in second_edges:
        sb, sc = _get_info(src, amap)
        db, dc = _get_info(dst, amap)
        t = t + timedelta(days=float(np.random.uniform(10, 28)), hours=float(np.random.uniform(1, 10)))
        amount = max(45.0, amount * float(np.random.uniform(0.90, 0.99)))
        txs.append(_tx(src, dst, amount, t, sb, db, sc, dc, tx_type="internal"))

    return _finalize_scenario(sid, "recursive_loop", accounts, amap, txs, [], start, False, True)


def _gen_scatter_gather(sid, available, amap, start):
    source = random.choice(available)
    remaining = [a for a in available if a != source]
    intermediaries = _pick_accounts(
        remaining,
        amap,
        random.randint(5, 6),
        require_multi_country=True,
        require_multi_bank=True,
    )
    dest_pool = [a for a in remaining if a not in intermediaries]
    if not dest_pool:
        return None
    destination = random.choice(dest_pool)
    txs = generate_scatter_gather(
        source,
        intermediaries,
        destination,
        amap,
        start,
        total_amount=float(np.random.uniform(6000, 22000)),
        delay_days=random.randint(5, 18),
    )
    accounts = [source] + intermediaries + [destination]
    return _finalize_scenario(sid, "scatter_gather", accounts, amap, txs, [], start, False, False)


def _gen_peel_off(sid, available, amap, start):
    source = random.choice(available)
    intermediaries = _pick_accounts(
        [a for a in available if a != source],
        amap,
        random.randint(3, 5),
        require_multi_bank=True,
    )
    txs = generate_peel_off(
        source,
        intermediaries,
        amap,
        start,
        initial_amount=float(np.random.uniform(3000, 12000)),
        delay_dist="human_mimicking",
    )
    accounts = [source] + intermediaries
    return _finalize_scenario(sid, "peel_off", accounts, amap, txs, [], start, False, False)


def _gen_fan_in(sid, available, amap, start):
    destination = random.choice(available)
    sources = _pick_accounts(
        [a for a in available if a != destination],
        amap,
        random.randint(3, 8),
        require_multi_country=True,
        require_multi_bank=True,
    )
    txs = generate_fan_in(
        sources,
        destination,
        amap,
        start,
        amount_range=(250, 3800),
        span_days=random.randint(2, 10),
    )
    accounts = sources + [destination]
    return _finalize_scenario(sid, "fan_in", accounts, amap, txs, [], start, False, False)


def _gen_fan_out(sid, available, amap, start):
    source = random.choice(available)
    destinations = _pick_accounts(
        [a for a in available if a != source],
        amap,
        random.randint(3, 8),
        require_multi_country=True,
        require_multi_bank=True,
    )
    txs = generate_fan_out(
        source,
        destinations,
        amap,
        start,
        amount_range=(200, 4200),
        span_days=random.randint(2, 8),
        amount_dist="uniform",
    )
    accounts = [source] + destinations
    return _finalize_scenario(sid, "fan_out", accounts, amap, txs, [], start, False, False)


def _gen_burst(sid, available, amap, start):
    account = random.choice(available)
    counterparties = _pick_accounts(
        [a for a in available if a != account],
        amap,
        random.randint(3, 6),
        require_multi_bank=True,
    )
    txs = generate_burst(
        account,
        counterparties,
        amap,
        start,
        window_hours=float(np.random.uniform(6, 24)),
        num_transactions=random.randint(10, 18),
        amount_range=(120, 2400),
        amount_dist="uniform",
    )
    accounts = [account] + counterparties
    return _finalize_scenario(sid, "burst", accounts, amap, txs, [], start, False, False)


def _gen_agentic_bot(sid, available, amap, start):
    bot = random.choice(available)
    targets = _pick_accounts(
        [a for a in available if a != bot],
        amap,
        random.randint(3, 6),
        require_multi_country=True,
        require_multi_bank=True,
    )
    txs, sessions = generate_agentic_bot(
        bot,
        targets,
        amap,
        start,
        num_sessions=random.randint(8, 16),
        amount_range=(120, 2600),
        amount_dist="uniform",
    )
    accounts = [bot] + targets
    return _finalize_scenario(sid, "agentic_bot", accounts, amap, txs, sessions, start, False, False)
