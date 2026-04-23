"""
Normal (benign) background traffic generator.

Includes hard negatives:
    - high volume
    - multi-country counterparties
    - fan-out behavior
    - multiple devices
without recursive loops/delayed return motifs.
"""

import hashlib
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np


BASE_TX_TEMPLATES = [
    ("salary_deposit", "ach", (1200, 9000)),
    ("utility_payment", "ach", (40, 350)),
    ("rent_mortgage", "ach", (600, 3200)),
    ("grocery_retail", "cash_withdrawal", (15, 350)),
    ("p2p_transfer", "wire", (40, 1800)),
    ("savings_transfer", "internal", (80, 4500)),
    ("subscription", "ach", (5, 70)),
]


def _stable_ratio(account_id: str, seed: int, salt: str) -> float:
    h = hashlib.sha256(f"{seed}:{salt}:{account_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def _is_hard_negative(account_id: str, seed: int, ratio: float = 0.08) -> bool:
    return _stable_ratio(account_id, seed, "hard-negative") < ratio


def _pick_counterparty_pool(
    acc_id: str,
    all_account_ids: List[str],
    accounts_map: Dict,
    rng: np.random.RandomState,
    pool_size: int,
    prefer_cross_country: bool,
) -> List[str]:
    src_info = accounts_map.get(acc_id, {})
    src_country = src_info.get("country_code", "US")
    src_bank = src_info.get("bank_id", "bank_a")

    eligible = [a for a in all_account_ids if a != acc_id]
    if prefer_cross_country:
        cross = [
            a for a in eligible
            if accounts_map.get(a, {}).get("country_code", "US") != src_country
        ]
        if len(cross) >= max(4, pool_size // 2):
            eligible = cross + [a for a in eligible if a not in cross]

    rng.shuffle(eligible)
    # Favor cross-bank counterparties for harder negatives.
    eligible.sort(key=lambda a: 0 if accounts_map.get(a, {}).get("bank_id", src_bank) != src_bank else 1)
    return eligible[:pool_size]


def generate_normal_transactions(
    accounts: List,
    accounts_map: Dict,
    all_account_ids: List[str],
    start_date: datetime,
    num_days: int = 365,
    seed: int = 42,
    tx_budget: int | None = None,
) -> Tuple[List[Dict], Dict]:
    """
    Generate benign transactions with hard negatives, bounded by tx_budget.
    """
    rng = np.random.RandomState(seed)
    transactions: List[Dict] = []
    hard_negative_accounts = set()
    hard_negative_tx = 0
    months = max(1, num_days // 30)

    shuffled_accounts = list(accounts)
    rng.shuffle(shuffled_accounts)

    for account in shuffled_accounts:
        if tx_budget is not None and len(transactions) >= tx_budget:
            break

        acc_id = account.account_id
        info = accounts_map.get(acc_id, {})
        bank_id = info.get("bank_id", "bank_a")
        country = info.get("country_code", "US")
        salary_band = max(1, int(getattr(account, "salary_band", 5)))
        is_hn = _is_hard_negative(acc_id, seed)

        if is_hn:
            hard_negative_accounts.add(acc_id)
            annual_target = int(rng.randint(120, 185))
            pool = _pick_counterparty_pool(
                acc_id, all_account_ids, accounts_map, rng, pool_size=32, prefer_cross_country=True
            )
        else:
            annual_target = int(rng.randint(10, 30))
            pool = _pick_counterparty_pool(
                acc_id, all_account_ids, accounts_map, rng, pool_size=12, prefer_cross_country=False
            )

        for i in range(annual_target):
            if tx_budget is not None and len(transactions) >= tx_budget:
                break
            if not pool:
                break

            # Hard-negative fan-out style: cycle across many recipients.
            if is_hn:
                dst = pool[i % len(pool)] if rng.rand() < 0.70 else rng.choice(pool)
                tx_type = rng.choice(["wire", "ach", "internal"], p=[0.35, 0.40, 0.25])
                if rng.rand() < 0.70:
                    amount = float(rng.uniform(180, 3600))
                else:
                    amount = float(rng.uniform(3600, 9800))
                # Wide temporal spread, no fixed intervals.
                day = int(rng.randint(0, num_days))
                hour = int(rng.choice(list(range(7, 22)) * 3 + list(range(0, 7)) + list(range(22, 24))))
                minute = int(rng.randint(0, 60))
                second = int(rng.randint(0, 60))
                ts = start_date + timedelta(days=day, hours=hour, minutes=minute, seconds=second)
            else:
                tx_name, tx_type, (amt_lo, amt_hi) = BASE_TX_TEMPLATES[int(rng.randint(0, len(BASE_TX_TEMPLATES)))]
                dst = rng.choice(pool)
                mult = float(np.clip(0.5 + salary_band / 8.0, 0.45, 1.8))
                amount = float(rng.uniform(amt_lo, amt_hi)) * mult
                amount *= float(rng.uniform(0.8, 1.2))
                day = int(rng.randint(0, num_days))
                hour = int(rng.choice(list(range(8, 20)) * 3 + list(range(0, 8)) + list(range(20, 24))))
                minute = int(rng.randint(0, 60))
                ts = start_date + timedelta(days=day, hours=hour, minutes=minute)
                tx_name = tx_name

            dst_info = accounts_map.get(dst, {})
            dst_bank = dst_info.get("bank_id", bank_id)
            dst_country = dst_info.get("country_code", country)

            memo = "Hard-negative benign fanout activity" if is_hn else f"Normal {tx_name}"
            transactions.append({
                "tx_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
                "src_account_id": acc_id,
                "dst_account_id": dst,
                "amount": round(max(5.0, amount), 2),
                "currency": "USD",
                "tx_type": str(tx_type),
                "timestamp": ts.isoformat(),
                "src_bank_id": bank_id,
                "dst_bank_id": dst_bank,
                "src_country": country,
                "dst_country": dst_country,
                "memo": memo,
            })
            if is_hn:
                hard_negative_tx += 1

    metadata = {
        "hard_negative_accounts": len(hard_negative_accounts),
        "hard_negative_transactions": hard_negative_tx,
        "generated_transactions": len(transactions),
        "avg_tx_account_month": len(transactions) / max(1.0, len(accounts) * months),
    }
    return transactions, metadata


def generate_sessions(
    accounts: List,
    accounts_map: Dict,
    start_date: datetime,
    num_days: int = 365,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Generate session metadata with harder benign behavior for hard negatives.
    """
    rng = np.random.RandomState(seed + 1)
    sessions_by_bank: Dict[str, List[Dict]] = {"bank_a": [], "bank_b": [], "bank_c": []}
    months = max(1, num_days // 30)

    for account in accounts:
        acc_id = account.account_id
        info = accounts_map.get(acc_id, {})
        bank_id = info.get("bank_id", "bank_a")
        country = info.get("country_code", "US")
        is_hn = _is_hard_negative(acc_id, seed)

        if is_hn:
            sessions_per_month = int(rng.randint(14, 28))
            num_devices = int(rng.randint(4, 8))
            login_mix = ["web", "mobile", "api", "mobile", "web", "api"]
        else:
            sessions_per_month = int(rng.randint(3, 10))
            num_devices = int(rng.randint(1, 4))
            login_mix = ["web", "web", "mobile", "mobile", "api"]

        device_hashes = [
            hashlib.sha256(f"dev_{acc_id}_{d}".encode("utf-8")).hexdigest()[:16]
            for d in range(num_devices)
        ]

        for month in range(months):
            month_start = start_date + timedelta(days=month * 30)
            for _ in range(sessions_per_month):
                day = int(rng.randint(0, 30))
                hour = int(rng.choice(list(range(7, 22)) * 3 + list(range(0, 7)) + list(range(22, 24))))
                minute = int(rng.randint(0, 60))
                second = int(rng.randint(0, 60))
                login_time = month_start + timedelta(days=day, hours=hour, minutes=minute, seconds=second)

                if is_hn:
                    duration = float(rng.uniform(45, 2600))
                    actions = int(rng.randint(2, 22))
                else:
                    duration = float(rng.uniform(30, 1800))
                    actions = int(rng.randint(1, 15))

                logout_time = login_time + timedelta(seconds=duration)
                login_method = str(rng.choice(login_mix))

                sessions_by_bank[bank_id].append({
                    "session_id": f"SESS-{uuid.uuid4().hex[:12].upper()}",
                    "account_id": acc_id,
                    "login_timestamp": login_time.isoformat(),
                    "logout_timestamp": logout_time.isoformat(),
                    "session_duration_seconds": round(duration, 2),
                    "device_fingerprint_hash": str(rng.choice(device_hashes)),
                    "ip_country": country,
                    "login_method": login_method,
                    "actions_count": actions,
                })

    return sessions_by_bank
