"""
Synthetic account generation for 5,000 accounts across 3 banks.

Generates KYC data, session metadata, and assigns accounts to banks.
Account IDs are hashed with SHA-256 for privacy.
"""

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np

# Import from project
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bank_node.kyc_codebook import (
    OCCUPATION_CODEBOOK, OCCUPATION_NAMES,
    SALARY_BANDS, salary_to_band,
    COUNTRY_RISK_TIERS, ALL_COUNTRIES, get_country_risk,
    EXPECTED_SALARY_BANDS,
)
from config import ACCOUNTS_PER_BANK


# ─── Name generation pools ───────────────────────────────────────────────────
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Raj", "Priya", "Amit", "Sunita", "Vikram", "Anjali", "Rahul", "Deepa",
    "Kenji", "Yuki", "Hassan", "Fatima", "Chen", "Wei", "Mohammed", "Aisha",
    "Carlos", "Maria", "Juan", "Ana", "Pierre", "Marie", "Hans", "Greta",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Patel", "Shah", "Kumar", "Singh",
    "Sharma", "Gupta", "Tanaka", "Sato", "Kim", "Park", "Wang", "Li",
    "Chen", "Ali", "Ahmed", "Mueller", "Schmidt", "Dubois", "Martin",
    "Lopez", "Gonzalez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson",
]

# Occupation codes available for sampling (exclude 0='other')
OCCUPATION_CODES = [code for code in set(OCCUPATION_CODEBOOK.values()) if code > 0]

# Countries weighted by tier for realistic distribution
# 60% low risk, 30% medium, 10% high
LOW_RISK = COUNTRY_RISK_TIERS["low"]["countries"]
MED_RISK = COUNTRY_RISK_TIERS["medium"]["countries"]
HIGH_RISK = COUNTRY_RISK_TIERS["high"]["countries"]


@dataclass
class SyntheticAccount:
    """A fully specified synthetic bank account."""
    account_id: str
    hashed_account_id: str
    bank_id: str
    customer_name: str
    pan: str
    aadhaar: str
    home_address: str
    exact_salary: float
    raw_occupation: str
    occupation_code: int
    salary_band: int
    country_code: str
    created_at: datetime = field(default_factory=datetime.utcnow)


def _generate_pan() -> str:
    """Generate a fake PAN (Permanent Account Number)."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choices(letters, k=5))
        + "".join(random.choices("0123456789", k=4))
        + random.choice(letters)
    )


def _generate_aadhaar() -> str:
    """Generate a fake 12-digit Aadhaar number."""
    return "".join(random.choices("0123456789", k=12))


def _sample_country() -> str:
    """Sample a country with tier-weighted distribution."""
    r = random.random()
    if r < 0.60:
        return random.choice(LOW_RISK)
    elif r < 0.90:
        return random.choice(MED_RISK)
    else:
        return random.choice(HIGH_RISK)


def _sample_salary(occupation_code: int, country_code: str) -> float:
    """
    Sample a realistic salary based on occupation and country.

    Uses expected salary bands when available, with noise.
    """
    key = (occupation_code, country_code)
    if key in EXPECTED_SALARY_BANDS:
        min_band, max_band = EXPECTED_SALARY_BANDS[key]
    else:
        # Default: bands 2-6
        min_band, max_band = 2, 6

    # Pick a band within expected range (80%) or outside (20% for diversity)
    if random.random() < 0.80:
        band = random.randint(min_band, max_band)
    else:
        band = random.randint(1, 10)

    lo, hi = SALARY_BANDS[band]
    hi = min(hi, 1_000_000)  # cap infinity
    return round(random.uniform(lo, hi), 2)


def generate_accounts(seed: int = 42) -> Dict[str, List[SyntheticAccount]]:
    """
    Generate 5,000 synthetic accounts across 3 banks.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping bank_id → list of SyntheticAccount.
    """
    random.seed(seed)
    np.random.seed(seed)

    accounts_by_bank: Dict[str, List[SyntheticAccount]] = {
        "bank_a": [],
        "bank_b": [],
        "bank_c": [],
    }

    # Global account counter for unique IDs
    global_idx = 0
    base_date = datetime(2024, 1, 1)

    for bank_id, count in ACCOUNTS_PER_BANK.items():
        bank_letter = bank_id[-1].upper()  # 'A', 'B', 'C'

        for i in range(count):
            # Generate unique account ID
            account_id = f"ACC-{bank_letter}-{global_idx:06d}"
            hashed_id = hashlib.sha256(account_id.encode()).hexdigest()

            # Personal info (NEVER leaves bank)
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            name = f"{first} {last}"

            # Occupation and country
            occ_code = random.choice(OCCUPATION_CODES)
            occ_name = OCCUPATION_NAMES.get(occ_code, "other")
            country = _sample_country()

            # Salary (correlated with occupation and country)
            salary = _sample_salary(occ_code, country)
            band = salary_to_band(salary)

            # Account creation date (spread over 2 years before simulation)
            days_ago = random.randint(30, 730)
            created = base_date - timedelta(days=days_ago)

            account = SyntheticAccount(
                account_id=account_id,
                hashed_account_id=hashed_id,
                bank_id=bank_id,
                customer_name=name,
                pan=_generate_pan(),
                aadhaar=_generate_aadhaar(),
                home_address=f"{random.randint(1,9999)} {random.choice(LAST_NAMES)} St, {country}",
                exact_salary=salary,
                raw_occupation=occ_name,
                occupation_code=occ_code,
                salary_band=band,
                country_code=country,
                created_at=created,
            )

            accounts_by_bank[bank_id].append(account)
            global_idx += 1

    total = sum(len(v) for v in accounts_by_bank.values())
    print(f"Generated {total} accounts across {len(accounts_by_bank)} banks.")
    return accounts_by_bank


def insert_accounts_to_db(conn, accounts: List[SyntheticAccount]) -> None:
    """Insert accounts into a bank's SQLite database."""
    cursor = conn.cursor()
    for acc in accounts:
        cursor.execute("""
        INSERT OR REPLACE INTO kyc
            (account_id, customer_name, pan, aadhaar, home_address,
             exact_salary, raw_occupation, occupation_code, salary_band,
             country_code, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            acc.account_id, acc.customer_name, acc.pan, acc.aadhaar,
            acc.home_address, acc.exact_salary, acc.raw_occupation,
            acc.occupation_code, acc.salary_band, acc.country_code,
            acc.created_at.isoformat(),
        ))

        # Initialize ledger summary
        cursor.execute("""
        INSERT OR IGNORE INTO ledger_summary (account_id, last_updated)
        VALUES (?, ?)
        """, (acc.account_id, datetime.utcnow().isoformat()))

        # Initialize label as benign
        cursor.execute("""
        INSERT OR IGNORE INTO labels
            (account_id, is_suspicious, motif_type, role, confidence)
        VALUES (?, 0, NULL, 'benign', 1.0)
        """, (acc.account_id,))

    conn.commit()
