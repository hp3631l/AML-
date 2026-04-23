"""
Occupation codebook, salary band mapping, and country risk tiers.

These are the core reference tables used by bank nodes to transform
raw KYC data into privacy-preserving embeddings.
"""

# ─── Occupation Codebook ─────────────────────────────────────────────────────
OCCUPATION_CODEBOOK = {
    "business owner": 1,
    "freelancer": 2,
    "lawyer": 3,
    "doctor": 4, "physician": 4,
    "nurse": 5,
    "teacher": 6,
    "software engineer": 7, "developer": 7,
    "data scientist": 8,
    "accountant": 9,
    "banker": 10,
    "salesperson": 11,
    "student": 12,
    "farmer": 13,
    "construction worker": 14,
    "retired": 15,
    "homemaker": 16,
    "military": 17,
    "unemployed": 18,
    "artist": 19,
    "journalist": 20,
    "pilot": 21,
    "government employee": 22,
    "police officer": 23,
    "real estate agent": 24,
    "import/export": 25,
    "restaurant owner": 26,
    "trader": 27, "broker": 27,
    "consultant": 28,
    "other": 0,
}

# Reverse lookup: code → canonical name
OCCUPATION_NAMES = {}
for name, code in OCCUPATION_CODEBOOK.items():
    if code not in OCCUPATION_NAMES:
        OCCUPATION_NAMES[code] = name

# ─── Salary Bands ────────────────────────────────────────────────────────────
SALARY_BANDS = {
    1:  (0,       10_000),
    2:  (10_001,  25_000),
    3:  (25_001,  50_000),
    4:  (50_001,  75_000),
    5:  (75_001,  100_000),
    6:  (100_001, 150_000),
    7:  (150_001, 250_000),
    8:  (250_001, 400_000),
    9:  (400_001, 500_000),
    10: (500_001, float("inf")),
}


def salary_to_band(exact_salary: float) -> int:
    """Convert exact salary (USD/year) to salary band 1-10."""
    for band, (low, high) in SALARY_BANDS.items():
        if low <= exact_salary <= high:
            return band
    return 1


def band_to_salary_range(band: int) -> tuple:
    """Get the salary range for a given band."""
    return SALARY_BANDS.get(band, (0, 10_000))


# ─── Country Risk Tiers ──────────────────────────────────────────────────────
COUNTRY_RISK_TIERS = {
    "low": {
        "countries": ["US", "GB", "DE", "FR", "JP", "CA", "AU", "SG", "NZ", "SE"],
        "risk_score": 0.1,
    },
    "medium": {
        "countries": ["AE", "TR", "TH", "BR", "MX", "ZA", "IN", "CN", "RU", "SA"],
        "risk_score": 0.5,
    },
    "high": {
        "countries": ["MM", "KP", "IR", "SY", "AF", "YE", "LY", "SO", "VU", "PK"],
        "risk_score": 0.9,
    },
}

# Flat lookup
_COUNTRY_RISK_MAP = {}
for tier, data in COUNTRY_RISK_TIERS.items():
    for cc in data["countries"]:
        _COUNTRY_RISK_MAP[cc] = data["risk_score"]


def get_country_risk(country_code: str) -> float:
    """Get risk score for a country code. Unknown → 0.5 (medium)."""
    return _COUNTRY_RISK_MAP.get(country_code, 0.5)


# All known countries for sampling
ALL_COUNTRIES = []
for tier_data in COUNTRY_RISK_TIERS.values():
    ALL_COUNTRIES.extend(tier_data["countries"])


# ─── Expected Salary Bands by Occupation × Country ──────────────────────────
EXPECTED_SALARY_BANDS = {
    # (occupation_code, country_code) → (min_expected_band, max_expected_band)
    (4, "US"): (7, 9),    # doctor in US: $150K–$500K
    (4, "IN"): (4, 7),    # doctor in India: $50K–$250K
    (4, "GB"): (6, 8),
    (4, "DE"): (6, 8),
    (7, "US"): (5, 8),    # software engineer in US
    (7, "IN"): (3, 6),
    (7, "GB"): (5, 7),
    (7, "DE"): (5, 7),
    (12, "US"): (1, 2),   # student
    (12, "IN"): (1, 2),
    (12, "GB"): (1, 2),
    (15, "US"): (3, 6),   # retired
    (15, "IN"): (2, 4),
    (18, "US"): (1, 2),   # unemployed
    (18, "IN"): (1, 1),
    (27, "US"): (5, 10),  # trader (high variance)
    (27, "IN"): (3, 8),
    (1, "US"): (3, 10),   # business owner (wide range)
    (1, "IN"): (2, 8),
    (3, "US"): (6, 9),    # lawyer
    (3, "IN"): (4, 7),
    (9, "US"): (4, 7),    # accountant
    (9, "IN"): (3, 5),
    (6, "US"): (3, 5),    # teacher
    (6, "IN"): (2, 4),
    (25, "US"): (3, 8),   # import/export
    (25, "AE"): (4, 9),
    (25, "MM"): (2, 7),
    (28, "US"): (5, 9),   # consultant
    (28, "IN"): (3, 7),
}
