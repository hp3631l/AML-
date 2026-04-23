# AML Prototype — Implementation Plan (Sections 3–4)

---

> [!WARNING]
> The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys.
> This prototype uses hardware-backed key management, not hardware-backed confidential computing.

Raw KYC data NEVER leaves the local bank node. Only the following may leave a bank: hashed account ID, occupation embedding, salary band, country risk vector, session vector, ledger vector, and recent transaction metadata.
Trust Score: 0-100, Laundering Probability: 0-1, Confidence Score: 0-1. Temporary hold recommendation only, never automatic permanent freeze.
Memory: Active memory for last 30–90 days, Older data compressed into a historical vector, historical_weight = exp(-lambda * age_in_days).

---

## SECTION 3 — MULTI-BANK PRIVACY FLOW

### Bank Configuration

| Bank | API Port | SQLite Path | Accounts |
|------|----------|-------------|----------|
| Bank A | 8001 | `data/bank_a/bank_a.db` | 2,000 |
| Bank B | 8002 | `data/bank_b/bank_b.db` | 1,500 |
| Bank C | 8003 | `data/bank_c/bank_c.db` | 1,500 |

Each bank's SQLite database contains these tables:

```sql
-- KYC Store (LOCAL ONLY — NEVER SHARED)
CREATE TABLE kyc (
    account_id TEXT PRIMARY KEY,
    customer_name TEXT,           -- NEVER leaves bank
    pan TEXT,                     -- NEVER leaves bank
    aadhaar TEXT,                 -- NEVER leaves bank
    home_address TEXT,            -- NEVER leaves bank
    exact_salary REAL,           -- NEVER leaves bank
    raw_occupation TEXT,          -- NEVER leaves bank
    occupation_code INTEGER,     -- derived, shareable as embedding
    salary_band INTEGER,         -- derived, shareable
    country_code TEXT,
    created_at TIMESTAMP
);

-- Transaction History Store
CREATE TABLE transactions (
    tx_id TEXT PRIMARY KEY,
    src_account_id TEXT,
    dst_account_id TEXT,
    amount REAL,
    currency TEXT,
    tx_type TEXT,  -- 'wire', 'ach', 'cash_deposit', 'cash_withdrawal', 'internal'
    timestamp TIMESTAMP,
    src_bank_id TEXT,
    dst_bank_id TEXT,
    src_country TEXT,
    dst_country TEXT,
    memo TEXT  -- NEVER leaves bank
);

-- Session Metadata Store
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    account_id TEXT,
    login_timestamp TIMESTAMP,
    logout_timestamp TIMESTAMP,
    session_duration_seconds REAL,
    device_fingerprint_hash TEXT,  -- hashed, not raw
    ip_country TEXT,
    login_method TEXT,  -- 'web', 'mobile', 'api'
    actions_count INTEGER
);

-- Ledger Metadata Store
CREATE TABLE ledger_summary (
    account_id TEXT PRIMARY KEY,
    avg_tx_amount_30d REAL,
    tx_count_30d INTEGER,
    avg_tx_amount_90d REAL,
    tx_count_90d INTEGER,
    unique_counterparties_30d INTEGER,
    unique_countries_30d INTEGER,
    max_single_tx_30d REAL,
    last_updated TIMESTAMP
);
```

### Data That NEVER Leaves the Bank (Under Any Condition)

- Customer name
- PAN (permanent account number)
- Aadhaar (or equivalent national ID)
- Home address
- Exact salary value
- Raw occupation text string
- Transaction memo fields
- Raw IP addresses

### KYC Codebook

```python
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
```

### Salary Band Construction

```python
SALARY_BANDS = {
    1:  (0,      10_000),      # 0–10K USD/year
    2:  (10_001, 25_000),      # 10K–25K
    3:  (25_001, 50_000),      # 25K–50K
    4:  (50_001, 75_000),      # 50K–75K
    5:  (75_001, 100_000),     # 75K–100K
    6:  (100_001, 150_000),    # 100K–150K
    7:  (150_001, 250_000),    # 150K–250K
    8:  (250_001, 400_000),    # 250K–400K
    9:  (400_001, 500_000),    # 400K–500K
    10: (500_001, float('inf')),  # 500K+
}

def salary_to_band(exact_salary: float) -> int:
    for band, (low, high) in SALARY_BANDS.items():
        if low <= exact_salary <= high:
            return band
    return 1
```

### Local Embedding Construction

```python
def generate_embeddings(account_id: str, db: sqlite3.Connection) -> dict:
    kyc = db.execute("SELECT * FROM kyc WHERE account_id=?", (account_id,)).fetchone()
    sessions = db.execute("SELECT * FROM sessions WHERE account_id=? ORDER BY login_timestamp DESC LIMIT 30", (account_id,)).fetchall()
    ledger = db.execute("SELECT * FROM ledger_summary WHERE account_id=?", (account_id,)).fetchone()

    # Occupation embedding: learned 8d embedding from integer code
    occupation_emb = occupation_embedding_table[kyc['occupation_code']]  # nn.Embedding(30, 8)

    # Salary band: integer 1–10
    salary_band = kyc['salary_band']

    # Country risk vector
    country_risk = COUNTRY_RISK_SCORES[kyc['country_code']]  # float 0–1

    # Session vector (8d)
    session_vec = compute_session_vector(sessions)
    # = [mean_session_duration, std_session_duration, login_frequency_per_week,
    #    unique_devices, fraction_api_logins, mean_actions_per_session,
    #    std_inter_login_interval, fraction_unusual_hours]

    # Ledger vector (8d)
    ledger_vec = compute_ledger_vector(ledger)
    # = [log(avg_tx_amount_30d), log(tx_count_30d+1), log(avg_tx_amount_90d),
    #    log(tx_count_90d+1), unique_counterparties_30d/100,
    #    unique_countries_30d/10, log(max_single_tx_30d),
    #    tx_count_30d / (tx_count_90d/3 + 1)]  # velocity ratio

    # Customer profile vector (combined 34d)
    profile_vec = np.concatenate([
        occupation_emb,        # 8d
        [salary_band / 10.0],  # 1d normalized
        [country_risk],        # 1d
        session_vec,           # 8d
        ledger_vec,            # 8d
        trust_score_history,   # 4d [current, mean_30d, mean_90d, lifetime_min]
        historical_patterns,   # 4d [confirmed_count, total_alerts, max_prob, days_since_last]
    ])  # Total: 34d

    return {
        "account_id": account_id,
        "bank_id": BANK_ID,
        "profile_vector": profile_vec.tolist(),
        "occupation_embedding": occupation_emb.tolist(),
        "salary_band": salary_band,
        "country_risk": country_risk,
        "session_vector": session_vec.tolist(),
        "ledger_vector": ledger_vec.tolist(),
    }
```

### Occupation-Salary Mismatch Signal

**Prohibition enforced:** Occupation is NEVER used as the sole feature. It must always be combined with salary band and transaction behavior.

**Expected salary band ranges by occupation and country:**

```python
EXPECTED_SALARY_BANDS = {
    # (occupation_code, country_code) → (min_expected_band, max_expected_band)
    (4, "US"): (7, 9),   # doctor in US: bands 7–9 ($150K–$500K)
    (4, "IN"): (4, 7),   # doctor in India: bands 4–7 ($50K–$250K)
    (7, "US"): (5, 8),   # software engineer in US: bands 5–8
    (12, "US"): (1, 2),  # student in US: bands 1–2
    (15, "US"): (3, 6),  # retired in US: bands 3–6
    (18, "US"): (1, 2),  # unemployed in US: bands 1–2
    (27, "US"): (5, 10), # trader in US: bands 5–10 (high variance)
    (1, "US"): (3, 10),  # business owner: wide range
    # ... extended for all occupation × country pairs
}
```

**Mismatch formula:**

```python
def occupation_salary_mismatch(occupation_code, country_code, actual_salary_band, ledger_vec):
    expected_min, expected_max = EXPECTED_SALARY_BANDS.get(
        (occupation_code, country_code), (1, 10)  # default: no mismatch
    )
    expected_midpoint = (expected_min + expected_max) / 2
    band_range = expected_max - expected_min + 1
    
    # Salary band mismatch
    salary_mismatch = abs(actual_salary_band - expected_midpoint) / band_range
    
    # Transaction-implied band: derived from avg tx amount in ledger
    tx_implied_band = infer_band_from_ledger(ledger_vec)  # uses avg_tx_amount_30d
    
    # Transaction vs salary mismatch
    tx_salary_mismatch = abs(tx_implied_band - actual_salary_band) / 10.0
    
    # Combined mismatch score
    mismatch_score = 0.6 * salary_mismatch + 0.4 * tx_salary_mismatch
    
    return mismatch_score  # range [0, 1], higher = more suspicious
```

**Worked example:**
```
Doctor (occupation_code=4) in US:
  expected_salary_band_range = [7, 9]
  expected_midpoint = 8.0
  band_range = 3
  actual_salary_band = 2
  salary_mismatch = |2 - 8.0| / 3 = 2.0 (clamped to 1.0)
  
  Transaction behavior: avg monthly volume = $450,000 → tx_implied_band = 10
  tx_salary_mismatch = |10 - 2| / 10 = 0.8
  
  mismatch_score = 0.6 × 1.0 + 0.4 × 0.8 = 0.92  (HIGH RISK)
  
  Interpretation: Declared doctor earning band-2 ($10K–$25K) but moving
  $450K/month. Both salary-vs-occupation and transaction-vs-salary signals fire.
```

### Country-Based Risk Categories

```python
COUNTRY_RISK_TIERS = {
    "low": {  # FATF compliant, stable AML regimes
        "countries": ["US", "GB", "DE", "FR", "JP", "CA", "AU", "SG", "NZ", "SE"],
        "risk_score": 0.1
    },
    "medium": {  # Partial compliance, elevated exposure
        "countries": ["AE", "TR", "TH", "BR", "MX", "ZA", "IN", "CN", "RU", "SA"],
        "risk_score": 0.5
    },
    "high": {  # FATF grey/black list, sanctions, high laundering index
        "countries": ["MM", "KP", "IR", "SY", "AF", "YE", "LY", "SO", "VU", "PK"],
        "risk_score": 0.9
    }
}

def get_country_risk(country_code: str) -> float:
    for tier, data in COUNTRY_RISK_TIERS.items():
        if country_code in data["countries"]:
            return data["risk_score"]
    return 0.5  # unknown → medium risk

# Country risk enters node feature vector as a 1d scalar in the profile_vector
# For transaction edges: country_pair risk = max(src_country_risk, dst_country_risk)
```

---

## SECTION 4 — TRANSACTION SIMULATOR

### Overview

- **5,000 accounts** across 3 banks (2000 + 1500 + 1500)
- **5,000 distinct suspicious laundering scenarios** — each a unique combination of motif type, parameters, accounts, timing, and amounts
- Normal (benign) transactions generated as background noise at ~80% of total volume
- Suspicious scenarios embedded within the normal activity

### Realistic Noise and Obfuscation

To ensure the GNN learns actual laundering behavior rather than memorizing a clean simulator, **at least 30–40% of all scenarios will incorporate realistic noise**. Real AML data is messy, and our simulated data must reflect this.

**Noise Mechanisms:**
- **Missing Edges:** 5–10% of transactions in a laundering chain will be randomly dropped (simulating out-of-network transfers or cash conversions).
- **Benign Insertion:** Random benign transfers (e.g., salary deposits, utility payments) inserted into the middle of laundering chains.
- **Delayed Transfers:** Transactions randomly delayed beyond expected motif windows to break strict temporal regularity.
- **Duplicate Counterparties:** Occasional reuse of intermediaries across different laundering rings to simulate shared money mules.

**Hard Negative Dataset:**
We will generate a subset of "hard negative" benign users who mimic laundering motifs (e.g., a legitimate small business doing frequent fan-in/fan-out payroll/vendor payments, or a legitimate cross-border remittance user). This forces the model to distinguish intent rather than just structural shape.

### Parameter Space per Scenario

Each of the 5,000 scenarios independently varies:

```python
@dataclass
class LaunderingScenario:
    scenario_id: int
    motif_type: str           # one of 8 base motifs or hybrid
    origin_salary_band: int   # 1–10
    origin_occupation_code: int
    origin_country: str
    destination_country: str
    origin_bank: str          # 'bank_a', 'bank_b', 'bank_c'
    destination_bank: str
    delay_distribution: str   # 'uniform', 'exponential', 'human_mimicking'
    delay_min_hours: float
    delay_max_hours: float
    amount_min: float
    amount_max: float
    amount_distribution: str  # 'uniform', 'lognormal', 'just_below_threshold'
    tx_frequency: float       # transactions per day
    num_accounts: int         # accounts involved in this scenario
    session_profile: str      # 'human', 'bot', 'mixed'
    ledger_profile: str       # 'consistent', 'erratic', 'escalating'
```

### Distribution Requirements

```python
# Scenario type distribution (of 5000 total)
SCENARIO_DISTRIBUTION = {
    "low_and_slow": 0.50,      # ≥ 2,500 scenarios, spread over 14+ days
    "cross_country": 0.30,     # ≥ 1,500 scenarios, at least 2 countries
    "hybrid": 0.20,            # ≥ 1,000 scenarios, 2+ motif types combined
    # Note: categories overlap (a scenario can be low_and_slow AND cross_country)
    # Total unique scenarios = 5,000
}
```

### Motif Definitions

#### 1. Recursive Loop

**Formal definition:** A directed cycle `C = (v₁, v₂, ..., vₖ, v₁)` in the transaction graph where:
- `k ≥ 3` (minimum cycle length = 3 accounts)
- Each edge `(vᵢ, vᵢ₊₁)` represents a transaction with timestamp `tᵢ`
- `t₁ < t₂ < ... < tₖ < t_{k+1}` (strict temporal ordering)
- The cycle repeats with period `T`: the same cycle is observed ≥ 2 times
- `T ∈ [7, 30]` days between cycle repetitions

```python
def generate_recursive_loop(accounts: list, period_days: int, num_cycles: int, amount_range: tuple):
    """
    accounts: list of ≥3 account IDs forming the cycle
    period_days: days between cycle repetitions (7–30)
    num_cycles: ≥2
    amount_range: (min_amount, max_amount)
    """
    transactions = []
    for cycle in range(num_cycles):
        base_time = start_date + timedelta(days=cycle * period_days)
        for i in range(len(accounts)):
            src = accounts[i]
            dst = accounts[(i + 1) % len(accounts)]
            delay = timedelta(hours=np.random.exponential(scale=12))  # hours between hops
            amount = np.random.uniform(*amount_range) * (0.95 ** i)  # slight decay per hop
            transactions.append(Transaction(src, dst, amount, base_time + delay * i))
    return transactions
```

**Parameters:** amount ∈ [$500, $9,500], delay between hops ~ Exp(λ=12h), span: 14–90 days, accounts: 3–8.

#### 2. Peel-Off

**Formal definition:** A sequence of transactions from a source `v₀` through intermediaries where each hop retains a fraction `r` and peels off `(1-r)`:
- `v₀ → v₁`: amount `A`
- `v₁ → v₂`: amount `A × r₁`, `v₁` retains `A × (1 - r₁)`
- `v₂ → v₃`: amount `A × r₁ × r₂`, ...
- Where `rᵢ ~ Uniform(0.70, 0.95)` (each intermediary keeps 5–30%)

```python
def generate_peel_off(source, intermediaries, initial_amount, retention_dist=(0.70, 0.95)):
    transactions = []
    current_amount = initial_amount
    current_src = source
    t = start_time
    for intermediary in intermediaries:
        retention = np.random.uniform(*retention_dist)
        forward_amount = current_amount * retention
        transactions.append(Transaction(current_src, intermediary, current_amount, t))
        current_amount = forward_amount
        current_src = intermediary
        t += timedelta(hours=np.random.exponential(scale=24))
    return transactions
```

**Parameters:** initial amount ∈ [$5,000, $50,000], retention `r ~ U(0.70, 0.95)`, hops: 3–7, delay ~ Exp(24h), span: 3–30 days.

#### 3. Scatter-Gather

**Formal definition:**
- **Fan-out phase:** Source `v₀` sends to `N` accounts `{u₁, ..., uₙ}`, where `N ≥ 5`
- **Delay phase:** Duration `D ∈ [2, 14]` days
- **Fan-in phase:** All `N` accounts send to a single destination `w₀`
- Timing constraint: all fan-out edges have timestamps in window `[t₀, t₀ + W_out]`, all fan-in edges in `[t₀ + D, t₀ + D + W_in]`
- `W_out, W_in ∈ [1, 48]` hours

```python
def generate_scatter_gather(source, intermediaries, destination, amount, delay_days):
    N = len(intermediaries)  # ≥ 5
    split_amount = amount / N * np.random.dirichlet(np.ones(N))  # random split
    txs = []
    # Fan-out
    t_out = start_time
    for i, inter in enumerate(intermediaries):
        txs.append(Transaction(source, inter, split_amount[i],
                               t_out + timedelta(hours=np.random.uniform(0, 48))))
    # Fan-in
    t_in = t_out + timedelta(days=delay_days)
    for i, inter in enumerate(intermediaries):
        txs.append(Transaction(inter, destination, split_amount[i] * 0.98,  # small fee
                               t_in + timedelta(hours=np.random.uniform(0, 48))))
    return txs
```

**Parameters:** N ∈ [5, 20], amount ∈ [$10,000, $100,000] total, delay: 2–14 days, span: 3–21 days.

#### 4. Fan-In (distinct from scatter-gather gather phase)

**Formal definition:** Multiple sources `{s₁, ..., sₘ}` send to a single destination `d₀` where:
- `m ≥ 3`
- There is NO preceding fan-out phase from a common source
- Sources are otherwise unrelated (different occupation, country, bank)
- Amounts are similar: `std(amounts) / mean(amounts) < 0.3` (coefficient of variation)

**Parameters:** m ∈ [3, 15], individual amounts ∈ [$1,000, $9,000], span: 1–14 days.

#### 5. Fan-Out (distinct from scatter-gather fan phase)

**Formal definition:** A single source `s₀` sends to multiple destinations `{d₁, ..., dₙ}` where:
- `n ≥ 3`
- There is NO subsequent fan-in phase to a common destination
- Amounts are structured to stay below reporting thresholds

**Parameters:** n ∈ [3, 20], individual amounts ∈ [$1,000, $9,500], span: 1–7 days.

#### 6. Burst

**Formal definition:** A set of transactions from account `v` within a short window `W`:
- `|{tx : tx.timestamp ∈ [t₀, t₀ + W]}| ≥ F`
- `W ≤ 24` hours
- `F ≥ 10` transactions in window W

**Parameters:** W ∈ [1, 24] hours, F ∈ [10, 50], amounts ∈ [$200, $9,500], span: <1 day.

#### 7. Chain

**Formal definition:** A linear sequence `A → B → C → ... → Z` where:
- Minimum chain length: 4 hops (5 accounts)
- Each hop has a delay `δᵢ ~ Exp(λ)` where `λ` controls the speed
- No branching (strictly linear)
- Amount decreases by a small random fee at each hop

**Parameters:** length ∈ [4, 10] hops, amount ∈ [$2,000, $20,000], delay per hop ~ Exp(48h) for slow chains, span: 5–60 days.

#### 8. Agentic Bot Signature

**Behavioral features distinguishing bots from humans:**

```python
BOT_SIGNATURE = {
    "inter_tx_interval_cv": "<0.15",   # coefficient of variation of inter-transaction
                                        # intervals is very low (humans: >0.5)
    "session_duration_cv": "<0.10",     # nearly identical session durations
    "login_method": "api",              # always API, never web/mobile
    "time_of_day_entropy": ">0.95",     # transactions at all hours uniformly
                                        # (humans cluster in waking hours)
    "actions_per_session": "==1",       # exactly 1 action per session (single tx)
    "device_fingerprint_unique": 1,     # always same device hash
    "response_time_ms": "<200",         # sub-200ms response (human: >2000ms)
}
```

**Parameters:** Same underlying motif (chain, fan-out, etc.) but with bot session characteristics overlaid.

### Hybrid Motif Definitions

#### Fan-Out + Peel-Off
Phase 1: Fan-out from source to N intermediaries. Phase 2: Each intermediary executes a peel-off chain independently. The fan-out amounts and peel-off retention rates are drawn from the same distributions as the base motifs. Span: 7–45 days.

#### Scatter-Gather + Recursive Loop
A scatter-gather pattern where the gather destination is part of a recursive loop. After aggregation, the funds cycle through the loop ≥2 times before extraction. Span: 14–60 days.

#### Low-and-Slow Chain + Cross-Country Transfer
A chain of ≥5 hops spanning ≥3 countries with ≥21 days total duration. Each hop crosses at least one country boundary. Delays between hops: 3–7 days each. Amounts: $1,000–$8,000 per hop.

### Ground Truth Labeling

```python
def label_scenario(scenario: LaunderingScenario) -> dict:
    """Every account involved in a scenario receives a label."""
    labels = {}
    for account_id in scenario.involved_accounts:
        labels[account_id] = {
            "is_suspicious": True,
            "motif_type": scenario.motif_type,
            "scenario_id": scenario.scenario_id,
            "role": infer_role(account_id, scenario),  # 'source', 'intermediary', 'destination'
            "confidence": 1.0,  # ground truth
        }
    return labels

# Benign accounts receive: {"is_suspicious": False, "motif_type": None, ...}
# Accounts in multiple scenarios receive the highest-risk label
```

