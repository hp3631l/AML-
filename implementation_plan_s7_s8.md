# AML Prototype — Implementation Plan (Sections 7–8)

---

> [!WARNING]
> The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys.
> This prototype uses hardware-backed key management, not hardware-backed confidential computing.

Raw KYC data NEVER leaves the local bank node. Only the following may leave a bank: hashed account ID, occupation embedding, salary band, country risk vector, session vector, ledger vector, and recent transaction metadata.
Trust Score: 0-100, Laundering Probability: 0-1, Confidence Score: 0-1. Temporary hold recommendation only, never automatic permanent freeze.
Memory: Active memory for last 30–90 days, Older data compressed into a historical vector, historical_weight = exp(-lambda * age_in_days).

---

## SECTION 7 — TRUST SCORE, LAUNDERING PROBABILITY, AND ACTIONS

### Trust Score (0–100)

> [!WARNING]
> **Heuristic Score, Not a Probability:** The Trust Score is a heuristic score used for prioritization and UI color-coding. It is **not** a statistically calibrated true risk probability. The weights below are manually chosen for demonstration purposes to combine GNN outputs with historical memory.

In a production environment, this manual formula should be replaced by a simple **logistic regression or XGBoost model** trained on the simulated labels, allowing the coefficients to be statistically learned from the data rather than arbitrarily assigned.

For the prototype, we use the following heuristic:

```python
def compute_trust_score(features: dict) -> float:
    """
    Inputs (all pre-computed from node features and model output):
        occ_sal_mismatch:       float [0, 1]  — from Section 3 formula
        sal_tx_mismatch:        float [0, 1]  — |salary_band - tx_implied_band| / 10
        country_risk:           float [0, 1]  — from country risk tiers
        session_anomaly:        float [0, 1]  — z-score of session features vs population
        motif_participation:    float [0, 1]  — max(current_window_motif_scores)
        hist_pattern_count:     int           — from compressed_memory[20]
        trust_trajectory_min:   float [0,100] — from compressed_memory[9]
        hold_count:             int           — from compressed_memory[16]
        cross_country_chain:    int           — longest cross-country chain length this window
    """
    # Weights (sum of absolute values = 1.0 for interpretability)
    w1 = -0.15   # occupation-salary mismatch (negative = reduces trust)
    w2 = -0.15   # salary-transaction mismatch
    w3 = -0.10   # country risk
    w4 = -0.10   # session anomaly
    w5 = -0.20   # current motif participation (highest weight — direct evidence)
    w6 = -0.10   # historical confirmed pattern count
    w7 = -0.05   # previous hold count
    w8 = -0.10   # cross-country chain involvement
    bias = 2.0   # baseline bias (positive = starts at high trust)
    
    raw = (
        bias +
        w1 * features['occ_sal_mismatch'] +
        w2 * features['sal_tx_mismatch'] +
        w3 * features['country_risk'] +
        w4 * features['session_anomaly'] +
        w5 * features['motif_participation'] +
        w6 * min(features['hist_pattern_count'] / 5.0, 1.0) +  # cap at 5
        w7 * min(features['hold_count'] / 3.0, 1.0) +          # cap at 3
        w8 * min(features['cross_country_chain'] / 5.0, 1.0)   # cap at 5
    )
    
    trust_score = 100.0 * (1.0 / (1.0 + math.exp(-raw)))  # sigmoid → [0, 100]
    return round(trust_score, 2)
```

**Weight justifications:**
- `w5 = -0.20` (motif participation): Highest magnitude because direct graph-structural evidence of laundering is the strongest signal.
- `w1 = w2 = -0.15` (mismatch signals): Second-highest because occupation/salary/transaction mismatches are strong behavioral indicators requiring multiple corroborating features.
- `w3 = -0.10` (country risk): Significant but lower than behavioral signals — country alone is insufficient, but high-risk country + other signals compound.
- `w4 = -0.10` (session anomaly): Bot-like session patterns are informative but can have innocent explanations (API-based banking).
- `w6 = -0.10` (historical patterns): Repeat offenders should lose trust, but historical patterns are decay-weighted and older signals have less certainty.
- `w7 = -0.05` (hold count): Prior holds indicate past suspicion but may have been resolved (lowest weight).
- `w8 = -0.10` (cross-country chain): Multi-hop cross-border transactions are a key AML red flag.

### Laundering Probability (0–1)

Output of the GNN classification head (`self.classifier` in Section 6 model).

**Output layer:** `Linear(16, 1) → Sigmoid()` — produces a scalar ∈ (0, 1). This is the true, statistically grounded risk probability. The GNN determines the actual risk based on structural graph features and can detect cases beyond explicitly defined rules. The handcrafted rules (motifs) are used primarily to generate the ground truth labels for the simulator and to provide human-readable explanations in the UI.

**Loss function:** Binary Cross-Entropy with positive class weight:

```python
# Class imbalance: ~20% of accounts are suspicious (in training data)
# pos_weight = n_negative / n_positive = 4000 / 1000 = 4.0
pos_weight = torch.tensor([4.0]).cuda()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Applied before sigmoid (use raw logits):
# loss = loss_fn(logits, labels)  # logits from Linear(16,1) before sigmoid
```

**Justification for pos_weight=4.0:** With ~20% suspicious accounts in the simulator (accounts involved in 5000 scenarios may number ~2000–3000 unique accounts across 5000 total), the class ratio is approximately 1:4. The weight ensures gradient contribution from positive (suspicious) examples is 4× that of negative examples, preventing the model from defaulting to "all clean."

### Confidence Score (0–1)

**Method: Monte Carlo (MC) Dropout.**

```python
def compute_confidence(model, x, edge_index, edge_attr, n_forward=10):
    """
    Run n_forward stochastic forward passes with dropout enabled.
    Confidence = 1 - std(predictions).
    """
    model.train()  # enable dropout
    predictions = []
    with torch.no_grad():
        for _ in range(n_forward):
            _, prob = model(x, edge_index, edge_attr)
            predictions.append(prob)
    
    preds = torch.stack(predictions)  # [n_forward, N, 1]
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    
    # Confidence: high when std is low
    confidence = 1.0 - std_pred.squeeze()  # [N]
    confidence = confidence.clamp(0.0, 1.0)
    
    model.eval()  # restore eval mode
    return mean_pred.squeeze(), confidence
```

**n_forward = 10:** 10 passes at ~100ms each = 1 second total. Acceptable for batch inference (not real-time, but AML doesn't require sub-second latency).

### Thresholds and Actions

**Trust Score thresholds:**

| Range | Risk Level | Action |
|-------|-----------|--------|
| > 70 | Low risk | No action. Normal monitoring. |
| 40–70 | Medium risk | Flag for review queue. Analyst sees account in "Monitor" list. |
| < 40 | High risk | Suspicious. Escalate to active investigation. |

**Laundering Probability thresholds:**

| Range | Action |
|-------|--------|
| > 0.85 | Recommend temporary hold. Immediate analyst notification. |
| 0.60–0.85 | Manual review required. Analyst must examine within 24h. |
| < 0.60 | Monitor only. Logged but no immediate action. |

**Country-Aware Risk Modifier:**

```python
def apply_country_modifier(laundering_prob: float, countries_involved: list,
                           chain_length: int) -> float:
    # Country risk multiplier
    max_country_risk = max(get_country_risk(c) for c in countries_involved)
    # Multiplier range: 1.0 (low risk) to 1.5 (high risk)
    country_multiplier = 1.0 + 0.5 * max_country_risk  # max_country_risk ∈ [0, 1]
    
    # Chain penalty for multi-country chains of length ≥ 3
    chain_penalty = 0.0
    if chain_length >= 3:
        chain_penalty = 0.05 * (chain_length - 2)  # +0.05 per hop beyond 2
        chain_penalty = min(chain_penalty, 0.20)    # capped at +0.20
    
    modified_prob = laundering_prob * country_multiplier + chain_penalty
    return min(modified_prob, 1.0)  # clamp to [0, 1]
```

**Example:**
```
laundering_prob = 0.65
countries = ["US", "MM"]  (US = low risk 0.1, Myanmar = high risk 0.9)
chain_length = 4

country_multiplier = 1.0 + 0.5 × 0.9 = 1.45
chain_penalty = 0.05 × (4 - 2) = 0.10

modified_prob = 0.65 × 1.45 + 0.10 = 0.9425 + 0.10 = 1.04 → clamped to 1.0
Action: RECOMMEND TEMPORARY HOLD (> 0.85)
```

### Alert Requirements

Every alert must identify:
1. **Motif type:** Which laundering pattern was detected (e.g., "scatter-gather", "low-and-slow chain")
2. **Specific pattern:** The account IDs and transaction IDs forming the pattern
3. **Confidence sufficiency:** Whether `confidence > 0.7` (threshold for hold justification)

### Temporary Hold Recommendation — Example JSON

```json
{
    "hold_id": "HOLD-2025-00142",
    "timestamp": "2025-01-15T14:23:00Z",
    "account_id": "ACC-B-003721",
    "bank_id": "bank_b",
    "recommended_hold_duration_hours": 48,
    "reason": {
        "motif_type": "scatter_gather_with_cross_country",
        "description": "Account ACC-B-003721 received funds from 7 accounts across 3 countries (US, AE, MM) within 36 hours, following a fan-out from ACC-A-001205 five days prior. The gather phase collected $67,340 total. Origin account occupation-salary mismatch score: 0.92 (declared student, salary band 1, but origin of $67K transfer chain).",
        "key_evidence": [
            "Fan-out: ACC-A-001205 → 7 intermediaries on 2025-01-10",
            "Gather: 7 intermediaries → ACC-B-003721 on 2025-01-15",
            "Cross-country: US → AE → MM → US path",
            "Origin account: student (occ=12), salary_band=1, tx_implied_band=8",
            "Bot-like session: inter-tx CV=0.08, all API logins, uniform timing"
        ],
        "related_accounts": [
            "ACC-A-001205", "ACC-A-001890", "ACC-C-000455",
            "ACC-B-002103", "ACC-A-000712", "ACC-C-001337", "ACC-B-003102"
        ],
        "countries_involved": ["US", "AE", "MM"]
    },
    "scores": {
        "trust_score": 18.4,
        "laundering_probability": 0.94,
        "confidence_score": 0.87,
        "country_risk_modifier": 1.45
    },
    "laundering_type_classification": "scatter_gather",
    "previous_history": {
        "prior_alerts": 2,
        "agent_confirmed": 1,
        "prior_holds": 1,
        "days_since_last_alert": 45
    }
}
```

---

## SECTION 8 — PATTERN MEMORY

### Suspicious Pattern Store

```sql
CREATE TABLE pattern_memory (
    alert_id TEXT PRIMARY KEY,
    scenario_timestamp TIMESTAMP,
    motif_type TEXT,
    account_ids TEXT,         -- JSON array of account IDs involved
    countries TEXT,           -- JSON array of country codes
    confidence_score REAL,
    laundering_probability REAL,
    agent_decision TEXT DEFAULT 'pending',  -- 'confirmed', 'rejected', 'pending'
    agent_decision_timestamp TIMESTAMP,
    agent_notes TEXT,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_pattern_accounts ON pattern_memory(account_ids);
CREATE INDEX idx_pattern_motif ON pattern_memory(motif_type);
```

### Feedback Loop

When a bank agent confirms an alert:

```python
def process_agent_decision(alert_id: str, decision: str, agent_notes: str):
    """
    decision: 'confirmed' or 'rejected'
    """
    # 1. Update pattern memory
    db.execute("""
        UPDATE pattern_memory
        SET agent_decision = ?, agent_decision_timestamp = ?, agent_notes = ?
        WHERE alert_id = ?
    """, (decision, datetime.utcnow(), agent_notes, alert_id))
    
    if decision == 'confirmed':
        # 2. Update compressed historical vectors for ALL involved accounts
        alert = get_alert(alert_id)
        account_ids = json.loads(alert['account_ids'])
        motif_idx = MOTIF_TYPE_TO_INDEX[alert['motif_type']]
        
        for acc_id in account_ids:
            hist = load_historical_vector(acc_id)
            
            # Boost motif participation (dims 0–7)
            hist[motif_idx] += CONFIRMATION_BOOST  # CONFIRMATION_BOOST = 2.0
            
            # Increment confirmed suspicious count (dim 20)
            hist[20] += 1.0
            
            # Update trust trajectory min (dim 9) if current trust is lower
            current_trust = get_current_trust_score(acc_id)
            hist[9] = min(hist[9], current_trust)
            
            save_historical_vector(acc_id, hist)
        
        # 3. (Optional) Add confirmed patterns to fine-tuning buffer
        fine_tune_buffer.append({
            'account_ids': account_ids,
            'motif_type': alert['motif_type'],
            'label': 1.0,  # confirmed suspicious
            'weight': 2.0,  # higher weight in next training cycle
        })
    
    elif decision == 'rejected':
        # Reduce (but don't eliminate) suspicion for involved accounts
        account_ids = json.loads(alert['account_ids'])
        for acc_id in account_ids:
            hist = load_historical_vector(acc_id)
            motif_idx = MOTIF_TYPE_TO_INDEX[alert['motif_type']]
            hist[motif_idx] = max(0, hist[motif_idx] - 0.5)  # partial reduction
            save_historical_vector(acc_id, hist)
```

### How Confirmation Accelerates Future Detection

**Mechanism:** The `CONFIRMATION_BOOST = 2.0` applied to the motif participation dimension in the compressed historical vector has the following effect:

1. The historical vector's dims 0–7 (motif participation) are part of the node feature vector fed to the GNN.
2. An account with `hist[motif_idx] = 3.0` (1 occurrence + 2.0 boost from confirmation) will have a higher feature value than an account with `hist[motif_idx] = 1.0` (1 occurrence, no confirmation).
3. The GNN's learned weights on these dimensions will produce a higher suspicion signal for the boosted account.
4. Additionally, `hist[20]` (confirmed count) directly enters the trust score formula via `w6 × min(hist_pattern_count / 5.0, 1.0)`, reducing trust score for repeat offenders.

**Explicit statement:** The system raises suspicion faster for accounts with previously confirmed patterns than for accounts with no history. This is achieved through both the `CONFIRMATION_BOOST` in the GNN input features and the `hist_pattern_count` term in the trust score formula.

### Update Mechanisms Summary

| Trigger | What Updates | How |
|---------|-------------|-----|
| Agent confirms alert | Compressed historical vector (dims 0–7, 20) | `+CONFIRMATION_BOOST` to motif dim, `+1` to confirmed count |
| Agent rejects alert | Compressed historical vector (dims 0–7) | `-0.5` to motif dim (clamped at 0) |
| Fine-tuning cycle (weekly) | Model weights | Retrain on accumulated fine-tuning buffer with confirmed labels at 2× weight |
| Threshold adjustment (monthly) | Action thresholds | Analyze false positive/negative rates, adjust ±0.05 if needed |

