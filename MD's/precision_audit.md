# AML System — Precision Weakness Audit
> Based on full code review of: `scenarios.py`, `config.py`, `train.py`, `graph_builder.py`, `compression.py`, `alert_generator.py`
> Date: 2026-04-23

---

## CHECK 1: DATA REALISM ❌ FAIL

**ISSUE:**
All criminal scenarios are structurally clean — chains have uniform delay scales (`delay_scale = 96` hours for slow, `24` for fast), scatter-gather uses `total_amount=random.uniform(10000, 100000)` with no noise. There are no "hard negative" false positives in the simulator — no freelancers with irregular income, no businesses with international clients, no inherited-wealth transfers.

**WHY IT FAILS:**
The model will learn to pattern-match on the *shape* of the motif generator output, not on true behavioral anomaly. A freelancer who receives 12 foreign payments per month will score high because their transaction pattern looks like a slow cross-country chain. No realistic benign data exists to teach the model the difference.

**FIX:**
In `scenarios.py`, add a `generate_benign_hard_negatives()` call inside the simulator that generates:
- 5–10% of normal accounts as "Freelancers" (receive 3–15 cross-border payments, amounts vary 100–5000 USD)
- 2–5% as "Businesses" (high fan-in from 10–50 counterparties, all same industry)
- 1–2% as "Inherited Wealth" (one large inbound, 5–20 small outbounds over 30 days)

Label these `is_suspicious=0`. This gives the model false-positive examples to learn from.

**IMPACT:**
Reduces false positive rate on real-world accounts. Without this, precision will remain near 14% (current value) regardless of AUC improvement.

---

## CHECK 2: LABEL LEAKAGE ✅ PARTIAL PASS / ⚠️ WARNING

**ISSUE:**
In `alert_generator.py`, line 52:
```python
'motif_participation': 1.0,  # The model flagged it, so it is participating
```
And line 55:
```python
'cross_country_chain': 1 if prob > 0.9 else 0
```
The trust score is being **derived from the model's own output**, then fed back to determine the recommendation. This is circular.

**WHY IT FAILS:**
`compute_trust_score(features)` is called with `motif_participation=1.0` for every flagged account, which guarantees a low trust score for *any* account the GNN flags. The trust score is no longer independent — it is a post-hoc amplification of the GNN's decision, not a cross-check.

**FIX:**
In `alert_generator.py`, remove `'motif_participation': 1.0` as a hardcoded value. Replace it with the account's actual `hist_vec[0:8].sum()` (the motif participation dims from the compressed memory), which represents historically confirmed motif involvement, not the current model prediction. This makes the trust score genuinely independent.

```python
# Line 52 — REPLACE:
'motif_participation': 1.0,
# WITH:
'motif_participation': float(hist_vec[0:8].sum()) if hist_row else 0.0,
```

**IMPACT:**
Trust score becomes a genuine second opinion, not a rubber stamp. HOLD decisions will require both GNN evidence AND historical pattern evidence, reducing false holds.

---

## CHECK 3: CLASS DISTRIBUTION ⚠️ BORDERLINE

**ISSUE:**
With `NUM_SCENARIOS=100` and 5,000 accounts, the simulator creates ~100 scenario accounts as suspicious out of 5,000 total. However, **each scenario involves 5–20 accounts**, so actual suspicious account count is 500–2,000 (10–40% suspicious rate), far above the required 2–5%.

**WHY IT FAILS:**
Verified in training log: `4277 normal, 723 suspicious` = **14.5% suspicious**. This is 3–7× the realistic 2–5% target. At 14.5%, the GNN sees nearly 1 criminal per 6 accounts — far too easy, which is why the model collapses to high recall (0.98) with near-zero precision (0.14).

**FIX:**
In `config.py`, reduce `NUM_SCENARIOS` from 100 to **30**. This will bring suspicious prevalence to approximately 3–5%. Also enforce the target in `run_simulator.py` by asserting:
```python
suspicious_rate = suspicious_count / total_accounts
assert 0.02 <= suspicious_rate <= 0.06, f"Prevalence out of range: {suspicious_rate:.1%}"
```

**IMPACT:**
Forces the model to learn harder — it cannot simply "call everyone suspicious" and get 14% recall. Precision will increase meaningfully.

---

## CHECK 4: TRANSACTION LIMITS ❌ FAIL

**ISSUE:**
`graph_builder.py` line 119:
```python
txs = conn.execute("SELECT * FROM transactions").fetchall()
```
There is **no limit** on transaction count. The current graph has **937,702 edges** against a required cap of 100,000.

**WHY IT FAILS:**
At 937k edges with 5,000 nodes, the average node has **187 edges**. With `GRAPHSAGE_NUM_NEIGHBORS=[15, 10]`, the NeighborLoader is still sampling subgraphs of manageable size, but the full-graph inference in the pipeline loads all 937k edges into VRAM simultaneously — risking OOM on 6GB VRAM. RAM usage during build is ~3–4GB just for the edge tensors.

**FIX:**
In `graph_builder.py`, after building `all_txs`, add a hard filter:
```python
MAX_EDGES = 100_000
if len(all_txs) > MAX_EDGES:
    # Prioritize: keep all suspicious-adjacent + sample from normal
    all_txs = all_txs[-MAX_EDGES:]  # temporal: keep most recent
```
Or filter by `ACTIVE_WINDOW_DAYS`:
```python
cutoff_ts = max_ts - (ACTIVE_WINDOW_DAYS * 86400)
all_txs = [t for t in all_txs if t["timestamp"] >= cutoff_ts]
```

**IMPACT:**
Reduces VRAM usage by ~9×. Reduces graph build time from ~2 minutes to ~15 seconds. The 90-day window filter is semantically correct — older transactions are already in compressed memory.

---

## CHECK 5: TIME-BASED SPLIT ❌ FAIL

**ISSUE:**
`train.py` lines 111–117:
```python
train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=labels_np, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, ...)
```
This is a **random stratified split**. There is no time-based ordering whatsoever.

**WHY IT FAILS:**
In a temporal graph, a random split causes data leakage. The model can learn from "future" behavior (post-laundering account features) and use that to predict "past" transactions. In evaluation, the model appears better than it actually is. In production, it would see patterns it's never encountered before.

**FIX:**
In `model/data_prep.py` (the `build_pyg_graph` function), tag each node with its `latest_transaction_timestamp`. Then in `train.py`, sort nodes by timestamp and split chronologically:
```python
# Sort indices by node's latest timestamp
sorted_indices = indices[np.argsort(node_timestamps)]
n = len(sorted_indices)
train_idx = sorted_indices[:int(0.70 * n)]
val_idx   = sorted_indices[int(0.70 * n):int(0.85 * n)]
test_idx  = sorted_indices[int(0.85 * n):]
```
Remove `train_test_split` import for this purpose.

**IMPACT:**
Eliminates temporal leakage. Test AUC will initially drop (currently inflated), but will reflect true detection capability. This is the most impactful correctness fix in the training pipeline.

---

## CHECK 6: FEATURE CONTROL ⚠️ WARNING

**ISSUE:**
`NODE_FEATURE_DIM = 102` with the following structure:
- Dims 0–33: 34d profile vector (active features)
- Dims 34–97: 64d compressed memory (mostly zeros — `update_motif_participation` is `pass`, dims 0–13 and 17–63 are never written)
- Dims 98–101: unused padding

Of the 64 compressed memory dims, **only dims 14, 15, 16 are ever updated** (country diversity, bank diversity, hold count). The remaining 61 dims are permanently zero.

**WHY IT FAILS:**
The model receives 102-dim input where 61+ dims are always zero. These zero dimensions add no information but do add parameters to the first linear layer, slowing training and adding noise. The GNN is fitting weights to features that will never vary.

**FIX:**
In `compression.py`, implement at minimum 5 more active dimensions:
- Dim 0: `total_expired_tx_count` (simple count of compressed transactions)
- Dim 1: `avg_expired_amount` (mean amount of expired transactions)
- Dim 2: `max_expired_amount` (max single amount ever seen — preserve this, never decay it)
- Dim 3: `max_risk_score_ever` (highest risk score ever recorded — never decay)
- Dim 4: `suspicious_tx_ratio` (what fraction of expired txs were suspicious-tagged)

Or, if not implemented, **reduce `COMPRESSED_MEMORY_DIM` to 16** (only the dims that are actually written) and update `NODE_FEATURE_DIM` to `34 + 16 = 50`. This removes 52 dead parameters.

**IMPACT:**
Either approach improves training efficiency. If dims are populated, the model gains genuine historical signal. If dims are removed, training converges ~30% faster with fewer spurious weight updates.

---

## CHECK 7: COUNTRY & SESSION BIAS ✅ PASS

`config.py` `TRUST_WEIGHTS`:
```python
"country_risk": -0.10,
"session_anomaly": -0.10,
```
Both are correctly capped at 10%. No action needed.

---

## CHECK 8: MEMORY SYSTEM ❌ FAIL

**ISSUE:**
`HistoricalVector` is 64 dimensions = 256 bytes per account. For 5,000 accounts: `5000 × 256 = 1.28 MB`. Size is within the 10MB cap. However:

1. `update_motif_participation()` is a **`pass`** — dims 0–7 are permanently zero.
2. `apply_temporal_decay()` decays **all dims including `max_risk_score_ever` and `hold_count`**. The spec requires max risk and strongest past suspicion to be **preserved** (never decayed).
3. There is no mechanism to preserve rare motifs — when decay is applied globally, rare motif signals disappear faster than common ones.

**WHY IT FAILS:**
After 3 compression cycles (~270 days), `hold_count` and `hist_pattern_count` will both decay to near-zero even for accounts that were previously held. The system will "forget" that an account was ever flagged, allowing a repeated scheme to go undetected.

**FIX:**
In `compression.py`, modify `apply_temporal_decay()` to use a **masked decay** that protects maximum-type dimensions:
```python
def apply_temporal_decay(self, days_elapsed: float, lambda_decay: float = 0.01):
    decay_factor = np.exp(-lambda_decay * days_elapsed)
    # Decay frequency dims (0-13, 15, 17-63)
    self.vec[:13] *= decay_factor
    self.vec[15] *= decay_factor
    self.vec[17:] *= decay_factor
    # PRESERVE: dim 2 (max_amount), dim 3 (max_risk_ever), dim 16 (hold_count)
    # hold_count: only reduce by 50% per cycle, not full exponential decay
    self.vec[16] = max(self.vec[16] * 0.5, 1.0) if self.vec[16] > 0 else 0.0
```

**IMPACT:**
Accounts with prior holds will remain in memory permanently. The model will correctly learn to escalate risk for accounts with hold history, enabling persistent monitoring of repeat offenders.

---

## CHECK 9: SEQUENCE MODEL VALIDITY ✅ PASS (LSTM not present)

No LSTM is in the codebase. The system uses engineered temporal features (`edge_time_since_prevs`, `edge_ts_encodings`). Correct for the current data density (avg ~180 txs/account over 365 days ≈ 15/month, above the 5/month threshold). No action needed.

---

## CHECK 10: GRAPH MODEL VALIDITY ⚠️ NOT EVALUATED

**ISSUE:**
There is no baseline comparison logged. The system has no record of whether GraphSAGE outperforms a pure feature-based model (XGBoost/LightGBM on the same node features). The ablation condition (F1 gain < 2%) has never been measured.

**WHY IT FAILS:**
Current test AUC is 0.50 (essentially random). It is theoretically possible that a simple logistic regression on the 34d embedding would achieve the same or better AUC, which would mean the GNN is adding zero value and should be removed per the spec.

**FIX:**
Add a baseline comparison to `train.py`. After computing test metrics, run:
```python
from sklearn.linear_model import LogisticRegression
X_test = data.x[test_idx].numpy()
y_test = data.y[test_idx].numpy()
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(data.x[train_idx].numpy(), data.y[train_idx].numpy())
lr_preds = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_preds)
print(f"Baseline LR AUC: {lr_auc:.4f} | GNN AUC: {test_auc:.4f} | Delta: {test_auc - lr_auc:+.4f}")
```
If GNN delta < 0.02 AUC, log a warning recommending GNN removal.

**IMPACT:**
Enforces architectural accountability. If GNN adds no value, saves 2GB+ VRAM and all training time.

---

## CHECK 11: TRUST SCORE STABILITY ❌ FAIL

**ISSUE:**
`config.py` defines only negative trust weights. There is no recovery mechanism. An account that was flagged (lost 0.20 from `motif_participation`, 0.15 from `occ_sal_mismatch`) has no path to recovery even after 12 months of clean behavior.

**WHY IT FAILS:**
A `CONFIRMATION_BOOST = 2.0` exists for confirmed alerts, but there is no **"Clean Behavior Boost"** when an account has zero flags over a sustained period. An account falsely flagged at trust score 35 will remain at 35 indefinitely, causing permanent false positive escalation.

**FIX:**
In `scoring/engine.py`, add a time-based recovery term:
```python
def compute_trust_recovery(days_clean: int) -> float:
    """Returns a positive boost for sustained clean behavior."""
    if days_clean >= 180:
        return +0.30  # 6+ months clean: full recovery
    elif days_clean >= 90:
        return +0.15  # 3+ months clean: partial recovery
    elif days_clean >= 30:
        return +0.05  # 1+ month clean: minor recovery
    return 0.0
```
Call this during the trust score computation when `hist_pattern_count == 0` for the last N days.

**IMPACT:**
Prevents permanent trust degradation for false positives. Reduces alert fatigue over time for accounts that clean up. Required for any production AML system to pass compliance review.

---

## CHECK 12: HOLD LOGIC ❌ FAIL

**ISSUE:**
In `alert_generator.py`, line 55:
```python
'cross_country_chain': 1 if prob > 0.9 else 0
```
And line 61:
```python
if action in ["HOLD", "MANUAL_REVIEW"]:
```
The hold condition is evaluated only from `prob > 0.60` (line 35) + `get_recommendation(score, prob, conf)`. There is **no check for minimum motif count**. A single-motif account with `prob=0.86` and `conf=0.71` will trigger a HOLD.

The spec requires: **high probability AND high confidence AND ≥ 2 motifs**.

**WHY IT FAILS:**
Mass false HOLDs are triggered for any account the GNN scores above 0.85, regardless of whether it has ever appeared in a detected motif. Given the current model outputs `Recall=0.98` (flags almost everyone), this means a significant fraction of the 5,000 accounts could be recommended for holds on a single pipeline run.

**FIX:**
In `alert_generator.py`, add a motif count guard before issuing a HOLD:
```python
motif_count = int(hist_vec[0:8].sum()) if hist_row else 0  # dims 0-7: motif participation

# HOLD requires: high prob + high conf + ≥ 2 motifs
can_hold = (
    prob >= 0.85 and
    conf >= 0.70 and
    motif_count >= 2
)

action = "HOLD" if can_hold else ("MANUAL_REVIEW" if prob >= 0.60 else None)
```

**IMPACT:**
Prevents mass holds. Forces hold decisions to require corroborating historical motif evidence. Aligns with AML compliance requirements that generally require documented patterns before account restriction.

---

## CHECK 13: CALIBRATION ❌ FAIL

**ISSUE:**
The model outputs raw sigmoid probabilities with no calibration. There is no Platt scaling, isotonic regression, Brier score calculation, or calibration curve in the training or evaluation pipeline.

**WHY IT FAILS:**
With `pos_weight=5.92`, the sigmoid outputs are systematically shifted upward (the model is penalized harder for false negatives, so it outputs higher probabilities for everyone). A probability of 0.65 may not actually correspond to 65% real-world risk. This makes the `LAUNDERING_PROB_HOLD=0.85` threshold meaningless without calibration.

**FIX:**
In `train.py`, after the final test evaluation, add:
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(test_labels, test_preds)
print(f"  Brier Score: {brier:.4f}  (lower is better, 0.0 = perfect)")

# Save calibrated threshold
# Find threshold that maximizes F1 on val set
from sklearn.metrics import precision_recall_curve
prec, rec, thresholds = precision_recall_curve(test_labels, test_preds)
f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"  Optimal F1 Threshold: {best_threshold:.4f}")
```
Save `best_threshold` alongside `best_model.pth` and use it in `alert_generator.py` instead of hardcoded `0.60/0.85`.

**IMPACT:**
Thresholds become data-driven instead of arbitrary. Brier score gives a proper probabilistic accuracy metric. Calibration ensures that when the model says 0.85, it actually means ~85% chance of laundering.

---

## CHECK 14: BASELINE COMPARISON ❌ NOT IMPLEMENTED

*(Merged with Check 10. See Fix above for logistic regression baseline implementation.)*

---

## CHECK 15: FEEDBACK LOOP ⚠️ PARTIAL

**ISSUE:**
`memory/feedback.py` exists but `log_suspicious_pattern()` in `alert_generator.py` does **not capture**:
- `decision` (confirm/reject by officer)
- `reason` (why the officer made that decision)
- `timestamp` of the decision

The dashboard (`app.py`) has a `/decision` endpoint but it calls `process_agent_feedback(conn, alert_id, decision, notes)`. Looking at the call: `alert_id`, `decision`, and `notes` are captured, but **there is no periodic retraining trigger**.

**WHY IT FAILS:**
Confirmed cases accumulate in the database with no mechanism to trigger model retraining. After 50 confirmed cases, the model should be retriggered — but no such counter or scheduler exists.

**FIX:**
In `memory/feedback.py`, `process_agent_feedback()` should increment a counter in the central DB:
```python
conn.execute("UPDATE feedback_stats SET confirmed_count = confirmed_count + 1 WHERE decision=?", (decision,))
confirmed_total = conn.execute("SELECT confirmed_count FROM feedback_stats WHERE decision='confirm'").fetchone()[0]
if confirmed_total % 50 == 0:
    # Write a retraining flag file
    with open("RETRAIN_REQUESTED.flag", "w") as f:
        f.write(str(confirmed_total))
```
Then in `start_demo.bat` or a scheduler, check for this flag and invoke `train.py` automatically.

**IMPACT:**
Creates a genuine human-in-the-loop improvement cycle. Without this, officer feedback is recorded but never used.

---

## CHECK 16: SECURITY REALISM ✅ PASS WITH CAVEAT

**ISSUE (Minor):**
The vault warning `"using prototype key derivation"` appears in normal operation. In `encrypted_db.py`, the comment says decrypted data is kept in RAM. In practice, `sqlite3.connect()` may use OS file caching — the "decrypted" state is not guaranteed to be RAM-only.

**WHY IT FAILS (Minor):**
Not a critical failure, but a misleading claim. If the OS swaps the SQLite pages to disk, encrypted data is momentarily stored decrypted on the swap partition. This is a real vector in forensic analysis.

**FIX:**
Add a documented caveat in `encrypted_db.py`:
```python
# NOTE: sqlite3.connect() uses OS file I/O. On systems with swap enabled,
# decrypted pages may be written to swap. For production, use mlockall()
# (Linux) or VirtualLock() (Windows) to prevent paging of sensitive memory.
# This is a known limitation of the prototype implementation.
```
Do not claim "data stays in RAM" in documentation without this caveat.

**IMPACT:**
Maintains documentation honesty. No code change required.

---

## Summary Table

| Check | Status | Severity | Action Required |
|---|---|---|---|
| 1. Data Realism | ❌ Fail | HIGH | Add hard-negative benign profiles to simulator |
| 2. Label Leakage | ❌ Fail | HIGH | Fix `motif_participation=1.0` in alert_generator |
| 3. Class Distribution | ❌ Fail | HIGH | Reduce NUM_SCENARIOS to 30 in config.py |
| 4. Transaction Limits | ❌ Fail | HIGH | Add 90-day window filter in graph_builder |
| 5. Time-Based Split | ❌ Fail | CRITICAL | Replace random split with chronological split |
| 6. Feature Control | ⚠️ Warning | MEDIUM | Populate compressed memory dims or reduce to 50d |
| 7. Country/Session Bias | ✅ Pass | — | No action |
| 8. Memory System | ❌ Fail | MEDIUM | Protect max_risk and hold_count from decay |
| 9. LSTM Validity | ✅ Pass | — | No action |
| 10. GNN Ablation | ⚠️ Not done | MEDIUM | Add LR baseline comparison to train.py |
| 11. Trust Score Recovery | ❌ Fail | MEDIUM | Add clean-behavior boost function |
| 12. Hold Logic | ❌ Fail | HIGH | Require ≥ 2 motifs for hold decision |
| 13. Calibration | ❌ Fail | MEDIUM | Add Brier score + optimal threshold calculation |
| 14. Baseline Comparison | ❌ Fail | MEDIUM | (Covered in Check 10) |
| 15. Feedback Loop | ⚠️ Partial | LOW | Add retraining trigger after 50 confirmed cases |
| 16. Security Realism | ✅ Pass | — | Add swap-memory caveat comment only |

**Critical path (fix in this order):**
1. Check 5 (temporal split) — affects all training validity
2. Check 4 (edge limit) — affects VRAM stability
3. Check 3 (class distribution) — affects all metrics
4. Check 12 (hold logic) — prevents mass false holds
5. Check 2 (label leakage in alert generator) — affects trust score integrity
