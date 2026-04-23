# Plan 03 — Aggregator Pipeline (The Missing Core)

The `aggregator/` directory is empty. This is the component that ties every other subsystem together into a working end-to-end pipeline. Without it, the bank APIs, model, scoring engine, and dashboard are disconnected islands.

---

## Current Architecture Gap

```
Simulator → Bank DBs → Bank APIs → ??? → Model → Scoring → Dashboard
                                     ^
                                     |
                              aggregator/ is empty
```

The `data_prep.py` script currently **bypasses** the bank APIs and reads SQLite files directly, which violates the privacy contract (raw KYC is accessible). The aggregator must enforce the API boundary.

---

## Proposed Implementation

### File: `aggregator/pipeline.py`

**Purpose:** Orchestrate the full inference cycle.

```
Step 1: Fetch embeddings from all 3 bank APIs (HTTP)
Step 2: Fetch transactions from central graph store  
Step 3: Build PyG graph from API responses (never touching raw KYC)
Step 4: Load trained model checkpoint
Step 5: Run inference with MC Dropout
Step 6: Compute trust scores for flagged accounts
Step 7: Generate alerts and insert into pattern_memory
Step 8: Log metrics and timing
```

**Key design decisions:**
- Embeddings are fetched via `GET /embeddings/batch/` — not by opening SQLite files.
- Transactions are stored in the central graph store (which only has anonymized tx metadata, no PII).
- The aggregator **never** sees customer names, PANs, or Aadhaar numbers.

### File: `aggregator/graph_builder.py`

**Purpose:** Construct the PyG `Data` object from API responses instead of raw DB queries.

This replaces the direct-DB-access pattern in `model/data_prep.py` for production use. `data_prep.py` remains as a training-only utility.

### File: `aggregator/alert_generator.py`

**Purpose:** Convert model predictions into structured alerts.

```python
def generate_alerts(predictions, confidences, account_ids, threshold=0.60):
    """
    For each account with laundering_prob > threshold:
      1. Compute trust score
      2. Determine recommendation (HOLD / MANUAL_REVIEW / ALLOW)
      3. Detect which motif type best matches the neighborhood structure
      4. Insert alert into pattern_memory
    """
```

### File: `aggregator/scheduler.py`

**Purpose:** Run the pipeline on a schedule (e.g., every 30 minutes).

For the prototype, this can be a simple `while True` loop with `time.sleep()`. In production, this would be replaced by a proper job scheduler (Celery, Airflow, or cron).

---

## Estimated Effort

| File | Lines | Complexity |
|------|-------|------------|
| `pipeline.py` | ~150 | Medium — HTTP calls + orchestration |
| `graph_builder.py` | ~120 | Medium — PyG graph construction from dicts |
| `alert_generator.py` | ~80 | Low — threshold logic + DB insert |
| `scheduler.py` | ~40 | Low — loop with sleep |

**Total:** ~390 lines, ~2–3 days.

---

## Privacy Verification

After implementing the aggregator, run a **privacy audit**:
1. Grep the aggregator code for any direct SQLite access to bank DBs → should be zero.
2. Grep for any PII field names (`customer_name`, `pan`, `aadhaar`, `home_address`) → should be zero.
3. Verify that the only data flowing into the aggregator comes from the `/embeddings/` API endpoints.
