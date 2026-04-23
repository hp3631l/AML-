# Plan 01 — Critical Bugs & Immediate Fixes

These are issues that currently compromise the correctness or reliability of the prototype. They should be addressed **before** any new feature work.

---

## 1. Training Produces `nan` AUC — Label Leak in Test Split

**File:** `model/train.py` (lines 72–76, 88–94)

**Problem:** The train/test split is done by **random node index**, but `NeighborLoader` with `num_neighbors=[-1, -1]` on the test set loads the **full neighborhood** including training nodes. This means:
- The test loader pulls in training-labeled nodes as neighbors, leaking signal.
- More critically, the test batches only contain nodes from a random 20% slice. If that slice happens to be all-positive or all-negative, `roc_auc_score` returns `nan` (as we observed: `"Only one class is present in y_true"`).

**Fix:**
1. Use `torch_geometric.transforms.RandomNodeSplit` or manually create `data.train_mask` / `data.test_mask` and ensure both classes appear in both splits via **stratified splitting**.
2. Change test loader to use `num_neighbors=[15, 10]` (same as training) instead of `[-1, -1]` to avoid OOM on larger graphs.
3. Wrap the `roc_auc_score` call in a try/except to handle the single-class edge case gracefully.

**Severity:** 🔴 Critical — the model has never been properly evaluated.

---

## 2. Node Features Are 68% Zero-Padded

**File:** `model/data_prep.py` (lines 52–57)

**Problem:** The 102-dimensional node feature vector is currently `[34d real data | 68d zeros]`. The model "learns" that 68 of its 102 input dimensions are always zero. This:
- Wastes model capacity.
- May cause the model to overfit to the 34 non-zero dimensions while ignoring the zero block entirely.
- Means the compressed memory system (`memory/compression.py`) is never actually wired into the graph.

**Fix:**
1. Either reduce `NODE_FEATURE_DIM` to 34 in `config.py` and adjust `gnn.py` accordingly.
2. Or wire the `HistoricalVector` (64d) from `memory/compression.py` into `data_prep.py` so the padded dims carry real signal.

**Severity:** 🟡 High — the model is training on mostly-empty features.

---

## 3. `memory/compression.py` Is Entirely Stub Code

**File:** `memory/compression.py` (lines 27–41)

**Problem:** The methods `update_motif_participation()`, `update_country_diversity()`, and `update_bank_diversity()` are all `pass` — they do nothing. The `compress_expired_transactions()` function queries a table (`active_transactions`) that may not exist in the bank databases (the simulator writes to `transactions`, not `active_transactions`).

**Fix:**
1. Implement the body of each update method to actually extract features from expired transactions.
2. Ensure the table name matches (`transactions` vs `active_transactions`).
3. Add a migration script or schema update if `active_transactions` is intended to be a separate table.

**Severity:** 🟡 High — the long-term memory system is non-functional.

---

## 4. Dashboard Hardcodes Dummy Alerts Instead of Using Real Model Output

**File:** `dashboard/app.py` (lines 36–49)

**Problem:** On startup, the dashboard inserts 3 hardcoded dummy alerts (`ALT-001`, `ALT-002`, `ALT-003`) into `pattern_memory`. There is no pipeline that connects the model's inference output to the dashboard. Real alerts from `test_integration.py` are inserted but the dummy data persists and mixes with them.

**Fix:**
1. Remove the dummy alert insertion.
2. Build an `aggregator/pipeline.py` that runs the trained model on the current graph, generates alerts for accounts exceeding the `LAUNDERING_PROB_REVIEW` threshold, and inserts them into `pattern_memory`.
3. The dashboard should only display alerts that came from the model pipeline.

**Severity:** 🟡 High — the dashboard doesn't reflect real model output.

---

## 5. `aggregator/` Directory Is Empty

**File:** `aggregator/` (empty directory)

**Problem:** The Central Aggregator — the component that ties everything together (fetches embeddings from bank APIs → builds graph → runs inference → generates alerts) — has no code at all. Currently, `data_prep.py` bypasses the bank APIs entirely and reads SQLite directly, which violates the privacy architecture.

**Fix:**
1. Implement `aggregator/pipeline.py` that:
   - Calls each bank's `/embeddings/batch/` endpoint to fetch embeddings over HTTP.
   - Constructs the PyG graph from API responses (not direct DB access).
   - Runs inference using the trained model checkpoint.
   - Pushes alerts to the central `pattern_memory` table.

**Severity:** 🟡 High — the federated architecture is bypassed during training.

---

## 6. `secure_delete_refs()` Can Destroy Live Data

**File:** `security/secure_cleanup.py` (lines 63–79)

**Problem:** `secure_delete_refs()` calls `.clear()` on any dict or list passed to it. In `bank_node/api.py`, this was previously called on `emb_data` (the API response dict) **before** returning it, which would have sent an empty `{}` to the client. The call was removed during debugging, but the function itself is dangerous — it mutates its arguments.

**Fix:**
1. Rename to `secure_delete_refs_destructive()` to make the side effect obvious.
2. Add a `copy()` step in the API endpoint if cleanup is desired after serialization.
3. Add unit tests that verify the API response is non-empty after cleanup.

**Severity:** 🟠 Medium — the bug was patched but the footgun remains.
