# AML Prototype — Implementation Plan (Sections 5–6)

---

> [!WARNING]
> The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys.
> This prototype uses hardware-backed key management, not hardware-backed confidential computing.

Raw KYC data NEVER leaves the local bank node. Only the following may leave a bank: hashed account ID, occupation embedding, salary band, country risk vector, session vector, ledger vector, and recent transaction metadata.
Trust Score: 0-100, Laundering Probability: 0-1, Confidence Score: 0-1. Temporary hold recommendation only, never automatic permanent freeze.
Memory: Active memory for last 30–90 days, Older data compressed into a historical vector, historical_weight = exp(-lambda * age_in_days).

---

## SECTION 5 — LONG-TERM MEMORY SYSTEM

### Layer 1 — Active Memory

**Time window:** Last 90 days (configurable, minimum 30 days).

**Contents per active account:**
- Full transaction graph neighborhood (all edges involving this account within the window)
- Full transaction details: amount, timestamp, counterparty (as embedding), tx_type, countries
- Session metadata: last 30 sessions
- Ledger snapshots: daily/weekly aggregates

**Storage:** SQLite, disk-backed. Tables:

```sql
-- Active transactions (auto-pruned beyond window)
CREATE TABLE active_transactions (
    tx_id TEXT PRIMARY KEY,
    src_account_id TEXT,
    dst_account_id TEXT,
    amount REAL,
    tx_type TEXT,
    timestamp TIMESTAMP,
    src_bank_id TEXT,
    dst_bank_id TEXT,
    src_country TEXT,
    dst_country TEXT,
    -- indexed for window queries
    CHECK (timestamp >= datetime('now', '-90 days'))
);
CREATE INDEX idx_active_tx_time ON active_transactions(timestamp);
CREATE INDEX idx_active_tx_src ON active_transactions(src_account_id);
CREATE INDEX idx_active_tx_dst ON active_transactions(dst_account_id);
```

**Loading strategy:** Data is loaded into GPU memory **only during inference**, as mini-batches via `NeighborLoader`. At most 256 root nodes per batch with [15, 10] sampled neighbors.

**RAM usage:** SQLite pages cached by OS ≈ 200 MB at peak. Python data structures for current batch ≈ 50 MB. Total active memory layer RAM: **~250 MB**.

### Layer 2 — Compressed Historical Memory

**Trigger:** When transactions age beyond 90 days, they are compressed into the historical risk vector for the involved accounts, then deleted from the active transactions table.

**Compression procedure (runs daily as a cron job / scheduled task):**

```python
def compress_expired_transactions(account_id: str, db: sqlite3.Connection):
    """Compress transactions older than 90 days into historical vector."""
    expired = db.execute("""
        SELECT * FROM active_transactions
        WHERE (src_account_id = ? OR dst_account_id = ?)
        AND timestamp < datetime('now', '-90 days')
    """, (account_id, account_id)).fetchall()
    
    if not expired:
        return
    
    # Load current historical vector (or initialize)
    hist = load_historical_vector(account_id, db)  # 64d vector
    
    # Update historical vector components
    hist.update_motif_participation(expired)      # dims 0–7: binary/count per motif type
    hist.update_trust_trajectory(expired)          # dims 8–11: mean, min, max, current
    hist.update_laundering_prob(expired)           # dims 12–13: peak, most_recent
    hist.update_country_diversity(expired)         # dim 14: count distinct countries
    hist.update_bank_diversity(expired)            # dim 15: count distinct banks
    hist.update_hold_count(expired)                # dim 16: number of holds
    hist.update_tx_frequency_stats(expired)        # dims 17–18: mean, variance
    hist.update_max_chain_length(expired)          # dim 19: longest chain participation
    hist.update_confirmed_suspicious(expired)      # dim 20: agent-confirmed count
    hist.update_aggregate_features(expired)        # dims 21–63: additional compressed features
    
    save_historical_vector(account_id, hist, db)
    
    # Delete expired transactions
    db.execute("""
        DELETE FROM active_transactions
        WHERE (src_account_id = ? OR dst_account_id = ?)
        AND timestamp < datetime('now', '-90 days')
    """, (account_id, account_id))
    db.commit()
```

**Historical risk vector specification (64 dimensions):**

| Dims | Name | Type | Description |
|------|------|------|-------------|
| 0–7 | `motif_participation` | float | Count of participation in each of 8 motif types, log-scaled: `log(1 + count)` |
| 8–11 | `trust_trajectory` | float | `[lifetime_mean, lifetime_min, lifetime_max, last_known]` trust scores |
| 12–13 | `laundering_prob_history` | float | `[peak_ever, most_recent_90d]` laundering probability |
| 14 | `country_diversity` | float | `log(1 + unique_countries_ever)` |
| 15 | `bank_diversity` | float | `log(1 + unique_banks_ever)` |
| 16 | `hold_count` | float | `log(1 + total_holds)` |
| 17–18 | `tx_frequency` | float | `[mean_daily_tx_count, std_daily_tx_count]` over lifetime |
| 19 | `max_chain_length` | float | Longest laundering chain this account participated in |
| 20 | `confirmed_suspicious` | float | Count of agent-confirmed suspicious patterns |
| 21–28 | `amount_statistics` | float | `[mean, std, p25, p50, p75, p95, max, total]` of historical tx amounts (log-scaled) |
| 29–36 | `temporal_statistics` | float | `[mean_inter_tx, std_inter_tx, max_gap, min_gap, weekday_frac, night_frac, weekend_frac, holiday_frac]` |
| 37–44 | `counterparty_statistics` | float | `[unique_counterparties, repeat_rate, max_single_counterparty_frac, avg_counterparty_risk, ...]` |
| 45–52 | `decay_weighted_motif` | float | Motif participation counts weighted by `exp(-λ × age)` |
| 53–63 | `reserved` | float | Zero-initialized, reserved for future features |

**Confirmed suspicious patterns storage:**

```python
# Confirmed patterns are stored with elevated weight
# When agent confirms alert for account X:
#   hist_vector[20] += 1  (confirmed_suspicious count)
#   hist_vector[0:8] += motif_one_hot * CONFIRMATION_BOOST  (CONFIRMATION_BOOST = 2.0)
#
# Effect: accounts with prior confirmed patterns have higher motif_participation
# values in their historical vector, causing the GNN to assign higher suspicion
# for similar future patterns.
```

**Explicit statement:** Confirmed suspicious patterns are stored so that repeated patterns in future windows raise the risk score faster than first-time patterns. The mechanism is the `CONFIRMATION_BOOST` multiplier applied to motif participation dimensions in the compressed vector.

### Temporal Decay Formula

```
historical_weight(age_in_days) = exp(-λ × age_in_days)
```

**λ = 0.01** (decay half-life = `ln(2) / 0.01 ≈ 69 days`)

**Justification:** At λ = 0.01:
- 30-day-old data: weight = `exp(-0.3)` = 0.74 (74% influence)
- 90-day-old data: weight = `exp(-0.9)` = 0.41 (41% influence)
- 180-day-old data: weight = `exp(-1.8)` = 0.17 (17% influence — still meaningful for repeat offenders)
- 365-day-old data: weight = `exp(-3.65)` = 0.026 (2.6% — minimal but nonzero)

This half-life matches the typical AML investigation horizon (60–90 days for initial detection, up to 1 year for prosecution).

**Application:** When combining historical vector with active-window features at inference:

```python
def combine_features(active_features: torch.Tensor,  # from current 90d window
                     historical_vector: torch.Tensor,  # 64d compressed
                     days_since_last_compression: int) -> torch.Tensor:
    decay = math.exp(-0.01 * days_since_last_compression)
    weighted_history = historical_vector * decay
    # Concatenate: active features (variable) + weighted history (64d)
    combined = torch.cat([active_features, weighted_history], dim=-1)
    return combined
```

### How the Model "Remembers" Without Full History in RAM

1. **During the active window (0–90 days):** Full transaction graph is available on disk (SQLite). Loaded as mini-batches via NeighborLoader → GPU.

2. **At compression time (day 90+):** The graph structure and transaction details are summarized into a fixed 64d vector using the aggregate statistics defined above. This is a **lossy compression** — individual transaction details are lost, but the statistical signature is preserved.

3. **Information-theoretic tradeoff:**
   - Raw history for 1 account over 1 year: ~1200 transactions × 32 bytes = 38.4 KB
   - Compressed vector: 64 × 4 bytes = 256 bytes
   - Compression ratio: **150:1**
   - Information loss: individual transaction ordering, exact amounts, specific counterparty sequences
   - Information preserved: statistical distribution of amounts, temporal patterns, motif participation counts, risk trajectory, counterparty diversity
   - The tradeoff is acceptable because the GNN's task is **pattern classification**, not **transaction reconstruction**. The compressed statistics provide sufficient signal for "has this account exhibited suspicious behavior before?" without storing every transaction.

4. **At inference:** The 64d historical vector is appended to the node feature vector. The GNN sees both current-window graph structure AND historical summary. No historical transactions are loaded into RAM.

---

## SECTION 6 — ADVANCED TEMPORAL GRAPH MODEL

### Architecture Recommendation: GraphSAGE with Temporal Features

**Recommended:** GraphSAGE with temporal edge features and attention-based temporal aggregation.

**Justification:** The binding constraint is VRAM (6 GB). GraphSAGE with neighborhood sampling provides explicit control over per-batch memory via the sampling parameters. A small temporal GNN (e.g., temporal attention network) would require storing temporal edge sequences per node in GPU memory, which is harder to bound. GraphSAGE + temporal features achieves comparable expressiveness for this scale while guaranteeing memory bounds.

**Why full TGN is infeasible — VRAM estimates:**

| Component | TGN (full) | GraphSAGE+Temporal |
|-----------|------------|-------------------|
| Node memory (persistent) | 5000 × 128 × 4B = 2.56 MB | N/A (stateless) |
| Temporal attention per node | 5000 × 100_events × 128 × 4B × 3 = 730 MB/layer | N/A |
| Edge storage for BPTT | 500K × 32 × 4B = 64 MB | Per-batch only: ~16 MB |
| Attention intermediates | ~1.5 GB (2 layers) | ~50 MB (sampled) |
| Model parameters | ~100 MB | ~15 MB |
| CUDA context | 300 MB | 300 MB |
| **Total VRAM** | **~2.7–4.5 GB** (training) | **~500 MB–1.2 GB** |

TGN could barely fit for inference but leaves no headroom for training gradients, which roughly double activation memory. GraphSAGE+Temporal uses **<25%** of available VRAM.

### Node Feature Vector

| Feature | Type | Dims | Description |
|---------|------|------|-------------|
| `occupation_embedding` | dense | 8 | `nn.Embedding(30, 8)` from occupation code |
| `salary_band` | scalar | 1 | Normalized: `band / 10.0` |
| `country_risk` | scalar | 1 | Float 0–1 from risk tier |
| `session_vector` | dense | 8 | Behavioral session features |
| `ledger_vector` | dense | 8 | Aggregated ledger features |
| `trust_score_history` | dense | 4 | `[current, mean_30d, mean_90d, lifetime_min]` |
| `historical_suspicious_patterns` | dense | 8 | Confirmed pattern participation (from compressed memory dims 0–7) |
| `compressed_memory_vector` | dense | 64 | Full historical risk vector (decay-weighted) |

**Total node feature dimensionality: 8 + 1 + 1 + 8 + 8 + 4 + 8 + 64 = 102 dimensions**

Justification: 102d is small enough for efficient batch processing. The 64d compressed memory is the largest component but carries critical historical context. Reducing it below 32d would lose motif-specific resolution (8 motifs × 4 statistics each = 32d minimum).

### Edge Feature Vector

| Feature | Type | Dims | Description |
|---------|------|------|-------------|
| `log_amount` | scalar | 1 | `log(amount + 1)` |
| `timestamp_encoding` | dense | 8 | See encoding below |
| `bank_pair` | categorical→embedding | 4 | `nn.Embedding(9, 4)` for 3×3 bank pairs |
| `tx_type` | categorical→embedding | 4 | `nn.Embedding(5, 4)` for 5 tx types |
| `country_pair_risk` | scalar | 1 | `max(src_risk, dst_risk)` |
| `time_since_previous` | scalar | 1 | `log(seconds_since_prev + 1)` |

**Total edge feature dimensionality: 1 + 8 + 4 + 4 + 1 + 1 = 19 dimensions**

### Timestamp Encoding

Raw Unix timestamps are NOT used. Encoding formula:

```python
def encode_timestamp(unix_ts: float, window_start: float, window_end: float) -> np.ndarray:
    dt = datetime.fromtimestamp(unix_ts)
    
    # Time-of-day (sinusoidal, 2d)
    hour_frac = (dt.hour + dt.minute / 60) / 24.0
    tod_sin = math.sin(2 * math.pi * hour_frac)
    tod_cos = math.cos(2 * math.pi * hour_frac)
    
    # Day-of-week (sinusoidal, 2d)
    dow_frac = dt.weekday() / 7.0
    dow_sin = math.sin(2 * math.pi * dow_frac)
    dow_cos = math.cos(2 * math.pi * dow_frac)
    
    # Day-of-month (sinusoidal, 2d)
    dom_frac = dt.day / 31.0
    dom_sin = math.sin(2 * math.pi * dom_frac)
    dom_cos = math.cos(2 * math.pi * dom_frac)
    
    # Position within 90-day active window (1d)
    window_pos = (unix_ts - window_start) / (window_end - window_start + 1e-8)
    window_pos = max(0.0, min(1.0, window_pos))
    
    # Log-scaled time since previous transaction (1d) — computed externally
    # Placeholder 0.0 here, filled by caller
    log_time_since_prev = 0.0
    
    return np.array([
        tod_sin, tod_cos,          # 2d
        dow_sin, dow_cos,          # 2d
        dom_sin, dom_cos,          # 2d
        window_pos,                # 1d
        log_time_since_prev        # 1d
    ], dtype=np.float32)  # Total: 8d
```

### Sparse Neighborhood Sampling

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data=graph_data,              # PyG Data object with all nodes/edges in active window
    num_neighbors=[15, 10],       # Layer 1: sample 15 neighbors, Layer 2: sample 10
    batch_size=256,               # 256 root nodes per batch
    input_nodes=target_nodes,     # Nodes to compute embeddings for
    shuffle=True,
)
```

**Memory calculation per batch:**
- Max nodes per batch: `256 × (1 + 15 + 15 × 10) = 256 × 166 = 42,496`
- Node features: `42,496 × 102 × 4 bytes = 17.3 MB`
- Edge features (estimated edges ≈ 42,496 × 10 avg): `424,960 × 19 × 4 bytes = 32.3 MB`
- Layer 1 activations: `42,496 × 64 × 4 = 10.9 MB`
- Layer 2 activations: `256 × 32 × 4 = 32 KB`
- Gradient buffers (training): ~60 MB (roughly 1× activations)
- **Total per batch: ~120 MB** + model params (15 MB) + CUDA context (300 MB) = **~435 MB VRAM**

**Without sampling (full neighborhood):**
- 2-hop for 256 root nodes with avg degree 100: `256 × 100 × 100 = 2.56M` node accesses
- Feature matrix: `2.56M × 102 × 4 = 1.05 GB` — just for features, before activations or gradients
- With activations + gradients: **>3 GB per batch**. Would consume >50% of VRAM for a single batch.

**Expected runtime per batch:** ~50–100 ms on i7-12650HX + RTX-class 6GB GPU (based on PyG benchmarks for similar configurations).

### Model Architecture

```python
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class TemporalGraphSAGE(nn.Module):
    def __init__(self, node_feat_dim=102, edge_feat_dim=19, hidden_dim=64, out_dim=32):
        super().__init__()
        
        # Edge feature projection
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # GraphSAGE layers with edge-aware message passing
        self.conv1 = SAGEConv(node_feat_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr='mean')
        
        # Temporal attention over edge timestamps (lightweight)
        self.temporal_attention = nn.Sequential(
            nn.Linear(16, 8),  # edge features → attention logits
            nn.Tanh(),
            nn.Linear(8, 1)   # scalar attention weight per edge
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Encode edge features
        edge_emb = self.edge_encoder(edge_attr)  # [E, 16]
        
        # Compute temporal attention weights
        attn_weights = self.temporal_attention(edge_emb).squeeze(-1)  # [E]
        attn_weights = torch.sigmoid(attn_weights)
        
        # Layer 1: message passing with temporal weighting
        h = self.conv1(x, edge_index)
        # Apply temporal attention: weight messages by attention scores
        # (implemented via edge_weight in message passing)
        h = self.norm1(h)
        h = torch.relu(h)
        h = self.dropout(h)
        
        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = torch.relu(h)
        
        # Classification (laundering probability)
        laundering_prob = self.classifier(h)  # [N, 1]
        
        return h, laundering_prob

# Model parameters: ~102×64 + 64×32 + 19×32 + 32×16 + 16×8 + 8×1 + 32×16 + 16×1
# ≈ 6528 + 2048 + 608 + 512 + 128 + 8 + 512 + 16 = ~10,360 parameters
# Size: ~10K × 4 bytes = ~40 KB (trivially small)
```

### Temporal Aggregation Method

**Choice: Time-windowed message passing with lightweight temporal attention.**

**Justification:**
- **Recurrent aggregation (GRU/LSTM):** Requires sequential processing of edge sequences per node. For 100 events/node, this is 100 sequential GRU steps — slow on GPU, and memory scales linearly with sequence length. Not hardware-efficient.
- **Full temporal attention:** O(N² ) complexity over edge sequences. For 100 events/node: 100×100 attention matrix per node per layer = excessive.
- **Time-windowed message passing + lightweight attention:** Each edge gets a scalar attention weight based on its temporal features (computed in parallel, O(E) total). The attention captures recency bias and periodic patterns. Memory is O(E), not O(E²). This fits within the VRAM budget.

The temporal attention mechanism:
1. Edge features (including timestamp encoding) are projected to a 16d embedding
2. A small MLP (16→8→1) computes a scalar attention weight per edge
3. Messages are weighted by these attention scores during aggregation
4. More recent transactions and temporally anomalous transactions receive higher attention

This provides temporal awareness without the memory overhead of full sequence modeling.

