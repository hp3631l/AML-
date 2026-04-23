# AML Prototype — Implementation Plan (Sections 1–2)

> [!IMPORTANT]
> This plan is split across multiple files due to size. This file covers Sections 1–2 and the required hardware justifications.

---

## HARDWARE JUSTIFICATIONS

### Why GraphSAGE over full TGN

A full TGN maintains a persistent memory vector (e.g., 128d) for **every** node, updated on every event. For 5,000 accounts with memory vectors, attention modules, and message functions:

- TGN memory store: `5000 × 128 × 4 bytes = 2.56 MB` (acceptable alone)
- But TGN requires storing **all** temporal edges in GPU memory for backpropagation through time. With ~500K edges over 90 days (100 tx/account): `500000 × (edge_features=32) × 4 bytes = 64 MB` raw + attention computation buffers, intermediate activations for temporal attention across full sequence ≈ **3–4 GB VRAM** during training.
- TGN's temporal attention over **all** historical interactions per node scales as O(N_events × d_model). With 100 events/account average, attention matrices alone: `5000 × 100 × 128 × 4 bytes × 3 (Q,K,V)` ≈ **730 MB** per layer.
- Total TGN training VRAM: **4–5 GB**, leaving <1 GB for PyTorch overhead, CUDA context (~300 MB), OS display. **Infeasible.**

GraphSAGE with temporal features:
- Samples fixed-size neighborhoods: [15, 10] = max 150 neighbors per root node.
- Mini-batch of 256 root nodes: `256 × 150 × feature_dim(96) × 4 bytes ≈ 14.7 MB` per batch.
- Model parameters (2-layer, 96→64→32): < 50 MB.
- Total VRAM per training step: **~500 MB–1.2 GB**. Fits with headroom.

### Why sparse neighborhood sampling

Without sampling, 2-hop neighborhoods for 5,000 accounts with avg degree 100:
- 1-hop: 5000 × 100 = 500K nodes
- 2-hop: 500K × 100 = 50M node accesses
- Feature matrix: `50M × 96 × 4 bytes = 19.2 GB` — exceeds both RAM (16 GB) and VRAM (6 GB).

With [15, 10] sampling per batch of 256:
- Max nodes per batch: `256 × (1 + 15 + 15×10) = 256 × 166 = 42,496`
- Feature matrix: `42496 × 96 × 4 bytes ≈ 16.3 MB`. Fits trivially.

### Why 90-day active window + compressed historical vector

At 100 transactions/account/month, 5000 accounts, 1 year:
- Full history: `5000 × 1200 × 32 bytes (edge features) = 192 MB` raw edges.
- Graph adjacency + temporal ordering metadata: ~300 MB additional.
- Over 2+ years: exceeds RAM budget allocated to graph data (target: ≤2 GB for graph).
- Compression: each account's pre-90-day history → single 64d vector = `5000 × 64 × 4 = 1.28 MB`. **150x reduction.**

### Why CUDA, not NPU + OpenVINO

- This workload is **training + inference** of a GNN with irregular, sparse graph structure. OpenVINO is optimized for dense, fixed-topology inference (CNNs, transformers with fixed sequence length).
- PyTorch Geometric's `NeighborLoader`, sparse matrix operations (`torch_sparse`), and scatter/gather ops have **no OpenVINO operator support**.
- NPU on this CPU (if present) is limited to ~10 TOPS INT8 for fixed models. GNN training requires FP32/FP16 gradient computation. CUDA provides native PyTorch autograd support.

### TPM + HashiCorp Vault Security Layer

The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys.

> This prototype uses hardware-backed key management, not hardware-backed confidential computing.

**Key Management Design — 5 encryption keys stored in TPM 2.0 through HashiCorp Vault:**

```text
bank_a_key       — encrypts Bank A's SQLite database
bank_b_key       — encrypts Bank B's SQLite database
bank_c_key       — encrypts Bank C's SQLite database
aml_cache_key    — encrypts the central AML transaction cache
backup_key       — encrypts backups and key escrow
```

Keys are NEVER hardcoded into source code. Keys are NEVER stored in plain text files. Keys are retrieved only when data must be decrypted.

**Data Decryption Workflow:**

```text
1. Bank A transaction database is encrypted on disk.
2. The encryption key for Bank A is stored in TPM-backed HashiCorp Vault.
3. When the AML model needs Bank A data, it requests the key from Vault.
4. Vault retrieves the key from TPM.
5. The database is decrypted temporarily in RAM.
6. The AML model extracts only:
   - occupation embedding
   - salary band
   - country risk vector
   - session vector
   - ledger vector
   - transaction metadata
7. Raw KYC data is discarded immediately after embedding generation.
8. Only the embeddings are passed into the graph model.
```

> The TPM protects the encryption keys. HashiCorp Vault manages the keys. RAM and VRAM are not hardware-protected.

**Security Limitations (stated honestly):**

- TPM + Vault protects data at rest
- TPM + Vault protects key storage
- TPM + Vault does NOT protect RAM or GPU VRAM
- Once the data is decrypted for GraphSAGE or the temporal GNN, it exists in normal memory
- Therefore this is not equivalent to SGX or TDX

**Encryption Details:**

- AES-256-GCM for encrypted bank databases
- HMAC-SHA256 for request signing
- SHA-256 hashed account IDs
- TLS or localhost HTTPS between mocked banks and the AML engine

**Encryption boundaries:**

| Boundary | Where | Method |
|----------|-------|--------|
| Data at rest (bank DBs) | On disk | AES-256-GCM per-bank key from Vault |
| Data at rest (AML cache) | On disk | AES-256-GCM with aml_cache_key from Vault |
| Data in transit | Between bank nodes and AML engine | TLS / localhost HTTPS + HMAC-SHA256 signed requests |
| Data in RAM | During embedding generation | Unprotected (cleared after use) |
| Data in VRAM | During GNN inference/training | Unprotected |

**Secure Cleanup:**

```text
After embeddings are generated:
- delete raw decrypted KYC structures from RAM
- clear temporary buffers
- retain only anonymized embeddings and salary bands
```

Decrypted data remains in memory only for the duration of embedding generation (milliseconds per account). After embedding extraction, raw KYC structures are explicitly deleted and temporary buffers are zeroed.

**Security module files:**

```text
security/
    vault_manager.py       — HashiCorp Vault client, key CRUD, lease management
    tpm_key_store.py       — TPM 2.0 interface, seal/unseal, PCR binding
    encrypted_db.py        — AES-256-GCM SQLite/DuckDB encryption/decryption
    secure_cleanup.py      — RAM cleanup, buffer zeroing, embedding isolation
```

**Additional constraints preserved:**

- Raw KYC data NEVER leaves the local bank node. Only the following may leave a bank: hashed account ID, occupation embedding, salary band, country risk vector, session vector, ledger vector, and recent transaction metadata.
- Trust Score: 0-100, Laundering Probability: 0-1, Confidence Score: 0-1. Temporary hold recommendation only, never automatic permanent freeze.
- Memory: Active memory for last 30–90 days, Older data compressed into a historical vector, historical_weight = exp(-lambda * age_in_days).

---

## SECTION 1 — PROJECT GOAL

### Low-and-Slow Money Laundering

**Definition:** A laundering strategy where illicit funds are moved in small, individually unremarkable transactions spread across days, weeks, or months. Each transaction is designed to remain below reporting thresholds (e.g., below $10,000 CTR threshold in the US, below €15,000 in EU).

**Difference from burst-pattern laundering:** Burst laundering concentrates many transactions in a short window (hours to 1-2 days) — e.g., rapid structuring of $9,500 deposits across branches in one afternoon. Low-and-slow distributes the same total across 30–90+ days with irregular intervals.

**Why it evades traditional AML:**
1. **Threshold rules** (e.g., flag if amount > $X) never trigger — each transaction is small.
2. **Velocity rules** (e.g., flag if >N transactions in T hours) never trigger — frequency is low per window.
3. **Per-transaction scoring** assigns low risk to each individual event.
4. **No temporal graph memory** — systems that score transactions independently cannot accumulate evidence across weeks.

**Required time horizon:** Minimum 90 days of behavioral context to detect patterns where the laundering cycle (placement → layering → integration) spans 30–60 days. Without this, the system sees only isolated normal-looking transactions.

### Asynchronous Agentic Smurfing

**Classical smurfing:** A coordinator recruits human "smurfs" who make structured deposits at multiple branches, typically within the same day or week, at the same bank.

**Asynchronous agentic smurfing:** Automated bots (agents) execute smurfing across **multiple banks** with:
- **Deliberate randomized delays** between transactions (hours to days, drawn from a distribution mimicking human behavior)
- **Cross-bank distribution** — each bot interacts with a different bank, so no single bank sees the full pattern
- **Non-human session signatures** — uniform inter-transaction timing, API-rate-consistent request patterns, minimal session duration variance
- **Coordination** — bots follow a shared laundering plan but execute independently, making the coordination invisible to any single institution

**Why cross-bank detection is mandatory:** Each bank sees only 1/N of the pattern. A single-bank system cannot distinguish a bot's 2 small transfers from legitimate activity. Only by correlating **embeddings** across banks (without sharing raw KYC) can the full fan-out/fan-in structure be reconstructed.

### Why Traditional AML Systems Fail

1. **Threshold-based rules:** Flag transactions > $X or > N transactions in T hours. Low-and-slow stays below all thresholds.
2. **Per-transaction scoring:** Each transaction is scored independently. Score = low. No accumulation mechanism.
3. **No temporal graph memory:** Systems do not maintain a persistent representation of account behavior over time. Each day is evaluated in isolation.
4. **Single-bank visibility:** Correspondent banking relationships are opaque. Cross-bank laundering chains are invisible.
5. **Static features:** Rules don't adapt. The same thresholds that miss low-and-slow also generate false positives on legitimate high-frequency merchants.

### Why Burst-Based Detection Fails

A burst detector is tuned for: `count(transactions in window W) > threshold_N AND sum(amounts in W) > threshold_A` where W is typically 1–24 hours. A low-and-slow pattern with 2 transactions/week at $3,000 each produces `count = 0-1` in any 24h window and `sum < $6,000`. The detector's recall for this pattern class is **zero**. Burst detectors optimize for precision on high-velocity patterns, creating a systematic blind spot for temporally distributed laundering.

### Why Long-Term Temporal Memory is Required

The minimum detectable laundering cycle is ~14 days (fast low-and-slow). Typical cycles span 30–90 days. The system must maintain at minimum:
- **30-day active window** for detecting in-progress laundering cycles
- **90-day active window** (configurable) for capturing complete cycles
- **Compressed historical memory** for detecting repeat offenders and escalating risk for accounts with prior suspicious behavior

Without 90-day memory, a chain `A→B (day 1) → C (day 30) → D (day 60) → E (day 85)` is invisible — the system forgets A→B before D→E occurs.

**"The system must detect suspicious behavior spread across days, weeks, or months, even when each individual transaction appears normal and below standard AML thresholds."**

---

## SECTION 2 — FINAL SYSTEM ARCHITECTURE

### Text Architecture Diagram

Mock Bank Nodes
→ Encrypted SQLite / DuckDB Databases
→ TPM 2.0 + HashiCorp Vault Key Retrieval
→ Local KYC / Salary / Occupation Embedding Generator
→ Sparse Temporal Graph AML Model
→ Trust Score + Laundering Probability + Temporary Hold Recommendation
→ Dashboard for Bank Agent

### Component Details

#### 1. Mock Bank Nodes (Bank A / B / C)

**What it does:** Each bank node is an independent FastAPI service with its own SQLite database containing raw KYC data, transaction history, session metadata, and ledger data. It transforms raw data into privacy-preserving embeddings.

**CPU computations:**
- KYC codebook lookup (occupation text → integer code)
- Salary → salary band mapping
- Embedding vector construction (concatenation + normalization)
- HMAC signing of outgoing embeddings

**GPU computations:** None. Bank nodes are CPU-only.

**Data that stays local (NEVER leaves):**
- Customer name, PAN, Aadhaar, home address, exact salary, raw occupation text
- Raw transaction details (counterparty names, memo fields)
- Raw session logs (IP addresses, full device fingerprints)

**Data shared (embeddings only):**
- `global_entity_hash` (hashed PAN/Aadhaar fragment) for **Entity Resolution** (allows linking accounts belonging to the same person across banks)
- `occupation_embedding` (8d float vector derived from integer code)
- `salary_band` (integer 1–10)
- `country_risk_vector` (1d float)
- `session_vector` (8d float)
- `ledger_vector` (8d float)
- `customer_profile_vector` (combined 34d)
- Transaction edges with: log-amount, encoded timestamp, bank_pair, tx_type, country_pair

**Memory allocation per bank node:**
- SQLite DB on disk: ~50–200 MB per bank (1,700 accounts × ~100 tx/month × 12 months)
- RAM during embedding generation: ~100 MB peak (batch processing)
- Total per bank: ~100 MB RAM, 0 VRAM

#### 2. TPM + HashiCorp Vault Security Layer

**What it does:** Manages per-bank encryption keys through TPM 2.0 hardware and HashiCorp Vault software. Encrypts all bank databases at rest using AES-256-GCM with bank-specific keys. Provides secure key retrieval for embedding generation and HMAC-SHA256 signing for inter-service communication.

> [!WARNING]
> The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys. This prototype uses hardware-backed key management, not hardware-backed confidential computing. The TPM protects the encryption keys. HashiCorp Vault manages the keys. RAM and VRAM are not hardware-protected.

**Interfaces:**
```python
class VaultManager:
    def get_bank_key(self, bank_id: str) -> bytes:
        """Retrieve per-bank AES-256-GCM key from Vault (backed by TPM)."""
        pass
    
    def get_cache_key(self) -> bytes:
        """Retrieve AML cache encryption key from Vault."""
        pass
    
    def rotate_key(self, key_name: str) -> None:
        """Rotate a specific key in Vault."""
        pass

class EncryptedDB:
    def open_encrypted(self, db_path: str, bank_id: str) -> sqlite3.Connection:
        """Decrypt database using key from Vault, return connection."""
        pass
    
    def close_and_reencrypt(self, conn: sqlite3.Connection) -> None:
        """Close connection and ensure data is re-encrypted on disk."""
        pass

class SecureCleanup:
    def clear_kyc_from_ram(self, raw_data: dict) -> None:
        """Delete raw KYC structures, zero buffers after embedding generation."""
        pass
```

**CPU/GPU:** CPU only. Crypto operations are lightweight.  
**RAM:** <10 MB for key management. Decrypted data held temporarily during embedding generation only.

#### 3. Sparse Temporal Graph AML Model (GraphSAGE with temporal features)

**What it does:** Constructs a cross-bank transaction graph from merged embeddings. Runs GraphSAGE with temporal edge features to produce per-account risk representations.

**CPU computations:**
- Graph construction from incoming embeddings
- `NeighborLoader` sampling
- Data loading and batching
- Post-inference score computation

**GPU (CUDA) computations:**
- GraphSAGE forward pass (message passing, aggregation)
- Loss computation and backpropagation during training
- Mini-batch inference

**Memory at peak load:**
- **RAM:** Graph structure (5000 nodes, ~200K active edges in 90d window): ~50 MB. NeighborLoader buffers: ~100 MB. SQLite transaction cache: ~200 MB. Python/FastAPI overhead: ~300 MB. Compressed historical vectors: ~1.3 MB. **Total RAM: ~1.5–2 GB** (well within 16 GB after OS ~4 GB).
- **VRAM:** Model parameters: ~15 MB. Per-batch feature tensors (256 × 166 nodes × 96 features × 4B): ~16 MB. Intermediate activations (2 layers): ~50 MB. Gradient buffers (training): ~80 MB. CUDA context: ~300 MB. **Total VRAM: ~500 MB–1.2 GB** (well within 6 GB).

#### 4. Trust Score + Laundering Probability + Confidence Score Engine

**What it does:** Takes GNN output embeddings and computes three scores per account. Trust score uses a weighted formula combining GNN features with historical memory. Laundering probability is the GNN classification head output. Confidence is computed via MC dropout.

- Trust Score: 0–100
- Laundering Probability: 0–1
- Confidence Score: 0–1

**CPU only.** Score computation is lightweight linear algebra.  
**RAM:** <50 MB.

#### 5. Temporary Hold Recommendation Engine

**What it does:** Applies threshold logic to scores, generates hold recommendations with motif classification, duration, and evidence summary. Temporary hold recommendation only, never automatic permanent freeze.

**CPU only.** Rule evaluation.  
**RAM:** <20 MB.

#### 6. Bank Agent Dashboard

**What it does:** Browser-based UI (served from FastAPI) showing account summaries, alerts, motif visualizations, and action buttons for compliance officers.

**CPU only.** HTML/JS rendering.  
**RAM:** ~50 MB for the web server.

### Total System Resource Usage (Peak)

| Resource | Usage | Limit | Headroom |
|----------|-------|-------|----------|
| RAM | ~4–5 GB (app) + ~4 GB (OS) = ~9 GB | 16 GB | ~7 GB |
| VRAM | ~1.2 GB (training peak) | 6 GB | ~4.8 GB |
| Disk | ~1–2 GB (all SQLite DBs + model checkpoints) | Laptop SSD | Ample |

