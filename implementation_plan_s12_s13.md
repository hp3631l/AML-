# AML Prototype — Implementation Plan (Sections 12–13)

---

> [!WARNING]
> The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys.
> This prototype uses hardware-backed key management, not hardware-backed confidential computing.

Raw KYC data NEVER leaves the local bank node. Only the following may leave a bank: hashed account ID, occupation embedding, salary band, country risk vector, session vector, ledger vector, and recent transaction metadata.
Trust Score: 0-100, Laundering Probability: 0-1, Confidence Score: 0-1. Temporary hold recommendation only, never automatic permanent freeze.
Memory: Active memory for last 30–90 days, Older data compressed into a historical vector, historical_weight = exp(-lambda * age_in_days).

---

## SECTION 12 — DEVELOPMENT PLAN

### Prioritization (Managing Scope Creep)
To mitigate the risk of scope creep and ensure a functional prototype, development is strictly prioritized. If time becomes constrained, we will aggressively remove complexity rather than preserving architectural purity.

**Must Build First:**
1. Transaction Simulator
2. One Bank API (Bank A first)
3. Temporal Graph Model
4. Trust Score Engine
5. Bank Agent Dashboard

**Only If Time Remains:**
- Multi-bank federation (Bank B and C APIs)
- TPM + HashiCorp Vault Security Layer
- Compressed historical memory
- Advanced action workflows

---

### Phase 1: Transaction Simulator

**Primary objective:** Build a deterministic, parameterized simulator that generates 5,000 accounts across 3 banks with 5,000 unique laundering scenarios and benign background traffic.

**Deliverables:**
- `simulator/` package with all motif generators
- 3 populated SQLite databases (`data/bank_a/bank_a.db`, etc.)
- Ground truth labels stored in each DB
- Distribution validation report: ≥50% low-and-slow, ≥30% cross-country, ≥20% hybrid

**Estimated time:** 5–7 days

**Justification:** 8 base motifs + 3 hybrids + parameter variation + benign traffic. Each motif requires individual implementation and testing. Distribution validation requires iteration.

**Key risks:**
- Motif overlap: two scenarios may inadvertently share accounts, creating ambiguous labels. Mitigation: enforce account reservation per scenario.
- Distribution skew: random parameter sampling may not hit distribution targets. Mitigation: stratified sampling with quotas.

**Definition of done:**
- `python scripts/run_simulator.py` completes without error
- 5,000 accounts exist across 3 DBs
- 5,000 distinct `scenario_id` values in labels table
- Distribution validation passes all three percentage thresholds
- Each motif type has ≥100 scenarios

---

### Phase 2: Bank Node APIs and Local Embedding Generators

**Primary objective:** Implement 3 independent FastAPI services that serve privacy-preserving embeddings derived from local KYC/transaction/session/ledger data, ensuring no raw PII is ever exposed.

**Deliverables:**
- `bank_node/` package with FastAPI app
- KYC codebook with all occupation codes
- Salary band mapping function
- Embedding generator producing 34d profile vectors
- Session and ledger feature extractors
- Occupation-salary mismatch calculator
- REST endpoints: `GET /embeddings/{account_id}`, `GET /embeddings/batch`, `GET /health`

**Estimated time:** 4–5 days

**Key risks:**
- PII leakage: accidentally including raw fields in API response. Mitigation: Pydantic response models that whitelist only embedding fields; explicit test that raw KYC fields are absent from all responses.
- Codebook coverage: occupations in simulated data not in codebook. Mitigation: default `occupation_code=0` for unknown.

**Definition of done:**
- All 3 bank services start on ports 8001, 8002, 8003
- `GET /embeddings/{account_id}` returns valid 34d vector
- `test_privacy.py` confirms no raw KYC fields in any response
- Mismatch score matches worked example from Section 3

---

### Phase 3: TPM + HashiCorp Vault Security Layer

> [!WARNING]
> The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys.
> This prototype uses hardware-backed key management, not hardware-backed confidential computing.

**Primary objective:** Implement the TPM 2.0 + HashiCorp Vault key management layer with per-bank AES-256-GCM database encryption, HMAC-SHA256 request signing, and secure RAM cleanup after embedding generation.

**Deliverables:**
- `security/` package with:
  - `vault_manager.py` — HashiCorp Vault client, key CRUD, lease management
  - `tpm_key_store.py` — TPM 2.0 interface, seal/unseal, PCR binding
  - `encrypted_db.py` — AES-256-GCM SQLite/DuckDB encryption/decryption
  - `secure_cleanup.py` — RAM cleanup, buffer zeroing, embedding isolation
- 5 encryption keys provisioned in Vault: `bank_a_key`, `bank_b_key`, `bank_c_key`, `aml_cache_key`, `backup_key`
- Integration with bank node API: all bank databases encrypted at rest, decrypted only during embedding generation
- Secure cleanup: raw KYC data deleted from RAM after embedding extraction
- Unit tests for encrypt/decrypt roundtrip and key retrieval workflow

**Estimated time:** 3–4 days

**Key risks:**
- TPM availability: not all laptops expose TPM 2.0 to userspace applications. Mitigation: provide a software fallback that logs a warning but still uses Vault without TPM binding.
- Vault setup complexity: HashiCorp Vault requires initialization and unsealing. Mitigation: provide a setup script that initializes Vault in dev mode for the prototype.
- Performance overhead: AES-256-GCM encryption on every database access. Mitigation: decrypt once per embedding generation cycle, not per-row; AES-GCM throughput exceeds 1 GB/s on modern CPUs.

**Definition of done:**
- All bank databases are encrypted at rest with per-bank keys from Vault
- Keys are NEVER hardcoded in source code or stored in plain text
- Decrypted data is cleared from RAM after embedding generation
- Encrypt/decrypt roundtrip produces byte-identical results
- Key retrieval from Vault completes in <100ms

---

### Phase 4: Temporal Graph Model (GraphSAGE with Temporal Features)

**Primary objective:** Implement the 2-layer GraphSAGE with temporal edge features, train it on the simulated data, and verify it fits within 6 GB VRAM.

**Deliverables:**
- `model/` package with TemporalGraphSAGE, edge encoder, timestamp encoding
- Training loop with NeighborLoader [15, 10], batch_size=256
- MC dropout confidence estimation
- Model checkpoint saving/loading
- VRAM usage measurement (<2 GB during training)
- Training metrics: loss convergence, AUC-ROC, precision, recall, F1

**Estimated time:** 7–10 days

**Justification:** Most complex component. Requires PyG data preparation, model architecture tuning, hyperparameter search (learning rate, dropout, pos_weight), and VRAM validation.

**Key risks:**
- VRAM overflow with edge features: if edge count per batch is higher than estimated. Mitigation: monitor with `torch.cuda.max_memory_allocated()`, reduce batch_size if needed.
- Poor convergence: GNN may not learn motif patterns from node features alone. Mitigation: ensure edge features (especially timestamp encoding and amount) carry sufficient signal; validate with simple baselines (random forest on node features).
- PyG version compatibility with CUDA toolkit. Mitigation: pin exact versions in requirements.txt.

**Definition of done:**
- `torch.cuda.max_memory_allocated() < 2 * 1024**3` (2 GB) during training
- Training loss decreases monotonically after warmup
- AUC-ROC > 0.80 on held-out test set (20% of accounts)
- Precision > 0.70 at recall > 0.60 for suspicious class

---

### Phase 5: Trust Score Engine and Long-Term Memory System

**Primary objective:** Implement the trust score formula, country risk modifier, hold recommendation engine, and the two-layer memory system (active window + compressed historical vectors).

**Deliverables:**
- `scoring/` package with trust score, country modifier, hold engine, thresholds
- `memory/` package with active memory window, compression, historical vectors, temporal decay
- Daily compression scheduler (prototype: manual trigger)
- Pattern memory store with agent feedback loop

**Estimated time:** 5–6 days

**Key risks:**
- Compression information loss: compressed vector may not preserve sufficient signal for repeat-offender detection. Mitigation: validate by comparing model performance with and without historical vectors on known repeat-offender scenarios.
- Temporal decay calibration: λ=0.01 may be too aggressive or too lenient. Mitigation: parameterize λ in config.py; test with λ ∈ {0.005, 0.01, 0.02}.

**Definition of done:**
- Trust score formula produces values matching the worked example (Section 7)
- Compression reduces data volume by >100x (measured)
- Historical vector update after agent confirmation increases motif participation dims by CONFIRMATION_BOOST
- Accounts with confirmed patterns score lower trust (higher suspicion) than first-time accounts with identical current behavior

---

### Phase 6: Bank Agent Dashboard

**Primary objective:** Build a functional browser-based dashboard for compliance officers to view alerts, examine account details, visualize motifs, and submit decisions.

**Deliverables:**
- `dashboard/` package with FastAPI routes and HTML templates
- Alert queue view with color-coded trust scores
- Account detail view with all fields from Section 9
- ASCII motif visualization
- Decision submission form (confirm/reject with notes)
- Auto-refresh capability (JS polling every 30 seconds)

**Estimated time:** 4–5 days

**Key risks:**
- Dashboard usability: compliance officers need clear, scannable information. Mitigation: follow the exact field list from Section 9; use color coding and progressive disclosure.
- API latency: dashboard queries aggregator which queries bank nodes. Mitigation: cache recent results at aggregator level (5-minute TTL).

**Definition of done:**
- Dashboard loads at `http://localhost:8080/dashboard`
- Alert queue displays all active alerts sorted by laundering probability
- Account detail page shows all 12 fields specified in Section 9
- Agent can submit confirm/reject decision through the UI
- Decision triggers feedback loop (pattern memory update)

---

### Phase 7: Integration Testing and Evaluation

**Primary objective:** Run end-to-end pipeline (simulate → embed → encrypt → aggregate → infer → score → alert → dashboard → feedback) and measure system performance.

**Deliverables:**
- `test_integration.py`: end-to-end test
- Performance report: precision, recall, F1, AUC-ROC by motif type
- Latency report: time per inference cycle, time per batch
- Memory report: peak RAM and VRAM during full pipeline execution
- False positive analysis: manual review of top 50 false positives
- Privacy audit: automated scan confirming no raw KYC in any inter-bank communication

**Estimated time:** 3–4 days

**Key risks:**
- Integration bugs: data format mismatches between components. Mitigation: Pydantic schemas enforce contracts at every boundary.
- Performance shortfall: model may not achieve target AUC-ROC. Mitigation: document performance and identify specific failure modes for future improvement.

**Definition of done:**
- Full pipeline runs without error on the target laptop
- **Recall > 0.80** for suspicious laundering accounts
- **Precision > 0.70** at detecting genuine laundering motifs
- **False Positive Rate < 0.10** across benign background traffic
- **Baseline Comparison:** Compare GraphSAGE performance against a simple rule engine baseline to demonstrate model lift
- Peak RAM < 10 GB (under 16 GB limit with OS overhead)
- Peak VRAM < 4 GB (under 6 GB limit)
- Privacy audit passes: zero raw KYC fields in inter-bank data
- All 8 motif types + 3 hybrids are detectable

### Total Estimated Timeline: 30–40 days

---

## SECTION 13 — FUTURE PRODUCTION SCALING

### Feasible Now (In This Prototype)

1. **5,000 accounts with synthetic data** — fully functional
2. **3 mock bank nodes with FastAPI** — running locally on different ports
3. **Privacy-preserving embeddings** — all KYC stays local, only vectors shared
4. **TPM + HashiCorp Vault Security Layer** — TPM 2.0 + Vault + AES-256-GCM per-bank encryption
5. **GraphSAGE with temporal features** — 2-layer, [15,10] sampling, fits 6GB VRAM
6. **90-day active memory + compressed historical vectors** — SQLite-backed
7. **Trust score + laundering probability + confidence** — fully specified formulas
8. **Agent dashboard with feedback loop** — browser-based, functional
9. **8 base motifs + 3 hybrid motifs** — mathematically defined
10. **Monte Carlo dropout confidence** — 10-pass inference

### Future Work Only (Do NOT Attempt in Prototype)

#### 2. Real Multi-Bank API Integration

**What changes:** Mock FastAPI services → real bank core system integrations (ISO 20022, SWIFT gpi, core banking APIs). Requires real authentication (OAuth2, mTLS), real KYC sources, real transaction feeds.

**Preserved:** Embedding generation logic, privacy layer interface, model architecture.

**Replaced:** `bank_node/app.py` routes → real bank adapters. `simulator/` → real data ingestion.

**Minimum requirement:** Legal agreements with partner banks, API access credentials, compliance certification.

#### 3. Millions of Accounts

**What changes:** Single-machine PyG graph → distributed graph storage (e.g., Amazon Neptune, TigerGraph, or DistDGL). NeighborLoader → distributed sampler. Single SQLite → PostgreSQL cluster or Apache Cassandra.

**Preserved:** Model architecture (GraphSAGE scales naturally with sampling). Feature engineering. Scoring formulas.

**Replaced:** Graph storage layer, data loading pipeline, batch scheduling.

**Minimum requirement:** Multi-node cluster (≥4 nodes, each with 64 GB RAM, 1 GPU). Distributed graph framework (DistDGL or PyG distributed).

#### 4. Larger Temporal GNN Models

**What changes:** 2-layer GraphSAGE → deeper models (4–6 layers), larger hidden dimensions (256–512), or full TGN with persistent memory.

**Preserved:** Feature engineering, privacy layer, scoring formulas, dashboard.

**Replaced:** Model definition, training pipeline (multi-GPU training with PyTorch DDP).

**Minimum requirement:** Multi-GPU server (≥2× NVIDIA A100 80GB or equivalent). Cost: ~$15,000–$30,000.

#### 5. Real-Time Streaming Ingestion

**What changes:** Batch processing → streaming pipeline. SQLite transaction inserts → Kafka topic consumption. Periodic inference → triggered inference on new events.

**Preserved:** Model architecture (inference is already batch-based, just triggered more frequently). Scoring formulas.

**Replaced:** Data ingestion layer (add Kafka/Flink). Inference scheduler (event-driven instead of batch).

**Minimum requirement:** Kafka cluster (≥3 brokers), stream processing framework (Flink or Kafka Streams), low-latency network.

#### 6. Real KYC Integration with Bank Core Systems

**What changes:** Simulated KYC data → real customer data from core banking systems. Requires data governance, access controls, audit logging, GDPR/privacy compliance.

**Preserved:** Embedding generation logic (same codebook, same band mapping). Privacy contract.

**Replaced:** Data source connections, KYC ingestion pipeline, audit logging infrastructure.

**Minimum requirement:** Core banking API access, data governance framework, legal/compliance approval.

#### 7. Production-Grade Privacy Auditing and Compliance Reporting

**What changes:** Add comprehensive audit logging for all data accesses, embedding generations, cross-bank transfers. Generate compliance reports for regulators (SAR filings, FATF compliance documentation).

**Preserved:** All existing logic. Auditing is additive.

**Replaced:** Nothing replaced — new audit layer added on top.

**Minimum requirement:** Centralized logging infrastructure (ELK stack or equivalent), compliance reporting templates, legal review.

---

> The prototype remains fully feasible on an i7-12650HX laptop with 16GB RAM and 6GB VRAM because it uses hardware-backed key storage through TPM + HashiCorp Vault, while keeping the AML model lightweight through GraphSAGE, sparse batching, 5,000 simulated accounts, and 30–90 day rolling memory.
