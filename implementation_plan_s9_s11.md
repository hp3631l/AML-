# AML Prototype — Implementation Plan (Sections 9–11)

---

> [!WARNING]
> The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys.
> This prototype uses hardware-backed key management, not hardware-backed confidential computing.

Raw KYC data NEVER leaves the local bank node. Only the following may leave a bank: hashed account ID, occupation embedding, salary band, country risk vector, session vector, ledger vector, and recent transaction metadata.
Trust Score: 0-100, Laundering Probability: 0-1, Confidence Score: 0-1. Temporary hold recommendation only, never automatic permanent freeze.
Memory: Active memory for last 30–90 days, Older data compressed into a historical vector, historical_weight = exp(-lambda * age_in_days).

---

## SECTION 9 — BANK AGENT DASHBOARD

### Dashboard Design

The dashboard is a browser-based application served via FastAPI (Jinja2 templates + vanilla JS). It displays:

1. **Alert queue** — list of active alerts sorted by laundering probability (descending)
2. **Account detail view** — full breakdown for a selected account
3. **Pattern visualization** — text-based graph rendering of detected motifs
4. **Action panel** — confirm / reject / hold buttons with notes field

### Account Summary Fields

Every account displayed includes:

| Field | Type | Display |
|-------|------|---------|
| Account ID | text | Anonymized (e.g., "ACC-B-003721") |
| Trust Score | 0–100 | Color-coded bar: green (>70), yellow (40–70), red (<40) |
| Laundering Probability | 0–1 | Numeric + bar |
| Confidence Score | 0–1 | Numeric + indicator icon |
| Detected Laundering Type | text | Motif name (e.g., "scatter_gather") |
| Countries Involved | list | Country codes with risk tier badges |
| Motif Visualization | text | ASCII graph of the pattern structure |
| Reason for Suspicion | text | Human-readable evidence summary |
| Related Accounts | list | Clickable account IDs in the same pattern |
| Previous History | struct | Prior alerts count, confirmed count, last alert date |
| Recommended Action | enum | "monitor" / "review" / "hold" |
| Hold Duration | text | "48h" (if applicable) |

### Complete Example Dashboard Output

```json
{
    "alert_queue": [
        {
            "alert_id": "ALERT-2025-00287",
            "account_id": "ACC-B-003721",
            "bank_id": "bank_b",
            "trust_score": 18.4,
            "trust_score_color": "red",
            "laundering_probability": 0.94,
            "confidence_score": 0.87,
            "detected_motif": "scatter_gather_cross_country",
            "countries": [
                {"code": "US", "risk_tier": "low"},
                {"code": "AE", "risk_tier": "medium"},
                {"code": "MM", "risk_tier": "high"}
            ],
            "motif_visualization": [
                "ACC-A-001205 (US, student, band=1)",
                "  ├──$9,200──→ ACC-A-001890 (US)",
                "  ├──$8,700──→ ACC-C-000455 (AE)",
                "  ├──$9,100──→ ACC-B-002103 (US)",
                "  ├──$8,950──→ ACC-A-000712 (AE)",
                "  ├──$9,400──→ ACC-C-001337 (MM)",
                "  ├──$9,300──→ ACC-B-003102 (US)",
                "  └──$9,500──→ ACC-B-004210 (MM)",
                "        [5-day delay]",
                "  ACC-A-001890 ──$9,016──→ ACC-B-003721",
                "  ACC-C-000455 ──$8,526──→ ACC-B-003721",
                "  ACC-B-002103 ──$8,918──→ ACC-B-003721",
                "  ACC-A-000712 ──$8,771──→ ACC-B-003721",
                "  ACC-C-001337 ──$9,212──→ ACC-B-003721",
                "  ACC-B-003102 ──$9,114──→ ACC-B-003721",
                "  ACC-B-004210 ──$9,310──→ ACC-B-003721",
                "Total gathered: $63,867 (from $64,150 dispersed)"
            ],
            "reason_for_suspicion": "Scatter-gather pattern detected: ACC-A-001205 (student, salary band 1) dispersed $64,150 across 7 accounts in 3 countries, which reconverged to ACC-B-003721 after a 5-day delay. Origin account has occupation-salary mismatch score 0.92 (student with $64K+ outflow). Session analysis shows bot-like characteristics: inter-transaction CV=0.08, all API logins. Myanmar (FATF grey list) involvement adds country risk modifier 1.45.",
            "related_accounts": [
                "ACC-A-001205", "ACC-A-001890", "ACC-C-000455",
                "ACC-B-002103", "ACC-A-000712", "ACC-C-001337",
                "ACC-B-003102", "ACC-B-004210"
            ],
            "previous_history": {
                "prior_alerts": 2,
                "agent_confirmed": 1,
                "prior_holds": 1,
                "days_since_last_alert": 45
            },
            "recommended_action": "hold",
            "recommended_hold_duration_hours": 48,
            "timestamp": "2025-01-15T14:23:00Z"
        }
    ]
}
```

### Dashboard Endpoints

```python
# Dashboard API routes (FastAPI)
@app.get("/dashboard")                      # Main dashboard view (HTML)
@app.get("/api/alerts")                     # Get all active alerts (JSON)
@app.get("/api/alerts/{alert_id}")          # Get single alert detail (JSON)
@app.get("/api/accounts/{account_id}")      # Get account detail (JSON)
@app.post("/api/alerts/{alert_id}/decide")  # Submit agent decision (JSON)
@app.get("/api/stats")                      # System-wide statistics (JSON)
```

---

## SECTION 10 — SOFTWARE STACK

### Required

| Library | Min Version | Role | Depends On |
|---------|-------------|------|------------|
| Python | 3.10+ | Runtime | All components |
| PyTorch | 2.1.0+ | Tensor operations, neural network, autograd | Model (Section 6), training, inference |
| PyTorch Geometric | 2.4.0+ | Graph neural network layers, NeighborLoader, Data objects | Model, graph construction, mini-batch sampling |
| CUDA Toolkit | 12.1+ | GPU acceleration for PyTorch | Must match PyTorch CUDA build; required for model training/inference |
| torch-scatter, torch-sparse | matching PyG version | Sparse tensor ops for message passing | PyTorch Geometric internals |
| SQLite3 | built-in (Python stdlib) | Transaction storage, KYC store, pattern memory, historical vectors | Bank nodes, memory system |
| FastAPI | 0.104+ | Bank node REST APIs, central aggregator API, dashboard server | All HTTP communication |
| uvicorn | 0.24+ | ASGI server for FastAPI | FastAPI runtime |
| NumPy | 1.24+ | Numerical operations, embedding construction | Embedding generator, simulator |
| Pydantic | 2.0+ | Request/response validation for APIs | FastAPI schemas |
| Jinja2 | 3.1+ | HTML template rendering for dashboard | Dashboard UI |

**Why SQLite (not DuckDB):** SQLite is chosen over DuckDB because:
1. Built into Python stdlib — zero additional installation.
2. Row-level CRUD operations (insert/update/delete individual transactions) are the primary access pattern for bank nodes. DuckDB is optimized for columnar analytics, not OLTP.
3. Simpler concurrent access model for the prototype (single-writer, multiple-reader with WAL mode).
4. Lower memory footprint per database instance (~1 MB overhead vs DuckDB's ~50 MB).

**Why FastAPI (not Flask):** FastAPI is chosen because:
1. Native async support — bank nodes can handle concurrent embedding requests without blocking.
2. Automatic OpenAPI documentation — useful for debugging inter-bank communication.
3. Pydantic integration — request/response validation is built-in.
4. Better performance for API-heavy workloads (async I/O).

### Optional (with conditions)

| Library | Condition | Role |
|---------|-----------|------|
| matplotlib / plotly | If visualization beyond ASCII is needed | Graph visualization, metric plots |
| scikit-learn | If additional baseline models needed | Evaluation metrics (precision, recall, F1), train/test split |
| cryptography | Required for AES-256-GCM encryption via TPM+Vault security layer | Encryption in security layer (encrypted_db.py) |
| httpx | If async HTTP client needed for inter-bank communication | Async REST calls between bank nodes |
| rich | If enhanced terminal output needed | Pretty-printing for debugging, colored logs |

---

## SECTION 11 — FILE STRUCTURE

```
aml_prototype/
│
├── README.md                          # Project overview, setup instructions, hardware requirements
├── requirements.txt                   # All Python dependencies with pinned versions
├── setup.py                           # Package configuration (optional, for development install)
├── config.py                          # Global configuration: ports, DB paths, model hyperparameters,
│                                      #   thresholds, decay constants, feature dimensions
│
├── data/                              # All persistent data (gitignored except schema files)
│   ├── bank_a/
│   │   └── bank_a.db                  # Bank A's SQLite database (KYC, transactions, sessions, ledger)
│   ├── bank_b/
│   │   └── bank_b.db                  # Bank B's SQLite database
│   ├── bank_c/
│   │   └── bank_c.db                  # Bank C's SQLite database
│   ├── central/
│   │   ├── graph_store.db             # Central aggregator's graph metadata + pattern memory
│   │   └── historical_vectors.db      # Compressed historical vectors for all accounts
│   └── model/
│       ├── checkpoints/               # Model checkpoint files (.pt)
│       └── training_logs/             # Training loss, metrics per epoch
│
├── simulator/                         # Section 4: Transaction Simulator
│   ├── __init__.py
│   ├── generator.py                   # Main simulator: generates 5000 accounts, 5000 scenarios
│   ├── motifs.py                      # Motif definitions: recursive_loop, peel_off, scatter_gather,
│   │                                  #   fan_in, fan_out, burst, chain, agentic_bot
│   ├── hybrids.py                     # Hybrid motif definitions: fanout+peeloff, scatter+loop, etc.
│   ├── accounts.py                    # Account generation: occupation, salary, country assignment
│   ├── scenarios.py                   # Scenario parameter sampling (5000 unique combinations)
│   ├── normal_traffic.py              # Benign transaction generation (background noise)
│   ├── labels.py                      # Ground truth labeling logic
│   └── distributions.py              # Amount, delay, frequency distribution functions
│
├── bank_node/                         # Section 3: Bank Node (runs as independent FastAPI service)
│   ├── __init__.py
│   ├── app.py                         # FastAPI application factory, routes for bank API
│   ├── database.py                    # SQLite connection management, table creation, queries
│   ├── kyc_codebook.py                # Occupation codebook, salary band mapping
│   ├── embedding_generator.py         # Transforms raw KYC → privacy-preserving embeddings
│   ├── session_features.py            # Computes session vector from session metadata
│   ├── ledger_features.py             # Computes ledger vector from transaction history
│   ├── country_risk.py                # Country risk tier assignments and scoring
│   ├── mismatch.py                    # Occupation-salary and salary-transaction mismatch formulas
│   └── schemas.py                     # Pydantic models for API request/response
│
├── security/                          # Section 2: TPM + HashiCorp Vault Security Layer
│   ├── __init__.py
│   ├── vault_manager.py               # HashiCorp Vault client, key CRUD, lease management
│   ├── tpm_key_store.py               # TPM 2.0 interface, seal/unseal, PCR binding
│   ├── encrypted_db.py                # AES-256-GCM SQLite/DuckDB encryption/decryption
│   └── secure_cleanup.py              # RAM cleanup, buffer zeroing, embedding isolation
│
├── aggregator/                        # Section 2: Central Aggregator
│   ├── __init__.py
│   ├── app.py                         # FastAPI application for central aggregator (port 8000)
│   ├── graph_builder.py               # Merges bank embeddings into PyG Data object
│   ├── inference.py                   # Runs model inference on constructed graph
│   └── schemas.py                     # Pydantic models for aggregator API
│
├── model/                             # Section 6: Temporal Graph Model
│   ├── __init__.py
│   ├── temporal_graphsage.py          # TemporalGraphSAGE model definition (PyTorch + PyG)
│   ├── edge_encoder.py                # Edge feature encoder (timestamp encoding, amount normalization)
│   ├── timestamp_encoding.py          # Sinusoidal timestamp encoding functions
│   ├── training.py                    # Training loop: NeighborLoader, loss, optimizer, checkpointing
│   ├── evaluation.py                  # Evaluation metrics: precision, recall, F1, AUC-ROC
│   └── mc_dropout.py                  # Monte Carlo dropout for confidence estimation
│
├── memory/                            # Section 5: Long-Term Memory System
│   ├── __init__.py
│   ├── active_memory.py               # Active window management (90-day window queries)
│   ├── compression.py                 # Historical vector compression logic
│   ├── historical_vector.py           # Historical risk vector storage, retrieval, update
│   ├── temporal_decay.py              # Decay formula: exp(-λ × age) with λ=0.01
│   └── scheduler.py                   # Daily compression scheduler (prune expired, compress)
│
├── scoring/                           # Section 7: Trust Score + Actions
│   ├── __init__.py
│   ├── trust_score.py                 # Trust score formula implementation
│   ├── country_modifier.py            # Country-aware risk modifier
│   ├── hold_engine.py                 # Temporary hold recommendation generation
│   └── thresholds.py                  # Threshold definitions and action mapping
│
├── patterns/                          # Section 8: Pattern Memory
│   ├── __init__.py
│   ├── pattern_store.py               # Pattern memory CRUD operations
│   ├── feedback_loop.py               # Agent decision processing, historical vector updates
│   └── fine_tuning.py                 # Fine-tuning buffer accumulation and periodic retraining
│
├── dashboard/                         # Section 9: Bank Agent Dashboard
│   ├── __init__.py
│   ├── app.py                         # Dashboard FastAPI routes (port 8080)
│   ├── templates/
│   │   ├── base.html                  # Base HTML template (layout, navigation, styles)
│   │   ├── dashboard.html             # Main alert queue view
│   │   ├── account_detail.html        # Single account detail view
│   │   └── pattern_view.html          # Motif visualization view
│   └── static/
│       ├── style.css                  # Dashboard CSS (dark theme, color-coded scores)
│       └── dashboard.js               # Frontend JS (fetch alerts, submit decisions, auto-refresh)
│
├── tests/                             # Integration and unit tests
│   ├── test_simulator.py              # Verify simulator generates correct distributions
│   ├── test_bank_node.py              # Verify embedding generation, privacy guarantees
│   ├── test_privacy.py                # Verify no raw KYC leaks in API responses
│   ├── test_model.py                  # Verify model forward pass, VRAM usage
│   ├── test_memory.py                 # Verify compression, decay, window management
│   ├── test_scoring.py                # Verify trust score formula, thresholds
│   ├── test_integration.py            # End-to-end: simulate → embed → infer → score → alert
│   └── test_feedback.py               # Verify agent decision feedback loop
│
└── scripts/                           # Utility scripts
    ├── run_simulator.py               # Generate synthetic data (run once)
    ├── start_banks.py                 # Start all 3 bank node services
    ├── start_aggregator.py            # Start central aggregator
    ├── start_dashboard.py             # Start dashboard server
    ├── train_model.py                 # Run model training
    ├── run_inference.py               # Run single inference cycle
    └── evaluate.py                    # Run full evaluation pipeline
```

Every directory maps to a Section:
- `simulator/` → Section 4
- `bank_node/` → Section 3
- `security/` → Section 2 (TPM + HashiCorp Vault Security Layer)
- `aggregator/` → Section 2
- `model/` → Section 6
- `memory/` → Section 5
- `scoring/` → Section 7
- `patterns/` → Section 8
- `dashboard/` → Section 9

