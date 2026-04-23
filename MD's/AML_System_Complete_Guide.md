# AML Federated Learning System — Complete Technical Guide

> **Anti-Money Laundering Detection using Temporal Graph Neural Networks, Federated Learning, and Hardware-Backed Security**

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Architecture Overview](#2-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [How to Run Everything](#4-how-to-run-everything)
5. [The Neural Network](#5-the-neural-network)
6. [The Data Pipeline](#6-the-data-pipeline)
7. [Training the Model](#7-training-the-model)
8. [The Dashboard (Command Center)](#8-the-dashboard-command-center)
9. [Security Layer (HashiCorp Vault & TPM)](#9-security-layer-hashicorp-vault--tpm)
10. [The Memory System](#10-the-memory-system)
11. [Low and Slow Detection](#11-low-and-slow-detection)
12. [Known Issues & Fixes Applied](#12-known-issues--fixes-applied)
13. [How to Improve the Model](#13-how-to-improve-the-model)
14. [Training Data Sources](#14-training-data-sources)
15. [Hardware Usage](#15-hardware-usage)
16. [Glossary](#16-glossary)

---

## 1. What This Project Is

This is a **Privacy-First Intelligence System** for detecting money laundering across multiple banks.

### The Problem
In the real world, Bank A cannot see what Bank B's customers are doing because of privacy laws (PII regulations). Criminals exploit this by moving money across banks to hide the trail:

```
Criminal → Bank A → Bank B → Bank C → Clean Money
```

### The Solution
Instead of sharing raw customer data (names, addresses, SSNs), each bank shares **"Embeddings"** — mathematical vectors that describe a customer's *behavior* without revealing their *identity*.

A **Central Aggregator** collects these embeddings from all banks, builds a massive **Transaction Graph**, and runs a **Graph Neural Network (GNN)** to find hidden patterns of laundering — including "Low and Slow" schemes that rule-based systems miss.

### The Flow (Plain English)
1. **Simulator** creates fake criminals and normal customers.
2. **Vault/TPM** encrypts the data so it's "safe."
3. **Bank APIs** turn the data into "Vibes" (Embeddings).
4. **Aggregator** builds a "Map" (Graph) of those vibes.
5. **Neural Network** finds the criminals on the map.
6. **Dashboard** shows you the alerts.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Central Aggregator (Pipeline)                │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Graph        │  │  GNN Model   │  │  Alert       │         │
│  │  Builder      │→ │  (Inference) │→ │  Generator   │         │
│  └──────┬───────┘  └──────────────┘  └──────────────┘         │
│         │                                                       │
│    Fetches embeddings via HTTP                                  │
└────┬────────────┬────────────┬──────────────────────────────────┘
     │            │            │
     ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Bank A  │ │ Bank B  │ │ Bank C  │
│ API     │ │ API     │ │ API     │
│ :8001   │ │ :8002   │ │ :8003   │
├─────────┤ ├─────────┤ ├─────────┤
│Encrypted│ │Encrypted│ │Encrypted│
│ SQLite  │ │ SQLite  │ │ SQLite  │
│ (Vault) │ │ (Vault) │ │ (Vault) │
└─────────┘ └─────────┘ └─────────┘
                │
                ▼
        ┌──────────────┐
        │  Dashboard   │
        │  :8080       │
        │  (Command    │
        │   Center)    │
        └──────────────┘
```

---

## 3. Directory Structure

```
projekt/
├── start_demo.bat              # Launches all 4 services
├── aml_prototype/
│   ├── config.py               # All hyperparameters and paths
│   │
│   ├── bank_node/              # The "Bank" layer
│   │   ├── api.py              # FastAPI server (one per bank)
│   │   ├── embedding_generator.py  # Turns raw data → 34d vector
│   │   └── database.py         # Schema definitions
│   │
│   ├── aggregator/             # The "Central Intelligence" layer
│   │   ├── pipeline.py         # Orchestrates fetch → inference → alerts
│   │   ├── graph_builder.py    # Builds the PyG graph from embeddings
│   │   └── alert_generator.py  # Converts model scores → alerts
│   │
│   ├── model/                  # The "Brain" (Neural Network)
│   │   ├── gnn.py              # TemporalGraphSAGE architecture
│   │   ├── encoder.py          # Edge feature encoding (19d)
│   │   └── train.py            # Training loop with early stopping
│   │
│   ├── simulator/              # Fake data generator
│   │   ├── generator.py        # Creates accounts, transactions, KYC
│   │   └── scenarios.py        # Defines laundering motifs
│   │
│   ├── security/               # The "Vault" layer
│   │   ├── vault_manager.py    # HashiCorp Vault integration
│   │   ├── encrypted_db.py     # Encrypts/decrypts SQLite at rest
│   │   └── secure_cleanup.py   # Memory scrubbing after use
│   │
│   ├── memory/                 # Long-term behavioral memory
│   │   ├── compression.py      # 64d historical vector compression
│   │   └── feedback.py         # Compliance officer feedback loop
│   │
│   ├── dashboard/              # The GUI
│   │   ├── app.py              # FastAPI backend with SSE streaming
│   │   ├── templates/
│   │   │   └── index.html      # Full Command Center UI
│   │   └── static/
│   │
│   ├── scripts/
│   │   └── run_simulator.py    # Entry point to generate synthetic data
│   │
│   └── data/                   # Generated data (gitignored)
│       ├── bank_a/bank_a.db
│       ├── bank_b/bank_b.db
│       ├── bank_c/bank_c.db
│       └── model/checkpoints/best_model.pth
│
└── .venv/                      # Python virtual environment (~2-3 GB)
```

---

## 4. How to Run Everything

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA (optional but recommended)
- Virtual environment already set up (`.venv/`)

### Step 1: Generate Synthetic Data (First Time Only)
```powershell
.venv\Scripts\python.exe aml_prototype\scripts\run_simulator.py
```
This creates `bank_a.db`, `bank_b.db`, `bank_c.db` with ~2,000 accounts each and hundreds of thousands of transactions.

### Step 2: Train the Neural Network
```powershell
.venv\Scripts\python.exe aml_prototype\model\train.py
```
This reads the databases, builds a graph, and trains the GNN. Takes 1-5 minutes depending on GPU. Saves the best model to `data/model/checkpoints/best_model.pth`.

### Step 3: Launch All Services
Double-click `start_demo.bat` or run it from PowerShell. This opens 4 windows:
- **Bank A API** → `http://127.0.0.1:8001`
- **Bank B API** → `http://127.0.0.1:8002`
- **Bank C API** → `http://127.0.0.1:8003`
- **Dashboard** → `http://127.0.0.1:8080`

### Step 4: Run the Pipeline
Wait ~5 seconds for the APIs to start, then:
```powershell
.venv\Scripts\python.exe aml_prototype\aggregator\pipeline.py
```
This fetches embeddings from all 3 banks, builds the federated graph, runs the GNN inference, and generates alerts.

### Step 5: Open the Dashboard
Open your browser to `http://127.0.0.1:8080`. You'll see the **AML Neural Network Command Center**.

---

## 5. The Neural Network

### Where Is It?
The neural network lives in **3 files**:

| File | Purpose |
|---|---|
| `model/gnn.py` | The architecture — defines the layers |
| `model/encoder.py` | Converts raw data → numbers the AI understands |
| `model/train.py` | The teacher — trains the model using ground truth |

### What Type Is It?
It's a **Temporal GraphSAGE** — a Graph Neural Network that understands both *connections* and *time*.

### How It Works (Simple Version)
1. Each account is a **Node** with 34 features (occupation, salary, transaction velocity, etc.)
2. Each transaction is an **Edge** with 19 features (amount, timestamp, country risk, etc.)
3. The GNN looks at each node, then looks at its **neighbors** (who it sent money to), then looks at *their* neighbors. This is called **2-hop message passing**.
4. After "reading" the neighborhood, it outputs a probability: `0.0` (definitely normal) to `1.0` (definitely criminal).

### Key Components in `gnn.py`
- **`EdgeAwareSAGEConv`**: A custom neuron that looks at both the person AND the transaction connecting them. Unlike normal SAGEConv which ignores edge data.
- **`TemporalGraphSAGE`**: Stacks two layers of `EdgeAwareSAGEConv` so it can see "Friends of Friends" (2-hop neighborhoods).

### The Trained Model (The "Brain Save")
After training, the model weights are saved to:
```
aml_prototype/data/model/checkpoints/best_model.pth
```
This binary file is what gets loaded during inference (pipeline.py).

---

## 6. The Data Pipeline

### Node Features (34 dimensions)
Generated by `bank_node/embedding_generator.py`:

| Dims | Feature | Description |
|---|---|---|
| 0-7 | Occupation (one-hot) | 8 occupation categories |
| 8 | Salary Band | Normalized income level |
| 9 | Country Risk | 0=low, 1=high risk jurisdiction |
| 10 | Avg TX Amount (30d) | Average transaction size, last 30 days |
| 11 | TX Count (30d) | Number of transactions, last 30 days |
| 12 | Avg TX Amount (90d) | Average transaction size, last 90 days |
| 13 | TX Count (90d) | Number of transactions, last 90 days |
| 14 | Unique Counterparties | How many different people they transact with |
| 15 | Unique Countries (30d) | Geographic diversity |
| 16 | Max Single TX (30d) | Largest single transfer |
| 17-33 | Profile Vector | Behavioral fingerprint |

### Edge Features (19 dimensions)
Generated by `model/encoder.py`:

| Feature | Description |
|---|---|
| Log Amount | Log-normalized transaction amount |
| Timestamp (sin/cos) | Time encoded as sine/cosine waves |
| Cross-Border Flag | Does this transaction cross country borders? |
| Bank-Pair Encoding | Which banks are involved? |
| TX Type | Wire, ACH, crypto, cash |
| Country Risk | Combined risk score of source+destination |
| Time Since Previous | Time gap between this and last transaction |

### The Graph
When the pipeline runs, it builds a **PyTorch Geometric `Data` object** with:
- ~5,000 nodes (accounts across all 3 banks)
- ~900,000+ edges (transactions)
- Node feature matrix: `[5000 × 102]` (34d embedding + padding)
- Edge feature matrix: `[900000 × 19]`

---

## 7. Training the Model

### How to Train
```powershell
.venv\Scripts\python.exe aml_prototype\model\train.py
```

### What Happens During Training
1. **Data Loading**: Reads all 3 bank databases, extracts nodes and edges.
2. **Graph Construction**: Builds a PyG graph with features.
3. **Train/Val/Test Split**: Randomly splits nodes into 60/20/20.
4. **Loss Function**: Binary Cross-Entropy with `pos_weight` (dynamic, based on class imbalance).
5. **Optimizer**: Adam with learning rate from `config.py`.
6. **Early Stopping**: Stops if Val AUC doesn't improve for 10 epochs.
7. **Checkpoint**: Saves the best model (by Val AUC) to disk.

### Key Training Metrics
| Metric | What It Means |
|---|---|
| **AUC-ROC** | How well the model distinguishes suspicious vs. normal. 0.5 = coin flip, 1.0 = perfect. |
| **AUC-PR** | Precision-Recall area. More important for imbalanced data. |
| **F1** | Balance between precision and recall. |
| **Precision** | "Of all the people I flagged, how many were actually criminals?" |
| **Recall** | "Of all the actual criminals, how many did I catch?" |

### Current Performance
- **AUC-ROC**: ~0.57 (above random, but needs improvement)
- **Recall**: ~0.98 (catches almost all criminals, but also flags many innocents)

### How to Train for "Low and Slow"
1. Increase `NUM_SCENARIOS` in `config.py` (e.g., 500) to generate more diverse crime patterns.
2. Increase `pos_weight` in `train.py` to penalize the model harder for missing criminals.
3. Run the simulator again, then retrain.

---

## 8. The Dashboard (Command Center)

### URL
```
http://127.0.0.1:8080
```

### Panels

| Panel | What It Shows |
|---|---|
| **Stat Cards** (top row) | Total accounts, suspicious flagged, alerts generated, injected transactions |
| **Training Monitor** (left) | Live streaming log of the training process (SSE). Click "Train Model" to start. |
| **Live Alert Feed** (right) | Real-time alerts from injected transactions |
| **Transaction Injector** (bottom-left) | Form to manually inject transactions into the live database |
| **Bank Node Statistics** (bottom-center) | Per-bank account counts, suspicion rates, stacked bar chart |
| **Transaction Graph** (bottom-right) | Visual sample of the node/edge graph. Red = suspicious, Blue = normal. |

### How to Use the Transaction Injector
1. Select source/destination bank
2. Enter amount
3. Choose countries
4. Check "Force High-Risk Pattern" for guaranteed HIGH alert
5. Click "⚡ Inject Transaction"
6. Watch the Live Alert Feed for the result

### Creating a HIGH Risk Transaction
- Source Country: Russia or Iran
- Destination Country: North Korea or UAE
- Amount: $50,000+
- Cross-bank: Bank A → Bank C
- Check: "Force High-Risk Pattern"

### Creating a LOW Risk Transaction
- Source Country: US
- Destination Country: US
- Amount: $500
- Same bank: Bank A → Bank A
- Uncheck: "Force High-Risk Pattern"

---

## 9. Security Layer (HashiCorp Vault & TPM)

### Where Is It?
```
aml_prototype/security/
├── vault_manager.py     # HashiCorp Vault integration
├── encrypted_db.py      # Database encryption at rest
└── secure_cleanup.py    # Memory scrubbing
```

### HashiCorp Vault (`vault_manager.py`)
Instead of storing database passwords in a text file (dangerous), the code asks **HashiCorp Vault** for the encryption key at runtime.
- In production: Vault runs as a separate service with audit logging.
- In this prototype: We use a simulated key derivation (you see the warning: `"VaultManager: no VAULT_TOKEN set — using prototype key derivation"`).

### TPM 2.0 (Trusted Platform Module)
The Vault itself is designed to be locked using your computer's **TPM chip** — a hardware security module soldered to your motherboard.
- **What it does**: Even if someone steals your hard drive, they can't open the bank databases because the encryption keys are physically tied to *your* motherboard.
- **In this prototype**: TPM integration is simulated. The architecture is ready for real TPM binding in production.

### Encrypted Databases (`encrypted_db.py`)
- All `.db` files are encrypted at rest.
- When a Bank API starts, it calls `_ENC_DB.open_encrypted()` which decrypts the database *in memory*.
- When the API shuts down, it calls `close_and_reencrypt()` to re-encrypt the file on disk.
- If you try to open `bank_a.db` with a normal SQLite viewer, it will look like gibberish.

### Secure Cleanup (`secure_cleanup.py`)
After processing a batch of embeddings, the system calls `secure_delete_refs()` to zero out any temporary memory that held raw data. This prevents sensitive data from lingering in RAM.

---

## 10. The Memory System

### Where Is It?
```
aml_prototype/memory/
├── compression.py    # Historical vector compression (64d)
└── feedback.py       # Compliance officer feedback loop
```

### The Problem
Storing 10 years of transactions for every customer is too expensive and slow for AI inference.

### The Solution: Compressed Historical Vectors (`compression.py`)
After 90 days, old transactions are "squished" into a **64-dimensional vector**.
- **Dims 0-7**: Motif participation counts (how often this account appeared in known laundering patterns)
- **Dim 14**: Country diversity (how many different countries appear in history)
- **Dim 15**: Bank diversity (how many different banks)
- **Dim 16**: Number of holds/freezes

Instead of remembering "He bought 500 coffees," the vector remembers "High frequency, low value, mostly local." This saves space while keeping the behavioral memory alive.

### Feedback Loop (`feedback.py`)
When a compliance officer reviews an alert on the dashboard and submits a decision (confirm/reject), that feedback is stored and can be used to retrain the model. This creates a virtuous cycle:
```
AI flags account → Officer reviews → Confirms → Model learns → AI gets smarter
```

---

## 11. Low and Slow Detection

### What Is "Low and Slow"?
Instead of sending one big $50,000 transfer (which triggers automatic bank alarms), a criminal sends **small amounts** (e.g., $400) every few days over **several months**.

### How This System Detects It

#### 1. Training Data
The simulator (`scenarios.py`) generates "Low and Slow" motifs as 50% of all criminal examples. The model has literally "seen" thousands of slow, small-amount laundering chains during training.

#### 2. Temporal Features
The model receives `edge_time_since_prevs` — the time elapsed since the last transaction. If this number is suspiciously consistent (e.g., exactly 3 days every time), the AI recognizes this as a "Bot" or "Scheduled Smurf."

#### 3. 90-Day Window
The node features include `tx_count_90d` — transaction count over 3 months, not just 1 day. This prevents the "Slow" criminal from hiding behind quiet weeks.

#### 4. Graph Neural Network (The Killer Feature)
A normal bank rule looks at one transaction at a time: $100 → "Fine."
The GNN connects the dots across the entire graph:
- "This account has no job (KYC)."
- "But it has sent exactly $100 to the same person every Tuesday for 12 weeks."
- "And that person just sent a large chunk to a high-risk country."

The GNN catches the "Slow" because it has **Temporal Memory** through 2-hop message passing.

### How to Train Specifically for Low and Slow
1. Increase `NUM_SCENARIOS` in `config.py` to 500+.
2. Increase `pos_weight` in `train.py` (e.g., 10.0 instead of 5.0).
3. Re-run the simulator, then retrain.

---

## 12. Known Issues & Fixes Applied

### Bug 1: Ledger Summary Was Empty
- **Symptom**: All node features were zero. AUC stuck at 0.50.
- **Root Cause**: `generator.py` was computing ledger stats using `datetime('now')` against 2024-era simulation data.
- **Fix**: Changed reference date to `2024-12-31` (simulation end date).
- **File**: `simulator/generator.py` lines 107-133.

### Bug 2: Data Imbalance
- **Symptom**: Model predicted everything as suspicious.
- **Root Cause**: `NUM_SCENARIOS=500` was creating ~50% suspicious accounts (unrealistic).
- **Fix**: Reduced to `NUM_SCENARIOS=100` for ~14.5% suspicious prevalence.
- **File**: `config.py`.

### Bug 3: Bank APIs Crashing with "Unknown BANK_ID"
- **Symptom**: All 3 bank APIs failed on startup.
- **Root Cause**: `BANK_ID` env var was read at Python module import time, before Windows had applied the `set` command.
- **Fix**: Moved `BANK_ID` resolution inside the `lifespan()` function (runtime, not import time).
- **File**: `bank_node/api.py`.

### Bug 4: Training Crash on NaN AUC
- **Symptom**: Training crashed when all predictions were the same class.
- **Root Cause**: `sklearn.metrics.roc_auc_score` throws an error when only one class is present.
- **Fix**: Added `math.isnan()` guards around AUC calculations.
- **File**: `model/train.py`.

### Bug 5: Memory Compression Row Factory
- **Symptom**: `compression.py` crashed when accessing DB rows by column name.
- **Root Cause**: Missing `sqlite3.Row` factory setting.
- **Fix**: Rewrote to properly set `row_factory` before queries.
- **File**: `memory/compression.py`.

### Bug 6: Windows Encoding Crash
- **Symptom**: `run_simulator.py` crashed on Windows with Unicode errors.
- **Root Cause**: Non-ASCII characters (checkmarks, etc.) in print statements.
- **Fix**: Replaced with ASCII-safe strings.
- **File**: `scripts/run_simulator.py`.

---

## 13. How to Improve the Model

### Current State
- AUC-ROC: ~0.57 (above random, room for improvement)
- The model is stable and trains without crashing

### Path to AUC 0.90+

#### A. Better Signal (Feature Engineering)
- Add `is_part_of_chain` feature to node embeddings.
- Add `total_tx_velocity_vs_peers` (how fast is this account compared to average).
- File to modify: `bank_node/embedding_generator.py`

#### B. Smarter Architecture
- Replace `SAGEConv` with `GATConv` (Graph Attention Network) in `gnn.py`.
- GAT can "attend" to specific suspicious edges rather than averaging all neighbors.

#### C. Graph Thinning
- Filter out transactions below a minimum amount threshold.
- Only keep edges within the active 90-day window.
- File to modify: `aggregator/graph_builder.py`

#### D. Hyperparameter Tuning
In `config.py`:
- Increase `GRAPHSAGE_HIDDEN_DIM` from 64 to 128.
- Adjust `LEARNING_RATE`.
- Increase `NUM_SCENARIOS` for more diverse training data.

#### E. External Datasets
- **Elliptic Dataset (Kaggle)**: 200,000 Bitcoin transactions, pre-labeled.
- **IBM AMLSim**: Professional-grade AML simulator.
- Write an ingestion script to load external data into our SQLite schema.

---

## 14. Training Data Sources

### Current: Built-in Simulator
The simulator (`scripts/run_simulator.py`) generates:
- 2,000 accounts per bank × 3 banks = 6,000 accounts
- ~300,000 transactions per bank = ~900,000 total
- Laundering motifs: chains, smurfing, round-tripping, low-and-slow

### External Options
| Dataset | Type | Size | Best For |
|---|---|---|---|
| Elliptic (Kaggle) | Bitcoin graph | 200K nodes | Graph classification |
| IBM AMLSim | Synthetic bank | Configurable | Realistic bank flows |
| SatoshiReveal | Blockchain | Large | Cross-chain analysis |

To use external data, write an ingestion script that maps it into our `kyc`, `transactions`, and `ledger_summary` SQLite tables.

---

## 15. Hardware Usage

| Component | What Uses It | How Much |
|---|---|---|
| **GPU (VRAM)** | `train.py` and `pipeline.py` (GNN inference) | ~1-2 GB VRAM |
| **System RAM** | Building the graph (900K edges) | ~2-4 GB |
| **Disk (SSD)** | Encrypted SQLite databases, model checkpoints | ~300 MB data |
| **CPU** | Simulator, Bank APIs, Dashboard | Light |

### Why is `.venv` so large (~2-3 GB)?
- **PyTorch**: ~1.5-2 GB (includes CUDA kernels)
- **Torch Geometric**: ~300 MB (graph-specific ML library)
- **CUDA binaries**: ~1 GB (NVIDIA GPU drivers)

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **AML** | Anti-Money Laundering |
| **GNN** | Graph Neural Network — AI that operates on graph-structured data |
| **GraphSAGE** | A specific GNN algorithm that learns node embeddings by sampling and aggregating neighbor features |
| **Temporal** | Time-aware — the model considers *when* transactions happened, not just *what* |
| **Embedding** | A fixed-size numerical vector that represents complex data (a 34-number "fingerprint" of an account) |
| **Node** | A point in the graph (an account) |
| **Edge** | A connection in the graph (a transaction) |
| **2-hop** | Looking at neighbors-of-neighbors (friends of friends) |
| **Message Passing** | The process where nodes "talk" to their neighbors to share information |
| **Federated Learning** | Training AI across multiple institutions without sharing raw data |
| **PII** | Personally Identifiable Information (names, SSNs, etc.) |
| **TPM** | Trusted Platform Module — hardware chip for cryptographic key storage |
| **HashiCorp Vault** | Software for managing secrets (passwords, encryption keys) |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic curve — model quality metric |
| **pos_weight** | How much extra penalty the model gets for missing a criminal vs. false-flagging a normal person |
| **Early Stopping** | Automatically stop training when the model stops improving |
| **SSE** | Server-Sent Events — one-way streaming from server to browser |
| **Motif** | A recurring pattern in transactions (e.g., chain, fan-out, round-trip) |
| **Smurfing** | Breaking large amounts into many small transactions to avoid detection |
| **Low and Slow** | Laundering small amounts over long periods to avoid triggering rules |
| **Layering** | Moving money through multiple intermediaries to obscure its origin |
| **KYC** | Know Your Customer — identity verification data banks collect |
| **PyG** | PyTorch Geometric — the graph ML library used in this project |
| **SAGEConv** | The specific convolution layer from GraphSAGE |
| **GATConv** | Graph Attention Network convolution — a more advanced alternative |

---

*Last updated: April 23, 2026*
*Generated from debugging and development session context.*
