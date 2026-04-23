# Replace SimulatedConfidentialLayer with TPM + HashiCorp Vault Security Layer

## Goal

Revise all 6 implementation plan files in `c:\Users\praka\Desktop\projekt` to remove every reference to `SimulatedConfidentialLayer`, SGX, Intel TDX, AMD SEV-SNP, hardware-backed confidential computing, and simulated enclaves — and replace them with a realistic, laptop-compatible security architecture using **TPM 2.0 + HashiCorp Vault + encrypted local databases + per-bank encryption keys**.

---

## Proposed Changes

### Global Preamble Change (All 6 Files)

Every file currently has this preamble:

```markdown
> [!WARNING]
> This prototype does not implement SGX, Intel TDX, AMD SEV-SNP, or any hardware-backed confidential computing. It uses a software-only SimulatedConfidentialLayer because the target hardware is an i7-12650HX laptop with 16GB RAM and 6GB VRAM.
```

**Replace with:**

```markdown
> [!WARNING]
> The laptop does not support SGX, Intel TDX, or other confidential-computing technologies. Instead, the prototype uses TPM-backed key management with HashiCorp Vault to protect bank-specific encryption keys.
> This prototype uses hardware-backed key management, not hardware-backed confidential computing.
```

This change applies to all 6 files:
- `implementation_plan_s1_s2.md`
- `implementation_plan_s3_s4.md`
- `implementation_plan_s5_s6.md`
- `implementation_plan_s7_s8.md`
- `implementation_plan_s9_s11.md`
- `implementation_plan_s12_s13.md`

---

### [MODIFY] [implementation_plan_s1_s2.md](file:///c:/Users/praka/Desktop/projekt/implementation_plan_s1_s2.md)

1. **Preamble** (lines 5–6): Replace with new TPM+Vault preamble
2. **Prototype Privacy Layer section** (lines 50–59): Replace `SimulatedConfidentialLayer` description with full TPM + Vault security architecture, including:
   - Explicit statement about no SGX/TDX support
   - 5 encryption keys (bank_a_key, bank_b_key, bank_c_key, aml_cache_key, backup_key)
   - Data-at-rest encryption workflow (8-step)
   - Security limitations (TPM protects keys, not RAM/VRAM)
   - AES-256-GCM, HMAC-SHA256, SHA-256 specifications
   - Secure cleanup step
3. **Architecture diagram** (lines 119–125): Replace with new architecture flow including encrypted databases and TPM+Vault
4. **Component 2 (Privacy Layer)** (lines 161–189): Replace `SimulatedConfidentialLayer` class with `VaultManager` + `TPMKeyStore` + `EncryptedDB` + `SecureCleanup` descriptions

---

### [MODIFY] [implementation_plan_s3_s4.md](file:///c:/Users/praka/Desktop/projekt/implementation_plan_s3_s4.md)

1. **Preamble** (lines 5–6): Replace with new TPM+Vault preamble
2. **Bank database description** (lines 17–23): Add encryption-at-rest note per bank DB
3. All existing KYC codebook, salary band, embedding, mismatch, and country risk content is **preserved unchanged**

---

### [MODIFY] [implementation_plan_s5_s6.md](file:///c:/Users/praka/Desktop/projekt/implementation_plan_s5_s6.md)

1. **Preamble** (lines 5–6): Replace with new TPM+Vault preamble
2. No other changes needed — memory system and model architecture are unrelated to the security layer

---

### [MODIFY] [implementation_plan_s7_s8.md](file:///c:/Users/praka/Desktop/projekt/implementation_plan_s7_s8.md)

1. **Preamble** (lines 5–6): Replace with new TPM+Vault preamble
2. No other changes needed — trust score, laundering probability, and pattern memory are unrelated to the security layer

---

### [MODIFY] [implementation_plan_s9_s11.md](file:///c:/Users/praka/Desktop/projekt/implementation_plan_s9_s11.md)

1. **Preamble** (lines 5–6): Replace with new TPM+Vault preamble
2. **Software stack** (line 152): Replace `cryptography` condition from "If AES-256-GCM encryption for Confidential Layer" to "Required for AES-256-GCM encryption via TPM+Vault security layer"
3. **File structure** (lines 207–211): Replace `privacy/` directory with new `security/` directory containing:
   - `vault_manager.py`
   - `tpm_key_store.py`
   - `encrypted_db.py`
   - `secure_cleanup.py`
4. **Section mapping** (lines 285): Update `privacy/` → `security/`

---

### [MODIFY] [implementation_plan_s12_s13.md](file:///c:/Users/praka/Desktop/projekt/implementation_plan_s12_s13.md)

1. **Preamble** (lines 5–6): Replace with new TPM+Vault preamble
2. **Phase 3** (lines 88–111): Replace `SimulatedConfidentialLayer` phase with TPM + HashiCorp Vault Security Layer phase, including:
   - `security/` package deliverables (vault_manager.py, tpm_key_store.py, encrypted_db.py, secure_cleanup.py)
   - Integration with bank node API for encrypted DB access
   - Key retrieval workflow testing
3. **Section 13 "Feasible Now"** (line 234): Replace privacy layer reference
4. **Final statement**: Add required conclusion about laptop feasibility
5. **"Only If Time Remains"** (line 29): Replace "Confidential Transport Layer Encryption" with "TPM + Vault integration"

---

## Security Architecture Details (to be written into s1_s2.md)

### Key Management Design

5 separate encryption keys stored in TPM 2.0 through HashiCorp Vault:

```
bank_a_key       — encrypts Bank A's SQLite database
bank_b_key       — encrypts Bank B's SQLite database
bank_c_key       — encrypts Bank C's SQLite database
aml_cache_key    — encrypts the central AML transaction cache
backup_key       — encrypts backups and key escrow
```

### Data Decryption Workflow

```
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

### Security Limitations

> The TPM protects the encryption keys. HashiCorp Vault manages the keys. RAM and VRAM are not hardware-protected.

- TPM + Vault protects data at rest
- TPM + Vault protects key storage
- TPM + Vault does NOT protect RAM or GPU VRAM
- Once the data is decrypted for GraphSAGE or the temporal GNN, it exists in normal memory
- Therefore this is not equivalent to SGX or TDX

> This prototype uses hardware-backed key management, not hardware-backed confidential computing.

### Encryption Details

- AES-256-GCM for encrypted bank databases
- HMAC-SHA256 for request signing
- SHA-256 hashed account IDs
- TLS or localhost HTTPS between mocked banks and the AML engine

### Secure Cleanup

```
After embeddings are generated:
- delete raw decrypted KYC structures from RAM
- clear temporary buffers
- retain only anonymized embeddings and salary bands
```

### New `security/` Directory

```
security/
    vault_manager.py       — HashiCorp Vault client, key CRUD, lease management
    tpm_key_store.py       — TPM 2.0 interface, seal/unseal, PCR binding
    encrypted_db.py        — AES-256-GCM SQLite/DuckDB encryption/decryption
    secure_cleanup.py      — RAM cleanup, buffer zeroing, embedding isolation
```

### Required Final Statement

> The prototype remains fully feasible on an i7-12650HX laptop with 16GB RAM and 6GB VRAM because it uses hardware-backed key storage through TPM + HashiCorp Vault, while keeping the AML model lightweight through GraphSAGE, sparse batching, 5,000 simulated accounts, and 30–90 day rolling memory.

---

## Verification Plan

### Automated Verification
- `grep -riE "(SimulatedConfidentialLayer|simulated_confidential|SGX|sev-snp|simulated enclave|confidential_layer\.py|simulated_confidential_layer\.py|attestation\.py)" *.md` → zero matches
- `grep -c "TPM" *.md` → matches in all 6 files
- `grep -c "HashiCorp Vault" *.md` → matches in at least s1_s2, s9_s11, s12_s13
- `grep -c "vault_manager.py" *.md` → matches in s9_s11 and s12_s13
- `grep -c "hardware-backed key management, not hardware-backed confidential computing" *.md` → matches in all 6 files

### Manual Verification
- Read each file's preamble to confirm consistent wording
- Verify the 8-step decryption workflow is present in s1_s2
- Verify the 5-key design is present in s1_s2
- Verify `security/` directory replaces `privacy/` in s9_s11 and s12_s13
- Verify all KYC/codebook/mismatch content is preserved unchanged
