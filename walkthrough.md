# Walkthrough: TPM + HashiCorp Vault Security Layer Revision

## What Changed

All 6 implementation plan files were updated to replace the `SimulatedConfidentialLayer` / SGX / TDX references with a realistic **TPM 2.0 + HashiCorp Vault** security architecture.

### Per-File Summary

| File | Changes |
|------|---------|
| `s1_s2.md` | Replaced "Prototype Privacy Layer" section with full TPM+Vault security layer (5 keys, 8-step workflow, limitations, encryption specs, secure cleanup). Replaced architecture diagram. Replaced Component 2 class interfaces (`VaultManager`, `EncryptedDB`, `SecureCleanup`). |
| `s3_s4.md` | Preamble updated. All KYC/codebook/mismatch content preserved. |
| `s5_s6.md` | Preamble updated. Memory system and model architecture unchanged. |
| `s7_s8.md` | Preamble updated. Trust score and pattern memory unchanged. |
| `s9_s11.md` | Preamble updated. `cryptography` library condition updated. `privacy/` directory → `security/` with 4 new files. Section mapping updated. |
| `s12_s13.md` | Preamble updated. Phase 3 rewritten (new deliverables, risks, definition of done). "Only If Time Remains" bullet updated. Section 13 feasibility reference updated. Final laptop feasibility statement added. |

### Key Additions

1. **5 encryption keys**: `bank_a_key`, `bank_b_key`, `bank_c_key`, `aml_cache_key`, `backup_key`
2. **8-step decryption workflow** from disk → Vault → TPM → RAM → embeddings → cleanup
3. **Security limitations** honestly stated (protects at-rest, not RAM/VRAM)
4. **Encryption boundary table** (at rest, in transit, in RAM, in VRAM)
5. **Secure cleanup** procedure after embedding generation
6. **New `security/` directory**: `vault_manager.py`, `tpm_key_store.py`, `encrypted_db.py`, `secure_cleanup.py`
7. **Required final statement** about laptop feasibility with TPM + Vault

## Verification

- **Zero** remaining `SimulatedConfidentialLayer` / `confidential_layer.py` / `attestation.py` / `simulated enclave` references in project plan files
- **All 6 files** contain `hardware-backed key management, not hardware-backed confidential computing`
- **`vault_manager.py`** referenced in `s1_s2`, `s9_s11`, and `s12_s13`
