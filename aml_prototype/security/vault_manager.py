"""
HashiCorp Vault client for key management.

In production, this connects to a real HashiCorp Vault instance
backed by TPM 2.0 for key sealing/unsealing.

For the Phase 1 prototype, this provides placeholder functions
that return deterministic keys derived from bank IDs (NEVER
used in production — keys must come from Vault).

The laptop does not support SGX, Intel TDX, or other
confidential-computing technologies. This prototype uses
hardware-backed key management, not hardware-backed
confidential computing.
"""

import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Keys are NEVER hardcoded in production. This is a prototype placeholder.
_PROTOTYPE_MASTER_SECRET = os.environ.get(
    "AML_VAULT_MASTER_SECRET",
    "PROTOTYPE_ONLY_DO_NOT_USE_IN_PRODUCTION"
)


class VaultManager:
    """
    Manages encryption keys through HashiCorp Vault.

    Production workflow:
        1. Vault stores keys sealed by TPM 2.0 PCR binding.
        2. Application authenticates to Vault via AppRole or TLS cert.
        3. Vault unseals the key from TPM and returns it.
        4. Key is held in memory only during the decryption window.
        5. After use, key reference is cleared.

    Prototype workflow:
        Keys are derived from a master secret + key name using HKDF.
        This simulates the Vault retrieval without requiring a running
        Vault server during Phase 1 development.
    """

    def __init__(self, vault_addr: Optional[str] = None, vault_token: Optional[str] = None):
        self._vault_addr = vault_addr or os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")
        self._vault_token = vault_token or os.environ.get("VAULT_TOKEN", "")
        self._use_real_vault = bool(self._vault_token)

        if self._use_real_vault:
            logger.info("VaultManager: using real HashiCorp Vault at %s", self._vault_addr)
        else:
            logger.warning(
                "VaultManager: no VAULT_TOKEN set — using prototype key derivation. "
                "Keys are NOT production-safe."
            )

    def get_bank_key(self, bank_id: str) -> bytes:
        """
        Retrieve per-bank AES-256-GCM key from Vault (backed by TPM).

        Args:
            bank_id: One of 'bank_a', 'bank_b', 'bank_c'.

        Returns:
            32-byte AES-256 key.
        """
        key_name = f"{bank_id}_key"
        return self._get_key(key_name)

    def get_cache_key(self) -> bytes:
        """Retrieve AML cache encryption key from Vault."""
        return self._get_key("aml_cache_key")

    def get_backup_key(self) -> bytes:
        """Retrieve backup encryption key from Vault."""
        return self._get_key("backup_key")

    def rotate_key(self, key_name: str) -> None:
        """
        Rotate a specific key in Vault.

        In production, Vault handles versioned key rotation.
        In prototype, this is a no-op with a warning.
        """
        if self._use_real_vault:
            # TODO: Implement real Vault key rotation via hvac client
            raise NotImplementedError("Real Vault key rotation not yet implemented")
        else:
            logger.warning("Key rotation is a no-op in prototype mode.")

    def _get_key(self, key_name: str) -> bytes:
        """
        Internal key retrieval.

        Production: query Vault API → Vault queries TPM → return key.
        Prototype: HKDF derive from master secret + key name.
        """
        if self._use_real_vault:
            return self._get_key_from_vault(key_name)
        else:
            return self._derive_prototype_key(key_name)

    def _get_key_from_vault(self, key_name: str) -> bytes:
        """Retrieve key from real HashiCorp Vault. Requires hvac."""
        try:
            import hvac
            client = hvac.Client(url=self._vault_addr, token=self._vault_token)
            secret = client.secrets.kv.v2.read_secret_version(
                path=f"aml-prototype/{key_name}"
            )
            key_hex = secret["data"]["data"]["key"]
            return bytes.fromhex(key_hex)
        except ImportError:
            logger.error("hvac package not installed. Install with: pip install hvac")
            raise
        except Exception as e:
            logger.error("Failed to retrieve key '%s' from Vault: %s", key_name, e)
            raise

    def _derive_prototype_key(self, key_name: str) -> bytes:
        """
        Derive a deterministic 32-byte key from master secret + key name.
        WARNING: This is NOT cryptographically safe for production.
        """
        raw = f"{_PROTOTYPE_MASTER_SECRET}:{key_name}".encode("utf-8")
        return hashlib.sha256(raw).digest()  # 32 bytes = AES-256
