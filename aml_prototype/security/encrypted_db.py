"""
AES-256-GCM encrypted database access.

Encrypts and decrypts SQLite database files using per-bank keys
retrieved from HashiCorp Vault (backed by TPM 2.0).

Encryption boundaries:
    - Data at rest: AES-256-GCM encrypted on disk.
    - Data in RAM: Unprotected during embedding generation.
    - Data in VRAM: Unprotected during GNN inference.

The TPM protects the encryption keys. HashiCorp Vault manages
the keys. RAM and VRAM are not hardware-protected.
"""

import logging
import os
import sqlite3
import struct
from typing import Optional

logger = logging.getLogger(__name__)


class EncryptedDB:
    """
    Manages AES-256-GCM encrypted SQLite databases.

    Production workflow:
        1. Database file is stored encrypted on disk.
        2. On open: key retrieved from Vault → file decrypted to temp → connection returned.
        3. On close: temp file re-encrypted → written to disk → temp cleared.

    Phase 1 prototype workflow:
        Databases are stored as plaintext SQLite files. The encryption
        layer is structurally present but operates in passthrough mode.
        This allows all downstream code to call open_encrypted / close_and_reencrypt
        without modification when real encryption is enabled later.
    """

    def __init__(self, vault_manager=None):
        """
        Args:
            vault_manager: VaultManager instance for key retrieval.
                           If None, operates in passthrough mode.
        """
        from security.vault_manager import VaultManager
        self._vault = vault_manager or VaultManager()
        self._active_connections: dict[str, sqlite3.Connection] = {}

    def open_encrypted(self, db_path: str, bank_id: str) -> sqlite3.Connection:
        """
        Open an encrypted database.

        In production: decrypt file using bank key → open SQLite connection.
        In prototype: open SQLite connection directly (passthrough).

        Args:
            db_path: Path to the .db file.
            bank_id: Bank identifier for key retrieval.

        Returns:
            sqlite3.Connection with WAL mode enabled.
        """
        logger.info("Opening encrypted DB: %s (bank: %s)", db_path, bank_id)

        # Retrieve key (proves the key path works even in prototype)
        _key = self._vault.get_bank_key(bank_id)
        logger.debug("Retrieved %d-byte key for %s", len(_key), bank_id)

        # In prototype: direct SQLite open (no decryption needed)
        # In production: decrypt file first, then open
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        self._active_connections[db_path] = conn
        return conn

    def close_and_reencrypt(self, conn: sqlite3.Connection, db_path: Optional[str] = None) -> None:
        """
        Close connection and ensure data is re-encrypted on disk.

        In production: close connection → re-encrypt file → overwrite.
        In prototype: close connection (passthrough).
        """
        if db_path and db_path in self._active_connections:
            del self._active_connections[db_path]

        conn.close()
        logger.info("Closed and (would) re-encrypt: %s", db_path or "unknown")

    def close_all(self) -> None:
        """Close all active connections."""
        for path, conn in list(self._active_connections.items()):
            try:
                conn.close()
            except Exception:
                pass
        self._active_connections.clear()
