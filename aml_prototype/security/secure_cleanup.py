"""
Secure cleanup utilities.

After embeddings are generated:
    - delete raw decrypted KYC structures from RAM
    - clear temporary buffers
    - retain only anonymized embeddings and salary bands

Decrypted data remains in memory only for the duration of
embedding generation. After embedding extraction, raw KYC
structures are explicitly deleted and temporary buffers are zeroed.
"""

import ctypes
import gc
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def clear_kyc_from_ram(raw_data: Any) -> None:
    """
    Delete raw KYC structures and zero their memory.

    This provides best-effort RAM cleanup. Python's garbage collector
    does not guarantee immediate deallocation, but we:
        1. Overwrite mutable containers with zeros.
        2. Delete all references.
        3. Force garbage collection.

    Note: This does NOT protect against memory forensics or swap file
    analysis. RAM and VRAM are not hardware-protected (no SGX/TDX).
    """
    if isinstance(raw_data, dict):
        for key in list(raw_data.keys()):
            _zero_value(raw_data[key])
            raw_data[key] = None
        raw_data.clear()
    elif isinstance(raw_data, list):
        for i in range(len(raw_data)):
            _zero_value(raw_data[i])
            raw_data[i] = None
        raw_data.clear()
    elif isinstance(raw_data, bytearray):
        for i in range(len(raw_data)):
            raw_data[i] = 0

    del raw_data
    gc.collect()
    logger.debug("Secure cleanup: raw data cleared from RAM.")


def clear_buffer(buf: bytearray) -> None:
    """Zero a bytearray buffer."""
    if isinstance(buf, bytearray):
        for i in range(len(buf)):
            buf[i] = 0
        logger.debug("Cleared %d-byte buffer.", len(buf))


def secure_wipe_inplace(*refs: Any) -> None:
    """
    WARNING: Mutates argument in place. Call only after the 
    caller has finished using the object.
    
    Delete multiple references and force garbage collection.

    Usage:
        secure_wipe_inplace(kyc_row, session_data, ledger_data)
    """
    for ref in refs:
        try:
            if isinstance(ref, dict):
                ref.clear()
            elif isinstance(ref, list):
                ref.clear()
        except Exception:
            pass
    del refs
    gc.collect()


def _zero_value(value: Any) -> None:
    """Attempt to zero out a value in-place."""
    if isinstance(value, bytearray):
        for i in range(len(value)):
            value[i] = 0
    elif isinstance(value, list):
        for i in range(len(value)):
            value[i] = 0
    # str, int, float are immutable — cannot be zeroed, only deleted
