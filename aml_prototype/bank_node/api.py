"""
Bank Node API (Phase 2).

Provides FastAPI endpoints to serve privacy-preserving embeddings.
Raw PII data is never exposed.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path if running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BANK_DB_PATHS
from security.encrypted_db import EncryptedDB
from security.secure_cleanup import secure_wipe_inplace
from bank_node.embedding_generator import generate_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: _BANK_ID and _DB_PATH are resolved at startup inside lifespan()
# so that env vars set by the shell are available (Windows timing issue).
_BANK_ID = None
_DB_PATH = None
_ENC_DB = EncryptedDB()
_DB_CONN = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _DB_CONN, _BANK_ID, _DB_PATH
    # Resolve at runtime — env var must be available when lifespan starts
    _BANK_ID = os.environ.get("BANK_ID", "bank_a")
    _DB_PATH = BANK_DB_PATHS.get(_BANK_ID)

    if not _DB_PATH:
        logger.error(f"Unknown BANK_ID: {_BANK_ID!r}. Valid: {list(BANK_DB_PATHS.keys())}")
        sys.exit(1)

    logger.info(f"Starting API for {_BANK_ID}...")
    try:
        _DB_CONN = _ENC_DB.open_encrypted(_DB_PATH, _BANK_ID)
        logger.info(f"Successfully opened encrypted DB for {_BANK_ID}")
    except Exception as e:
        logger.error(f"Failed to open DB: {e}")
        sys.exit(1)

    yield

    if _DB_CONN:
        _ENC_DB.close_and_reencrypt(_DB_CONN, _DB_PATH)
        logger.info(f"Closed and secured DB for {_BANK_ID}")


app = FastAPI(title="Bank Node API", lifespan=lifespan)


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class EmbeddingResponse(BaseModel):
    account_id: str
    bank_id: str
    profile_vector: List[float]
    occupation_code: int
    salary_band: int
    country_code: str
    mismatch_score: float

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[EmbeddingResponse]
    missing_accounts: List[str]

class HealthResponse(BaseModel):
    status: str
    bank_id: str


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "bank_id": _BANK_ID}


@app.get("/embeddings/{account_id}", response_model=EmbeddingResponse)
def get_embedding(account_id: str):
    """
    Generate and return the 34d privacy-preserving embedding for a given account.
    Raw KYC fields are never returned.
    """
    if not _DB_CONN:
        raise HTTPException(status_code=500, detail="Database connection not ready")

    logger.info(f"Looking for account {repr(account_id)} in {_BANK_ID}")
    emb_data = generate_embedding(account_id, _DB_CONN, _BANK_ID)

    if not emb_data:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found in {_BANK_ID}")

    assert isinstance(emb_data, dict) and len(emb_data) > 0, \
        "API response was wiped before return"

    return emb_data


class BatchEmbeddingRequest(BaseModel):
    account_ids: List[str]

@app.post("/embeddings/batch/", response_model=BatchEmbeddingResponse)
def get_embeddings_batch(req: BatchEmbeddingRequest):
    """
    Generate and return embeddings for a batch of accounts.
    """
    if not _DB_CONN:
        raise HTTPException(status_code=500, detail="Database connection not ready")

    embeddings = []
    missing = []

    for acc in req.account_ids:
        emb_data = generate_embedding(acc, _DB_CONN, _BANK_ID)
        if emb_data:
            assert isinstance(emb_data, dict) and len(emb_data) > 0, \
                "API response was wiped before return"
            embeddings.append(emb_data)
        else:
            missing.append(acc)

    return {"embeddings": embeddings, "missing_accounts": missing}


if __name__ == "__main__":
    import uvicorn
    from config import BANK_PORTS
    _runtime_bank_id = os.environ.get("BANK_ID", "bank_a")
    port = BANK_PORTS.get(_runtime_bank_id, 8001)
    uvicorn.run(app, host="127.0.0.1", port=port)
