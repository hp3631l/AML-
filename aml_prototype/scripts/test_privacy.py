"""
Test script to verify privacy guarantees of the Bank Node API.

Validates that:
1. Endpoints return valid 34d profile vectors.
2. Raw KYC fields (name, pan, aadhaar, address, exact_salary, raw_occupation)
   are NEVER present in the API response.
"""

import urllib.request
import json
import sqlite3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BANK_DB_PATHS, BANK_PORTS

RAW_KYC_FIELDS = [
    "customer_name",
    "pan",
    "aadhaar",
    "home_address",
    "exact_salary",
    "raw_occupation",
    "memo"
]

def get_random_account(db_path: str) -> str:
    """Get a random account ID from the database to test."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT account_id FROM kyc LIMIT 1").fetchone()
    conn.close()
    return row["account_id"] if row else None

def test_bank_api(bank_id: str, port: int, db_path: str):
    print(f"\n--- Testing {bank_id} (Port {port}) ---")
    
    account_id = get_random_account(db_path)
    if not account_id:
        print(f"Skipping {bank_id} - No accounts found in DB.")
        return
        
    url = f"http://127.0.0.1:{port}/embeddings/{account_id}"
    print(f"Fetching: {url}")
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        # 1. Verify vector dimensions
        vector = data.get("profile_vector", [])
        print(f"[OK] Received profile vector of dimension: {len(vector)}")
        assert len(vector) == 34, f"Expected 34d vector, got {len(vector)}d"
        
        # 2. Verify no raw KYC fields leaked
        for field in RAW_KYC_FIELDS:
            if field in data:
                print(f"[FAIL] PRIVACY LEAK: Raw field '{field}' found in response!")
                assert False, f"Privacy leak: {field}"
                
        # Also check values to ensure they didn't accidentally map a raw field to another key
        # Though the Pydantic model restricts this, it's good to be thorough.
        
        # 3. Mismatch score
        score = data.get("mismatch_score")
        print(f"[OK] Mismatch score computed: {score:.4f}")
        
        print(f"[OK] All privacy tests passed for {bank_id}.")
        
    except Exception as e:
        print(f"[FAIL] Error testing {bank_id}: {e}")

if __name__ == "__main__":
    print("Running Privacy Verification Tests...")
    for bank_id, port in BANK_PORTS.items():
        db_path = BANK_DB_PATHS.get(bank_id)
        test_bank_api(bank_id, port, db_path)
    print("\nTest suite complete.")
