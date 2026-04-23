"""
Runner script to start all Bank Node APIs concurrently.

Usage:
    python scripts/run_banks.py
"""

import os
import sys
import time
import subprocess
import signal

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BANK_PORTS

processes = []

def signal_handler(sig, frame):
    print("\nShutting down bank nodes...")
    for p in processes:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def main():
    print("========================================")
    print("STARTING BANK NODE APIS")
    print("========================================")
    
    for bank_id, port in BANK_PORTS.items():
        env = os.environ.copy()
        env["BANK_ID"] = bank_id
        
        # Use python -m uvicorn to ensure it runs from current environment
        cmd = [sys.executable, "-m", "uvicorn", "bank_node.api:app", "--host", "127.0.0.1", "--port", str(port)]
        
        print(f"Starting {bank_id} on port {port}...")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        
    print("All banks started. Press Ctrl+C to terminate.")
    
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
