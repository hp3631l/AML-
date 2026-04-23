"""
Aggregator Scheduler.

Runs the pipeline periodically (simulating a cron job).
"""

import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aggregator.pipeline import run_pipeline

def start_scheduler(interval_seconds=600):
    print(f"Starting pipeline scheduler. Interval: {interval_seconds} seconds.")
    try:
        while True:
            run_pipeline()
            print(f"Sleeping for {interval_seconds} seconds...")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nScheduler stopped.")

if __name__ == "__main__":
    start_scheduler(interval_seconds=300) # Every 5 minutes for the prototype
