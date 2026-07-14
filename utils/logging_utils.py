"""
Lightweight metrics logger. Swap `print` calls for LitLogger / W&B in production --
kept dependency-free here so the template runs without extra setup.
"""

import json
import os
import time
from typing import Dict, Any


class MetricsLogger:
    def __init__(self, log_dir: str, run_name: str = None):
        self.run_name = run_name or f"run_{int(time.time())}"
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{self.run_name}.jsonl")

    def log(self, step: int, metrics: Dict[str, Any]):
        record = {"step": step, "ts": time.time(), **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[step {step}] " + " ".join(f"{k}={v}" for k, v in metrics.items()))
