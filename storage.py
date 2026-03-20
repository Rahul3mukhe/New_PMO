import os
from datetime import datetime

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_run_dir(base: str = "output") -> str:
    ensure_dir(base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"run_{ts}")
    ensure_dir(run_dir)
    return run_dir