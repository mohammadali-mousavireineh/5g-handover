#!/usr/bin/env python
"""Run the default reproducible experiment from the repository root."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
cmd = [sys.executable, str(root / "anatel_grid_compare.py"), "--data-dir", str(root / "data/processed"), "--output-dir", str(root / "results")]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd, cwd=root)
