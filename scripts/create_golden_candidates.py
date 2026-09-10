from __future__ import annotations

import argparse
from pathlib import Path

from hiver_support.evaluation.golden import create_candidates

parser = argparse.ArgumentParser(description="Create held-out human-label candidates.")
parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
parser.add_argument("--size", type=int, default=200)
args = parser.parse_args()
print(f"Wrote {create_candidates(args.config, args.size)}")

