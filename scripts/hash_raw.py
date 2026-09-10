from __future__ import annotations

import hashlib
from pathlib import Path

path = Path("data/raw/twcs.csv")
if not path.is_file():
    raise SystemExit(f"Missing {path}")
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(f"{digest.hexdigest()}  {path}")

