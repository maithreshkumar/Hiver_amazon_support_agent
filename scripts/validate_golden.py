from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from hiver_support.evaluation.golden_labels import initialize_golden_set, validate_golden_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", default="data/reports/golden_validation.json")
    args = parser.parse_args()
    path = initialize_golden_set(
        output_path=os.getenv("GOLDEN_SET_PATH", "data/golden/golden_set.csv")
    )
    result = validate_golden_set(path, require_complete=not args.allow_incomplete)
    artifact = {
        "validated_at": datetime.now(UTC).isoformat(),
        "golden_path": str(Path(path).resolve()),
        "require_complete": not args.allow_incomplete,
        **result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2))
