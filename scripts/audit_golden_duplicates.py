from __future__ import annotations

import argparse
import json

from hiver_support.evaluation.duplicates import run_near_duplicate_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit the sealed golden set for near duplicates")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    result = run_near_duplicate_audit(args.threshold)
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))
    print(json.dumps(result["pairs"][:10], indent=2))
