from __future__ import annotations

import argparse
import json

from hiver_support.evaluation.golden_labels import import_reviewed_golden_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="data/golden/human_labels_import.csv",
        help="Reviewed 200-row human-label CSV",
    )
    args = parser.parse_args()
    print(json.dumps(import_reviewed_golden_set(args.path), indent=2))
