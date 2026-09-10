from __future__ import annotations

import argparse
import json

from hiver_support.evaluation.reply_ratings import import_reviewed_reply_ratings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atomically import validated human reply-quality ratings")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/golden/reply_quality_human_ratings_import.csv",
    )
    args = parser.parse_args()
    print(json.dumps(import_reviewed_reply_ratings(args.path), indent=2))
