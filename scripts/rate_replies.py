from __future__ import annotations

import argparse
import os

from hiver_support.evaluation.reply_ratings import (
    DIMENSIONS,
    initialize_reply_ratings,
    load_reply_ratings,
    save_reply_rating,
    validate_reply_ratings,
)


class RatingQuit(Exception):
    pass


def _score(name: str) -> int:
    while True:
        value = input(f"{name.replace('_', ' ').title()} (1-5, or q to stop): ").strip()
        if value.casefold() == "q":
            raise RatingQuit
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)
        print("Enter an integer from 1 to 5.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blinded human rating for generated support replies")
    parser.add_argument("--labeler", help="Human rater ID")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    path = initialize_reply_ratings(
        response_path=os.getenv("REPLY_RESPONSE_PATH", "data/reports/golden_response_outputs.json"),
        output_path=os.getenv("REPLY_RATINGS_PATH", "data/golden/reply_quality_human_ratings.csv"),
    )
    status = validate_reply_ratings(path)
    if args.status:
        print(f"{status['completed']} / {status['rows']} rated")
        return
    if not args.labeler:
        parser.error("--labeler is required unless --status is used")
    while True:
        frame = load_reply_ratings(path)
        remaining = frame.index[~frame["human_rating_complete"].fillna(False).astype(bool)].tolist()
        if not remaining:
            print(f"{len(frame)} / {len(frame)} rated")
            return
        index = int(remaining[0])
        row = frame.loc[index]
        print(f"\n[{len(frame) - len(remaining) + 1}/{len(frame)}] {row['example_id']}")
        print(f"\nCustomer message:\n{row['customer_message']}")
        print(f"\nHistorical evidence:\n{row['historical_evidence']}")
        print(f"\nGenerated response:\n{row['generated_response']}\n")
        try:
            scores = {dimension: _score(dimension) for dimension in DIMENSIONS}
        except RatingQuit:
            current = validate_reply_ratings(path)
            print(f"Stopped safely. {current['completed']} / {current['rows']} rated.")
            return
        reason = input("Optional rating note: ").strip()
        confirm = input("Save this rating? [y/N]: ").strip().casefold()
        if confirm != "y":
            print("Not saved.")
            continue
        save_reply_rating(frame, index, scores, args.labeler, reason, path)
        print(f"Saved {row['example_id']} atomically.")


if __name__ == "__main__":
    main()
