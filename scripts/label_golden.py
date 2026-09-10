from __future__ import annotations

import argparse
import os
import sys

from hiver_support.evaluation.golden_labels import (
    _complete_mask,
    initialize_golden_set,
    load_golden_set,
    next_unlabelled_index,
    save_human_label,
    validate_golden_set,
)
from hiver_support.intents.taxonomy import load_approved_taxonomy


def _select(prompt: str, allowed: dict[str, str]) -> str | None:
    while True:
        value = input(prompt).strip()
        if value.casefold() in {"q", "quit", "exit"}:
            return None
        selected = allowed.get(value.casefold())
        if selected:
            return selected
        print("Invalid choice. Enter one of the displayed options, or q to save and quit.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Unbiased human labeling for the sealed golden set")
    parser.add_argument("--labeler", default=os.getenv("HUMAN_LABELER_ID", ""))
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--example", help="Edit a specific example ID, such as gold-057")
    args = parser.parse_args()
    path = initialize_golden_set(
        output_path=os.getenv("GOLDEN_SET_PATH", "data/golden/golden_set.csv")
    )
    status = validate_golden_set(path, require_complete=False)
    if args.status:
        print(f"{status['completed']} / {status['rows']} labelled ({status['remaining']} remaining)")
        return
    labeler = args.labeler.strip() or input("Human labeler ID: ").strip()
    if not labeler:
        raise SystemExit("A non-empty human labeler ID is required")

    taxonomy = load_approved_taxonomy()
    print("\nApproved intents (no model predictions are shown):")
    for number, item in enumerate(taxonomy, 1):
        print(f"  {number:2d}. {item.name} — {item.description}")
    intent_choices = {str(number): item.name for number, item in enumerate(taxonomy, 1)}
    intent_choices.update({item.name.casefold(): item.name for item in taxonomy})
    route_choices = {"a": "AUTO_HANDLE", "auto_handle": "AUTO_HANDLE", "e": "ESCALATE", "escalate": "ESCALATE"}

    while True:
        frame = load_golden_set(path)
        complete = _complete_mask(frame)
        if args.example:
            matches = frame.index[frame["example_id"].eq(args.example)].tolist()
            if not matches:
                raise SystemExit(f"Unknown example ID: {args.example}")
            row_index = matches[0]
        else:
            row_index = next_unlabelled_index(frame)
            if row_index is None:
                print("All 200 examples are labelled. Run: python scripts/validate_golden.py")
                return
        row = frame.loc[row_index]
        done = int(complete.sum())
        print("\n" + "=" * 78)
        print(f"Progress: {done} / {len(frame)} labelled | Example: {row['example_id']}")
        print("\nCustomer message:\n")
        print(str(row["customer_message"]))
        if bool(row["human_label_complete"]):
            print(f"\nExisting human label: {row['human_intent']} / {row['human_escalation']}")
        intent = _select("\nIntent number/name (q to quit): ", intent_choices)
        if intent is None:
            return
        route = _select("Route [A]UTO_HANDLE or [E]SCALATE (q to quit): ", route_choices)
        if route is None:
            return
        reason = input("Short human routing reason: ").strip()
        if not reason:
            print("Reason cannot be blank; this example was not saved.")
            continue
        confirmation = input(f"Save {intent} / {route}? [y/N]: ").strip().casefold()
        if confirmation != "y":
            print("Not saved.")
            continue
        save_human_label(
            frame,
            row_index,
            intent=intent,
            escalation=route,
            reason=reason,
            labeler_id=labeler,
            output_path=path,
            audit_path=os.getenv("GOLDEN_AUDIT_PATH", "data/golden/human_label_audit.jsonl"),
        )
        print(f"Saved {row['example_id']} with label_source=human_gold.")
        if args.example:
            return


if __name__ == "__main__":
    main()
