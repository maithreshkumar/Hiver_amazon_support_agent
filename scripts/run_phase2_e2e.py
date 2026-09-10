from __future__ import annotations

import json
import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from hiver_support.agent import AmazonSupportAgent


MESSAGES = [
    "My parcel was promised by Tuesday, but tracking has not moved for three days. Where is it?",
    "The app marks my package delivered this morning, but nothing arrived at my home.",
    "I need to send these headphones back because they are not suitable for me.",
    "I do not recognize this charge and I am worried somebody accessed my Amazon account.",
    "Prime Video keeps showing an error whenever I try to play the next episode.",
    "The marketplace seller stopped replying after sending me the wrong color.",
    "Hi Amazon, can somebody help me please?",
]


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        help="Optional 1-based example indices to rerun and merge into the audit artifact.",
    )
    args = parser.parse_args()
    requested = set(args.indices or range(1, len(MESSAGES) + 1))
    if not requested <= set(range(1, len(MESSAGES) + 1)):
        raise SystemExit("Example indices must be between 1 and 7")
    training = pd.read_parquet(
        "data/processed/retrieval_corpus.parquet", columns=["customer_text"]
    )
    seen = {normalized(value) for value in training["customer_text"].astype(str)}
    agent = AmazonSupportAgent()
    output = Path("data/reports/phase2_e2e_outputs.json")
    prior: dict[str, dict[str, object]] = {}
    if args.indices and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        prior = {str(item["message"]): item for item in existing.get("examples", [])}
    outputs: list[dict[str, object]] = []
    for index, message in enumerate(MESSAGES, 1):
        if index not in requested and message in prior:
            outputs.append(prior[message])
            continue
        print(f"Running end-to-end example {index}/{len(MESSAGES)}", flush=True)
        started = time.perf_counter()
        result = agent.handle(message)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["exact_training_duplicate"] = normalized(message) in seen
        outputs.append(result)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": "Phase 2 full local classifier + semantic retrieval + qwen3:4b generation + routing smoke test",
        "examples": outputs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # ASCII escaping keeps Windows consoles with legacy code pages from failing on
    # emoji in historical evidence after the auditable UTF-8 artifact is written.
    print(json.dumps(report, indent=2, ensure_ascii=True))
