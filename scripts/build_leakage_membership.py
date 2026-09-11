from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


SOURCES = {
    "train": Path("data/processed/train.parquet"),
    "test": Path("data/processed/test.parquet"),
    "weak_labels": Path("data/processed/weak_labels.parquet"),
    "retrieval_index": Path("indexes/amazon_support/metadata.parquet"),
    "golden": Path("data/golden/golden_set.csv"),
}
OUTPUT = Path("data/evaluation/leakage_membership.parquet")
SUMMARY = Path("data/evaluation/leakage_validation.json")
NAMESPACE = "hiver-amazon-support-thread-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_thread_id(value: object) -> str:
    normalized = str(value).strip()
    return hashlib.sha256(f"{NAMESPACE}:{normalized}".encode("utf-8")).hexdigest()


def read_thread_ids(path: Path) -> set[str]:
    frame = pd.read_parquet(path, columns=["thread_id"]) if path.suffix == ".parquet" else pd.read_csv(path, usecols=["thread_id"], dtype={"thread_id": "string"})
    return {str(value).strip() for value in frame["thread_id"].dropna()}


def main() -> None:
    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot rebuild compact leakage evidence; missing local source artifacts: "
            + ", ".join(missing)
        )
    memberships = {name: read_thread_ids(path) for name, path in SOURCES.items()}
    all_ids = sorted(set().union(*memberships.values()), key=lambda value: hash_thread_id(value))
    rows = []
    for thread_id in all_ids:
        rows.append(
            {
                "thread_id_sha256": hash_thread_id(thread_id),
                **{f"in_{name}": thread_id in values for name, values in memberships.items()},
            }
        )
    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False, compression="zstd")

    golden_hashes = set(frame.loc[frame["in_golden"], "thread_id_sha256"])
    overlaps = {
        name: sorted(golden_hashes & set(frame.loc[frame[f"in_{name}"], "thread_id_sha256"]))
        for name in ("train", "weak_labels", "retrieval_index")
    }
    if any(overlaps.values()):
        raise RuntimeError("Golden thread leakage detected while building compact evidence")
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "algorithm": f"sha256({NAMESPACE} + ':' + normalized_thread_id)",
        "rows": len(frame),
        "membership_counts": {
            name: int(frame[f"in_{name}"].sum()) for name in memberships
        },
        "golden_overlap_counts": {name: len(values) for name, values in overlaps.items()},
        "source_sha256": {name: sha256_file(path) for name, path in SOURCES.items()},
        "membership_sha256": sha256_file(OUTPUT),
        "raw_thread_ids_included": False,
        "validation": "PASS",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
