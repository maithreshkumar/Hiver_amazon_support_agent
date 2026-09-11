from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from hiver_support.evaluation.judge import run_reply_judge


PRE_REPAIR_RESPONSES = Path("data/reports/golden_response_outputs.json")
POST_REPAIR_RESPONSES = Path(
    "data/reports/post_repair/golden_response_outputs_post_repair.json"
)
HISTORICAL_JUDGE = Path("data/reports/reply_quality_judge.json")
FINAL_JUDGE_JSONL = Path("data/reports/post_repair/reply_quality_judge_post_repair.jsonl")
FINAL_JUDGE_JSON = Path("data/reports/post_repair/reply_quality_judge_post_repair.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _judge_inputs(record: dict[str, object]) -> dict[str, object]:
    return {
        "message": record["message"],
        "retrieved_evidence": record["retrieved_evidence"][:3],
        "trivial_reply": record["trivial_reply"],
        "tfidf_reply": record["tfidf_reply"],
        "final_safe_response": record["final_safe_response"],
    }


def seed_byte_identical_judgments() -> int:
    """Reuse only judgments whose complete blinded evaluation input is unchanged."""
    if FINAL_JUDGE_JSONL.exists() and FINAL_JUDGE_JSONL.stat().st_size:
        return 0

    before = json.loads(PRE_REPAIR_RESPONSES.read_text(encoding="utf-8"))["records"]
    after = json.loads(POST_REPAIR_RESPONSES.read_text(encoding="utf-8"))["records"]
    historical = json.loads(HISTORICAL_JUDGE.read_text(encoding="utf-8"))["records"]
    before_by_id = {str(row["example_id"]): row for row in before}
    judge_by_id = {str(row["example_id"]): row for row in historical}
    if [str(row["example_id"]) for row in before] != [str(row["example_id"]) for row in after]:
        raise ValueError("Pre- and post-repair response samples differ in IDs or order")

    source_judge_sha256 = _sha256(HISTORICAL_JUDGE)
    seeded: list[dict[str, object]] = []
    for record in after:
        example_id = str(record["example_id"])
        if _judge_inputs(before_by_id[example_id]) != _judge_inputs(record):
            continue
        copied = dict(judge_by_id[example_id])
        copied["reused_byte_identical_pre_repair_judgment"] = True
        copied["source_pre_repair_judge_sha256"] = source_judge_sha256
        seeded.append(copied)

    FINAL_JUDGE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    temporary = FINAL_JUDGE_JSONL.with_suffix(".tmp.jsonl")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeded),
        encoding="utf-8",
    )
    os.replace(temporary, FINAL_JUDGE_JSONL)
    return len(seeded)


def main() -> None:
    seed_byte_identical_judgments()
    payload = run_reply_judge(
        response_path=POST_REPAIR_RESPONSES,
        output_path=FINAL_JUDGE_JSONL,
        summary_output_path=FINAL_JUDGE_JSON,
        allow_legacy_cache=True,
    )
    reused = sum(
        bool(row.get("reused_byte_identical_pre_repair_judgment"))
        for row in payload["records"]
    )
    payload["evaluation_stage"] = "final_post_repair"
    payload["historical_judge_sha256"] = _sha256(HISTORICAL_JUDGE)
    payload["reused_byte_identical_records"] = reused
    payload["newly_judged_post_repair_records"] = len(payload["records"]) - reused
    payload["judge_was_not_tuned_against_human_ratings"] = True
    FINAL_JUDGE_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(payload["records"]),
                "reused_byte_identical": reused,
                "newly_judged_post_repair": len(payload["records"]) - reused,
                "output": str(FINAL_JUDGE_JSON),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
