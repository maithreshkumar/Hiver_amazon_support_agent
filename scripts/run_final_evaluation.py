from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

from hiver_support.evaluation.duplicates import run_near_duplicate_audit
from hiver_support.evaluation.final_intent import run_final_intent_evaluation
from hiver_support.evaluation.golden_labels import validate_golden_set
from hiver_support.evaluation.phase3 import analyze_response_outputs
from hiver_support.evaluation.reply_agreement import calculate_human_judge_agreement


def main() -> None:
    golden = Path("data/golden/golden_set.csv")
    before = hashlib.sha256(golden.read_bytes()).hexdigest()
    ratings_file = Path("data/golden/reply_quality_human_ratings.csv")
    judge_file = Path("data/reports/reply_quality_judge.json")
    ratings_before = hashlib.sha256(ratings_file.read_bytes()).hexdigest()
    judge_before = hashlib.sha256(judge_file.read_bytes()).hexdigest()
    validation = validate_golden_set(require_complete=True)
    intent = run_final_intent_evaluation()
    duplicates = run_near_duplicate_audit(0.35)
    response_path = Path("data/reports/golden_response_outputs.json")
    judge_path = Path("data/reports/reply_quality_judge.json")
    if not response_path.exists() or not judge_path.exists():
        raise FileNotFoundError(
            "Persisted response/judge artifacts are missing. Run run_golden_responses.py and judge_replies.py first."
        )
    response = analyze_response_outputs(response_path)
    agreement = calculate_human_judge_agreement()
    runpy.run_path("scripts/build_phase3_report.py", run_name="__main__")
    after = hashlib.sha256(golden.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Evaluation modified the human golden labels")
    if ratings_before != hashlib.sha256(ratings_file.read_bytes()).hexdigest():
        raise RuntimeError("Evaluation modified the human reply ratings")
    if judge_before != hashlib.sha256(judge_file.read_bytes()).hexdigest():
        raise RuntimeError("Evaluation modified or reran the frozen judge output")
    print(json.dumps({
        "golden_validation": validation,
        "distilroberta_accuracy": intent["metrics"]["models"]["distilroberta"]["accuracy"],
        "near_duplicate_pairs": duplicates["near_duplicate_pairs"],
        "routing_rows": response["routing_metrics.json"]["rows"],
        "human_judge_exact_agreement": agreement["overall_pooled_dimensions"]["exact_agreement"],
        "human_judge_within_one": agreement["overall_pooled_dimensions"]["agreement_within_one"],
        "human_labels_unchanged": True,
        "human_ratings_unchanged": True,
        "frozen_judge_unchanged": True,
        "report": "PHASE_3_EVALUATION_REPORT.md",
    }, indent=2))


if __name__ == "__main__":
    main()
