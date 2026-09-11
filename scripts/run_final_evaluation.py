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
    ratings_file = Path("data/golden/reply_quality_human_ratings_post_repair.csv")
    judge_file = Path("data/reports/post_repair/reply_quality_judge_post_repair.json")
    response_path = Path("data/reports/post_repair/golden_response_outputs_post_repair.json")
    historical_ratings = Path("data/golden/reply_quality_human_ratings.csv")
    historical_judge = Path("data/reports/reply_quality_judge.json")
    for required in (ratings_file, judge_file, response_path, historical_ratings, historical_judge):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required final evaluation artifact is missing: {required}. "
                "Run the documented artifact fetch/verification workflow; pre-repair fallback is forbidden."
            )
    ratings_before = hashlib.sha256(ratings_file.read_bytes()).hexdigest()
    judge_before = hashlib.sha256(judge_file.read_bytes()).hexdigest()
    response_before = hashlib.sha256(response_path.read_bytes()).hexdigest()
    historical_ratings_before = hashlib.sha256(historical_ratings.read_bytes()).hexdigest()
    historical_judge_before = hashlib.sha256(historical_judge.read_bytes()).hexdigest()
    validation = validate_golden_set(require_complete=True)
    intent = run_final_intent_evaluation()
    duplicates = run_near_duplicate_audit(0.35)
    response = analyze_response_outputs(response_path, "data/reports/post_repair")
    agreement = calculate_human_judge_agreement(
        ratings_path=ratings_file,
        judge_path=judge_file,
        output_path="data/reports/post_repair/human_judge_agreement_post_repair.json",
        scope=(
            "48 human-rated final post-repair RAG+safety responses matched to the frozen "
            "post-repair LLM judge artifact"
        ),
        judge_rerun_after_human_ratings=True,
        post_hoc_judge_experiment=True,
    )
    runpy.run_path("scripts/build_phase3_report.py", run_name="__main__")
    after = hashlib.sha256(golden.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Evaluation modified the human golden labels")
    if ratings_before != hashlib.sha256(ratings_file.read_bytes()).hexdigest():
        raise RuntimeError("Evaluation modified the human reply ratings")
    if judge_before != hashlib.sha256(judge_file.read_bytes()).hexdigest():
        raise RuntimeError("Evaluation modified or reran the frozen judge output")
    if response_before != hashlib.sha256(response_path.read_bytes()).hexdigest():
        raise RuntimeError("Evaluation modified the frozen post-repair response output")
    if historical_ratings_before != hashlib.sha256(historical_ratings.read_bytes()).hexdigest():
        raise RuntimeError("Evaluation modified historical pre-repair human ratings")
    if historical_judge_before != hashlib.sha256(historical_judge.read_bytes()).hexdigest():
        raise RuntimeError("Evaluation modified historical pre-repair judge output")
    print(json.dumps({
        "golden_validation": validation,
        "distilroberta_accuracy": intent["metrics"]["models"]["distilroberta"]["accuracy"],
        "near_duplicate_pairs": duplicates["near_duplicate_pairs"],
        "routing_rows": response["routing_metrics.json"]["rows"],
        "human_judge_exact_agreement": agreement["overall_pooled_dimensions"]["exact_agreement"],
        "human_judge_within_one": agreement["overall_pooled_dimensions"]["agreement_within_one"],
        "artifact_set_version": "hiver-submission-v1",
        "final_response_artifact": str(response_path),
        "final_rating_artifact": str(ratings_file),
        "final_judge_artifact": str(judge_file),
        "human_labels_unchanged": True,
        "human_ratings_unchanged": True,
        "frozen_judge_unchanged": True,
        "historical_pre_repair_artifacts_unchanged": True,
        "report": "PHASE_3_EVALUATION_REPORT.md",
    }, indent=2))


if __name__ == "__main__":
    main()
