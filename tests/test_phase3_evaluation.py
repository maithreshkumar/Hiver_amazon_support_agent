import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from hiver_support.evaluation.judge import _candidate_mapping
from hiver_support.evaluation.phase3 import analyze_response_outputs, calculate_routing_metrics
from hiver_support.evaluation.reply_ratings import (
    DIMENSIONS,
    import_reviewed_reply_ratings,
    initialize_reply_ratings,
    load_reply_ratings,
    save_reply_rating,
    validate_reply_ratings,
)
from hiver_support.evaluation.reply_agreement import calculate_human_judge_agreement


def _response(example: int, human: str = "ESCALATE", predicted: str = "AUTO_HANDLE") -> dict[str, object]:
    return {
        "example_id": f"gold-{example:03d}",
        "message": f"Customer message {example}",
        "human_intent": "general_or_context_missing",
        "predicted_intent": "delivery_tracking_or_delay",
        "intent_confidence": 0.8,
        "intent_correct": False,
        "human_route": human,
        "predicted_route": predicted,
        "routing_reason": "This is a low-risk, high-confidence request.",
        "retrieved_evidence": [{
            "thread_id": str(example),
            "customer_text": "Historical question",
            "amazon_response": "Historical answer",
            "similarity": 0.6,
        }],
        "final_safe_response": f"Generated response {example}",
        "trivial_reply": "Generic reply",
        "tfidf_reply": "Retrieved reply",
        "provider_error": None,
        "unsupported_claim_detected": False,
        "safety_override": False,
        "provider": "ollama",
        "model": "qwen3:4b",
        "latency_seconds": {
            "intent": 0.1,
            "embedding_and_retrieval": 0.2,
            "generation": 1.0,
            "safety_and_routing": 0.01,
            "total_component_sum": 1.31,
        },
    }


def test_routing_metrics_use_actual_labels() -> None:
    result = calculate_routing_metrics(
        ["ESCALATE", "ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE"],
        ["AUTO_HANDLE", "ESCALATE", "ESCALATE", "AUTO_HANDLE"],
    )
    assert result["accuracy"] == 0.5
    assert result["false_auto_handle_count"] == 1
    assert result["false_auto_handle_rate_among_human_escalations"] == 0.5
    assert result["false_escalation_count"] == 1
    assert result["confusion_matrix"]["values"] == [[1, 1], [1, 1]]


def test_phase3_reports_are_derived_from_records(tmp_path: Path) -> None:
    golden = Path("data/golden/golden_set.csv")
    golden_before = hashlib.sha256(golden.read_bytes()).hexdigest()
    source = tmp_path / "responses.json"
    source.write_text(json.dumps({"scope": "test", "records": [_response(1), _response(2, "AUTO_HANDLE", "AUTO_HANDLE")]}))
    result = analyze_response_outputs(source, tmp_path / "reports")
    assert result["routing_metrics.json"]["rows"] == 2
    assert result["routing_metrics.json"]["false_auto_handle_count"] == 1
    assert result["retrieval_analysis.json"]["top1_similarity"]["mean"] == pytest.approx(0.6)
    written = json.loads((tmp_path / "reports" / "routing_metrics.json").read_text())
    assert written["correct"] == 1
    assert hashlib.sha256(golden.read_bytes()).hexdigest() == golden_before


def test_blinded_reply_rating_save_reload_and_validation(tmp_path: Path) -> None:
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps({"records": [_response(index) for index in range(1, 41)]}))
    output = initialize_reply_ratings(responses, tmp_path / "ratings.csv")
    frame = load_reply_ratings(output)
    assert len(frame) == 40
    assert not frame["human_rating_complete"].any()
    forbidden = {"human_intent", "predicted_intent", "intent_correct", "judge_score", "judge_rationale"}
    assert not forbidden & set(frame.columns)
    save_reply_rating(frame, 0, {name: 4 for name in DIMENSIONS}, "human-rater", output_path=output)
    reloaded = load_reply_ratings(output)
    assert bool(reloaded.at[0, "human_rating_complete"])
    assert all(int(reloaded.at[0, f"human_{name}"]) == 4 for name in DIMENSIONS)
    assert validate_reply_ratings(output) == {"rows": 40, "completed": 1, "remaining": 39}
    with pytest.raises(ValueError, match="incomplete"):
        validate_reply_ratings(output, require_complete=True)


def test_judge_blinding_is_deterministic() -> None:
    first = _candidate_mapping("gold-001")
    assert first == _candidate_mapping("gold-001")
    assert set(first) == {"A", "B", "C"}
    assert set(first.values()) == {"trivial", "tfidf_retrieval_only", "rag_with_safety"}


def test_reply_rating_cli_quits_without_partial_save(tmp_path: Path) -> None:
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps({"records": [_response(index) for index in range(1, 41)]}))
    output = tmp_path / "ratings.csv"
    result = subprocess.run(
        [sys.executable, "scripts/rate_replies.py", "--labeler", "human-rater"],
        input="q\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env={**os.environ, "REPLY_RESPONSE_PATH": str(responses), "REPLY_RATINGS_PATH": str(output)},
        check=False,
    )
    assert result.returncode == 0
    assert "Stopped safely. 0 / 40 rated." in result.stdout
    assert validate_reply_ratings(output) == {"rows": 40, "completed": 0, "remaining": 40}


def test_completed_reply_rating_import_is_exact_and_atomic(tmp_path: Path) -> None:
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps({"records": [_response(index) for index in range(1, 41)]}))
    canonical = initialize_reply_ratings(responses, tmp_path / "canonical.csv")
    blank_bytes = canonical.read_bytes()
    reviewed = load_reply_ratings(canonical)
    now = "2026-09-10T12:00:00+00:00"
    for dimension in DIMENSIONS:
        reviewed[f"human_{dimension}"] = 4
    reviewed["human_rating_complete"] = True
    reviewed["labeler_id"] = "human-rater"
    reviewed["human_rating_reason"] = "Human final decision."
    reviewed["rated_at"] = now
    reviewed["updated_at"] = now
    source = tmp_path / "reviewed.csv"
    reviewed.to_csv(source, index=False)
    result = import_reviewed_reply_ratings(
        source,
        canonical,
        tmp_path / "backup.csv",
        tmp_path / "provenance.json",
    )
    assert result["completed"] == 40
    assert canonical.read_bytes() == source.read_bytes()
    assert (tmp_path / "backup.csv").read_bytes() == blank_bytes
    assert validate_reply_ratings(canonical, require_complete=True)["remaining"] == 0


def test_human_judge_agreement_uses_frozen_scores(tmp_path: Path) -> None:
    responses = tmp_path / "responses.json"
    records = [_response(index) for index in range(1, 41)]
    responses.write_text(json.dumps({"records": records}))
    ratings_path = initialize_reply_ratings(responses, tmp_path / "ratings.csv")
    ratings = load_reply_ratings(ratings_path)
    now = "2026-09-10T12:00:00+00:00"
    for dimension in DIMENSIONS:
        ratings[f"human_{dimension}"] = 4
    ratings["human_rating_complete"] = True
    ratings["labeler_id"] = "human-rater"
    ratings["human_rating_reason"] = "Human final decision."
    ratings["rated_at"] = now
    ratings["updated_at"] = now
    ratings.to_csv(ratings_path, index=False)
    judge_records = []
    for index, record in enumerate(records):
        score = 4 if index < 20 else 3
        judge_records.append({
            "example_id": record["example_id"],
            "scores_by_system": {
                "rag_with_safety": {
                    "scores": {dimension: score for dimension in DIMENSIONS},
                    "aggregate": float(score),
                    "rationale": "Frozen test rationale.",
                }
            },
        })
    judge_path = tmp_path / "judge.json"
    judge_path.write_text(json.dumps({
        "judge_provider": "ollama",
        "judge_model": "separate-judge",
        "prompt_version": "judge-v1",
        "rubric_version": "rubric-v1",
        "records": judge_records,
    }))
    judge_before = judge_path.read_bytes()
    ratings_before = ratings_path.read_bytes()
    result = calculate_human_judge_agreement(
        ratings_path, judge_path, tmp_path / "reports" / "agreement.json"
    )
    assert result["overall_pooled_dimensions"]["exact_agreement"] == 0.5
    assert result["overall_pooled_dimensions"]["agreement_within_one"] == 1.0
    assert result["per_dimension"]["safety"]["judge_minus_human_mean_bias"] == -0.5
    assert result["judge_rerun_after_human_ratings"] is False
    assert judge_path.read_bytes() == judge_before
    assert ratings_path.read_bytes() == ratings_before
