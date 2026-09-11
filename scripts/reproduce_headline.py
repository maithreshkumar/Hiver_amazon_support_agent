from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from hiver_support.evaluation.duplicates import run_near_duplicate_audit
from hiver_support.evaluation.final_intent import run_final_intent_evaluation
from hiver_support.evaluation.phase3 import analyze_response_outputs
from hiver_support.evaluation.reply_agreement import calculate_human_judge_agreement
from hiver_support.submission import load_artifact_manifest


GOLDEN = Path("data/golden/golden_set.csv")
FINAL_RATINGS = Path("data/golden/reply_quality_human_ratings_post_repair.csv")
FINAL_RESPONSES = Path("data/reports/post_repair/golden_response_outputs_post_repair.json")
FINAL_JUDGE = Path("data/reports/post_repair/reply_quality_judge_post_repair.json")
OUTPUT_DIR = Path("data/reproduction/latest")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_close(name: str, observed: float, expected: float) -> None:
    if abs(observed - expected) > 1e-12:
        raise RuntimeError(f"Recomputed {name}={observed} differs from frozen value {expected}")


def main() -> None:
    started = time.perf_counter()
    subprocess.run([sys.executable, "scripts/preflight_submission.py"], check=True)
    manifest = load_artifact_manifest()
    protected = {
        path: sha256(path) for path in (GOLDEN, FINAL_RATINGS, FINAL_RESPONSES, FINAL_JUDGE)
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    intent = run_final_intent_evaluation(OUTPUT_DIR, validate_inputs=False)
    duplicates = run_near_duplicate_audit(
        0.35,
        OUTPUT_DIR / "near_duplicate_audit.json",
        validate_inputs=False,
    )
    response = analyze_response_outputs(FINAL_RESPONSES, OUTPUT_DIR)
    agreement = calculate_human_judge_agreement(
        ratings_path=FINAL_RATINGS,
        judge_path=FINAL_JUDGE,
        output_path=OUTPUT_DIR / "human_judge_agreement_post_repair.json",
        scope=(
            "48 human-rated final post-repair RAG+safety responses matched to the frozen "
            "post-repair LLM judge artifact"
        ),
        judge_rerun_after_human_ratings=True,
        post_hoc_judge_experiment=True,
    )

    frozen_intent = json.loads(Path("data/reports/final_intent_metrics.json").read_text(encoding="utf-8"))
    for model, observed in intent["metrics"]["models"].items():
        expected = frozen_intent["models"][model]
        if int(observed["correct"]) != int(expected["correct"]):
            raise RuntimeError(f"Recomputed correct count changed for {model}")
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
            _assert_close(f"{model}.{metric}", float(observed[metric]), float(expected[metric]))

    frozen_routing = json.loads(Path("data/reports/post_repair/routing_metrics.json").read_text(encoding="utf-8"))
    observed_routing = response["routing_metrics.json"]
    for metric in (
        "accuracy", "escalation_precision", "escalation_recall", "escalation_f1",
        "false_auto_handle_rate_among_human_escalations",
    ):
        _assert_close(f"routing.{metric}", float(observed_routing[metric]), float(frozen_routing[metric]))

    frozen_agreement = json.loads(
        Path("data/reports/post_repair/human_judge_agreement_post_repair.json").read_text(encoding="utf-8")
    )
    for metric in (
        "exact_agreement", "agreement_within_one", "linear_weighted_cohen_kappa",
        "quadratic_weighted_cohen_kappa",
    ):
        _assert_close(
            f"agreement.{metric}",
            float(agreement["overall_pooled_dimensions"][metric]),
            float(frozen_agreement["overall_pooled_dimensions"][metric]),
        )

    judge = json.loads(FINAL_JUDGE.read_text(encoding="utf-8"))
    leakage = json.loads(Path("data/evaluation/leakage_validation.json").read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - started
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_set_version": manifest["artifact_set_version"],
        "runtime_seconds": elapsed,
        "runtime_target_seconds": 900,
        "runtime_target_met": elapsed <= 900,
        "intent": {
            name: {
                key: values[key]
                for key in ("rows", "correct", "accuracy", "macro_f1", "weighted_f1")
            }
            for name, values in intent["metrics"]["models"].items()
        },
        "routing": {
            key: observed_routing[key]
            for key in (
                "rows", "accuracy", "escalation_precision", "escalation_recall",
                "escalation_f1", "false_auto_handle_count",
                "false_auto_handle_rate_among_human_escalations",
            )
        },
        "human_judge_agreement": {
            key: agreement["overall_pooled_dimensions"][key]
            for key in (
                "exact_agreement", "agreement_within_one", "linear_weighted_cohen_kappa",
                "quadratic_weighted_cohen_kappa",
            )
        },
        "reply_quality_judge_means": judge["mean_scores"],
        "near_duplicate_pairs": duplicates["near_duplicate_pairs"],
        "leakage": leakage["golden_overlap_counts"],
        "expensive_llm_generation_or_judging_rerun": False,
        "protected_artifacts_unchanged": all(sha256(path) == value for path, value in protected.items()),
    }
    if not result["protected_artifacts_unchanged"]:
        raise RuntimeError("Headline reproduction modified a sealed human or frozen LLM artifact")
    output = OUTPUT_DIR / "headline_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
