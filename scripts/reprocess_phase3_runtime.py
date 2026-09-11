from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from hiver_support.evaluation.phase3 import analyze_response_outputs
from hiver_support.generation.reply import DraftValidation, validate_draft
from hiver_support.routing.final_response import build_final_response
from hiver_support.routing.policy import RoutingPolicy


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reapply the final runtime validator/router to frozen Phase 3 raw drafts"
    )
    parser.add_argument("--source", default="data/reports/golden_response_outputs.json")
    parser.add_argument(
        "--output", default="data/reports/post_repair/golden_response_outputs_post_repair.json"
    )
    parser.add_argument("--metrics-dir", default="data/reports/post_repair")
    args = parser.parse_args()

    source_path = Path(args.source)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    policy = RoutingPolicy()
    repaired: list[dict[str, object]] = []
    changed_replies = 0
    changed_routes = 0

    for original in source["records"]:
        record = dict(original)
        message = str(record["message"])
        intent = str(record["predicted_intent"])
        raw_draft = str(record.get("raw_generated_draft", ""))
        similarities = [float(item["similarity"]) for item in record["retrieved_evidence"]]
        if record.get("provider_error"):
            validation = DraftValidation()
            routing = policy.provider_failure_decision(
                message=message,
                intent=intent,
                intent_confidence=float(record["intent_confidence"]),
                ambiguity_score=float(record.get("ambiguity_score", 0.0)),
            )
        else:
            validation = validate_draft(
                message,
                raw_draft,
                model_unsupported_claim=bool(record.get("unsupported_claim_detected", False)),
            )
            routing = policy.decide(
                message=message,
                intent=intent,
                intent_confidence=float(record["intent_confidence"]),
                best_similarity=max(similarities or [0.0]),
                grounding_confidence=float(record.get("grounding_confidence", 0.0)),
                unsupported_claim=validation.unsupported_claim_detected,
                ambiguity_score=float(record.get("ambiguity_score", 0.0)),
                safety_flags=validation.flags,
            )
        final = build_final_response(
            message=message,
            intent=intent,
            raw_draft=raw_draft,
            validation=validation,
            routing=routing,
        )
        previous_reply = str(record.get("final_safe_response", ""))
        previous_route = str(record.get("predicted_route", ""))
        changed_replies += int(previous_reply != final.reply)
        changed_routes += int(previous_route != final.decision)
        record.update(
            {
                "pre_repair_final_response": previous_reply,
                "pre_repair_predicted_route": previous_route,
                "final_safe_response": final.reply,
                "predicted_route": final.decision,
                "routing_reason": final.reason,
                "routing_reason_codes": final.reason_codes,
                "safety_validation_flags": list(validation.flags),
                "unsupported_claim_detected": validation.unsupported_claim_detected,
                "safety_override": final.safety_override_applied,
                "final_response_template": final.template,
                "runtime_reprocessed_at": datetime.now(UTC).isoformat(),
                "runtime_policy_version": policy.version,
            }
        )
        repaired.append(record)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "Post-repair routing/safety reprocessing of the frozen 48 Phase 3 raw drafts",
        "source": str(source_path),
        "method_note": (
            "No LLM was rerun. The saved raw drafts, classifier outputs, retrieval evidence, and "
            "grounding values were passed through the repaired deterministic validator, router, "
            "and canonical final-response builder. Human labels and ratings were not modified."
        ),
        "records": repaired,
    }
    output_path = Path(args.output)
    _atomic_json(output_path, payload)
    artifacts = analyze_response_outputs(output_path, args.metrics_dir)
    summary = {
        "records": len(repaired),
        "changed_routes": changed_routes,
        "changed_final_replies": changed_replies,
        "routing": artifacts["routing_metrics.json"],
        "response": artifacts["response_evaluation_summary.json"],
    }
    _atomic_json(Path(args.metrics_dir) / "runtime_repair_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
