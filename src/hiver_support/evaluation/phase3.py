from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def calculate_routing_metrics(expected: list[str], predicted: list[str]) -> dict[str, object]:
    """Calculate binary routing metrics with ESCALATE as the safety-positive class."""
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Routing labels must be non-empty and have equal lengths")
    allowed = {"AUTO_HANDLE", "ESCALATE"}
    if set(expected) - allowed or set(predicted) - allowed:
        raise ValueError("Routing labels must be AUTO_HANDLE or ESCALATE")
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=["AUTO_HANDLE", "ESCALATE"], zero_division=0
    )
    matrix = confusion_matrix(expected, predicted, labels=["AUTO_HANDLE", "ESCALATE"])
    false_escalation = int(matrix[0, 1])
    false_auto_handle = int(matrix[1, 0])
    human_escalations = int(matrix[1].sum())
    return {
        "rows": len(expected),
        "correct": int(sum(a == b for a, b in zip(expected, predicted, strict=True))),
        "incorrect": int(sum(a != b for a, b in zip(expected, predicted, strict=True))),
        "accuracy": float(np.mean(np.asarray(expected) == np.asarray(predicted))),
        "escalation_precision": float(precision[1]),
        "escalation_recall": float(recall[1]),
        "escalation_f1": float(f1[1]),
        "auto_handle_precision": float(precision[0]),
        "auto_handle_recall": float(recall[0]),
        "auto_handle_f1": float(f1[0]),
        "support": {"AUTO_HANDLE": int(support[0]), "ESCALATE": int(support[1])},
        "false_auto_handle_count": false_auto_handle,
        "false_auto_handle_rate_among_human_escalations": (
            false_auto_handle / human_escalations if human_escalations else 0.0
        ),
        "false_escalation_count": false_escalation,
        "confusion_matrix": {
            "labels": ["AUTO_HANDLE", "ESCALATE"],
            "values": matrix.tolist(),
            "rows_are_human_columns_are_system": True,
        },
    }


def _routing_disagreement_category(record: dict[str, object]) -> str:
    reason = str(record.get("routing_reason", "")).casefold()
    if not bool(record.get("intent_correct", False)):
        return "classifier_error"
    if "provider" in reason:
        return "provider_failure"
    if "security" in reason or "fraud" in reason or "payment" in reason or "legal" in reason:
        return "security_or_risk_signal"
    if "not confident" in reason:
        return "low_confidence"
    if "similar historical" in reason:
        return "weak_retrieval"
    if "not supported" in reason:
        return "unsupported_claim"
    if "ambiguous" in reason:
        return "ambiguous_or_context_missing"
    if "ground" in reason:
        return "low_grounding"
    if bool(record.get("safety_override", False)):
        return "safety_override"
    return "overly_conservative_or_policy_mismatch"


def _failure_examples(records: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked_groups = [
        ("unsafe_false_auto_handle", lambda r: r["human_route"] == "ESCALATE" and r["predicted_route"] == "AUTO_HANDLE"),
        ("high_confidence_classifier_error", lambda r: not r["intent_correct"] and float(r["intent_confidence"]) >= 0.72),
        ("weak_retrieval", lambda r: max([float(e["similarity"]) for e in r["retrieved_evidence"]] or [0.0]) < 0.55),
        ("context_missing_or_taxonomy_gap", lambda r: r["human_intent"] == "general_or_context_missing" and not r["intent_correct"]),
        ("false_escalation", lambda r: r["human_route"] == "AUTO_HANDLE" and r["predicted_route"] == "ESCALATE"),
    ]
    selected: list[tuple[str, dict[str, object]]] = []
    used: set[str] = set()
    for category, predicate in ranked_groups:
        candidates = [r for r in records if predicate(r) and str(r["example_id"]) not in used]
        if not candidates:
            continue
        if category == "unsafe_false_auto_handle":
            candidates.sort(key=lambda r: float(r["intent_confidence"]), reverse=True)
        elif category == "weak_retrieval":
            candidates.sort(key=lambda r: max([float(e["similarity"]) for e in r["retrieved_evidence"]] or [0.0]))
        else:
            candidates.sort(key=lambda r: float(r["intent_confidence"]), reverse=True)
        chosen = candidates[0]
        used.add(str(chosen["example_id"]))
        selected.append((category, chosen))
    if len(selected) < 5:
        remaining = [r for r in records if str(r["example_id"]) not in used and (not r["intent_correct"] or r["human_route"] != r["predicted_route"])]
        remaining.sort(key=lambda r: float(r["intent_confidence"]), reverse=True)
        for chosen in remaining[: 5 - len(selected)]:
            selected.append(("additional_material_disagreement", chosen))
    output = []
    for category, record in selected[:5]:
        similarities = [float(e["similarity"]) for e in record["retrieved_evidence"]]
        output.append({
            "category": category,
            "example_id": record["example_id"],
            "customer_message": record["message"],
            "expected_intent": record["human_intent"],
            "predicted_intent": record["predicted_intent"],
            "expected_route": record["human_route"],
            "predicted_route": record["predicted_route"],
            "intent_confidence": record["intent_confidence"],
            "top1_similarity": max(similarities or [0.0]),
            "generated_response": record["final_safe_response"],
            "why": _routing_disagreement_category(record),
            "likely_root_cause": (
                "The frozen classifier or policy signals did not align with the human-reviewed decision; "
                "the retrieved historical evidence may amplify lexical rather than task-level similarity."
            ),
            "improvement": (
                "Add independently adjudicated examples for this failure pattern, calibrate on development data, "
                "and evaluate intent-aware reranking without inspecting this sealed test label."
            ),
        })
    return output


def analyze_response_outputs(
    input_path: str | Path = "data/reports/golden_response_outputs.json",
    output_dir: str | Path = "data/reports",
) -> dict[str, object]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records = list(payload["records"])
    if not records:
        raise ValueError("No response records were found")
    generated_at = datetime.now(UTC).isoformat()
    scope = payload.get("scope", "response-evaluation sample")

    routing = calculate_routing_metrics(
        [str(r["human_route"]) for r in records], [str(r["predicted_route"]) for r in records]
    )
    disagreements = [r for r in records if r["human_route"] != r["predicted_route"]]
    breakdown: dict[str, int] = {}
    for record in disagreements:
        category = _routing_disagreement_category(record)
        breakdown[category] = breakdown.get(category, 0) + 1
    routing.update({
        "generated_at": generated_at,
        "scope": scope,
        "scope_warning": "These routing metrics cover the real generated-response sample, not all 200 intent-evaluation rows.",
        "disagreement_breakdown_primary_category": breakdown,
        "disagreement_example_ids": [r["example_id"] for r in disagreements],
    })

    top1, top3_means, all_scores = [], [], []
    weak_ids, context_weak_ids, high_similarity_risk_ids = [], [], []
    evidence_rows, legacy_style_rows = 0, 0
    for record in records:
        scores = [float(item["similarity"]) for item in record["retrieved_evidence"]]
        top = max(scores or [0.0])
        top1.append(top)
        top3_means.append(float(np.mean(scores[:3])) if scores else 0.0)
        all_scores.extend(scores)
        if top < 0.55:
            weak_ids.append(record["example_id"])
            if record["human_intent"] == "general_or_context_missing":
                context_weak_ids.append(record["example_id"])
        if top >= 0.65 and not record["intent_correct"]:
            high_similarity_risk_ids.append(record["example_id"])
        for evidence in record["retrieved_evidence"]:
            evidence_rows += 1
            reply = str(evidence["amazon_response"])
            if re.search(r"@\d+|\^[A-Z]{2}\b|https?://|amzn\.to", reply):
                legacy_style_rows += 1
    retrieval = {
        "generated_at": generated_at,
        "scope": scope,
        "terminology_warning": (
            "No human retrieval-relevance labels exist, so these are similarity/evidence-usefulness diagnostics, not retrieval accuracy."
        ),
        "embedding_model": "qwen3-embedding:0.6b",
        "top1_similarity": _distribution(top1),
        "top3_mean_similarity": _distribution(top3_means),
        "all_top_k_similarities": _distribution(all_scores),
        "weak_retrieval_threshold": 0.55,
        "weak_retrieval_count": len(weak_ids),
        "weak_retrieval_frequency": len(weak_ids) / len(records),
        "weak_retrieval_example_ids": weak_ids,
        "context_missing_with_weak_retrieval_count": len(context_weak_ids),
        "context_missing_with_weak_retrieval_example_ids": context_weak_ids,
        "intent_relevance_available": False,
        "intent_relevance_reason": "The frozen retrieval metadata has no independently human-labeled relevance judgments.",
        "high_similarity_classifier_mismatch_proxy_count": len(high_similarity_risk_ids),
        "high_similarity_classifier_mismatch_proxy_ids": high_similarity_risk_ids,
        "legacy_social_format_evidence_count": legacy_style_rows,
        "legacy_social_format_evidence_rate": legacy_style_rows / evidence_rows if evidence_rows else 0.0,
        "legacy_language_note": "Mentions, agent initials, or historical links are proxies for stale social-support style, not proof of irrelevance.",
    }

    latency_keys = ["intent", "embedding_and_retrieval", "generation", "safety_and_routing", "total_component_sum"]
    latency = {
        "generated_at": generated_at,
        "scope": scope,
        "provider": records[0].get("provider"),
        "model": records[0].get("model"),
        "measurements_seconds": {
            key: _distribution([float(r["latency_seconds"][key]) for r in records]) for key in latency_keys
        },
        "measurement_note": "Embedding was batched and its wall time amortized across pending examples; generation is real per-request wall time.",
    }
    response = {
        "generated_at": generated_at,
        "scope": scope,
        "records": len(records),
        "provider_failures": sum(bool(r.get("provider_error")) for r in records),
        "unsupported_claim_flags": sum(bool(r.get("unsupported_claim_detected")) for r in records),
        "safety_overrides": sum(bool(r.get("safety_override")) for r in records),
        "auto_handle": sum(r["predicted_route"] == "AUTO_HANDLE" for r in records),
        "escalate": sum(r["predicted_route"] == "ESCALATE" for r in records),
        "raw_and_final_drafts_preserved": True,
    }
    failures = {
        "generated_at": generated_at,
        "selection_method": "Deterministic category-first selection from real errors; no examples were manually substituted.",
        "scope": scope,
        "top_five": _failure_examples(records),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "routing_metrics.json": routing,
        "retrieval_analysis.json": retrieval,
        "latency_analysis.json": latency,
        "response_evaluation_summary.json": response,
        "failure_analysis.json": failures,
    }
    for name, value in artifacts.items():
        (directory / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {name: value for name, value in artifacts.items()}
