from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

from hiver_support.config import load_yaml
from hiver_support.errors import ProviderUnavailable
from hiver_support.llm.factory import create_provider


SYSTEMS = ("trivial", "tfidf_retrieval_only", "rag_with_safety")
DIMENSIONS = (
    "groundedness", "helpfulness", "actionability", "safety",
    "historical_consistency", "unsupported_claim_avoidance",
)


def _candidate_mapping(example_id: str) -> dict[str, str]:
    seed = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:16], 16)
    systems = list(SYSTEMS)
    random.Random(seed).shuffle(systems)
    return dict(zip(("A", "B", "C"), systems, strict=True))


def _validate_result(value: dict[str, object]) -> dict[str, object]:
    candidates = value.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {"A", "B", "C"}:
        raise ValueError("Judge output must score blinded candidates A, B, and C")
    normalized: dict[str, object] = {}
    for key in ("A", "B", "C"):
        item = candidates[key]
        if not isinstance(item, dict) or not isinstance(item.get("scores"), dict):
            raise ValueError(f"Judge output for {key} is malformed")
        scores = item["scores"]
        if set(scores) != set(DIMENSIONS):
            raise ValueError(f"Judge scores for {key} do not match the rubric")
        numeric = {name: int(scores[name]) for name in DIMENSIONS}
        if any(score < 1 or score > 5 for score in numeric.values()):
            raise ValueError("Judge scores must be integers from 1 to 5")
        rationale = str(item.get("rationale", "")).strip()
        if not rationale:
            raise ValueError("Judge rationale may not be blank")
        normalized[key] = {
            "scores": numeric,
            "aggregate": sum(numeric.values()) / len(numeric),
            "rationale": rationale,
        }
    return normalized


def run_reply_judge(
    response_path: str | Path = "data/reports/golden_response_outputs.json",
    output_path: str | Path = "data/reports/reply_quality_judge.jsonl",
    summary_output_path: str | Path = "data/reports/reply_quality_judge.json",
    *,
    allow_legacy_cache: bool = True,
) -> dict[str, object]:
    response_file = Path(response_path)
    response_payload = json.loads(response_file.read_text(encoding="utf-8"))
    records = list(response_payload["records"])
    config = load_yaml("configs/judge.yaml")
    provider_config = dict(config)
    provider_config["provider"] = os.getenv("JUDGE_PROVIDER", str(config["provider"]))
    provider_config["model"] = os.getenv("JUDGE_MODEL", str(config["model"]))
    provider = create_provider(provider_config)
    if records and provider.name == records[0].get("provider") and provider.model == records[0].get("model"):
        raise ValueError("Judge must be separate from the response-generation model")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, object]] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[str(item["example_id"])] = item
    schema = {
        "candidates": {
            "A": {"scores": {name: "integer 1-5" for name in DIMENSIONS}, "rationale": "brief string"},
            "B": {"scores": {name: "integer 1-5" for name in DIMENSIONS}, "rationale": "brief string"},
            "C": {"scores": {name: "integer 1-5" for name in DIMENSIONS}, "rationale": "brief string"},
        }
    }
    system_prompt = (
        "You are an independent support-response evaluator. Score each blinded candidate from 1 (poor) to 5 "
        "(excellent) on exactly six dimensions. Groundedness and historical consistency refer only to the supplied "
        "historical AmazonHelp examples; they are not current policy. Penalize invented transactions, guarantees, "
        "unsafe requests for sensitive data, copied stale links, and claims of confirmed resolution. Evaluate all "
        "three candidates using the same standard. Human ratings and model correctness are unavailable."
    )

    for position, record in enumerate(records, 1):
        example_id = str(record["example_id"])
        mapping = _candidate_mapping(example_id)
        replies = {
            "trivial": str(record["trivial_reply"]),
            "tfidf_retrieval_only": str(record["tfidf_reply"]),
            "rag_with_safety": str(record["final_safe_response"]),
        }
        evidence = [
            {
                "historical_customer_message": item["customer_text"],
                "historical_support_reply": item["amazon_response"],
                "similarity": round(float(item["similarity"]), 4),
            }
            for item in record["retrieved_evidence"][:3]
        ]
        user_prompt = json.dumps({
            "customer_message": record["message"],
            "historical_evidence": evidence,
            "candidates": {blind: replies[system] for blind, system in mapping.items()},
        }, ensure_ascii=False)
        evaluation_input_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "judge_model": provider.model,
                    "prompt_version": str(config["prompt_version"]),
                    "rubric_version": str(config["rubric_version"]),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if example_id in existing:
            cached_hash = existing[example_id].get("evaluation_input_sha256")
            if cached_hash is None and allow_legacy_cache:
                continue
            if cached_hash != evaluation_input_sha256:
                raise ValueError(
                    f"Cached judge input does not match {example_id}; use a new output path"
                )
            continue
        last_error = ""
        judged = None
        for attempt in range(5):
            try:
                format_retry = (
                    ""
                    if attempt == 0
                    else (
                        "\nYour previous response was structurally invalid. Return one complete JSON object "
                        "with exactly candidates A, B, and C; include all six integer scores and one brief "
                        "rationale for each. Do not omit a candidate or add prose outside JSON."
                    )
                )
                judged = _validate_result(
                    provider.structured_generate(system_prompt + format_retry, user_prompt, schema)
                )
                break
            except (ProviderUnavailable, ValueError, TypeError, KeyError) as exc:
                last_error = str(exc)
        if judged is None:
            raise ProviderUnavailable(f"Judge failed five times for {example_id}: {last_error}")
        by_system = {mapping[blind]: judged[blind] for blind in ("A", "B", "C")}
        result = {
            "example_id": example_id,
            "judge_provider": provider.name,
            "judge_model": provider.model,
            "prompt_version": str(config["prompt_version"]),
            "rubric_version": str(config["rubric_version"]),
            "blinded_candidate_mapping": mapping,
            "scores_by_system": by_system,
            "human_ratings_visible_to_judge": False,
            "evaluation_input_sha256": evaluation_input_sha256,
            "reused_byte_identical_pre_repair_judgment": False,
            "judged_at": datetime.now(UTC).isoformat(),
        }
        existing[example_id] = result
        ordered = [existing[str(r["example_id"])] for r in records if str(r["example_id"]) in existing]
        temporary = target.with_suffix(".tmp.jsonl")
        temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")
        os.replace(temporary, target)
        print(f"Judged {position}/{len(records)} ({example_id})", flush=True)

    ordered = [existing[str(r["example_id"])] for r in records]
    aggregates: dict[str, dict[str, float]] = {}
    for system in SYSTEMS:
        aggregates[system] = {
            name: sum(float(item["scores_by_system"][system]["scores"][name]) for item in ordered) / len(ordered)
            for name in DIMENSIONS
        }
        aggregates[system]["overall"] = sum(float(item["scores_by_system"][system]["aggregate"]) for item in ordered) / len(ordered)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": response_payload.get("scope"),
        "judge_provider": provider.name,
        "judge_model": provider.model,
        "generator_model": records[0].get("model") if records else None,
        "separate_from_generator": True,
        "human_ratings_visible_to_judge": False,
        "prompt_version": config["prompt_version"],
        "rubric_version": config["rubric_version"],
        "source_response_path": str(response_file.as_posix()),
        "source_response_sha256": hashlib.sha256(response_file.read_bytes()).hexdigest(),
        "mean_scores": aggregates,
        "records": ordered,
    }
    summary = Path(summary_output_path)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
