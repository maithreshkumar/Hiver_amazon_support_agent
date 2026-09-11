from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from hiver_support.config import load_yaml
from hiver_support.errors import ProviderUnavailable
from hiver_support.evaluation.golden_labels import load_golden_set, validate_golden_set
from hiver_support.generation.reply import DraftValidation, GroundedReplyGenerator
from hiver_support.intents.predict import IntentClassifier
from hiver_support.intents.taxonomy import load_approved_taxonomy
from hiver_support.llm.factory import create_provider
from hiver_support.retrieval.embedder import OllamaEmbedder
from hiver_support.retrieval.index import LocalVectorIndex
from hiver_support.routing.final_response import build_final_response
from hiver_support.routing.policy import RoutingPolicy


TRIVIAL_REPLY = "Thanks for contacting Amazon support. A support specialist will review your request."


def select_reply_sample(size: int = 48) -> pd.DataFrame:
    if size < 40 or size > 50:
        raise ValueError("Human reply-quality sample must contain 40–50 examples")
    golden = load_golden_set()
    predictions = pd.read_csv("data/reports/final_intent_predictions.csv", dtype="string")
    frame = golden.merge(
        predictions[["example_id", "predicted_intent", "intent_confidence", "intent_correct"]],
        on="example_id", validate="one_to_one",
    )
    frame["intent_correct"] = frame["intent_correct"].astype(str).str.casefold().eq("true")
    frame["selection_stratum"] = (
        frame["human_intent"].astype(str) + "|" + frame["human_escalation"].astype(str)
        + "|" + frame["intent_correct"].map({True: "correct", False: "incorrect"})
    )
    shuffled = frame.sample(frac=1, random_state=314159).reset_index(drop=True)
    groups = {name: group.index.tolist() for name, group in shuffled.groupby("selection_stratum")}
    selected: list[int] = []
    while len(selected) < size and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(selected) < size:
                selected.append(groups[name].pop(0))
    sample = shuffled.loc[selected].copy().reset_index(drop=True)
    sample["sample_order"] = range(1, len(sample) + 1)
    sample.to_csv("data/golden/reply_evaluation_sample_internal.csv", index=False)
    return sample


def _tfidf_reply_baseline(messages: list[str]) -> list[dict[str, object]]:
    corpus = pd.read_parquet(
        "data/processed/retrieval_corpus.parquet",
        columns=["thread_id", "customer_text", "amazon_response"],
    ).drop_duplicates("thread_id")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus["customer_text"].astype(str))
    queries = vectorizer.transform(messages)
    results: list[dict[str, object]] = []
    for row in range(len(messages)):
        scores = (queries[row] @ matrix.T).toarray()[0]
        index = int(np.argmax(scores))
        results.append({
            "thread_id": str(corpus.iloc[index]["thread_id"]),
            "reply": str(corpus.iloc[index]["amazon_response"]),
            "similarity": float(scores[index]),
        })
    return results


def run_response_evaluation(size: int = 48) -> dict[str, object]:
    validate_golden_set(require_complete=True)
    sample = select_reply_sample(size)
    runtime, retrieval = load_yaml("configs/runtime.yaml"), load_yaml("configs/retrieval.yaml")
    labels = {item.name for item in load_approved_taxonomy()}
    classifier = IntentClassifier(runtime["intent"]["model_path"], labels)
    index = LocalVectorIndex(
        retrieval["index_dir"], OllamaEmbedder(retrieval["model"], retrieval["base_url"])
    )
    generator = GroundedReplyGenerator(create_provider(runtime["llm"]))
    policy = RoutingPolicy(runtime["routing"]["policy_path"])
    simple_replies = _tfidf_reply_baseline(sample["customer_message"].astype(str).tolist())

    output_path = Path("data/reports/golden_response_outputs.jsonl")
    existing: dict[str, dict[str, object]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[str(item["example_id"])] = item

    prepared: list[dict[str, object]] = []
    for position, row in sample.iterrows():
        example_id = str(row["example_id"])
        if example_id in existing:
            continue
        message = str(row["customer_message"])
        intent_started = time.perf_counter()
        prediction = classifier.predict(message)
        intent_latency = time.perf_counter() - intent_started
        prepared.append({
            "position": position,
            "row": row,
            "prediction": prediction,
            "intent_latency": intent_latency,
        })
        print(f"Prepared classifier {len(prepared)}/{len(sample) - len(existing)}", flush=True)

    if prepared:
        embedding_started = time.perf_counter()
        query_vectors = index.embedder.embed(
            [str(item["row"]["customer_message"]) for item in prepared]
        )
        embedding_total = time.perf_counter() - embedding_started
        amortized_embedding = embedding_total / len(prepared)
        for item, query_vector in zip(prepared, query_vectors, strict=True):
            search_started = time.perf_counter()
            item["matches"] = index.search_vector(
                query_vector, int(retrieval["top_k"]), item["prediction"].label
            )
            item["retrieval_latency"] = amortized_embedding + (time.perf_counter() - search_started)
        print(f"Batched {len(prepared)} query embeddings in {embedding_total:.1f}s", flush=True)

    for offset, item in enumerate(prepared, 1):
        row = item["row"]
        prediction = item["prediction"]
        matches = item["matches"]
        message = str(row["customer_message"])
        provider_error = None
        generation_started = time.perf_counter()
        try:
            generated = generator.generate(message, prediction.label, matches)
        except ProviderUnavailable as exc:
            provider_error = str(exc)
            generated = None
        generation_latency = time.perf_counter() - generation_started
        routing_started = time.perf_counter()
        if generated is None:
            final = build_final_response(
                message=message,
                intent=prediction.label,
                raw_draft="",
                validation=DraftValidation(),
                routing=policy.provider_failure_decision(
                    message=message,
                    intent=prediction.label,
                    intent_confidence=prediction.confidence,
                    ambiguity_score=prediction.ambiguity_score,
                ),
            )
            reply, route, route_reason = final.reply, final.decision, final.reason
            reason_codes, validation_flags, template = final.reason_codes, [], final.template
            raw_draft, grounding, unsupported, override, evidence_ids = "", 0.0, False, True, []
        else:
            best_similarity = max((match.similarity for match in matches), default=0.0)
            decision = policy.decide(
                message=message, intent=prediction.label, intent_confidence=prediction.confidence,
                best_similarity=best_similarity, grounding_confidence=generated.grounding_confidence,
                unsupported_claim=generated.unsupported_claim_detected,
                ambiguity_score=prediction.ambiguity_score,
                safety_flags=generated.validation.flags,
            )
            final = build_final_response(
                message=message,
                intent=prediction.label,
                raw_draft=generated.raw_draft,
                validation=generated.validation,
                routing=decision,
            )
            route, route_reason = final.decision, final.reason
            reply, raw_draft = final.reply, generated.raw_draft
            grounding, unsupported = generated.grounding_confidence, generated.unsupported_claim_detected
            override, evidence_ids = final.safety_override_applied, generated.evidence_thread_ids
            reason_codes = final.reason_codes
            validation_flags, template = list(generated.validation.flags), final.template
        routing_latency = time.perf_counter() - routing_started
        simple = simple_replies[int(item["position"])]
        record = {
            "example_id": str(row["example_id"]), "thread_id": str(row["thread_id"]),
            "message": message, "human_intent": str(row["human_intent"]),
            "human_route": str(row["human_escalation"]), "predicted_intent": prediction.label,
            "intent_confidence": prediction.confidence, "intent_correct": prediction.label == str(row["human_intent"]),
            "secondary_intent": prediction.secondary_intent, "ambiguity_score": prediction.ambiguity_score,
            "retrieved_evidence": [asdict(match) for match in matches],
            "generator_evidence_thread_ids": evidence_ids, "raw_generated_draft": raw_draft,
            "final_safe_response": reply, "grounding_confidence": grounding,
            "unsupported_claim_detected": unsupported, "safety_override": override,
            "predicted_route": route, "routing_reason": route_reason,
            "routing_reason_codes": reason_codes, "safety_validation_flags": validation_flags,
            "final_response_template": template,
            "provider": generator.provider.name, "model": generator.provider.model,
            "prompt_version": "amazon-grounded-v1", "provider_error": provider_error,
            "trivial_reply": TRIVIAL_REPLY, "tfidf_reply": simple["reply"],
            "tfidf_reply_thread_id": simple["thread_id"], "tfidf_reply_similarity": simple["similarity"],
            "latency_seconds": {
                "intent": item["intent_latency"], "embedding_and_retrieval": item["retrieval_latency"],
                "generation": generation_latency, "safety_and_routing": routing_latency,
                "total_component_sum": item["intent_latency"] + item["retrieval_latency"] + generation_latency + routing_latency,
            },
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
        existing[str(row["example_id"])] = record
        ordered = [existing[str(value)] for value in sample["example_id"] if str(value) in existing]
        temporary = output_path.with_suffix(".tmp.jsonl")
        temporary.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in ordered), encoding="utf-8")
        os.replace(temporary, output_path)
        print(f"Generated response {offset}/{len(prepared)} ({row['example_id']}, {generation_latency:.1f}s)", flush=True)

    records = [existing[str(value)] for value in sample["example_id"] if str(value) in existing]
    final = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "Deterministic 48-example stratified human-gold response/routing sample",
        "justification": (
            "Phase 2 measured 102–109 second local CPU generation. A 48-example sample satisfies the "
            "required 40–50 human-rating range while covering intent, route, and correctness strata."
        ),
        "records": records,
    }
    Path("data/reports/golden_response_outputs.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final
