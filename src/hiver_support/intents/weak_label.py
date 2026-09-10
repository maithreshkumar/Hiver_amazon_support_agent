from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import numpy as np

from hiver_support.config import load_yaml
from hiver_support.errors import ProviderUnavailable
from hiver_support.intents.taxonomy import IntentDefinition, load_approved_taxonomy
from hiver_support.llm.base import LLMProvider
from hiver_support.llm.factory import create_provider
from hiver_support.retrieval.embedder import OllamaEmbedder

_RULES = {
    "delivered_but_missing": r"(?:says|marked|showing).{0,30}delivered.{0,50}(?:not|never|missing|can.?t find)|delivered.{0,40}(?:not here|never received|missing)",
    "delivery_tracking_or_delay": r"\b(?:deliver(?:y|ed|ing)?|package|parcel|tracking|shipment|one[ -]?day shipping|two[ -]?day shipping|guaranteed delivery)\b",
    "return_request": r"\breturn(?:ed|ing|s)?\b|return label|send (?:it|this) back",
    "refund_or_payment_issue": r"\brefund(?:ed|ing|s)?\b|money back|\bcharged?\b|\bpayment\b|credit card|debit card|charged twice",
    "order_cancellation": r"\bcancel(?:led|ing|ation|s)?\b",
    "damaged_or_defective_item": r"\b(?:damag(?:ed|e)|broken|defective|doesn.?t work|not working|cracked)\b",
    "wrong_or_incomplete_item": r"wrong (?:item|product)|not what i ordered|ordered .{0,35} (?:got|received)|missing (?:item|part|piece)",
    "account_access_or_security": r"account.{0,40}(?:locked|login|log in|password|access|hack|unauthori[sz]ed)|(?:login|log in|password).{0,40}account|\bphishing\b|\bscam\b|someone accessed",
    "prime_video_issue": r"prime video|\bvideo.{0,25}(?:play|stream|watch|subtitle|episode|season)|(?:movie|show|episode|season).{0,30}(?:prime|available)",
    "prime_membership": r"\bprime\b.{0,40}(?:membership|member|renew|fee|subscription|benefit)|(?:membership|member).{0,40}\bprime\b",
    "seller_or_marketplace_issue": r"\b(?:seller|vendor|marketplace|third[ -]?party)\b",
    "general_or_context_missing": r"^\s*(?:(?:@\w+|MENTION)\s*)*(?:help|hi|hey|hello|URL|\?|please help)?\s*[!?]*\s*$",
}


def _matching_rules(text: str) -> list[str]:
    matches = [label for label, pattern in _RULES.items() if re.search(pattern, text, re.IGNORECASE)]
    if "delivered_but_missing" in matches and "delivery_tracking_or_delay" in matches:
        matches.remove("delivery_tracking_or_delay")
    if "prime_video_issue" in matches and "prime_membership" in matches:
        matches.remove("prime_membership")
    return matches


def _taxonomy_prompt(taxonomy: list[IntentDefinition]) -> str:
    return json.dumps(
        [
            {
                "name": item.name,
                "description": item.description,
                "inclusion_criteria": item.inclusion_criteria,
                "exclusion_criteria": item.exclusion_criteria,
                "common_confusions": item.common_confusions,
            }
            for item in taxonomy
        ],
        ensure_ascii=False,
    )


def label_batch(
    provider: LLMProvider, taxonomy: list[IntentDefinition], rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    names = {item.name for item in taxonomy}
    system = (
        "Weak-label Amazon support messages using only the approved taxonomy. Do not force ambiguous "
        "messages into an operational class. Confidence is 0..1; rationale is one sentence.\nTaxonomy:\n"
        + _taxonomy_prompt(taxonomy)
    )
    result = provider.structured_generate(
        system,
        json.dumps({"messages": rows}, ensure_ascii=False),
        {"labels": [{"row_id": "string", "intent": "approved name", "confidence": "0..1", "rationale": "string"}]},
    )
    parsed: list[dict[str, object]] = []
    for item in result.get("labels", []):
        label = str(item.get("intent", ""))
        confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        if label in names:
            parsed.append(
                {
                    "row_id": str(item.get("row_id", "")),
                    "predicted_intent": label,
                    "confidence": confidence,
                    "rationale": str(item.get("rationale", "")),
                }
            )
    return parsed


def run_weak_labeling(training_config_path: str | Path = "configs/training.yaml") -> dict[str, int]:
    config = load_yaml(training_config_path)
    runtime = load_yaml("configs/runtime.yaml")
    taxonomy = load_approved_taxonomy(config["taxonomy_path"])
    pairs = pd.read_parquet("data/processed/retrieval_corpus.parquet")
    pairs = pairs.loc[pairs["language"] == "en"].drop_duplicates("thread_id")
    settings = config["weak_labeling"]
    seed = int(settings["random_seed"])
    per_intent = int(settings["per_intent_target"])
    pairs["rule_labels"] = pairs["customer_text"].astype(str).map(_matching_rules)
    selected: list[pd.DataFrame] = []
    for offset, intent in enumerate(item.name for item in taxonomy):
        pool = pairs.loc[pairs["rule_labels"].map(lambda values: intent in values)]
        take = min(per_intent, len(pool))
        if take:
            selected.append(pool.sample(take, random_state=seed + offset))
    if not selected:
        raise RuntimeError("No weak-label candidates matched the approved taxonomy signals")
    pairs = pd.concat(selected, ignore_index=True).drop_duplicates("thread_id").reset_index(drop=True)
    pairs["row_id"] = [f"weak-{index:06d}" for index in range(len(pairs))]
    weak_llm_config = dict(runtime["llm"])
    weak_llm_config["model"] = str(settings.get("adjudication_model", weak_llm_config["model"]))
    provider = create_provider(weak_llm_config)
    embedding_model = "qwen3-embedding:0.6b"
    embedder = OllamaEmbedder(embedding_model)
    prototype_texts = [
        f"{item.name}. {item.description} Includes: {'; '.join(item.inclusion_criteria)}. Excludes: {'; '.join(item.exclusion_criteria)}"
        for item in taxonomy
    ]
    prototypes = embedder.embed(prototype_texts)
    embedding_batches: list[np.ndarray] = []
    embedding_batch_size = int(settings["embedding_batch_size"])
    for start in range(0, len(pairs), embedding_batch_size):
        embedding_batches.append(
            embedder.embed(pairs["customer_text"].iloc[start : start + embedding_batch_size].astype(str).tolist())
        )
        print(f"Embedded weak-label candidates {min(start + embedding_batch_size, len(pairs)):,}/{len(pairs):,}", flush=True)
    vectors = np.vstack(embedding_batches)
    similarities = vectors @ prototypes.T
    label_names = [item.name for item in taxonomy]
    predictions: dict[str, dict[str, object]] = {}
    review_priority: list[tuple[float, int]] = []
    for index, row in enumerate(pairs.itertuples()):
        order = np.argsort(similarities[index])[::-1]
        embedding_label = label_names[int(order[0])]
        margin = float(similarities[index, order[0]] - similarities[index, order[1]])
        rules = list(row.rule_labels)
        rule_label = rules[0] if len(rules) == 1 else None
        agreed = rule_label == embedding_label
        predicted = rule_label or embedding_label
        confidence = 0.92 if agreed else (0.82 if rule_label else min(0.84, 0.70 + max(0.0, margin)))
        predictions[row.row_id] = {
            "row_id": row.row_id,
            "predicted_intent": predicted,
            "confidence": confidence,
            "rationale": f"taxonomy signal={rule_label or 'ambiguous'}; embedding top={embedding_label}; margin={margin:.3f}",
            "label_method": "rule_plus_embedding" if agreed else "rule_or_embedding",
        }
        review_priority.append((0.0 if len(rules) != 1 else (1.0 if agreed else 0.2), index))
    max_reviews = min(int(settings["max_llm_reviews"]), len(pairs))
    ranked_indices = [index for _, index in sorted(review_priority)]
    review_indices: list[int] = []
    represented: set[str] = set()
    for index in ranked_indices:
        label = str(predictions[pairs.iloc[index]["row_id"]]["predicted_intent"])
        if label not in represented:
            review_indices.append(index)
            represented.add(label)
        if len(review_indices) >= max_reviews:
            break
    for index in ranked_indices:
        if len(review_indices) >= max_reviews:
            break
        if index not in review_indices:
            review_indices.append(index)
    batch_size = int(settings["batch_size"])
    for start in range(0, len(review_indices), batch_size):
        indices = review_indices[start : start + batch_size]
        rows = [
            {"row_id": pairs.iloc[index]["row_id"], "text": pairs.iloc[index]["customer_text"]}
            for index in indices
        ]
        try:
            adjudicated = label_batch(provider, taxonomy, rows)
        except (ProviderUnavailable, ValueError) as exc:
            print(f"LLM adjudication batch retained provisional labels: {exc}", flush=True)
            adjudicated = []
        for prediction in adjudicated:
            prediction["label_method"] = "structured_llm_adjudication"
            predictions[str(prediction["row_id"])] = prediction
        print(f"LLM-adjudicated {min(start + batch_size, len(review_indices)):,}/{len(review_indices):,}", flush=True)
    generated_at = datetime.now(UTC).isoformat()
    records: list[dict[str, object]] = []
    for row in pairs.itertuples():
        prediction = predictions.get(
            row.row_id,
            {"predicted_intent": "", "confidence": 0.0, "rationale": "Provider omitted row"},
        )
        records.append(
            {
                "row_id": row.row_id,
                "thread_id": str(row.thread_id),
                "customer_text": row.customer_text,
                **prediction,
                "label_source": "weak_ai",
                "provider": provider.name,
                "model": provider.model,
                "embedding_model": embedding_model,
                "generated_at": generated_at,
            }
        )
    output = pd.DataFrame(records)
    accepted = output.loc[output["confidence"] >= float(settings["minimum_confidence"])].copy()
    quarantine = output.loc[output["confidence"] < float(settings["minimum_confidence"])].copy()
    Path(config["weak_labels_path"]).parent.mkdir(parents=True, exist_ok=True)
    accepted.to_parquet(config["weak_labels_path"], index=False)
    Path(config["quarantine_path"]).parent.mkdir(parents=True, exist_ok=True)
    quarantine.to_parquet(config["quarantine_path"], index=False)
    return {"requested": len(output), "accepted": len(accepted), "quarantined": len(quarantine)}
