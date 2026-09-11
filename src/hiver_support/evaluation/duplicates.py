from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from hiver_support.evaluation.golden_labels import load_golden_set, validate_golden_set


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"https?://\S+|@\w+", " ", text.casefold()).split())


def run_near_duplicate_audit(
    threshold: float = 0.35,
    output_path: str | Path = "data/reports/near_duplicate_audit.json",
    *,
    validate_inputs: bool = True,
) -> dict[str, object]:
    if validate_inputs:
        validate_golden_set(require_complete=True)
    frame = load_golden_set()
    normalized = frame["customer_message"].astype(str).map(_normalize)
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(normalized)
    similarity = (vectors @ vectors.T).toarray()
    pairs: list[dict[str, object]] = []
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            score = float(similarity[left, right])
            if score >= threshold:
                pairs.append({
                    "left_example_id": str(frame.at[left, "example_id"]),
                    "right_example_id": str(frame.at[right, "example_id"]),
                    "similarity": score,
                    "exact_normalized_duplicate": normalized.iloc[left] == normalized.iloc[right],
                    "left_intent": str(frame.at[left, "human_intent"]),
                    "right_intent": str(frame.at[right, "human_intent"]),
                })
    pairs.sort(key=lambda row: float(row["similarity"]), reverse=True)
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "character TF-IDF cosine similarity after lowercasing and removing URLs/mentions",
        "threshold": threshold,
        "rows": len(frame),
        "exact_normalized_duplicate_pairs": sum(bool(row["exact_normalized_duplicate"]) for row in pairs),
        "near_duplicate_pairs": len(pairs),
        "examples_in_near_duplicate_pairs": len({row[key] for row in pairs for key in ("left_example_id", "right_example_id")}),
        "human_review_noted_semantic_pairs": [
            {
                "left_example_id": "gold-004",
                "right_example_id": "gold-117",
                "theme": "accidental Kindle purchase cancellation",
                "detected_at_configured_threshold": any(
                    {row["left_example_id"], row["right_example_id"]} == {"gold-004", "gold-117"}
                    for row in pairs
                ),
                "note": "Preserved as a qualitative near-duplicate noted during human review; lexical character TF-IDF may miss paraphrases.",
            }
        ],
        "sampling_effect_note": (
            "Near-duplicate themes can overweight a behavior in a 200-example set. No examples were removed after labels were observed."
        ),
        "pairs": pairs,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
