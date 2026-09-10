from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from hiver_support.intents.predict import IntentClassifier
from hiver_support.intents.taxonomy import load_approved_taxonomy
from hiver_support.intents.weak_label import _matching_rules


def build_development_silver_set(
    dev_path: str | Path = "data/processed/dev.parquet",
    output_path: str | Path = "data/processed/dev_silver_labels.parquet",
    per_intent: int = 120,
    random_seed: int = 20260909,
) -> pd.DataFrame:
    """Create an evaluation-only silver reference from unambiguous taxonomy signals.

    These labels are never added to training. They provide an honest chronological-dev
    diagnostic while the human-labelled golden test set remains sealed for Phase 3.
    """
    taxonomy = load_approved_taxonomy()
    labels = [item.name for item in taxonomy]
    frame = pd.read_parquet(dev_path)
    frame = frame.loc[frame["language"].eq("en")].copy()
    frame["customer_text"] = frame["first_customer_message"].astype(str)
    frame["rule_labels"] = frame["customer_text"].map(_matching_rules)
    selected: list[pd.DataFrame] = []
    for offset, label in enumerate(labels):
        if label == "general_or_context_missing":
            pool = frame.loc[
                frame["rule_labels"].map(len).eq(0)
                & frame["customer_text"].str.split().map(len).le(6)
            ].copy()
            method = "short_unmatched_context_signal"
        else:
            pool = frame.loc[frame["rule_labels"].map(lambda values: values == [label])].copy()
            method = "single_high_precision_taxonomy_signal"
        if pool.empty:
            continue
        sample = pool.sample(min(per_intent, len(pool)), random_state=random_seed + offset)
        sample["reference_intent"] = label
        sample["reference_method"] = method
        selected.append(sample)
    silver = pd.concat(selected, ignore_index=True).drop_duplicates("thread_id")
    silver = silver[
        ["thread_id", "customer_text", "reference_intent", "reference_method", "split_time"]
    ].copy()
    silver["reference_source"] = "evaluation_only_silver_rules"
    silver["taxonomy_version"] = "1.0"
    silver["created_at"] = datetime.now(UTC).isoformat()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(output, index=False)
    return silver


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_development(
    model_dir: str | Path = "models/intent",
    baseline_dir: str | Path = "models/baselines",
    output_path: str | Path = "data/reports/development_metrics.json",
) -> dict[str, object]:
    taxonomy = load_approved_taxonomy()
    labels = [item.name for item in taxonomy]
    silver = build_development_silver_set()
    texts = silver["customer_text"].astype(str).tolist()
    expected = silver["reference_intent"].astype(str).tolist()

    classifier = IntentClassifier(model_dir, set(labels))
    neural_predictions = classifier.predict_batch(texts, batch_size=32)
    neural_labels = [item.label for item in neural_predictions]
    confidences = [item.confidence for item in neural_predictions]

    trivial = joblib.load(Path(baseline_dir) / "trivial.joblib")
    simple = joblib.load(Path(baseline_dir) / "simple.joblib")
    trivial_labels = [str(trivial.predict(text)["intent"]) for text in texts]
    simple_labels = [str(simple.predict_intent(text)["intent"]) for text in texts]

    threshold_rows: list[dict[str, object]] = []
    for threshold in (0.50, 0.60, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90):
        indices = [index for index, confidence in enumerate(confidences) if confidence >= threshold]
        correct = sum(neural_labels[index] == expected[index] for index in indices)
        threshold_rows.append(
            {
                "threshold": threshold,
                "covered": len(indices),
                "coverage": len(indices) / len(expected),
                "accuracy_when_covered": correct / len(indices) if indices else None,
            }
        )

    result: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "split": "chronological development split",
        "test_split_or_golden_set_used": False,
        "reference_type": "evaluation-only silver labels from single high-precision taxonomy signals",
        "important_limitation": (
            "These are not human ground truth and favor messages containing explicit lexical signals. "
            "Metrics are diagnostic weak-label agreement, not Phase 3 headline performance."
        ),
        "rows": len(silver),
        "support": dict(Counter(expected)),
        "labels": labels,
        "models": {
            "distilroberta": _metrics(expected, neural_labels, labels),
            "tfidf_logistic_regression": _metrics(expected, simple_labels, labels),
            "most_frequent_trivial": _metrics(expected, trivial_labels, labels),
        },
        "confidence_diagnostic": threshold_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metadata_path = Path(model_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["chronological_development_diagnostic"] = {
        "reference_type": result["reference_type"],
        "rows": result["rows"],
        **result["models"]["distilroberta"],
    }
    metadata["chronological_development_diagnostic"].pop("classification_report")
    metadata["chronological_development_diagnostic"].pop("confusion_matrix")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return result

