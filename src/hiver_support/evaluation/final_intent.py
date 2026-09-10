from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from hiver_support.evaluation.golden_labels import load_golden_set, validate_golden_set
from hiver_support.intents.predict import IntentClassifier
from hiver_support.intents.taxonomy import load_approved_taxonomy


def _metrics(expected: list[str], predicted: list[str], labels: list[str]) -> dict[str, object]:
    correct = int(sum(a == b for a, b in zip(expected, predicted, strict=True)))
    return {
        "rows": len(expected),
        "correct": correct,
        "incorrect": len(expected) - correct,
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_precision": float(precision_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(expected, predicted, labels=labels, average="weighted", zero_division=0)),
        "per_intent": classification_report(expected, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels, "values": confusion_matrix(expected, predicted, labels=labels).tolist()},
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values), "min": float(array.min()), "p25": float(np.quantile(array, .25)),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "p75": float(np.quantile(array, .75)), "max": float(array.max()),
    }


def run_final_intent_evaluation() -> dict[str, object]:
    validate_golden_set(require_complete=True)
    golden = load_golden_set()
    taxonomy = load_approved_taxonomy()
    labels = [item.name for item in taxonomy]
    texts = golden["customer_message"].astype(str).tolist()
    expected = golden["human_intent"].astype(str).tolist()

    classifier = IntentClassifier("models/intent", set(labels))
    neural = classifier.predict_batch(texts, batch_size=32)
    neural_labels = [item.label for item in neural]
    confidences = [item.confidence for item in neural]
    simple = joblib.load("models/baselines/simple.joblib")
    trivial = joblib.load("models/baselines/trivial.joblib")
    simple_labels = [str(simple.predict_intent(text)["intent"]) for text in texts]
    trivial_labels = [str(trivial.predict(text)["intent"]) for text in texts]

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "ground_truth": "200 human-reviewed golden examples; assisted annotation with human final decisions",
        "taxonomy_version": "1.0",
        "models": {
            "most_frequent_trivial": _metrics(expected, trivial_labels, labels),
            "tfidf_logistic_regression": _metrics(expected, simple_labels, labels),
            "distilroberta": _metrics(expected, neural_labels, labels),
        },
    }
    Path("data/reports/final_intent_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    per_intent_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for model_name, model_metrics in metrics["models"].items():
        report = model_metrics["per_intent"]
        for label in labels:
            per_intent_rows.append({
                "model": model_name,
                "intent": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            })
        matrix = model_metrics["confusion_matrix"]["values"]
        for expected_label, values in zip(labels, matrix, strict=True):
            confusion_rows.append({"model": model_name, "expected_intent": expected_label, **dict(zip(labels, values, strict=True))})
    pd.DataFrame(per_intent_rows).to_csv("data/reports/final_intent_per_intent.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv("data/reports/final_intent_confusion_matrices.csv", index=False)

    thresholds: list[dict[str, object]] = []
    for threshold in (0.0, 0.50, 0.60, 0.70, 0.72, 0.75, 0.80, 0.90):
        selected = [i for i, value in enumerate(confidences) if value >= threshold]
        selected_expected = [expected[i] for i in selected]
        selected_predicted = [neural_labels[i] for i in selected]
        thresholds.append({
            "threshold": threshold,
            "covered": len(selected),
            "coverage": len(selected) / len(expected),
            "low_confidence_escalations": len(expected) - len(selected),
            "accuracy_on_covered": float(accuracy_score(selected_expected, selected_predicted)) if selected else None,
            "macro_f1_on_covered": float(f1_score(selected_expected, selected_predicted, labels=labels, average="macro", zero_division=0)) if selected else None,
        })
    confidence = {
        "generated_at": metrics["generated_at"],
        "configured_threshold": 0.72,
        "threshold_source": "Frozen Phase 2 development-time policy; not selected on golden results.",
        "correct_predictions": _distribution([c for c, a, b in zip(confidences, expected, neural_labels, strict=True) if a == b]),
        "incorrect_predictions": _distribution([c for c, a, b in zip(confidences, expected, neural_labels, strict=True) if a != b]),
        "thresholds": thresholds,
    }
    Path("data/reports/confidence_analysis.json").write_text(json.dumps(confidence, indent=2) + "\n", encoding="utf-8")

    predictions = golden[["example_id", "thread_id", "customer_message", "human_intent", "human_escalation"]].copy()
    predictions["trivial_intent"] = trivial_labels
    predictions["tfidf_intent"] = simple_labels
    predictions["predicted_intent"] = neural_labels
    predictions["intent_confidence"] = confidences
    predictions["intent_correct"] = [a == b for a, b in zip(expected, neural_labels, strict=True)]
    predictions["secondary_intent"] = [item.secondary_intent or "" for item in neural]
    predictions["ambiguity_score"] = [item.ambiguity_score for item in neural]
    predictions.to_csv("data/reports/final_intent_predictions.csv", index=False)
    return {"metrics": metrics, "confidence": confidence}
