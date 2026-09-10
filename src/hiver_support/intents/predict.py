from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hiver_support.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    secondary_intent: str | None = None
    ambiguity_score: float = 0.0


class IntentClassifier:
    """Runtime loader for a persisted compact transformer sequence classifier."""

    def __init__(self, model_dir: str | Path, allowed_labels: set[str]) -> None:
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise ConfigurationError(
                f"Trained intent model is missing at {self.model_dir}. "
                "Approve the taxonomy, weak-label training data, and run the training command first."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = self.metadata.get("labels", [])
        if not labels or not set(labels) <= allowed_labels:
            raise ConfigurationError("Intent model labels do not match the approved taxonomy.")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ConfigurationError("Install the Phase 2 neural extras: transformers and torch.") from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.labels = labels

    def predict(self, text: str) -> IntentPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[IntentPrediction]:
        """Run deterministic batched CPU/GPU inference without changing prediction semantics."""
        if not texts:
            return []
        rows: list[IntentPrediction] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            with self._torch.no_grad():
                logits = self.model(**encoded).logits
            for probabilities in self._torch.softmax(logits, dim=-1).cpu().numpy().astype(float):
                rows.append(self._to_prediction(probabilities))
        return rows

    def _to_prediction(self, probabilities: np.ndarray) -> IntentPrediction:
        order = np.argsort(probabilities)[::-1]
        primary, secondary = int(order[0]), int(order[1])
        gap = float(probabilities[primary] - probabilities[secondary])
        return IntentPrediction(
            label=self.labels[primary],
            confidence=float(probabilities[primary]),
            probabilities={label: float(probabilities[index]) for index, label in enumerate(self.labels)},
            model_version=str(self.metadata.get("version", "unknown")),
            secondary_intent=self.labels[secondary] if gap < 0.18 else None,
            ambiguity_score=float(max(0.0, 1.0 - gap)),
        )
