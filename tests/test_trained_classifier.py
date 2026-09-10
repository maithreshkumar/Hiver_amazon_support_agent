from pathlib import Path

import pytest

from hiver_support.intents.predict import IntentClassifier
from hiver_support.intents.taxonomy import load_approved_taxonomy


def test_trained_classifier_artifact_loads_and_predicts() -> None:
    if not Path("models/intent/metadata.json").exists():
        pytest.skip("Trained classifier artifact has not been created")
    labels = {item.name for item in load_approved_taxonomy()}
    classifier = IntentClassifier("models/intent", labels)
    outputs = classifier.predict_batch(
        [
            "Tracking says my order will arrive late.",
            "Prime Video will not play this episode.",
        ]
    )
    assert len(outputs) == 2
    assert all(output.label in labels for output in outputs)
    assert all(0.0 <= output.confidence <= 1.0 for output in outputs)
