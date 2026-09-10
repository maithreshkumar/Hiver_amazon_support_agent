from hiver_support.intents.taxonomy import IntentDefinition
from hiver_support.intents.weak_label import label_batch
from hiver_support.llm.base import LLMProvider


class FakeLabeler(LLMProvider):
    name, model = "fake", "fake"

    def generate(self, system_prompt, user_prompt):
        return ""

    def structured_generate(self, system_prompt, user_prompt, schema=None):
        return {
            "labels": [
                {"row_id": "1", "intent": "delivery", "confidence": 1.2, "rationale": "tracking"},
                {"row_id": "2", "intent": "invented", "confidence": 0.9, "rationale": "bad"},
            ]
        }


def test_weak_labels_are_validated_and_confidence_clamped() -> None:
    taxonomy = [IntentDefinition("delivery", "", [], [], [])]
    result = label_batch(FakeLabeler(), taxonomy, [{"row_id": "1", "text": "where"}])
    assert result == [
        {"row_id": "1", "predicted_intent": "delivery", "confidence": 1.0, "rationale": "tracking"}
    ]

