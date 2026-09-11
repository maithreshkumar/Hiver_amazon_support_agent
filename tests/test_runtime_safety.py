from types import SimpleNamespace

import pytest

from hiver_support.agent import AmazonSupportAgent
from hiver_support.generation.reply import GroundedReplyGenerator
from hiver_support.intents.predict import IntentPrediction
from hiver_support.llm.base import LLMProvider
from hiver_support.retrieval.index import RetrievalMatch
from hiver_support.routing.policy import RoutingPolicy


class _Classifier:
    def __init__(self, prediction: IntentPrediction) -> None:
        self.prediction = prediction

    def predict(self, text: str) -> IntentPrediction:
        return self.prediction


class _Index:
    def __init__(self, similarity: float = 0.90) -> None:
        self.match = RetrievalMatch(
            "t1",
            "Historical customer message",
            "Historical support handling without a current-policy guarantee.",
            similarity,
        )

    def search(self, message: str, top_k: int, intent: str):
        return [self.match]


class _Provider(LLMProvider):
    name, model = "fake", "deterministic"

    def __init__(self, draft: str, grounding: float = 0.90, model_flag: bool = False) -> None:
        self.draft = draft
        self.grounding = grounding
        self.model_flag = model_flag

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "unused"

    def structured_generate(self, system_prompt: str, user_prompt: str, schema=None):
        return {
            "draft_reply": self.draft,
            "evidence_thread_ids": ["t1"],
            "grounding_confidence": self.grounding,
            "unsupported_claim_detected": self.model_flag,
        }


def _prediction(intent: str, confidence: float, ambiguity: float = 0.10) -> IntentPrediction:
    return IntentPrediction(intent, confidence, {intent: confidence}, "test", ambiguity_score=ambiguity)


def _agent(prediction: IntentPrediction, draft: str, *, grounding: float = 0.90) -> AmazonSupportAgent:
    agent = AmazonSupportAgent.__new__(AmazonSupportAgent)
    agent.classifier = _Classifier(prediction)
    agent.index = _Index()
    agent.generator = GroundedReplyGenerator(_Provider(draft, grounding))
    agent.policy = RoutingPolicy()
    agent.top_k = 5
    return agent


def _assert_safe_escalation(result: dict[str, object]) -> None:
    assert result["decision"] == "ESCALATE"
    assert result["reply"] != result["raw_draft"]
    assert result["provenance"]["final_response_template"]
    assert result["provenance"]["safety_override_applied"] is True


@pytest.mark.parametrize(
    "message",
    [
        "My package is three days late. Where is it?",
        "Where is my order? It was supposed to arrive yesterday.",
    ],
)
def test_low_confidence_delivery_uses_final_delivery_escalation(message: str) -> None:
    result = _agent(
        _prediction("delivery_tracking_or_delay", 0.56),
        "Could you share the tracking number?",
    ).handle(message)
    _assert_safe_escalation(result)
    assert "LOW_INTENT_CONFIDENCE" in result["provenance"]["reason_codes"]
    assert "tracking number" not in result["reply"].casefold()


def test_hacked_account_never_echoes_phone_or_requests_transaction_details() -> None:
    phone = "+1 415-555-0199"
    result = _agent(
        _prediction("account_access_or_security", 0.95),
        f"Please provide transaction details and confirm your number {phone}.",
    ).handle(f"My account was hacked. Call me at {phone} about purchases I did not make.")
    _assert_safe_escalation(result)
    assert result["provenance"]["final_response_template"] == "security"
    assert {"SECURITY_RISK", "PII_REQUEST", "PII_ECHO"} <= set(
        result["provenance"]["reason_codes"]
    )
    assert phone not in result["reply"]
    assert "transaction details" not in result["reply"].casefold()


def test_delivered_but_missing_stays_distinct_and_never_leaks_historical_url() -> None:
    result = _agent(
        _prediction("delivered_but_missing", 0.95),
        "Please continue at https://t.co/old-link.",
    ).handle("My order says delivered, but I cannot find the package.")
    _assert_safe_escalation(result)
    assert result["intent"]["label"] == "delivered_but_missing"
    assert "STALE_URL" in result["provenance"]["reason_codes"]
    assert "http" not in result["reply"].casefold()


def test_placeholder_and_unsupported_promise_never_reach_final_reply() -> None:
    result = _agent(
        _prediction("damaged_or_defective_item", 0.95),
        "Use [link]. We'll replace the item immediately.",
    ).handle("The item arrived broken.")
    _assert_safe_escalation(result)
    assert {"PLACEHOLDER_OUTPUT", "UNSUPPORTED_PROMISE"} <= set(
        result["provenance"]["reason_codes"]
    )
    assert "[link]" not in result["reply"].casefold()


def test_safe_high_confidence_return_can_still_auto_handle() -> None:
    draft = "Was the item sold or shipped by Amazon, or by a third-party seller?"
    result = _agent(_prediction("return_request", 0.96), draft).handle(
        "I want to return the headphones I received."
    )
    assert result["decision"] == "AUTO_HANDLE"
    assert result["reply"] == draft
    assert result["provenance"]["validation_flags"] == []
    assert result["provenance"]["safety_override_applied"] is False
    assert result["provenance"]["final_response_template"] is None
