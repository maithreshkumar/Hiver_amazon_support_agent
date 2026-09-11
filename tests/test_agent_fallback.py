from types import SimpleNamespace

from hiver_support.agent import AmazonSupportAgent
from hiver_support.errors import ProviderUnavailable
from hiver_support.intents.predict import IntentPrediction
from hiver_support.routing.policy import RoutingPolicy


class _Classifier:
    def predict(self, text: str) -> IntentPrediction:
        return IntentPrediction("prime_video_issue", 0.9, {"prime_video_issue": 0.9}, "test")


class _UnavailableIndex:
    def search(self, message: str, top_k: int, intent: str):
        raise ProviderUnavailable("Ollama HTTP 500")


def test_provider_failure_returns_safe_escalation() -> None:
    agent = AmazonSupportAgent.__new__(AmazonSupportAgent)
    agent.classifier = _Classifier()
    agent.index = _UnavailableIndex()
    agent.generator = SimpleNamespace(provider=SimpleNamespace(name="ollama", model="qwen3:4b"))
    agent.policy = RoutingPolicy()
    agent.top_k = 5
    result = agent.handle("Prime Video will not play")
    assert result["decision"] == "ESCALATE"
    assert result["provenance"]["safety_override_applied"] is True
    assert "secure support channel" in result["reply"]
    assert "PROVIDER_UNAVAILABLE" in result["provenance"]["reason_codes"]
    assert "WEAK_RETRIEVAL" in result["provenance"]["reason_codes"]
    assert "WEAK_GROUNDING" in result["provenance"]["reason_codes"]
