from hiver_support.generation.reply import GroundedReplyGenerator
from hiver_support.llm.base import LLMProvider
from hiver_support.retrieval.index import RetrievalMatch


class FakeProvider(LLMProvider):
    name, model = "fake", "fake-v1"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "unused"

    def structured_generate(self, system_prompt: str, user_prompt: str, schema=None):
        assert "historical_evidence" in user_prompt
        return {"draft_reply": "Please contact support securely so this can be checked.", "evidence_thread_ids": ["t1", "invented"], "grounding_confidence": .8, "unsupported_claim_detected": False}


def test_generation_passes_and_filters_evidence_ids() -> None:
    evidence = [RetrievalMatch("t1", "Where is it?", "Please contact us.", .8)]
    result = GroundedReplyGenerator(FakeProvider()).generate("Where is it?", "delivery", evidence)
    assert result.evidence_thread_ids == ["t1"]
    assert 0 <= result.grounding_confidence <= 1
    assert result.safety_override_applied is False


class UnsafeFakeProvider(FakeProvider):
    def structured_generate(self, system_prompt: str, user_prompt: str, schema=None):
        return {"draft_reply": "Please provide your order number and delivery address.", "evidence_thread_ids": ["t1"], "grounding_confidence": .9, "unsupported_claim_detected": False}


def test_sensitive_request_triggers_safe_override() -> None:
    evidence = [RetrievalMatch("t1", "Where is it?", "Please contact us.", .8)]
    result = GroundedReplyGenerator(UnsafeFakeProvider()).generate("Where is it?", "delivery", evidence)
    assert result.safety_override_applied is True
    assert result.unsupported_claim_detected is True
    assert "do not post" in result.draft_reply.lower()


class StaleLinkProvider(FakeProvider):
    def structured_generate(self, system_prompt: str, user_prompt: str, schema=None):
        return {"draft_reply": "Request a refund through https://t.co/old-link.", "evidence_thread_ids": ["t1"], "grounding_confidence": .9, "unsupported_claim_detected": False}


def test_stale_historical_link_or_transaction_instruction_triggers_override() -> None:
    evidence = [RetrievalMatch("t1", "Wrong item", "Use this old link.", .8)]
    result = GroundedReplyGenerator(StaleLinkProvider()).generate(
        "Wrong item", "wrong_or_incomplete_item", evidence
    )
    assert result.safety_override_applied is True
    assert result.unsupported_claim_detected is True
