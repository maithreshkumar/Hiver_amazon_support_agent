import pytest

from hiver_support.generation.reply import (
    PII_ECHO,
    PII_REQUEST,
    PLACEHOLDER_OUTPUT,
    REDUNDANT_QUESTION,
    STALE_URL,
    UNSUPPORTED_PROMISE,
    GroundedReplyGenerator,
    validate_draft,
)
from hiver_support.llm.base import LLMProvider
from hiver_support.retrieval.index import RetrievalMatch


class FakeProvider(LLMProvider):
    name, model = "fake", "fake-v1"
    draft = "Have you received any recent tracking updates?"
    model_flag = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "unused"

    def structured_generate(self, system_prompt: str, user_prompt: str, schema=None):
        assert "historical_evidence" in user_prompt
        assert "Never copy URLs" in system_prompt
        return {
            "draft_reply": self.draft,
            "evidence_thread_ids": ["t1", "invented"],
            "grounding_confidence": 0.8,
            "unsupported_claim_detected": self.model_flag,
        }


def _generate(provider: FakeProvider, message: str = "Where is it?"):
    evidence = [RetrievalMatch("t1", "Where is it?", "Please contact us.", 0.8)]
    return GroundedReplyGenerator(provider).generate(message, "delivery", evidence)


def test_generation_preserves_raw_draft_and_filters_evidence_ids() -> None:
    result = _generate(FakeProvider())
    assert result.evidence_thread_ids == ["t1"]
    assert result.raw_draft == FakeProvider.draft
    assert result.validation.safe
    assert 0 <= result.grounding_confidence <= 1


@pytest.mark.parametrize(
    ("draft", "flag"),
    [
        ("Please provide your order number and delivery address.", PII_REQUEST),
        ("Continue at https://t.co/old-link.", STALE_URL),
        ("Please use [link] to continue.", PLACEHOLDER_OUTPUT),
        ("We'll contact the delivery team directly and resolve this today.", UNSUPPORTED_PROMISE),
    ],
)
def test_deterministic_draft_violations_are_flagged(draft: str, flag: str) -> None:
    validation = validate_draft("Please help", draft)
    assert flag in validation.flags


def test_customer_phone_email_and_identifier_are_not_echoed() -> None:
    message = "Call me at +1 415-555-0199; email me@example.com about order AB12-XY99."
    draft = "We'll use +1 415-555-0199 and me@example.com for order AB12-XY99."
    assert PII_ECHO in validate_draft(message, draft).flags


def test_negative_privacy_instruction_is_not_mistaken_for_a_request() -> None:
    draft = "Please use secure support. Do not post account or payment details publicly."
    assert PII_REQUEST not in validate_draft("My account was hacked", draft).flags


def test_supported_paraphrase_is_not_rejected_for_wording_difference() -> None:
    validation = validate_draft(
        "My order has not arrived.",
        "Have you received any recent tracking updates?",
    )
    assert validation.safe


def test_redundant_delivery_question_is_rejected() -> None:
    validation = validate_draft(
        "It was supposed to arrive yesterday.",
        "Has the estimated delivery date passed?",
    )
    assert REDUNDANT_QUESTION in validation.flags


def test_pronoun_based_redundant_delivery_question_is_rejected() -> None:
    validation = validate_draft(
        "It was supposed to arrive yesterday.",
        "Could you let us know if this date has passed?",
    )
    assert REDUNDANT_QUESTION in validation.flags


def test_security_case_cannot_request_charge_amount_or_date() -> None:
    validation = validate_draft(
        "I do not recognize this charge on my account.",
        "Please provide the amount of the charge and the date it was applied.",
    )
    assert PII_REQUEST in validation.flags


def test_indirect_investigation_promise_is_rejected() -> None:
    validation = validate_draft(
        "The seller is not replying.",
        "Please provide order details. This will allow us to investigate and take action on your behalf.",
    )
    assert UNSUPPORTED_PROMISE in validation.flags


def test_investigation_promise_without_to_is_rejected() -> None:
    validation = validate_draft(
        "The courier did not deliver my package.",
        "More details will help us investigate and resolve it promptly.",
    )
    assert UNSUPPORTED_PROMISE in validation.flags


def test_unsupported_cancellation_policy_claim_is_rejected() -> None:
    validation = validate_draft(
        "Amazon canceled my order.",
        "If it shipped, cancellation may not be possible.",
    )
    assert "UNSUPPORTED_CLAIM" in validation.flags
