import pytest

from hiver_support.routing.policy import RoutingPolicy


def test_risk_always_escalates_with_reason() -> None:
    result = RoutingPolicy().decide(message="Someone hacked my account", intent="general", intent_confidence=.99, best_similarity=.99, grounding_confidence=.99, unsupported_claim=False)
    assert result.decision == "ESCALATE"
    assert result.reason
    assert "SECURITY_RISK" in result.signals


def test_strong_low_risk_case_can_auto_handle() -> None:
    result = RoutingPolicy().decide(message="Where is my delivery?", intent="delivery_tracking_or_delay", intent_confidence=.94, best_similarity=.88, grounding_confidence=.85, unsupported_claim=False)
    assert result.decision == "AUTO_HANDLE"
    assert result.reason


def test_all_material_escalation_signals_are_preserved() -> None:
    result = RoutingPolicy().decide(
        message="My account was hacked and there is an unauthorized purchase",
        intent="account_access_or_security",
        intent_confidence=0.71,
        best_similarity=0.50,
        grounding_confidence=0.40,
        unsupported_claim=True,
        safety_flags=["PII_REQUEST"],
    )
    assert result.decision == "ESCALATE"
    assert {
        "SECURITY_RISK",
        "PII_REQUEST",
        "UNSUPPORTED_CLAIM",
        "LOW_INTENT_CONFIDENCE",
        "WEAK_RETRIEVAL",
        "WEAK_GROUNDING",
    } <= set(result.signals)
    assert "below the configured threshold" in result.reason
    assert "sensitive information" in result.reason


def test_delivered_but_missing_requires_order_specific_investigation() -> None:
    result = RoutingPolicy().decide(
        message="My order says delivered, but I cannot find it.",
        intent="delivered_but_missing",
        intent_confidence=0.95,
        best_similarity=0.90,
        grounding_confidence=0.90,
        unsupported_claim=False,
    )
    assert result.decision == "ESCALATE"
    assert "ACCOUNT_SPECIFIC_INVESTIGATION" in result.signals


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("The delivery driver was rude to me.", "delivery_tracking_or_delay"),
        ("Amazon canceled my order when I only requested an update.", "order_cancellation"),
    ],
)
def test_conduct_and_involuntary_cancellation_escalate(message: str, intent: str) -> None:
    result = RoutingPolicy().decide(
        message=message,
        intent=intent,
        intent_confidence=0.95,
        best_similarity=0.90,
        grounding_confidence=0.90,
        unsupported_claim=False,
    )
    assert result.decision == "ESCALATE"
    assert "ACCOUNT_SPECIFIC_INVESTIGATION" in result.signals


def test_provider_failure_preserves_security_and_confidence_signals() -> None:
    result = RoutingPolicy().provider_failure_decision(
        message="My account was hacked.",
        intent="account_access_or_security",
        intent_confidence=0.60,
        ambiguity_score=0.90,
    )
    assert {
        "SECURITY_RISK",
        "LOW_INTENT_CONFIDENCE",
        "AMBIGUOUS_INTENT",
        "WEAK_RETRIEVAL",
        "WEAK_GROUNDING",
        "PROVIDER_UNAVAILABLE",
    } <= set(result.signals)
