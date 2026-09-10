from hiver_support.routing.policy import RoutingPolicy


def test_risk_always_escalates_with_reason() -> None:
    result = RoutingPolicy().decide(message="Someone hacked my account", intent="general", intent_confidence=.99, best_similarity=.99, grounding_confidence=.99, unsupported_claim=False)
    assert result.decision == "ESCALATE"
    assert result.reason


def test_strong_low_risk_case_can_auto_handle() -> None:
    result = RoutingPolicy().decide(message="Where is my delivery?", intent="delivery_tracking_or_delay", intent_confidence=.94, best_similarity=.88, grounding_confidence=.85, unsupported_claim=False)
    assert result.decision == "AUTO_HANDLE"
    assert result.reason

