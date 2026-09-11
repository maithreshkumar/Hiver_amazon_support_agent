from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


LOW_INTENT_CONFIDENCE = "LOW_INTENT_CONFIDENCE"
WEAK_RETRIEVAL = "WEAK_RETRIEVAL"
WEAK_GROUNDING = "WEAK_GROUNDING"
AMBIGUOUS_INTENT = "AMBIGUOUS_INTENT"
SECURITY_RISK = "SECURITY_RISK"
PAYMENT_RISK = "PAYMENT_RISK"
SENSITIVE_RISK = "SENSITIVE_RISK"
ACCOUNT_SPECIFIC_INVESTIGATION = "ACCOUNT_SPECIFIC_INVESTIGATION"
PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"

_SECURITY_TERMS = (
    "unauthorized",
    "unauthorised",
    "hacked",
    "compromised",
    "account takeover",
    "fraud",
    "suspicious transaction",
    "stolen payment",
    "did not make",
    "don't recognize",
    "do not recognize",
)
_PAYMENT_TERMS = ("payment dispute", "chargeback", "charged twice", "refund", "charge")
_CONDUCT_ACTORS = ("driver", "courier", "delivery person", "delivery agent", "staff")
_CONDUCT_ISSUES = (
    "rude",
    "not delivering",
    "refused",
    "complaint",
    "conduct",
    "threat",
    "harass",
)
_INVOLUNTARY_CANCELLATION = (
    "amazon canceled my order",
    "amazon cancelled my order",
    "you canceled my order",
    "you cancelled my order",
    "order was canceled",
    "order was cancelled",
    "order got canceled",
    "order got cancelled",
)

_REASON_PARTS = {
    SECURITY_RISK: "the message may involve account compromise, fraud, or unauthorized activity",
    PAYMENT_RISK: "the payment or refund issue requires secure account-specific review",
    SENSITIVE_RISK: "the message contains a legal, physical-safety, or other sensitive risk signal",
    ACCOUNT_SPECIFIC_INVESTIGATION: "the issue requires account- or order-specific investigation",
    "PII_REQUEST": "the generated draft requested sensitive information that must not be posted publicly",
    "PII_ECHO": "the generated draft repeated customer PII",
    "UNSUPPORTED_CLAIM": "the generated draft contains a claim not supported by the evidence",
    "UNSUPPORTED_PROMISE": "the generated draft promises an action the system is not authorized to perform",
    "STALE_URL": "the generated draft contains an unverified historical URL",
    "PLACEHOLDER_OUTPUT": "the generated draft contains an unresolved placeholder",
    "REDUNDANT_QUESTION": "the generated draft asks for information already supplied by the customer",
    "EMPTY_DRAFT": "the response generator returned no usable draft",
    AMBIGUOUS_INTENT: "the classifier found competing or ambiguous intents",
    LOW_INTENT_CONFIDENCE: "the intent confidence is below the configured threshold",
    WEAK_RETRIEVAL: "the retrieved historical evidence is below the configured similarity threshold",
    WEAK_GROUNDING: "the draft grounding confidence is below the configured threshold",
    PROVIDER_UNAVAILABLE: "the configured local model provider is unavailable",
}


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    decision: str
    reason: str
    signals: list[str] = field(default_factory=list)


def escalation_reason(signals: Iterable[str]) -> str:
    ordered = list(dict.fromkeys(signals))
    parts = [_REASON_PARTS.get(signal, signal.casefold().replace("_", " ")) for signal in ordered]
    return "Escalation is required because " + "; ".join(parts) + "."


class RoutingPolicy:
    def __init__(self, path: str | Path = "configs/escalation_policy.yaml") -> None:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.version = str(config.get("version", "unknown"))
        self.min_intent = float(config["confidence"]["minimum_intent"])
        self.min_retrieval = float(config["confidence"]["minimum_retrieval_similarity"])
        self.risk_intents = set(config["risk_rules"]["always_escalate"])
        self.risk_signals = [str(value).casefold() for value in config["risk_rules"]["text_signals"]]

    def _risk_code(self, message: str, intent: str) -> str | None:
        lowered = message.casefold()
        if intent == "account_access_or_security" or any(term in lowered for term in _SECURITY_TERMS):
            return SECURITY_RISK
        if intent == "refund_or_payment_issue" or any(term in lowered for term in _PAYMENT_TERMS):
            return PAYMENT_RISK
        if intent in self.risk_intents or any(signal in lowered for signal in self.risk_signals):
            return SENSITIVE_RISK
        return None

    @staticmethod
    def _requires_account_specific_investigation(message: str, intent: str) -> bool:
        lowered = message.casefold()
        if intent == "delivered_but_missing":
            return True
        conduct_complaint = any(actor in lowered for actor in _CONDUCT_ACTORS) and any(
            issue in lowered for issue in _CONDUCT_ISSUES
        )
        involuntary_cancellation = any(value in lowered for value in _INVOLUNTARY_CANCELLATION)
        return conduct_complaint or involuntary_cancellation

    def decide(
        self,
        *,
        message: str,
        intent: str,
        intent_confidence: float,
        best_similarity: float,
        grounding_confidence: float,
        unsupported_claim: bool,
        ambiguity_score: float = 0.0,
        safety_flags: Iterable[str] = (),
    ) -> RoutingDecision:
        signals: list[str] = []
        risk_code = self._risk_code(message, intent)
        if risk_code:
            signals.append(risk_code)
        if self._requires_account_specific_investigation(message, intent):
            signals.append(ACCOUNT_SPECIFIC_INVESTIGATION)
        signals.extend(str(flag) for flag in safety_flags)
        if unsupported_claim and not ({"UNSUPPORTED_CLAIM", "UNSUPPORTED_PROMISE"} & set(signals)):
            signals.append("UNSUPPORTED_CLAIM")
        if ambiguity_score >= 0.82:
            signals.append(AMBIGUOUS_INTENT)
        if intent_confidence < self.min_intent:
            signals.append(LOW_INTENT_CONFIDENCE)
        if best_similarity < self.min_retrieval:
            signals.append(WEAK_RETRIEVAL)
        if grounding_confidence < 0.60:
            signals.append(WEAK_GROUNDING)
        signals = list(dict.fromkeys(signals))
        if signals:
            return RoutingDecision("ESCALATE", escalation_reason(signals), signals)
        return RoutingDecision(
            "AUTO_HANDLE",
            "This is a low-risk, high-confidence request with strong retrieval and grounding evidence.",
            ["HIGH_INTENT_CONFIDENCE", "STRONG_RETRIEVAL", "STRONG_GROUNDING"],
        )

    def provider_failure_decision(
        self,
        *,
        message: str,
        intent: str,
        intent_confidence: float,
        ambiguity_score: float = 0.0,
    ) -> RoutingDecision:
        """Preserve every active risk signal when retrieval or generation is unavailable."""
        base = self.decide(
            message=message,
            intent=intent,
            intent_confidence=intent_confidence,
            best_similarity=0.0,
            grounding_confidence=0.0,
            unsupported_claim=False,
            ambiguity_score=ambiguity_score,
        )
        signals = list(dict.fromkeys([*base.signals, PROVIDER_UNAVAILABLE]))
        return RoutingDecision("ESCALATE", escalation_reason(signals), signals)
