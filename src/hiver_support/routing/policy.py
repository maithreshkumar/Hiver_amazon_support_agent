from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    decision: str
    reason: str
    signals: list[str] = field(default_factory=list)


class RoutingPolicy:
    def __init__(self, path: str | Path = "configs/escalation_policy.yaml") -> None:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.version = str(config.get("version", "unknown"))
        self.min_intent = float(config["confidence"]["minimum_intent"])
        self.min_retrieval = float(config["confidence"]["minimum_retrieval_similarity"])
        self.risk_intents = set(config["risk_rules"]["always_escalate"])
        self.risk_signals = [str(value).casefold() for value in config["risk_rules"]["text_signals"]]

    def decide(self, *, message: str, intent: str, intent_confidence: float, best_similarity: float, grounding_confidence: float, unsupported_claim: bool, ambiguity_score: float = 0.0) -> RoutingDecision:
        lowered = message.casefold()
        if intent in self.risk_intents or any(signal in lowered for signal in self.risk_signals):
            return RoutingDecision("ESCALATE", "This message may involve security, fraud, payment, legal, or safety risk requiring sensitive account-specific investigation.", ["forced_risk"])
        if unsupported_claim:
            return RoutingDecision("ESCALATE", "The draft may contain a claim that is not supported by the retrieved historical evidence.", ["unsupported_claim"])
        if ambiguity_score >= 0.82:
            return RoutingDecision("ESCALATE", "The message appears to contain ambiguous or competing issues, so one automated intent would be unsafe.", ["intent_ambiguity"])
        if intent_confidence < self.min_intent:
            return RoutingDecision("ESCALATE", "The Amazon-specific intent classifier is not confident enough to select a reliable handling path.", ["low_intent_confidence"])
        if best_similarity < self.min_retrieval:
            return RoutingDecision("ESCALATE", "No sufficiently similar historical AmazonHelp handling pattern was retrieved to support a reliable automated response.", ["low_retrieval_evidence"])
        if grounding_confidence < 0.60:
            return RoutingDecision("ESCALATE", "The response generator could not ground the draft strongly enough in historical AmazonHelp evidence.", ["low_grounding"])
        return RoutingDecision("AUTO_HANDLE", "This is a low-risk, high-confidence request with a strong and consistent historical AmazonHelp handling pattern.", ["high_confidence", "strong_evidence"])

