from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from hiver_support.config import load_yaml
from hiver_support.errors import ProviderUnavailable
from hiver_support.generation.reply import DraftValidation, GroundedReplyGenerator
from hiver_support.intents.predict import IntentClassifier
from hiver_support.intents.taxonomy import load_approved_taxonomy
from hiver_support.llm.factory import create_provider
from hiver_support.retrieval.embedder import OllamaEmbedder
from hiver_support.retrieval.index import LocalVectorIndex
from hiver_support.routing.final_response import assert_response_invariants, build_final_response
from hiver_support.routing.policy import RoutingPolicy


class AmazonSupportAgent:
    def __init__(self, runtime_path: str | Path = "configs/runtime.yaml", retrieval_path: str | Path = "configs/retrieval.yaml") -> None:
        runtime, retrieval = load_yaml(runtime_path), load_yaml(retrieval_path)
        taxonomy = load_approved_taxonomy()
        labels = {intent.name for intent in taxonomy}
        self.classifier = IntentClassifier(runtime["intent"]["model_path"], labels)
        embedder = OllamaEmbedder(retrieval["model"], retrieval["base_url"])
        self.index = LocalVectorIndex(retrieval["index_dir"], embedder)
        self.generator = GroundedReplyGenerator(create_provider(runtime["llm"]))
        self.policy = RoutingPolicy(runtime["routing"]["policy_path"])
        self.top_k = int(retrieval["top_k"])

    def handle(self, message: str) -> dict[str, object]:
        if not message.strip():
            raise ValueError("Customer message cannot be empty")
        prediction = self.classifier.predict(message)
        try:
            matches = self.index.search(message, self.top_k, prediction.label)
            generated = self.generator.generate(message, prediction.label, matches)
        except ProviderUnavailable as exc:
            provider = getattr(self.generator.provider, "name", "unknown")
            model = getattr(self.generator.provider, "model", "unknown")
            final = build_final_response(
                message=message,
                intent=prediction.label,
                raw_draft="",
                validation=DraftValidation(),
                routing=self.policy.provider_failure_decision(
                    message=message,
                    intent=prediction.label,
                    intent_confidence=prediction.confidence,
                    ambiguity_score=prediction.ambiguity_score,
                ),
            )
            response = {
                "message": message,
                "intent": asdict(prediction),
                "raw_draft": "",
                "reply": final.reply,
                "decision": final.decision,
                "reason": final.reason,
                "evidence": [],
                "provenance": {
                    "provider": provider,
                    "model": model,
                    "intent_model_version": prediction.model_version,
                    "prompt_version": "provider-failure-fallback-v1",
                    "routing_policy_version": self.policy.version,
                    "grounding_confidence": 0.0,
                    "unsupported_claim_detected": False,
                    "validation_flags": [],
                    "reason_codes": final.reason_codes,
                    "final_response_template": final.template,
                    "safety_override_applied": final.safety_override_applied,
                    "provider_error": str(exc),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            }
            assert_response_invariants(response)
            return response
        best_similarity = max((item.similarity for item in matches), default=0.0)
        decision = self.policy.decide(
            message=message, intent=prediction.label, intent_confidence=prediction.confidence,
            best_similarity=best_similarity, grounding_confidence=generated.grounding_confidence,
            unsupported_claim=generated.unsupported_claim_detected, ambiguity_score=prediction.ambiguity_score,
            safety_flags=generated.validation.flags,
        )
        final = build_final_response(
            message=message,
            intent=prediction.label,
            raw_draft=generated.raw_draft,
            validation=generated.validation,
            routing=decision,
        )
        response = {
            "message": message,
            "intent": asdict(prediction),
            "raw_draft": generated.raw_draft,
            "reply": final.reply,
            "decision": final.decision,
            "reason": final.reason,
            "evidence": [asdict(item) for item in matches],
            "provenance": {
                "provider": generated.provider, "model": generated.model,
                "intent_model_version": prediction.model_version,
                "prompt_version": generated.prompt_version,
                "grounding_confidence": generated.grounding_confidence,
                "unsupported_claim_detected": generated.unsupported_claim_detected,
                "routing_policy_version": self.policy.version,
                "validation_flags": list(generated.validation.flags),
                "reason_codes": final.reason_codes,
                "final_response_template": final.template,
                "safety_override_applied": final.safety_override_applied,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }
        assert_response_invariants(response)
        return response
