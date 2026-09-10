from __future__ import annotations

import json
import re
from dataclasses import dataclass

from hiver_support.llm.base import LLMProvider
from hiver_support.retrieval.index import RetrievalMatch

PROMPT_VERSION = "amazon-grounded-v1"


def _unsafe_or_unsupported_draft(draft: str) -> bool:
    """Conservative deterministic backstop for public-channel or policy claims."""
    has_secure_channel = bool(
        re.search(r"\b(?:secure|private|direct message|dm|official support channel)\b", draft, re.IGNORECASE)
    )
    direct_sensitive_request = bool(
        re.search(
            r"\b(?:provide|send|share|confirm|tell me)\b.{0,90}"
            r"\b(?:password|card|account|order number|address|payment|charge)\b",
            draft,
            re.IGNORECASE,
        )
    ) and not has_secure_channel
    stale_link = bool(re.search(r"https?://|reit\.ly|amzn\.to", draft, re.IGNORECASE))
    ungrounded_transaction = bool(
        re.search(
            r"\b(?:you can|please|eligible|would you like)\b.{0,80}"
            r"\b(?:refund|replacement|return)\b|"
            r"\b(?:refund|replacement)\b.{0,80}\b(?:request|order details|processed|issued)\b",
            draft,
            re.IGNORECASE,
        )
    )
    hard_claim = bool(
        re.search(
            r"\b(?:guarantee|ensure).{0,30}\bdeliver|\brefund (?:has been|will be)\b",
            draft,
            re.IGNORECASE,
        )
    )
    credential_request = bool(re.search(r"\b(?:password|full card|card number)\b", draft, re.IGNORECASE))
    return direct_sensitive_request or stale_link or ungrounded_transaction or hard_claim or credential_request


@dataclass(frozen=True, slots=True)
class GeneratedReply:
    raw_draft: str
    draft_reply: str
    evidence_thread_ids: list[str]
    grounding_confidence: float
    unsupported_claim_detected: bool
    safety_override_applied: bool
    provider: str
    model: str
    prompt_version: str = PROMPT_VERSION


class GroundedReplyGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generate(self, message: str, intent: str, evidence: list[RetrievalMatch]) -> GeneratedReply:
        evidence_payload = [
            {"thread_id": item.thread_id, "customer_text": item.customer_text, "amazon_response": item.amazon_response, "similarity": round(item.similarity, 4)}
            for item in evidence
        ]
        system = (
            "You draft cautious Amazon customer-support replies using only supplied historical AmazonHelp evidence. "
            "Historical tweets are behavior examples, not current official policy and not proof of resolution. "
            "Never invent order/account facts, refunds, guarantees, policies, delivery dates, or resolution. "
            "Never request passwords or full payment details. If evidence is insufficient, produce a brief safe handoff reply "
            "and set low grounding confidence. Do not copy an old tweet verbatim."
        )
        user = json.dumps({"customer_message": message, "predicted_intent": intent, "historical_evidence": evidence_payload}, ensure_ascii=False)
        schema = {"draft_reply": "string", "evidence_thread_ids": ["string"], "grounding_confidence": "number 0..1", "unsupported_claim_detected": "boolean"}
        result = self.provider.structured_generate(system, user, schema)
        allowed_ids = {item.thread_id for item in evidence}
        returned_ids = [str(value) for value in result.get("evidence_thread_ids", []) if str(value) in allowed_ids]
        draft = str(result.get("draft_reply", "")).strip()
        raw_draft = draft
        model_flag = bool(result.get("unsupported_claim_detected", False))
        unsafe_request = _unsafe_or_unsupported_draft(draft)
        safety_override = model_flag or unsafe_request or not draft
        if safety_override:
            draft = (
                "I'm sorry this needs further investigation. Please contact Amazon through a secure "
                "support channel, and do not post order, address, account, or payment details publicly."
            )
        return GeneratedReply(
            raw_draft=raw_draft,
            draft_reply=draft,
            evidence_thread_ids=returned_ids,
            grounding_confidence=min(1.0, max(0.0, float(result.get("grounding_confidence", 0.0)))),
            unsupported_claim_detected=model_flag or unsafe_request,
            safety_override_applied=safety_override,
            provider=self.provider.name,
            model=self.provider.model,
        )
