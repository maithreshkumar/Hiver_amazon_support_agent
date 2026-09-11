from __future__ import annotations

import json
import re
from dataclasses import dataclass

from hiver_support.llm.base import LLMProvider
from hiver_support.retrieval.index import RetrievalMatch

PROMPT_VERSION = "amazon-grounded-v2"

PII_REQUEST = "PII_REQUEST"
PII_ECHO = "PII_ECHO"
UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
UNSUPPORTED_PROMISE = "UNSUPPORTED_PROMISE"
STALE_URL = "STALE_URL"
PLACEHOLDER_OUTPUT = "PLACEHOLDER_OUTPUT"
REDUNDANT_QUESTION = "REDUNDANT_QUESTION"
EMPTY_DRAFT = "EMPTY_DRAFT"

_SENSITIVE_REQUEST = re.compile(
    r"\b(?:provide|send|share|confirm|post|enter|include|reply with|tell (?:me|us)|give (?:me|us))\b"
    r".{0,120}\b(?:password|passcode|authentication code|verification code|otp|card(?: number)?|cvv|"
    r"payment details?|bank details?|account (?:number|id|details?)|order (?:number|id|details?)|"
    r"transaction (?:number|id|details?)|tracking (?:number|id)|phone(?: number)?|email(?: address)?|"
    r"delivery address|home address)\b",
    re.IGNORECASE | re.DOTALL,
)
_URL = re.compile(r"(?:https?://|www\.|(?:t|reit)\.ly/|amzn\.to/)", re.IGNORECASE)
_PLACEHOLDER = re.compile(
    r"\[(?:link|name|order(?: number)?|customer|url|insert[^\]]*)\]|"
    r"<(?:url|link|name|order(?: number)?|customer)>|"
    r"\{(?:link|url|name|order(?:_number)?|customer)\}|"
    r"\$\{[^}]+\}|\{\{[^}]+\}\}|\bTODO\b",
    re.IGNORECASE,
)
_UNSUPPORTED_PROMISE = re.compile(
    r"\b(?:we|i)(?:\s+will|'ll|\s+can)\s+(?:"
    r"refund|replace|investigate|call|contact (?:the )?(?:delivery team|carrier|seller)|"
    r"fix|resolve|guarantee|ensure)\b|"
    r"\byou\s+will\s+(?:receive|get)\s+(?:a\s+)?(?:refund|replacement)\b|"
    r"\b(?:refund|replacement)\s+(?:will be|has been)\s+(?:issued|processed|sent)\b|"
    r"\b(?:resolve|fix)\s+(?:this|it|the issue)\s+(?:promptly|immediately|today)\b|"
    r"\b(?:allow|help|enable)\s+us\s+(?:to\s+)?(?:investigate|take (?:appropriate )?action|resolve|fix)\b|"
    r"\btake (?:appropriate )?action on your behalf\b",
    re.IGNORECASE,
)
_UNSUPPORTED_POLICY_CLAIM = re.compile(
    r"\bcancell?ation\s+(?:may|might|will)\s+not\s+be\s+possible\b|"
    r"\b(?:we|amazon)\s+(?:have|has)\s+(?:approved|issued|processed|sent)\b",
    re.IGNORECASE,
)
_PHONE_OR_LONG_NUMBER = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{5,}\d)(?!\w)")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_LABELED_IDENTIFIER = re.compile(
    r"\b(?:account|order|transaction|tracking|customer|authentication|verification)"
    r"(?:\s+(?:number|id|code))?\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{3,})\b",
    re.IGNORECASE,
)
_STREET_ADDRESS = re.compile(
    r"\b\d{1,6}\s+[A-Z0-9][A-Z0-9 .'-]{1,50}\s+"
    r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|court|ct|way)\b",
    re.IGNORECASE,
)
_PAST_DELIVERY = re.compile(
    r"\b(?:yesterday|late|overdue|past due|was supposed to (?:arrive|be delivered)|"
    r"should have (?:arrived|been delivered)|delivery date (?:has )?passed)\b",
    re.IGNORECASE,
)
_ASKS_IF_PAST = re.compile(
    r"\b(?:has|did|is)\b.{0,55}\b(?:delivery|arrival|due)\b.{0,35}\b(?:passed|late|yesterday)\b|"
    r"\bhas the (?:estimated )?delivery date passed\b|"
    r"\b(?:if|whether) (?:this|the) (?:delivery )?date has passed\b",
    re.IGNORECASE,
)
_SECURITY_MESSAGE = re.compile(
    r"\b(?:unauthori[sz]ed|hacked|compromised|account takeover|fraud|suspicious transaction|"
    r"stolen payment|did not make|do not recognize|don't recognize)\b",
    re.IGNORECASE,
)
_SECURITY_DETAIL_REQUEST = re.compile(
    r"\b(?:provide|send|share|confirm|tell (?:me|us)|give (?:me|us))\b.{0,100}"
    r"\b(?:amount|date|purchase|charge|transaction|order|payment|detail|information)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class DraftValidation:
    flags: tuple[str, ...] = ()

    @property
    def safe(self) -> bool:
        return not self.flags

    @property
    def unsupported_claim_detected(self) -> bool:
        return bool(
            {UNSUPPORTED_CLAIM, UNSUPPORTED_PROMISE, STALE_URL, PLACEHOLDER_OUTPUT}
            & set(self.flags)
        )


def _normalise_number(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def _pii_echoed(message: str, draft: str) -> bool:
    """Detect exact customer PII copied into a public-facing draft."""
    message_numbers = {
        _normalise_number(match.group(0))
        for match in _PHONE_OR_LONG_NUMBER.finditer(message)
        if len(_normalise_number(match.group(0))) >= 7
    }
    draft_numbers = {
        _normalise_number(match.group(0))
        for match in _PHONE_OR_LONG_NUMBER.finditer(draft)
        if len(_normalise_number(match.group(0))) >= 7
    }
    if message_numbers & draft_numbers:
        return True

    message_emails = {match.group(0).casefold() for match in _EMAIL.finditer(message)}
    if any(value in draft.casefold() for value in message_emails):
        return True

    identifiers = {match.group(1).casefold() for match in _LABELED_IDENTIFIER.finditer(message)}
    if any(value in draft.casefold() for value in identifiers):
        return True

    addresses = {" ".join(match.group(0).casefold().split()) for match in _STREET_ADDRESS.finditer(message)}
    normalised_draft = " ".join(draft.casefold().split())
    return any(value in normalised_draft for value in addresses)


def _requests_sensitive_data(draft: str) -> bool:
    for match in _SENSITIVE_REQUEST.finditer(draft):
        prefix = draft[max(0, match.start() - 18) : match.start()].casefold()
        if re.search(r"(?:do not|don't|never)\s*$", prefix):
            continue
        return True
    return False


def validate_draft(
    message: str,
    draft: str,
    *,
    model_unsupported_claim: bool = False,
) -> DraftValidation:
    """Apply deterministic, public-channel safety checks to an LLM draft.

    Historical evidence is deliberately not compared lexically. Supported paraphrases are
    allowed; only substantive capability claims and concrete unsafe output are blocked here.
    """
    flags: list[str] = []
    if not draft.strip():
        flags.append(EMPTY_DRAFT)
    if model_unsupported_claim:
        flags.append(UNSUPPORTED_CLAIM)
    if _requests_sensitive_data(draft):
        flags.append(PII_REQUEST)
    if _SECURITY_MESSAGE.search(message) and _SECURITY_DETAIL_REQUEST.search(draft):
        flags.append(PII_REQUEST)
    if _pii_echoed(message, draft):
        flags.append(PII_ECHO)
    if _UNSUPPORTED_PROMISE.search(draft):
        flags.append(UNSUPPORTED_PROMISE)
    if _UNSUPPORTED_POLICY_CLAIM.search(draft):
        flags.append(UNSUPPORTED_CLAIM)
    if _URL.search(draft):
        flags.append(STALE_URL)
    if _PLACEHOLDER.search(draft):
        flags.append(PLACEHOLDER_OUTPUT)
    if _PAST_DELIVERY.search(message) and _ASKS_IF_PAST.search(draft):
        flags.append(REDUNDANT_QUESTION)
    return DraftValidation(tuple(dict.fromkeys(flags)))


@dataclass(frozen=True, slots=True)
class GeneratedReply:
    raw_draft: str
    evidence_thread_ids: list[str]
    grounding_confidence: float
    validation: DraftValidation
    provider: str
    model: str
    prompt_version: str = PROMPT_VERSION

    @property
    def unsupported_claim_detected(self) -> bool:
        return self.validation.unsupported_claim_detected


class GroundedReplyGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generate(self, message: str, intent: str, evidence: list[RetrievalMatch]) -> GeneratedReply:
        evidence_payload = [
            {
                "thread_id": item.thread_id,
                "customer_text": item.customer_text,
                "amazon_response": item.amazon_response,
                "similarity": round(item.similarity, 4),
            }
            for item in evidence
        ]
        system = (
            "You draft cautious Amazon customer-support replies using only supplied historical AmazonHelp evidence. "
            "Historical tweets are behavior examples, not current policy or verified support-link sources. "
            "Never copy URLs, unresolved placeholders, customer PII, account identifiers, or contact details. "
            "Never ask for order, transaction, account, payment, address, phone, email, or authentication details "
            "in this public channel. Never invent order/account facts, refunds, replacements, guarantees, policies, "
            "delivery dates, investigations, calls, or resolution. Do not ask a question the customer already answered. "
            "If evidence is insufficient, draft a brief secure handoff and set low grounding confidence. "
            "Do not copy an old tweet verbatim."
        )
        user = json.dumps(
            {
                "customer_message": message,
                "predicted_intent": intent,
                "historical_evidence": evidence_payload,
            },
            ensure_ascii=False,
        )
        schema = {
            "draft_reply": "string",
            "evidence_thread_ids": ["string"],
            "grounding_confidence": "number 0..1",
            "unsupported_claim_detected": "boolean",
        }
        result = self.provider.structured_generate(system, user, schema)
        allowed_ids = {item.thread_id for item in evidence}
        returned_ids = [
            str(value)
            for value in result.get("evidence_thread_ids", [])
            if str(value) in allowed_ids
        ]
        raw_draft = str(result.get("draft_reply", "")).strip()
        validation = validate_draft(
            message,
            raw_draft,
            model_unsupported_claim=bool(result.get("unsupported_claim_detected", False)),
        )
        return GeneratedReply(
            raw_draft=raw_draft,
            evidence_thread_ids=returned_ids,
            grounding_confidence=min(
                1.0, max(0.0, float(result.get("grounding_confidence", 0.0)))
            ),
            validation=validation,
            provider=self.provider.name,
            model=self.provider.model,
        )
