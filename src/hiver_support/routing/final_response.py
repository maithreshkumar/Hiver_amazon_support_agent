from __future__ import annotations

from dataclasses import dataclass

from hiver_support.generation.reply import DraftValidation, validate_draft
from hiver_support.routing.policy import (
    PAYMENT_RISK,
    SECURITY_RISK,
    RoutingDecision,
    escalation_reason,
)

GENERAL_ESCALATION = (
    "I'm sorry this needs further investigation. Please contact Amazon through a secure "
    "support channel, and do not post order, address, account, or payment details publicly."
)
SECURITY_ESCALATION = (
    "I'm sorry you're dealing with this. Because this may involve account security or "
    "unauthorized activity, please contact Amazon through a secure support channel. Do not "
    "post account, order, payment, transaction, or authentication details publicly."
)
DELIVERY_ESCALATION = (
    "I'm sorry your order hasn't arrived as expected. This appears to require order-specific "
    "investigation, so please contact Amazon through a secure support channel and avoid posting "
    "order or address details publicly."
)
PAYMENT_ESCALATION = (
    "I'm sorry you're having trouble with this payment or refund. Because it requires "
    "account-specific review, please contact Amazon through a secure support channel and do not "
    "post payment or account details publicly."
)


@dataclass(frozen=True, slots=True)
class FinalResponse:
    reply: str
    decision: str
    reason: str
    reason_codes: list[str]
    safety_override_applied: bool
    template: str | None


def _escalation_template(intent: str, signals: list[str]) -> tuple[str, str]:
    if SECURITY_RISK in signals or intent == "account_access_or_security":
        return SECURITY_ESCALATION, "security"
    if PAYMENT_RISK in signals or intent == "refund_or_payment_issue":
        return PAYMENT_ESCALATION, "payment"
    if intent in {"delivery_tracking_or_delay", "delivered_but_missing"}:
        return DELIVERY_ESCALATION, "delivery"
    return GENERAL_ESCALATION, "general"


def build_final_response(
    *,
    message: str,
    intent: str,
    raw_draft: str,
    validation: DraftValidation,
    routing: RoutingDecision,
) -> FinalResponse:
    """Build the sole user-facing response after safety and routing are final."""
    decision = routing.decision
    signals = list(dict.fromkeys([*routing.signals, *validation.flags]))
    reason = routing.reason

    # Defensive backstop: a caller cannot auto-handle a draft that failed validation.
    if validation.flags and decision != "ESCALATE":
        decision = "ESCALATE"
        reason = escalation_reason(signals)

    if decision == "ESCALATE":
        reply, template = _escalation_template(intent, signals)
        override = reply.strip() != raw_draft.strip()
    elif decision == "AUTO_HANDLE":
        reply, template, override = raw_draft.strip(), None, False
    else:
        raise ValueError(f"Unsupported routing decision: {decision}")

    final_validation = validate_draft(message, reply)
    if not final_validation.safe:
        raise RuntimeError(
            "Canonical final-response builder produced an unsafe response: "
            + ", ".join(final_validation.flags)
        )
    if decision == "AUTO_HANDLE" and not reply:
        raise RuntimeError("AUTO_HANDLE cannot return an empty response")
    return FinalResponse(
        reply=reply,
        decision=decision,
        reason=reason,
        reason_codes=signals,
        safety_override_applied=override,
        template=template,
    )


def assert_response_invariants(response: dict[str, object]) -> None:
    """Fail closed when returned fields contradict the final decision."""
    decision = str(response.get("decision", ""))
    reply = str(response.get("reply", ""))
    raw_draft = str(response.get("raw_draft", ""))
    provenance = dict(response.get("provenance", {}))
    flags = [str(value) for value in provenance.get("validation_flags", [])]

    if decision not in {"AUTO_HANDLE", "ESCALATE"}:
        raise RuntimeError("Response contains an invalid routing decision")
    if not reply:
        raise RuntimeError("Response contains an empty final reply")
    if decision == "AUTO_HANDLE" and flags:
        raise RuntimeError("Unsafe draft was marked AUTO_HANDLE")
    if flags and reply == raw_draft:
        raise RuntimeError("A draft with safety violations was returned unchanged")
    if bool(provenance.get("unsupported_claim_detected")) and reply == raw_draft:
        raise RuntimeError("An unsupported raw draft was returned unchanged")
    if decision == "ESCALATE" and provenance.get("final_response_template") is None:
        raise RuntimeError("Escalation response did not use a canonical safe template")
