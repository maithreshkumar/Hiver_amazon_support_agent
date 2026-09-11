from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from hiver_support.agent import AmazonSupportAgent
from hiver_support.generation.reply import validate_draft
from hiver_support.intents.predict import IntentClassifier
from hiver_support.intents.taxonomy import load_approved_taxonomy


CASES = [
    ("A_delivery_late", "My package is three days late. Where is it?", "delivery_tracking_or_delay"),
    ("C_delivery_yesterday", "Where is my order? It was supposed to arrive yesterday.", "delivery_tracking_or_delay"),
    ("D_delivered_missing", "My order says delivered, but I cannot find the package.", "delivered_but_missing"),
    ("E_simple_return", "I want to return the headphones I received.", "return_request"),
    ("damaged_item", "The coffee maker arrived cracked and does not work.", "damaged_or_defective_item"),
    ("wrong_item", "I ordered blue shoes but received a red shirt.", "wrong_or_incomplete_item"),
    ("refund", "My refund has not appeared in my bank account.", "refund_or_payment_issue"),
    ("unauthorized_charge", "I do not recognize this charge on my account.", "refund_or_payment_issue"),
    ("B_hacked_account", "My account was hacked and there are purchases I did not make.", "account_access_or_security"),
    ("prime_membership", "How do I cancel my Prime membership?", "prime_membership"),
    ("prime_video", "Prime Video shows an error whenever I play a movie.", "prime_video_issue"),
    ("seller_problem", "The marketplace seller has ignored all my messages.", "seller_or_marketplace_issue"),
    ("vague", "This is not what I expected. Please help.", "general_or_context_missing"),
    ("rude_input", "This delivery is damn ridiculous and your service is useless.", "delivery_tracking_or_delay"),
    ("staff_conduct", "The delivery driver was rude to me.", "general_or_context_missing"),
    (
        "F_hacked_phone",
        "My account was hacked. Call me at +1 415-555-0199 about purchases I did not make.",
        "account_access_or_security",
    ),
]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen classifier or complete repaired runtime regression set")
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--output", default="data/reports/runtime_behavior_regression.json")
    args = parser.parse_args()

    if args.classification_only:
        labels = {item.name for item in load_approved_taxonomy()}
        classifier = IntentClassifier("models/intent", labels)
        predictions = classifier.predict_batch([message for _, message, _ in CASES])
        records = [
            {
                "case": name,
                "message": message,
                "expected_intent": expected,
                "predicted_intent": prediction.label,
                "confidence": prediction.confidence,
                "ambiguity_score": prediction.ambiguity_score,
                "matches_expected": prediction.label == expected,
            }
            for (name, message, expected), prediction in zip(CASES, predictions, strict=True)
        ]
    else:
        agent = AmazonSupportAgent()
        records = []
        for position, (name, message, expected) in enumerate(CASES, 1):
            result = agent.handle(message)
            final_validation = validate_draft(message, str(result["reply"]))
            if not final_validation.safe:
                raise RuntimeError(f"{name} produced unsafe final output: {final_validation.flags}")
            records.append(
                {
                    "case": name,
                    "message": message,
                    "expected_intent": expected,
                    "predicted_intent": result["intent"]["label"],
                    "confidence": result["intent"]["confidence"],
                    "ambiguity_score": result["intent"]["ambiguity_score"],
                    "matches_expected": result["intent"]["label"] == expected,
                    "top_retrieval_similarities": [
                        item["similarity"] for item in result["evidence"][:3]
                    ],
                    "raw_draft": result["raw_draft"],
                    "safety_flags": result["provenance"]["validation_flags"],
                    "decision": result["decision"],
                    "reason_codes": result["provenance"]["reason_codes"],
                    "reason": result["reason"],
                    "final_reply": result["reply"],
                    "safety_override_applied": result["provenance"]["safety_override_applied"],
                    "final_response_template": result["provenance"]["final_response_template"],
                    "provider_error": result["provenance"].get("provider_error"),
                    "final_reply_safe": True,
                }
            )
            print(f"Validated {position}/{len(CASES)}: {name}", flush=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "classification_only": args.classification_only,
        "cases": len(records),
        "records": records,
    }
    _atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
