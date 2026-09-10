from __future__ import annotations

from collections import Counter


class TrivialBaseline:
    def fit(self, labels: list[str]) -> "TrivialBaseline":
        if not labels:
            raise ValueError("Training labels are required")
        self.most_frequent = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, text: str) -> dict[str, object]:
        return {
            "intent": self.most_frequent,
            "reply": "Thanks for contacting Amazon support. A support specialist will review your request.",
            "decision": "ESCALATE",
            "reason": "The trivial baseline always escalates.",
        }

