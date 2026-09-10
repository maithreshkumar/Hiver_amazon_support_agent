from __future__ import annotations

import json

from hiver_support.evaluation.development import evaluate_development


if __name__ == "__main__":
    result = evaluate_development()
    compact = {
        "rows": result["rows"],
        "support": result["support"],
        "models": {
            name: {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
            for name, metrics in result["models"].items()
        },
        "confidence_diagnostic": result["confidence_diagnostic"],
    }
    print(json.dumps(compact, indent=2))
