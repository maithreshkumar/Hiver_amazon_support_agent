from __future__ import annotations

import json

from hiver_support.evaluation.final_intent import run_final_intent_evaluation


if __name__ == "__main__":
    result = run_final_intent_evaluation()
    print(json.dumps({
        "models": {
            name: {key: value[key] for key in ("correct", "incorrect", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}
            for name, value in result["metrics"]["models"].items()
        },
        "configured_threshold": next(row for row in result["confidence"]["thresholds"] if row["threshold"] == 0.72),
    }, indent=2))
