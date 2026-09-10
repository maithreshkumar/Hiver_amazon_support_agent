from __future__ import annotations

import json

from hiver_support.evaluation.judge import run_reply_judge


if __name__ == "__main__":
    result = run_reply_judge()
    print(json.dumps({"rows": len(result["records"]), "mean_scores": result["mean_scores"]}, indent=2))
