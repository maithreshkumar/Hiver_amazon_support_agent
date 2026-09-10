from __future__ import annotations

import json

from hiver_support.evaluation.phase3 import analyze_response_outputs


if __name__ == "__main__":
    result = analyze_response_outputs()
    print(json.dumps({name: value.get("rows", value.get("records", "written")) for name, value in result.items()}, indent=2))
