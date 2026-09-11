import json
from pathlib import Path

from hiver_support.evaluation.phase3 import analyze_response_outputs


if __name__ == "__main__":
    repaired = Path("data/reports/post_repair/golden_response_outputs_post_repair.json")
    if not repaired.is_file():
        raise FileNotFoundError(
            f"Final post-repair response artifact is required: {repaired}. "
            "Pre-repair fallback is forbidden for headline evaluation."
        )
    result = analyze_response_outputs(repaired, "data/reports/post_repair")
    print(json.dumps({name: value.get("rows", value.get("records", "written")) for name, value in result.items()}, indent=2))
