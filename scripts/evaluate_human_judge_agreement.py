from __future__ import annotations

import json

from hiver_support.evaluation.reply_agreement import calculate_human_judge_agreement


if __name__ == "__main__":
    result = calculate_human_judge_agreement()
    print(json.dumps({
        "rows": result["human_rating_rows"],
        "paired_dimension_ratings": result["paired_dimension_ratings"],
        "overall_exact_agreement": result["overall_pooled_dimensions"]["exact_agreement"],
        "overall_agreement_within_one": result["overall_pooled_dimensions"]["agreement_within_one"],
        "overall_linear_weighted_kappa": result["overall_pooled_dimensions"]["linear_weighted_cohen_kappa"],
        "overall_quadratic_weighted_kappa": result["overall_pooled_dimensions"]["quadratic_weighted_cohen_kappa"],
        "aggregate": result["per_response_aggregate"],
    }, indent=2))
