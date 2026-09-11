from __future__ import annotations

import json

from hiver_support.evaluation.reply_agreement import calculate_human_judge_agreement


if __name__ == "__main__":
    result = calculate_human_judge_agreement(
        ratings_path="data/golden/reply_quality_human_ratings_post_repair.csv",
        judge_path="data/reports/post_repair/reply_quality_judge_post_repair.json",
        output_path="data/reports/post_repair/human_judge_agreement_post_repair.json",
        scope=(
            "48 human-rated final post-repair RAG+safety responses matched to the frozen "
            "post-repair LLM judge artifact"
        ),
        judge_rerun_after_human_ratings=True,
        post_hoc_judge_experiment=True,
    )
    print(
        json.dumps(
            {
                "rows": result["human_rating_rows"],
                "exact_agreement": result["overall_pooled_dimensions"]["exact_agreement"],
                "agreement_within_one": result["overall_pooled_dimensions"]["agreement_within_one"],
                "linear_weighted_kappa": result["overall_pooled_dimensions"]["linear_weighted_cohen_kappa"],
                "quadratic_weighted_kappa": result["overall_pooled_dimensions"]["quadratic_weighted_cohen_kappa"],
            },
            indent=2,
        )
    )
