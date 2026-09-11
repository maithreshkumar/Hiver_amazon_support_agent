from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from hiver_support.evaluation.reply_ratings import DIMENSIONS, load_reply_ratings, validate_reply_ratings


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _distribution(values: np.ndarray) -> dict[str, object]:
    counts = pd.Series(values).value_counts().reindex(range(1, 6), fill_value=0)
    return {
        "counts": {str(score): int(counts.loc[score]) for score in range(1, 6)},
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _dimension_metrics(human: np.ndarray, judge: np.ndarray) -> dict[str, object]:
    difference = judge - human
    matrix = confusion_matrix(human, judge, labels=[1, 2, 3, 4, 5])
    human_ranks = pd.Series(human).rank(method="average").to_numpy()
    judge_ranks = pd.Series(judge).rank(method="average").to_numpy()
    return {
        "rows": len(human),
        "exact_agreement": float(np.mean(difference == 0)),
        "agreement_within_one": float(np.mean(np.abs(difference) <= 1)),
        "linear_weighted_cohen_kappa": float(cohen_kappa_score(human, judge, weights="linear")),
        "quadratic_weighted_cohen_kappa": float(cohen_kappa_score(human, judge, weights="quadratic")),
        "pearson_correlation": _correlation(human.astype(float), judge.astype(float)),
        "spearman_rank_correlation": _correlation(human_ranks, judge_ranks),
        "correlation_note": "Correlation is omitted when either rater has zero variance; ordinal scores make weighted kappa primary.",
        "human_distribution": _distribution(human),
        "judge_distribution": _distribution(judge),
        "judge_minus_human_mean_bias": float(np.mean(difference)),
        "judge_overrated_count": int(np.sum(difference > 0)),
        "judge_underrated_count": int(np.sum(difference < 0)),
        "equal_count": int(np.sum(difference == 0)),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "confusion_matrix": {
            "labels": [1, 2, 3, 4, 5],
            "values": matrix.tolist(),
            "rows_are_human_columns_are_judge": True,
        },
    }


def calculate_human_judge_agreement(
    ratings_path: str | Path = "data/golden/reply_quality_human_ratings.csv",
    judge_path: str | Path = "data/reports/reply_quality_judge.json",
    output_path: str | Path = "data/reports/human_judge_agreement.json",
    scope: str = "48 human-rated RAG+safety responses matched to the original frozen LLM judge run",
    judge_rerun_after_human_ratings: bool = False,
    post_hoc_judge_experiment: bool = False,
) -> dict[str, object]:
    validate_reply_ratings(ratings_path, require_complete=True)
    ratings = load_reply_ratings(ratings_path)
    judge_file = Path(judge_path)
    judge_payload = json.loads(judge_file.read_text(encoding="utf-8"))
    judge_records = list(judge_payload["records"])
    judge_by_id = {str(item["example_id"]): item for item in judge_records}
    if len(judge_by_id) != len(judge_records):
        raise ValueError("Judge output contains duplicate example IDs")
    rating_ids = ratings["example_id"].astype(str).tolist()
    if set(rating_ids) != set(judge_by_id) or len(rating_ids) != len(judge_records):
        raise ValueError("Human ratings and frozen judge output do not cover the same examples")

    per_dimension: dict[str, dict[str, object]] = {}
    long_rows: list[dict[str, object]] = []
    for dimension in DIMENSIONS:
        human = ratings[f"human_{dimension}"].astype(int).to_numpy()
        judge = np.asarray([
            int(judge_by_id[example_id]["scores_by_system"]["rag_with_safety"]["scores"][dimension])
            for example_id in rating_ids
        ])
        per_dimension[dimension] = _dimension_metrics(human, judge)
        for index, example_id in enumerate(rating_ids):
            long_rows.append({
                "example_id": example_id,
                "dimension": dimension,
                "human_score": int(human[index]),
                "judge_score": int(judge[index]),
                "signed_difference_judge_minus_human": int(judge[index] - human[index]),
                "absolute_difference": int(abs(judge[index] - human[index])),
                "customer_message": str(ratings.at[index, "customer_message"]),
                "generated_response": str(ratings.at[index, "generated_response"]),
                "human_reason": str(ratings.at[index, "human_rating_reason"]),
                "judge_rationale": str(judge_by_id[example_id]["scores_by_system"]["rag_with_safety"]["rationale"]),
            })

    long_frame = pd.DataFrame(long_rows)
    pooled_human = long_frame["human_score"].to_numpy(dtype=int)
    pooled_judge = long_frame["judge_score"].to_numpy(dtype=int)
    overall = _dimension_metrics(pooled_human, pooled_judge)

    human_aggregate = ratings[[f"human_{dimension}" for dimension in DIMENSIONS]].astype(float).mean(axis=1).to_numpy()
    judge_aggregate = np.asarray([
        float(judge_by_id[example_id]["scores_by_system"]["rag_with_safety"]["aggregate"])
        for example_id in rating_ids
    ])
    aggregate = {
        "rows": len(ratings),
        "human_mean": float(np.mean(human_aggregate)),
        "judge_mean": float(np.mean(judge_aggregate)),
        "judge_minus_human_mean_bias": float(np.mean(judge_aggregate - human_aggregate)),
        "mean_absolute_error": float(np.mean(np.abs(judge_aggregate - human_aggregate))),
        "pearson_correlation": _correlation(human_aggregate, judge_aggregate),
        "spearman_rank_correlation": _correlation(
            pd.Series(human_aggregate).rank(method="average").to_numpy(),
            pd.Series(judge_aggregate).rank(method="average").to_numpy(),
        ),
    }

    bias_rows = []
    for dimension, metrics in per_dimension.items():
        bias = float(metrics["judge_minus_human_mean_bias"])
        direction = "judge_overrates" if bias > 0 else "judge_underrates" if bias < 0 else "no_mean_bias"
        bias_rows.append({
            "dimension": dimension,
            "direction": direction,
            "mean_difference": bias,
            "mean_absolute_error": metrics["mean_absolute_error"],
        })
    bias_rows.sort(key=lambda row: abs(float(row["mean_difference"])), reverse=True)

    largest = long_frame.sort_values(
        ["absolute_difference", "dimension", "example_id"], ascending=[False, True, True]
    ).head(15)
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_set_version": "hiver-submission-v1",
        "scope": scope,
        "rubric_dimensions": list(DIMENSIONS),
        "human_rating_rows": len(ratings),
        "paired_dimension_ratings": len(long_frame),
        "judge_provider": judge_payload["judge_provider"],
        "judge_model": judge_payload["judge_model"],
        "judge_prompt_version": judge_payload["prompt_version"],
        "judge_rubric_version": judge_payload["rubric_version"],
        "judge_rerun_after_human_ratings": judge_rerun_after_human_ratings,
        "post_hoc_judge_experiment": post_hoc_judge_experiment,
        "ratings_sha256": hashlib.sha256(Path(ratings_path).read_bytes()).hexdigest(),
        "judge_sha256": hashlib.sha256(judge_file.read_bytes()).hexdigest(),
        "overall_pooled_dimensions": overall,
        "per_dimension": per_dimension,
        "per_response_aggregate": aggregate,
        "systematic_biases_ranked": bias_rows,
        "largest_disagreements": largest.to_dict(orient="records"),
        "interpretation_limits": [
            "One human rater supplied the final scores after semi-automated preparation.",
            "There are 48 responses, so per-dimension agreement estimates are noisy.",
            "The 1-5 rubric is ordinal; weighted kappa is more appropriate than treating gaps as exact intervals.",
            "The same six dimensions are pooled only as a summary; per-dimension results remain primary.",
        ],
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame([
        {
            "dimension": dimension,
            "exact_agreement": metrics["exact_agreement"],
            "agreement_within_one": metrics["agreement_within_one"],
            "linear_weighted_cohen_kappa": metrics["linear_weighted_cohen_kappa"],
            "quadratic_weighted_cohen_kappa": metrics["quadratic_weighted_cohen_kappa"],
            "pearson_correlation": metrics["pearson_correlation"],
            "spearman_rank_correlation": metrics["spearman_rank_correlation"],
            "human_mean": metrics["human_distribution"]["mean"],
            "judge_mean": metrics["judge_distribution"]["mean"],
            "judge_minus_human_mean_bias": metrics["judge_minus_human_mean_bias"],
            "mean_absolute_error": metrics["mean_absolute_error"],
        }
        for dimension, metrics in per_dimension.items()
    ]).to_csv(target.parent / "human_judge_agreement_by_dimension.csv", index=False)
    largest.to_csv(target.parent / "human_judge_largest_disagreements.csv", index=False)
    return result
