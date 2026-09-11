from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from hiver_support.submission import sha256_file


ARTIFACTS = {
    "configuration": [
        "pyproject.toml",
        "requirements.txt",
        "requirements-lock.txt",
        "configs/intents.yaml",
        "configs/escalation_policy.yaml",
        "configs/runtime.yaml",
        "configs/retrieval.yaml",
        "configs/judge.yaml",
        "configs/training.yaml",
        "data/governance/HUMAN_APPROVAL.md",
    ],
    "intent_model": [
        "models/intent/model.safetensors",
        "models/intent/config.json",
        "models/intent/metadata.json",
        "models/intent/tokenizer.json",
        "models/intent/tokenizer_config.json",
    ],
    "baselines": [
        "models/baselines/simple.joblib",
        "models/baselines/trivial.joblib",
    ],
    "retrieval_index": [
        "indexes/amazon_support/manifest.json",
        "indexes/amazon_support/metadata.parquet",
        "indexes/amazon_support/vectors.npy",
    ],
    "gold_and_human_ratings": [
        "data/golden/golden_candidates.csv",
        "data/golden/golden_set.csv",
        "data/golden/import_provenance.json",
        "data/golden/reply_quality_human_ratings.csv",
        "data/golden/reply_rating_import_provenance.json",
        "data/golden/reply_quality_human_ratings_post_repair.csv",
        "data/golden/reply_quality_post_repair_provenance.json",
    ],
    "compact_leakage_evidence": [
        "data/evaluation/leakage_membership.parquet",
        "data/evaluation/leakage_validation.json",
    ],
    "final_intent_metrics": [
        "data/reports/final_intent_metrics.json",
        "data/reports/final_intent_per_intent.csv",
        "data/reports/final_intent_confusion_matrices.csv",
        "data/reports/final_intent_predictions.csv",
        "data/reports/confidence_analysis.json",
        "data/reports/near_duplicate_audit.json",
        "data/reports/data_quality.json",
    ],
    "final_post_repair_evaluation": [
        "data/reports/post_repair/golden_response_outputs_post_repair.json",
        "data/reports/post_repair/reply_quality_judge_post_repair.json",
        "data/reports/post_repair/reply_quality_judge_post_repair.jsonl",
        "data/reports/post_repair/human_judge_agreement_post_repair.json",
        "data/reports/post_repair/human_judge_agreement_by_dimension.csv",
        "data/reports/post_repair/human_judge_largest_disagreements.csv",
        "data/reports/post_repair/routing_metrics.json",
        "data/reports/post_repair/retrieval_analysis.json",
        "data/reports/post_repair/latency_analysis.json",
        "data/reports/post_repair/response_evaluation_summary.json",
        "data/reports/post_repair/failure_analysis.json",
        "data/reports/post_repair/runtime_repair_summary.json",
    ],
    "historical_pre_repair_evaluation": [
        "data/reports/golden_response_outputs.json",
        "data/reports/reply_quality_judge.json",
        "data/reports/reply_quality_judge.jsonl",
        "data/reports/human_judge_agreement.json",
    ],
}

LFS_PATHS = {
    "models/intent/model.safetensors",
    "indexes/amazon_support/vectors.npy",
}


def purpose(group: str) -> str:
    return group.replace("_", " ")


def main() -> None:
    rows = []
    for group, paths in ARTIFACTS.items():
        for relative in paths:
            path = Path(relative)
            if not path.is_file():
                raise FileNotFoundError(f"Required submission artifact is missing: {relative}")
            rows.append(
                {
                    "path": relative,
                    "purpose": purpose(group),
                    "artifact_version": "submission-v1",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "required": True,
                    "source": {
                        "type": "git_lfs" if relative in LFS_PATHS else "repository",
                        "location": relative,
                    },
                }
            )
    manifest = {
        "schema_version": 1,
        "artifact_set_version": "hiver-submission-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "delivery_strategy": {
            "type": "git_lfs",
            "large_binary_paths": sorted(LFS_PATHS),
            "note": (
                "Required large binaries are Git LFS objects. Repository-tracked artifacts are "
                "small final inference/evaluation files; raw Twitter data and training checkpoints are excluded."
            ),
        },
        "artifacts": rows,
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
    }
    target = Path("artifacts/manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": len(rows), "bytes": manifest["total_size_bytes"], "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
