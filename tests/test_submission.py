from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from hiver_support.evaluation.reply_ratings import load_reply_ratings, validate_reply_ratings
from hiver_support.submission import (
    ArtifactVerificationError,
    sha256_file,
    verify_artifact_files,
    verify_submission_artifacts,
)


GOLDEN_SHA256 = "9273e9c34ffa28579b898af20f8954c76fc5d386c39ed9a3a259d168cd36cd0f"
FINAL_RATINGS_SHA256 = "b06709c781980fe56866f201c9342ea5c912e047a9ca1b6a43c6c43373bf5745"


def _one_file_manifest(path: str, payload: bytes) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifacts": [
            {
                "path": path,
                "required": True,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }


def test_manifest_file_verification_detects_missing_and_corrupt_artifacts(tmp_path: Path) -> None:
    expected = b"sealed artifact"
    manifest = _one_file_manifest("artifact.bin", expected)
    with pytest.raises(ArtifactVerificationError, match="missing required artifact"):
        verify_artifact_files(manifest, tmp_path)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"corrupt artifact")
    with pytest.raises(ArtifactVerificationError, match="size mismatch|SHA-256 mismatch"):
        verify_artifact_files(manifest, tmp_path)
    path.write_bytes(expected)
    assert len(verify_artifact_files(manifest, tmp_path)) == 1


def test_manifest_file_verification_rejects_unresolved_lfs_pointer(tmp_path: Path) -> None:
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n"
    path = tmp_path / "model.safetensors"
    path.write_bytes(pointer)
    manifest = _one_file_manifest("model.safetensors", pointer)
    with pytest.raises(ArtifactVerificationError, match="Git LFS pointer"):
        verify_artifact_files(manifest, tmp_path)


def test_final_submission_artifacts_and_semantic_structure_pass() -> None:
    result = verify_submission_artifacts()
    assert result["status"] == "PASS"
    assert result["semantic_checks"]["golden_rows"] == 200
    assert result["semantic_checks"]["final_rating_rows"] == 48


def test_sealed_gold_and_final_human_ratings_are_immutable() -> None:
    assert sha256_file(Path("data/golden/golden_set.csv")) == GOLDEN_SHA256
    assert (
        sha256_file(Path("data/golden/reply_quality_human_ratings_post_repair.csv"))
        == FINAL_RATINGS_SHA256
    )
    assert validate_reply_ratings(
        "data/golden/reply_quality_human_ratings_post_repair.csv", require_complete=True
    ) == {"rows": 48, "completed": 48, "remaining": 0}


def test_final_ratings_judge_and_responses_cover_identical_ordered_examples() -> None:
    ratings = load_reply_ratings("data/golden/reply_quality_human_ratings_post_repair.csv")
    responses = json.loads(
        Path("data/reports/post_repair/golden_response_outputs_post_repair.json").read_text(
            encoding="utf-8"
        )
    )["records"]
    judge = json.loads(
        Path("data/reports/post_repair/reply_quality_judge_post_repair.json").read_text(
            encoding="utf-8"
        )
    )
    expected = ratings["example_id"].astype(str).tolist()
    assert [str(row["example_id"]) for row in responses] == expected
    assert [str(row["example_id"]) for row in judge["records"]] == expected
    assert judge["human_ratings_visible_to_judge"] is False
    assert all(
        str(response["final_safe_response"]) == str(ratings.at[index, "generated_response"])
        for index, response in enumerate(responses)
    )


def test_compact_leakage_membership_has_zero_golden_overlap() -> None:
    frame = pd.read_parquet("data/evaluation/leakage_membership.parquet")
    golden = set(frame.loc[frame["in_golden"], "thread_id_sha256"])
    assert len(golden) == 200
    assert golden <= set(frame.loc[frame["in_test"], "thread_id_sha256"])
    for column in ("in_train", "in_weak_labels", "in_retrieval_index"):
        assert not golden & set(frame.loc[frame[column], "thread_id_sha256"])


def test_final_evaluation_has_no_pre_repair_fallback() -> None:
    source = Path("scripts/run_final_evaluation.py").read_text(encoding="utf-8")
    assert "golden_response_outputs_post_repair.json" in source
    assert 'else Path("data/reports/golden_response_outputs.json")' not in source
    assert "pre-repair fallback is forbidden" in source


def test_reviewer_facing_files_have_no_user_specific_absolute_path() -> None:
    paths = [Path("README.md"), *Path("scripts").glob("*.py"), *Path("configs").glob("*.yaml")]
    paths += list(Path("data/golden").glob("*.json"))
    paths += list(Path("data/governance").glob("*.md"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users\\Maithresh" not in text, path
        assert "C:/Users/Maithresh" not in text, path
