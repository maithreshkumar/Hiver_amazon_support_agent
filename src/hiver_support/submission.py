from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ArtifactVerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact_manifest(path: str | Path = "artifacts/manifest.json") -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ArtifactVerificationError(
            f"Artifact manifest is missing: {manifest_path}. Restore it from the repository."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError(f"Artifact manifest is unreadable: {exc}") from exc
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("artifacts"), list):
        raise ArtifactVerificationError("Artifact manifest schema is unsupported or incomplete")
    return manifest


def _resolve_inside_root(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ArtifactVerificationError(f"Manifest contains unsafe artifact path: {relative}")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactVerificationError(f"Artifact path escapes repository root: {relative}") from exc
    return resolved


def _is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def verify_artifact_files(manifest: dict[str, Any], root: str | Path = ".") -> list[dict[str, Any]]:
    root = Path(root).resolve()
    verified: list[dict[str, Any]] = []
    problems: list[str] = []
    seen: set[str] = set()
    for entry in manifest["artifacts"]:
        relative = str(entry.get("path", ""))
        required = bool(entry.get("required", False))
        if not relative or relative in seen:
            problems.append(f"duplicate or blank manifest path: {relative!r}")
            continue
        seen.add(relative)
        path = _resolve_inside_root(root, relative)
        if not path.is_file():
            if required:
                problems.append(
                    f"missing required artifact {relative}; run python scripts/fetch_submission_artifacts.py"
                )
            continue
        if _is_lfs_pointer(path):
            problems.append(
                f"{relative} is only a Git LFS pointer; run git lfs pull or "
                "python scripts/fetch_submission_artifacts.py"
            )
            continue
        expected_size = int(entry.get("size_bytes", -1))
        actual_size = path.stat().st_size
        if expected_size != actual_size:
            problems.append(
                f"size mismatch for {relative}: expected {expected_size}, found {actual_size}"
            )
            continue
        expected_hash = str(entry.get("sha256", "")).casefold()
        actual_hash = sha256_file(path)
        if expected_hash != actual_hash:
            problems.append(
                f"SHA-256 mismatch for {relative}: expected {expected_hash}, found {actual_hash}; "
                "restore or refetch the artifact"
            )
            continue
        verified.append(entry)
    if problems:
        raise ArtifactVerificationError("\n- ".join(["Artifact verification failed:", *problems]))
    return verified


def _semantic_checks(root: Path) -> dict[str, Any]:
    import joblib
    import numpy as np
    import pandas as pd
    from safetensors import safe_open

    from hiver_support.evaluation.golden_labels import load_golden_set
    from hiver_support.evaluation.reply_ratings import load_reply_ratings, validate_reply_ratings
    from hiver_support.intents.taxonomy import load_approved_taxonomy

    taxonomy = [item.name for item in load_approved_taxonomy(root / "configs/intents.yaml")]
    golden_path = root / "data/golden/golden_set.csv"
    golden = load_golden_set(golden_path)
    complete = golden["human_label_complete"].fillna(False).astype(bool)
    if len(golden) != 200 or not golden["example_id"].is_unique or not golden["thread_id"].is_unique:
        raise ArtifactVerificationError("Golden set must contain 200 unique examples and thread IDs")
    if int(complete.sum()) != 200:
        raise ArtifactVerificationError(f"Golden set is incomplete: {int(complete.sum())}/200")
    if not set(golden["human_intent"]).issubset(set(taxonomy)):
        raise ArtifactVerificationError("Golden set contains an intent outside the approved taxonomy")
    if set(golden["human_escalation"]) - {"AUTO_HANDLE", "ESCALATE"}:
        raise ArtifactVerificationError("Golden set contains an invalid routing label")
    if not golden["label_source"].eq("human_gold").all():
        raise ArtifactVerificationError("Golden set contains non-human-gold provenance")

    ratings_path = root / "data/golden/reply_quality_human_ratings_post_repair.csv"
    rating_status = validate_reply_ratings(ratings_path, require_complete=True)
    ratings = load_reply_ratings(ratings_path)

    model_dir = root / "models/intent"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if list(metadata.get("labels", [])) != taxonomy:
        raise ArtifactVerificationError("Intent model metadata does not match the approved taxonomy")
    if set(config.get("label2id", {})) != set(taxonomy):
        raise ArtifactVerificationError("Intent model config labels do not match the approved taxonomy")
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        tensor_count = len(handle.keys())
    if tensor_count < 10:
        raise ArtifactVerificationError("Intent model safetensors file has an implausible tensor structure")

    simple = joblib.load(root / "models/baselines/simple.joblib")
    trivial = joblib.load(root / "models/baselines/trivial.joblib")
    if not hasattr(simple, "predict_intent") or not hasattr(trivial, "predict"):
        raise ArtifactVerificationError("Saved baseline artifacts do not expose the expected inference APIs")

    index_dir = root / "indexes/amazon_support"
    index_manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(index_dir / "vectors.npy", mmap_mode="r")
    index_metadata = pd.read_parquet(index_dir / "metadata.parquet")
    expected_shape = (int(index_manifest["records"]), int(index_manifest["dimensions"]))
    if vectors.shape != expected_shape or len(index_metadata) != expected_shape[0]:
        raise ArtifactVerificationError(
            f"Retrieval index structure mismatch: vectors={vectors.shape}, metadata={len(index_metadata)}, "
            f"manifest={expected_shape}"
        )
    if not bool(index_manifest.get("training_only")):
        raise ArtifactVerificationError("Retrieval index is not marked training-only")

    response_path = root / "data/reports/post_repair/golden_response_outputs_post_repair.json"
    responses = json.loads(response_path.read_text(encoding="utf-8"))["records"]
    if len(responses) != 48 or [str(row["example_id"]) for row in responses] != ratings["example_id"].astype(str).tolist():
        raise ArtifactVerificationError("Post-repair responses do not match the final 48-row rating sample")
    if any(
        str(response["message"]) != str(ratings.at[index, "customer_message"])
        or str(response["final_safe_response"]) != str(ratings.at[index, "generated_response"])
        for index, response in enumerate(responses)
    ):
        raise ArtifactVerificationError("Final human ratings are attached to different response text")

    judge_path = root / "data/reports/post_repair/reply_quality_judge_post_repair.json"
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    judge_ids = [str(row["example_id"]) for row in judge["records"]]
    if judge_ids != ratings["example_id"].astype(str).tolist() or len(judge_ids) != 48:
        raise ArtifactVerificationError("Frozen post-repair judge output does not match the rating sample")
    if not bool(judge.get("human_ratings_visible_to_judge") is False):
        raise ArtifactVerificationError("Judge artifact does not confirm blinded human ratings")
    if judge.get("source_response_sha256") != sha256_file(response_path):
        raise ArtifactVerificationError("Judge artifact is not tied to the final post-repair responses")

    agreement_path = root / "data/reports/post_repair/human_judge_agreement_post_repair.json"
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    if agreement.get("ratings_sha256") != sha256_file(ratings_path):
        raise ArtifactVerificationError("Agreement artifact is not tied to the final human ratings")
    if agreement.get("judge_sha256") != sha256_file(judge_path):
        raise ArtifactVerificationError("Agreement artifact is not tied to the frozen post-repair judge")
    if int(agreement.get("human_rating_rows", 0)) != 48:
        raise ArtifactVerificationError("Agreement artifact does not contain 48 paired responses")

    intent_metrics = json.loads(
        (root / "data/reports/final_intent_metrics.json").read_text(encoding="utf-8")
    )
    if any(int(row.get("rows", 0)) != 200 for row in intent_metrics["models"].values()):
        raise ArtifactVerificationError("Final intent metrics do not cover all 200 golden examples")
    routing = json.loads(
        (root / "data/reports/post_repair/routing_metrics.json").read_text(encoding="utf-8")
    )
    if int(routing.get("rows", 0)) != 48:
        raise ArtifactVerificationError("Final routing metrics do not cover the sealed 48-response sample")

    leakage = pd.read_parquet(root / "data/evaluation/leakage_membership.parquet")
    required_columns = {
        "thread_id_sha256", "in_train", "in_test", "in_weak_labels",
        "in_retrieval_index", "in_golden"
    }
    if set(leakage.columns) != required_columns or leakage["thread_id_sha256"].duplicated().any():
        raise ArtifactVerificationError("Compact leakage membership schema or uniqueness is invalid")
    golden_hashes = set(leakage.loc[leakage["in_golden"], "thread_id_sha256"])
    overlap = {
        name: len(golden_hashes & set(leakage.loc[leakage[column], "thread_id_sha256"]))
        for name, column in (
            ("train", "in_train"),
            ("weak_labels", "in_weak_labels"),
            ("retrieval_index", "in_retrieval_index"),
        )
    }
    if len(golden_hashes) != 200 or any(overlap.values()):
        raise ArtifactVerificationError(f"Compact leakage verification failed: {overlap}")
    test_hashes = set(leakage.loc[leakage["in_test"], "thread_id_sha256"])
    if not golden_hashes <= test_hashes:
        raise ArtifactVerificationError("Compact leakage verification does not place every golden row in test")

    return {
        "golden_rows": len(golden),
        "final_rating_rows": rating_status["completed"],
        "intent_model_tensors": tensor_count,
        "retrieval_records": len(index_metadata),
        "post_repair_response_rows": len(responses),
        "post_repair_judge_rows": len(judge_ids),
        "golden_leakage_overlap": overlap,
    }


def verify_submission_artifacts(
    manifest_path: str | Path = "artifacts/manifest.json",
    root: str | Path = ".",
) -> dict[str, Any]:
    repository = Path(root).resolve()
    manifest = load_artifact_manifest(repository / manifest_path)
    verified = verify_artifact_files(manifest, repository)
    semantic = _semantic_checks(repository)
    return {
        "artifact_set_version": manifest["artifact_set_version"],
        "verified_files": len(verified),
        "required_files": sum(bool(row.get("required")) for row in manifest["artifacts"]),
        "semantic_checks": semantic,
        "status": "PASS",
    }
