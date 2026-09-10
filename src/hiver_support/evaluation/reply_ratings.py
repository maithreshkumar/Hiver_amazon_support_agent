from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype


DIMENSIONS = (
    "groundedness", "helpfulness", "actionability", "safety",
    "historical_consistency", "unsupported_claim_avoidance",
)
SCORE_COLUMNS = [f"human_{name}" for name in DIMENSIONS]
STRING_COLUMNS = [
    "example_id", "customer_message", "generated_response", "historical_evidence",
    "labeler_id", "human_rating_reason",
]
TIMESTAMP_COLUMNS = ["rated_at", "updated_at"]
COLUMNS = STRING_COLUMNS[:4] + SCORE_COLUMNS + [
    "human_rating_complete", "labeler_id", "human_rating_reason", "rated_at", "updated_at",
]


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Reply-rating schema must be exactly {COLUMNS}")
    typed = frame.copy(deep=True)
    for column in STRING_COLUMNS:
        typed[column] = typed[column].astype("string").fillna("")
    for column in SCORE_COLUMNS:
        values = pd.to_numeric(typed[column].replace("", pd.NA), errors="coerce")
        invalid = typed[column].astype("string").fillna("").str.strip().ne("") & values.isna()
        if invalid.any():
            raise ValueError(f"Invalid numeric values in {column}")
        typed[column] = values.astype("Int64")
    raw = typed["human_rating_complete"]
    if is_bool_dtype(raw.dtype):
        typed["human_rating_complete"] = raw.fillna(False).astype("boolean")
    else:
        normalized = raw.astype("string").fillna("").str.strip().str.casefold()
        if not normalized.isin({"", "false", "0", "no", "true", "1", "yes"}).all():
            raise ValueError("Invalid human_rating_complete value")
        typed["human_rating_complete"] = normalized.isin({"true", "1", "yes"}).astype("boolean")
    for column in TIMESTAMP_COLUMNS:
        raw_time = typed[column].astype("string").fillna("").str.strip()
        parsed = pd.to_datetime(raw_time.mask(raw_time.eq("")), errors="coerce", utc=True)
        if (raw_time.ne("") & parsed.isna()).any():
            raise ValueError(f"Invalid timestamp in {column}")
        typed[column] = pd.Series(pd.array(parsed, dtype="datetime64[us, UTC]"), index=typed.index)
    return typed


def load_reply_ratings(path: str | Path = "data/golden/reply_quality_human_ratings.csv") -> pd.DataFrame:
    return _coerce(pd.read_csv(path, dtype="string", keep_default_na=False))


def _write_atomic(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    typed = _coerce(frame)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".tmp.csv", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        typed.to_csv(temporary, index=False, na_rep="", date_format="%Y-%m-%dT%H:%M:%S.%f%z")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = load_reply_ratings(temporary)
        if len(reloaded) != len(typed):
            raise ValueError("Atomic reply-rating write changed row count")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_reply_ratings(
    response_path: str | Path = "data/reports/golden_response_outputs.json",
    output_path: str | Path = "data/golden/reply_quality_human_ratings.csv",
) -> Path:
    output = Path(output_path)
    payload = json.loads(Path(response_path).read_text(encoding="utf-8"))
    records = list(payload["records"])
    if not 40 <= len(records) <= 50:
        raise ValueError("Human reply-rating sample must contain 40–50 records")
    identity = pd.DataFrame({
        "example_id": [r["example_id"] for r in records],
        "customer_message": [r["message"] for r in records],
        "generated_response": [r["final_safe_response"] for r in records],
        "historical_evidence": [
            "\n\n".join(
                f"Historical customer: {item['customer_text']}\nHistorical AmazonHelp reply: {item['amazon_response']}"
                for item in r["retrieved_evidence"][:3]
            )
            for r in records
        ],
    })
    if output.exists():
        current = load_reply_ratings(output)
        if not current[list(identity.columns)].reset_index(drop=True).equals(identity.astype("string")):
            raise ValueError("Existing human-rating artifact does not match the sealed response sample")
        return output
    frame = identity.copy()
    for column in SCORE_COLUMNS:
        frame[column] = pd.Series([pd.NA] * len(frame), dtype="Int64")
    frame["human_rating_complete"] = pd.Series([False] * len(frame), dtype="boolean")
    frame["labeler_id"] = ""
    frame["human_rating_reason"] = ""
    frame["rated_at"] = pd.NaT
    frame["updated_at"] = pd.NaT
    _write_atomic(frame[COLUMNS], output)
    return output


def validate_reply_ratings(
    path: str | Path = "data/golden/reply_quality_human_ratings.csv",
    *,
    require_complete: bool = False,
) -> dict[str, int]:
    frame = load_reply_ratings(path)
    if not 40 <= len(frame) <= 50 or not frame["example_id"].is_unique:
        raise ValueError("Reply-rating artifact must have 40–50 unique example IDs")
    complete = frame["human_rating_complete"].fillna(False).astype(bool)
    for index, row in frame.iterrows():
        scores = row[SCORE_COLUMNS]
        if complete.at[index]:
            if scores.isna().any() or not all(1 <= int(value) <= 5 for value in scores):
                raise ValueError(f"Completed rating {row['example_id']} must have all six 1–5 scores")
            if not str(row["labeler_id"]).strip() or pd.isna(row["rated_at"]):
                raise ValueError(f"Completed rating {row['example_id']} needs labeler and timestamp")
        elif scores.notna().any():
            raise ValueError(f"Incomplete rating {row['example_id']} contains partial scores")
    completed = int(complete.sum())
    if require_complete and completed != len(frame):
        raise ValueError(f"Human reply ratings incomplete: {completed}/{len(frame)}")
    return {"rows": len(frame), "completed": completed, "remaining": len(frame) - completed}


def save_reply_rating(
    frame: pd.DataFrame,
    row_index: int,
    scores: dict[str, int],
    labeler_id: str,
    reason: str = "",
    output_path: str | Path = "data/golden/reply_quality_human_ratings.csv",
) -> None:
    if set(scores) != set(DIMENSIONS) or any(not 1 <= int(value) <= 5 for value in scores.values()):
        raise ValueError("All six rubric dimensions require integer scores from 1 to 5")
    if not labeler_id.strip():
        raise ValueError("A labeler ID is required")
    working = _coerce(frame)
    if row_index not in working.index:
        raise IndexError(row_index)
    now = pd.Timestamp.now(tz="UTC").as_unit("us")
    for dimension, value in scores.items():
        working.at[row_index, f"human_{dimension}"] = int(value)
    working.at[row_index, "human_rating_complete"] = True
    working.at[row_index, "labeler_id"] = labeler_id.strip()
    working.at[row_index, "human_rating_reason"] = reason.strip()
    if pd.isna(working.at[row_index, "rated_at"]):
        working.at[row_index, "rated_at"] = now
    working.at[row_index, "updated_at"] = now
    _write_atomic(working, output_path)


def import_reviewed_reply_ratings(
    import_path: str | Path,
    canonical_path: str | Path = "data/golden/reply_quality_human_ratings.csv",
    backup_path: str | Path = "data/golden/reply_quality_human_ratings.pre-human-import.csv",
    provenance_path: str | Path = "data/golden/reply_rating_import_provenance.json",
) -> dict[str, object]:
    """Validate and atomically import completed ratings without reserializing their values."""
    source = Path(import_path)
    canonical = Path(canonical_path)
    reviewed = load_reply_ratings(source)
    validation = validate_reply_ratings(source, require_complete=True)
    current = load_reply_ratings(canonical)
    identity = ["example_id", "customer_message", "generated_response", "historical_evidence"]
    if not current[identity].reset_index(drop=True).equals(reviewed[identity].reset_index(drop=True)):
        raise ValueError("Reviewed ratings do not match the sealed 48-response sample")

    backup = Path(backup_path)
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(canonical, backup)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{canonical.stem}.", suffix=".tmp.csv", dir=canonical.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        validate_reply_ratings(temporary, require_complete=True)
        os.replace(temporary, canonical)
    finally:
        if temporary.exists():
            temporary.unlink()

    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    canonical_sha256 = hashlib.sha256(canonical.read_bytes()).hexdigest()
    if source_sha256 != canonical_sha256:
        raise RuntimeError("Canonical rating bytes differ from the validated import")
    canonical_validation = validate_reply_ratings(canonical, require_complete=True)
    provenance = {
        "imported_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_path": str(source.resolve()),
        "source_sha256": source_sha256,
        "canonical_path": str(canonical.resolve()),
        "canonical_sha256": canonical_sha256,
        "backup_path": str(backup.resolve()),
        "annotation_methodology": (
            "Human-reviewed reply-quality ratings prepared with semi-automation; the human made "
            "the final rating decisions for every rubric dimension."
        ),
        "human_ratings_modified_during_import": False,
        "safety_distribution_confirmed_by_human": {
            str(score): int(count)
            for score, count in reviewed["human_safety"].value_counts().sort_index().items()
        },
        "safety_outlier_reason": (
            "The single Safety=1 rating is deliberate because the generated response repeated "
            "the customer's phone number publicly."
        ),
        "validation": canonical_validation,
    }
    provenance_file = Path(provenance_path)
    provenance_file.parent.mkdir(parents=True, exist_ok=True)
    provenance_temporary = provenance_file.with_suffix(".tmp.json")
    provenance_temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    os.replace(provenance_temporary, provenance_file)
    return {**canonical_validation, "source_sha256": source_sha256, "canonical_sha256": canonical_sha256}
