from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype

from hiver_support.intents.taxonomy import load_approved_taxonomy


CANONICAL_COLUMNS = [
    "example_id", "thread_id", "customer_message", "human_intent",
    "human_escalation", "human_reason", "human_label_complete", "label_source",
    "labeler_id", "labeled_at", "updated_at",
]
STRING_COLUMNS = [
    "example_id", "thread_id", "customer_message", "human_intent",
    "human_escalation", "human_reason", "label_source", "labeler_id",
]
TIMESTAMP_COLUMNS = ["labeled_at", "updated_at"]


def _coerce_golden_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stable pandas dtypes independent of CSV inference."""
    if list(frame.columns) != CANONICAL_COLUMNS:
        raise ValueError(f"Golden-set schema must be exactly {CANONICAL_COLUMNS}")
    typed = frame.copy(deep=True)
    for column in STRING_COLUMNS:
        typed[column] = typed[column].astype("string").fillna("")

    raw_complete = typed["human_label_complete"]
    if is_bool_dtype(raw_complete.dtype):
        typed["human_label_complete"] = raw_complete.fillna(False).astype("boolean")
    else:
        normalized = raw_complete.astype("string").fillna("").str.strip().str.casefold()
        valid = {"", "false", "0", "no", "true", "1", "yes"}
        invalid = sorted(set(normalized[~normalized.isin(valid)].tolist()))
        if invalid:
            raise ValueError(f"Invalid human_label_complete values: {invalid}")
        typed["human_label_complete"] = normalized.isin({"true", "1", "yes"}).astype("boolean")

    for column in TIMESTAMP_COLUMNS:
        if is_datetime64_any_dtype(typed[column].dtype):
            parsed = pd.to_datetime(typed[column], utc=True)
            typed[column] = pd.Series(
                pd.array(parsed, dtype="datetime64[us, UTC]"), index=typed.index
            )
            continue
        raw = typed[column].astype("string").fillna("").str.strip()
        parsed = pd.to_datetime(raw.mask(raw.eq("")), errors="coerce", utc=True)
        invalid_count = int((raw.ne("") & parsed.isna()).sum())
        if invalid_count:
            raise ValueError(f"{invalid_count} invalid UTC timestamps in {column}")
        typed[column] = pd.Series(
            pd.array(parsed, dtype="datetime64[us, UTC]"), index=typed.index
        )
    return typed


def load_golden_set(path: str | Path = "data/golden/golden_set.csv") -> pd.DataFrame:
    """Load the CSV using the canonical nullable-string/Boolean/UTC schema."""
    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    return _coerce_golden_schema(raw)


def _write_golden_set_atomic(frame: pd.DataFrame, output_path: str | Path) -> None:
    """Validate and atomically replace the canonical CSV on the same filesystem."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    typed = _coerce_golden_schema(frame)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".tmp.csv", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        typed.to_csv(temporary, index=False, na_rep="", date_format="%Y-%m-%dT%H:%M:%S.%f%z")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = load_golden_set(temporary)
        if len(reloaded) != len(typed):
            raise ValueError("Atomic golden-set write changed the row count")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _complete_mask(frame: pd.DataFrame) -> pd.Series:
    if not is_bool_dtype(frame["human_label_complete"].dtype):
        frame = _coerce_golden_schema(frame)
    return frame["human_label_complete"].fillna(False).astype(bool)


def next_unlabelled_index(frame: pd.DataFrame) -> int | None:
    remaining = frame.index[~_complete_mask(frame)].tolist()
    return int(remaining[0]) if remaining else None


def initialize_golden_set(
    candidates_path: str | Path = "data/golden/golden_candidates.csv",
    output_path: str | Path = "data/golden/golden_set.csv",
) -> Path:
    """Create the unlabelled canonical file without copying AI suggestion columns."""
    candidates = pd.read_csv(candidates_path, dtype="string", keep_default_na=False)
    required = {"example_id", "thread_id", "customer_message"}
    if missing := required - set(candidates.columns):
        raise ValueError(f"Golden candidates are missing columns: {sorted(missing)}")
    if len(candidates) != 200 or not candidates["example_id"].is_unique or not candidates["thread_id"].is_unique:
        raise ValueError("Golden candidates must contain exactly 200 unique examples and threads")
    output = Path(output_path)
    if output.exists():
        existing = load_golden_set(output)
        expected = candidates[["example_id", "thread_id", "customer_message"]].reset_index(drop=True)
        observed = existing[["example_id", "thread_id", "customer_message"]].reset_index(drop=True)
        if not expected.equals(observed):
            raise ValueError("Existing golden_set.csv no longer matches the sealed candidates")
        return output
    frame = candidates[["example_id", "thread_id", "customer_message"]].copy()
    frame["human_intent"] = ""
    frame["human_escalation"] = ""
    frame["human_reason"] = ""
    frame["human_label_complete"] = False
    frame["label_source"] = ""
    frame["labeler_id"] = ""
    frame["labeled_at"] = ""
    frame["updated_at"] = ""
    frame["labeled_at"] = pd.NaT
    frame["updated_at"] = pd.NaT
    _write_golden_set_atomic(_coerce_golden_schema(frame), output)
    return output


def validate_golden_set(
    golden_path: str | Path = "data/golden/golden_set.csv",
    *,
    require_complete: bool = True,
    candidates_path: str | Path = "data/golden/golden_candidates.csv",
    train_path: str | Path = "data/processed/train.parquet",
    test_path: str | Path = "data/processed/test.parquet",
    retrieval_path: str | Path = "data/processed/retrieval_corpus.parquet",
    weak_labels_path: str | Path = "data/processed/weak_labels.parquet",
) -> dict[str, object]:
    frame = load_golden_set(golden_path)
    errors: list[str] = []
    if list(frame.columns) != CANONICAL_COLUMNS:
        errors.append(f"schema must be exactly {CANONICAL_COLUMNS}")
    if len(frame) != 200:
        errors.append(f"expected 200 rows, found {len(frame)}")
    if not frame["example_id"].is_unique:
        errors.append("example_id values are not unique")
    if not frame["thread_id"].is_unique:
        errors.append("thread_id values are not unique")

    candidates = pd.read_csv(candidates_path, dtype="string", keep_default_na=False)
    candidate_view = candidates[["example_id", "thread_id", "customer_message"]].reset_index(drop=True)
    golden_view = frame[["example_id", "thread_id", "customer_message"]].reset_index(drop=True)
    if not candidate_view.equals(golden_view):
        errors.append("canonical examples/messages differ from the sealed candidate file")

    ids = set(frame["thread_id"])
    train_ids = set(pd.read_parquet(train_path, columns=["thread_id"])["thread_id"].astype(str))
    test_ids = set(pd.read_parquet(test_path, columns=["thread_id"])["thread_id"].astype(str))
    retrieval_ids = set(pd.read_parquet(retrieval_path, columns=["thread_id"])["thread_id"].astype(str))
    weak_ids = set(pd.read_parquet(weak_labels_path, columns=["thread_id"])["thread_id"].astype(str))
    if ids - test_ids:
        errors.append(f"{len(ids - test_ids)} golden threads are outside the test split")
    if ids & train_ids:
        errors.append(f"{len(ids & train_ids)} golden threads overlap the train split")
    if ids & retrieval_ids:
        errors.append(f"{len(ids & retrieval_ids)} golden threads overlap retrieval")
    if ids & weak_ids:
        errors.append(f"{len(ids & weak_ids)} golden threads overlap weak labels")

    complete = _complete_mask(frame)
    allowed_intents = {item.name for item in load_approved_taxonomy()}
    completed = frame.loc[complete]
    invalid_intents = sorted(set(completed["human_intent"]) - allowed_intents)
    invalid_routes = sorted(set(completed["human_escalation"]) - {"AUTO_HANDLE", "ESCALATE"})
    if invalid_intents:
        errors.append(f"completed rows contain invalid intents: {invalid_intents}")
    if invalid_routes:
        errors.append(f"completed rows contain invalid routes: {invalid_routes}")
    for field in ("human_intent", "human_escalation", "human_reason", "labeler_id"):
        missing = int(completed[field].str.strip().eq("").sum())
        if missing:
            errors.append(f"{missing} completed rows have blank {field}")
    missing_labeled_at = int(completed["labeled_at"].isna().sum())
    if missing_labeled_at:
        errors.append(f"{missing_labeled_at} completed rows have blank labeled_at")
    wrong_source = int(completed["label_source"].ne("human_gold").sum())
    if wrong_source:
        errors.append(f"{wrong_source} completed rows do not have label_source=human_gold")
    if require_complete and int(complete.sum()) != 200:
        errors.append(f"human labeling incomplete: {int(complete.sum())}/200 complete")
    if errors:
        raise ValueError("Golden-set validation failed:\n- " + "\n- ".join(errors))
    return {
        "rows": len(frame), "unique_examples": int(frame["example_id"].nunique()),
        "unique_threads": int(frame["thread_id"].nunique()), "completed": int(complete.sum()),
        "remaining": int(len(frame) - complete.sum()), "train_overlap": len(ids & train_ids),
        "retrieval_overlap": len(ids & retrieval_ids), "weak_label_overlap": len(ids & weak_ids),
        "all_in_test_split": not bool(ids - test_ids),
    }


def save_human_label(
    frame: pd.DataFrame,
    row_index: int,
    *,
    intent: str,
    escalation: str,
    reason: str,
    labeler_id: str,
    output_path: str | Path = "data/golden/golden_set.csv",
    audit_path: str | Path = "data/golden/human_label_audit.jsonl",
) -> None:
    allowed = {item.name for item in load_approved_taxonomy()}
    if intent not in allowed:
        raise ValueError(f"Intent must be one of the approved labels: {sorted(allowed)}")
    if escalation not in {"AUTO_HANDLE", "ESCALATE"}:
        raise ValueError("Escalation must be AUTO_HANDLE or ESCALATE")
    if not reason.strip() or not labeler_id.strip():
        raise ValueError("A human reason and labeler ID are required")
    working = _coerce_golden_schema(frame)
    if row_index not in working.index:
        raise IndexError(f"Golden-set row index does not exist: {row_index}")
    now = pd.Timestamp.now(tz="UTC").as_unit("us")
    previous = {key: str(working.at[row_index, key]) for key in ("human_intent", "human_escalation", "human_reason", "human_label_complete")}
    working.at[row_index, "human_intent"] = intent
    working.at[row_index, "human_escalation"] = escalation
    working.at[row_index, "human_reason"] = reason.strip()
    working.at[row_index, "human_label_complete"] = True
    working.at[row_index, "label_source"] = "human_gold"
    working.at[row_index, "labeler_id"] = labeler_id.strip()
    if pd.isna(working.at[row_index, "labeled_at"]):
        working.at[row_index, "labeled_at"] = now
    working.at[row_index, "updated_at"] = now
    _write_golden_set_atomic(working, output_path)
    audit = {
        "event": "human_gold_label_saved", "example_id": str(frame.at[row_index, "example_id"]),
        "thread_id": str(frame.at[row_index, "thread_id"]), "labeler_id": labeler_id.strip(),
        "timestamp": now.isoformat(), "previous": previous,
        "new": {"human_intent": intent, "human_escalation": escalation, "human_reason": reason.strip()},
    }
    audit_file = Path(audit_path)
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with audit_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit, ensure_ascii=False) + "\n")


def import_reviewed_golden_set(
    import_path: str | Path,
    canonical_path: str | Path = "data/golden/golden_set.csv",
    backup_path: str | Path = "data/golden/golden_set.pre-human-import.csv",
    provenance_path: str | Path = "data/golden/import_provenance.json",
) -> dict[str, object]:
    """Validate and atomically import reviewed human labels without changing examples."""
    source = Path(import_path)
    canonical = Path(canonical_path)
    reviewed = load_golden_set(source)
    validation = validate_golden_set(source, require_complete=True)
    if canonical.exists():
        current = load_golden_set(canonical)
        identity = ["example_id", "thread_id", "customer_message"]
        if not current[identity].reset_index(drop=True).equals(reviewed[identity].reset_index(drop=True)):
            raise ValueError("Reviewed import does not match the canonical sealed examples")
        backup = Path(backup_path)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(canonical, backup)
    _write_golden_set_atomic(reviewed, canonical)
    canonical_validation = validate_golden_set(canonical, require_complete=True)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    canonical_sha256 = hashlib.sha256(canonical.read_bytes()).hexdigest()
    provenance = {
        "imported_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_path": str(source.resolve()),
        "source_sha256": source_sha256,
        "canonical_path": str(canonical.resolve()),
        "canonical_sha256": canonical_sha256,
        "annotation_methodology": (
            "Human-reviewed golden set where the human made the final intent and routing "
            "decisions after reviewing the examples and correcting disagreements during "
            "assisted annotation."
        ),
        "training_or_tuning_use": False,
        "validation": canonical_validation,
    }
    provenance_file = Path(provenance_path)
    provenance_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = provenance_file.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, provenance_file)
    return {**validation, "source_sha256": source_sha256, "canonical_sha256": canonical_sha256}
