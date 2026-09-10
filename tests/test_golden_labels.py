import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_string_dtype

from hiver_support.evaluation.golden_labels import (
    CANONICAL_COLUMNS,
    initialize_golden_set,
    load_golden_set,
    next_unlabelled_index,
    save_human_label,
    validate_golden_set,
)


def test_unbiased_canonical_golden_file_is_isolated() -> None:
    path = initialize_golden_set()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(frame.columns) == CANONICAL_COLUMNS
    assert not any(column.startswith("ai_") for column in frame.columns)
    result = validate_golden_set(path, require_complete=False)
    assert result["rows"] == 200
    assert result["train_overlap"] == 0
    assert result["retrieval_overlap"] == 0
    assert result["weak_label_overlap"] == 0
    assert result["all_in_test_split"] is True


def test_untouched_csv_loads_with_explicit_schema(tmp_path: Path) -> None:
    frame = load_golden_set(initialize_golden_set(output_path=tmp_path / "untouched.csv"))
    assert is_bool_dtype(frame["human_label_complete"].dtype)
    assert not frame["human_label_complete"].any()
    for column in (
        "human_intent", "human_escalation", "human_reason", "label_source", "labeler_id"
    ):
        assert is_string_dtype(frame[column].dtype)
        assert frame[column].eq("").all()
    assert is_datetime64_any_dtype(frame["labeled_at"].dtype)
    assert is_datetime64_any_dtype(frame["updated_at"].dtype)
    assert frame["labeled_at"].isna().all()


def test_save_reload_resume_status_and_validation(tmp_path: Path) -> None:
    output = initialize_golden_set(output_path=tmp_path / "golden_set.csv")
    audit = tmp_path / "audit.jsonl"
    frame = load_golden_set(output)
    assert next_unlabelled_index(frame) == 0
    save_human_label(
        frame,
        0,
        intent="delivery_tracking_or_delay",
        escalation="ESCALATE",
        reason="Multiple issues and a delivery promise require review.",
        labeler_id="test-human",
        output_path=output,
        audit_path=audit,
    )

    reloaded = load_golden_set(output)
    assert is_bool_dtype(reloaded["human_label_complete"].dtype)
    assert bool(reloaded.at[0, "human_label_complete"]) is True
    assert not bool(reloaded.at[1, "human_label_complete"])
    assert reloaded.at[0, "label_source"] == "human_gold"
    assert reloaded.at[0, "labeler_id"] == "test-human"
    assert pd.notna(reloaded.at[0, "labeled_at"])
    assert pd.notna(reloaded.at[0, "updated_at"])
    assert next_unlabelled_index(reloaded) == 1
    status = validate_golden_set(output, require_complete=False)
    assert status["completed"] == 1
    assert status["remaining"] == 199
    with pytest.raises(ValueError, match="human labeling incomplete: 1/200"):
        validate_golden_set(output, require_complete=True)
    assert audit.read_text(encoding="utf-8").count("human_gold_label_saved") == 1


def test_cli_quit_and_resume_status_count(tmp_path: Path) -> None:
    output = initialize_golden_set(output_path=tmp_path / "golden_set.csv")
    frame = load_golden_set(output)
    save_human_label(
        frame,
        0,
        intent="delivery_tracking_or_delay",
        escalation="ESCALATE",
        reason="Test-only temporary label.",
        labeler_id="test-human",
        output_path=output,
        audit_path=tmp_path / "audit.jsonl",
    )
    environment = {
        **os.environ,
        "GOLDEN_SET_PATH": str(output),
        "GOLDEN_AUDIT_PATH": str(tmp_path / "audit.jsonl"),
    }
    quit_result = subprocess.run(
        [sys.executable, "scripts/label_golden.py", "--labeler", "test-human"],
        input="q\n",
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=environment,
        check=False,
    )
    assert quit_result.returncode == 0
    assert "Example: gold-002" in quit_result.stdout
    status_result = subprocess.run(
        [sys.executable, "scripts/label_golden.py", "--status"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=environment,
        check=False,
    )
    assert status_result.returncode == 0
    assert "1 / 200 labelled" in status_result.stdout
    assert validate_golden_set(output, require_complete=False)["completed"] == 1


def test_blank_and_valid_boolean_strings_survive_loading(tmp_path: Path) -> None:
    output = initialize_golden_set(output_path=tmp_path / "golden_set.csv")
    raw = pd.read_csv(output, dtype="string", keep_default_na=False)
    raw.at[0, "human_label_complete"] = "True"
    raw.at[1, "human_label_complete"] = "False"
    raw.at[2, "human_label_complete"] = ""
    raw.to_csv(output, index=False)
    loaded = load_golden_set(output)
    assert loaded["human_label_complete"].tolist()[:3] == [True, False, False]


def test_failed_serialization_does_not_replace_canonical(tmp_path: Path, monkeypatch) -> None:
    output = initialize_golden_set(output_path=tmp_path / "golden_set.csv")
    original = output.read_bytes()
    frame = load_golden_set(output)

    def fail_to_csv(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        save_human_label(
            frame,
            0,
            intent="delivery_tracking_or_delay",
            escalation="ESCALATE",
            reason="Temporary test.",
            labeler_id="test-human",
            output_path=output,
            audit_path=tmp_path / "audit.jsonl",
        )
    assert output.read_bytes() == original


def test_completed_validation_rejects_nonhuman_or_invalid_labels(tmp_path: Path) -> None:
    canonical = initialize_golden_set()
    frame = pd.read_csv(canonical, dtype=str, keep_default_na=False)
    frame["human_intent"] = "not_an_approved_intent"
    frame["human_escalation"] = "ESCALATE"
    frame["human_reason"] = "Human review required."
    frame["human_label_complete"] = True
    frame["label_source"] = "weak_ai"
    frame["labeler_id"] = "test-human"
    frame["labeled_at"] = "2026-09-10T00:00:00+00:00"
    output = tmp_path / "invalid_gold.csv"
    frame.to_csv(output, index=False)
    with pytest.raises(ValueError, match="invalid intents"):
        validate_golden_set(output)
