import pytest

from hiver_support.data.schema import REQUIRED_COLUMNS, SchemaError, parse_bool, parse_id_list, validate_columns


def test_valid_schema() -> None:
    validate_columns(list(REQUIRED_COLUMNS))


def test_missing_schema_column() -> None:
    with pytest.raises(SchemaError, match="text"):
        validate_columns([column for column in REQUIRED_COLUMNS if column != "text"])


def test_parsers() -> None:
    assert parse_bool("True") is True
    assert parse_bool("false") is False
    assert parse_bool("not-a-bool") is None
    assert parse_id_list("1, 2") == ["1", "2"]
    assert parse_id_list("") == []

