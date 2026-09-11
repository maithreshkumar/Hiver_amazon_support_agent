import pytest

from hiver_support.errors import ProviderUnavailable
from hiver_support.llm.http import parse_json_object


def test_truncated_structured_output_becomes_provider_error() -> None:
    with pytest.raises(ProviderUnavailable, match="malformed or truncated JSON"):
        parse_json_object('{"draft_reply":"unterminated}')


def test_missing_final_container_closures_are_repaired() -> None:
    assert parse_json_object('{"candidates":{"A":{"score":3}}') == {
        "candidates": {"A": {"score": 3}}
    }


def test_mismatched_container_is_not_repaired() -> None:
    with pytest.raises(ProviderUnavailable, match="malformed or truncated JSON"):
        parse_json_object('{"candidates":[1,2}}')
