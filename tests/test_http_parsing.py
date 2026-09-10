import pytest

from hiver_support.errors import ProviderUnavailable
from hiver_support.llm.http import parse_json_object


def test_truncated_structured_output_becomes_provider_error() -> None:
    with pytest.raises(ProviderUnavailable, match="malformed or truncated JSON"):
        parse_json_object('{"draft_reply":"unterminated}')
