import pytest

from hiver_support.errors import ConfigurationError
from hiver_support.llm.factory import create_provider


def test_missing_hosted_key_fails_gracefully(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="API key"):
        create_provider({"provider": "openai", "model": "test-model"})

