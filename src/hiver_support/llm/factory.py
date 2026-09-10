from __future__ import annotations

import os

from hiver_support.errors import ConfigurationError
from hiver_support.llm.base import LLMProvider
from hiver_support.llm.hosted import hosted_from_environment
from hiver_support.llm.ollama import OllamaProvider


def create_provider(config: dict[str, object]) -> LLMProvider:
    name = os.getenv("LLM_PROVIDER", str(config.get("provider", "ollama"))).lower()
    timeout = int(config.get("timeout_seconds", 180))
    if name == "ollama":
        model = os.getenv("OLLAMA_MODEL", str(config.get("model", "")))
        if not model:
            raise ConfigurationError("OLLAMA_MODEL is required")
        return OllamaProvider(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", str(config.get("base_url", "http://localhost:11434"))),
            temperature=float(config.get("temperature", 0.1)),
            timeout=timeout,
            num_predict=int(config.get("num_predict", 400)),
        )
    if name not in {"openai", "gemini", "anthropic"}:
        raise ConfigurationError(f"Unsupported LLM provider: {name}")
    model = os.getenv(f"{name.upper()}_MODEL", str(config.get("model", "")))
    if not model:
        raise ConfigurationError(f"{name.upper()}_MODEL is required")
    return hosted_from_environment(name, model, timeout)
