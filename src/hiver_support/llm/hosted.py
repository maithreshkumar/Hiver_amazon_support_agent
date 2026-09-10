from __future__ import annotations

import os
from typing import Any, Callable

from hiver_support.errors import ConfigurationError
from hiver_support.llm.base import LLMProvider
from hiver_support.llm.http import parse_json_object, post_json


class HostedProvider(LLMProvider):
    def __init__(self, name: str, model: str, api_key: str, timeout: int = 180) -> None:
        if not api_key:
            raise ConfigurationError(f"{name.upper()} API key is not configured")
        self.name, self.model, self.api_key, self.timeout = name, model, api_key, timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.name == "openai":
            response = post_json(
                "https://api.openai.com/v1/chat/completions",
                {"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1},
                {"Authorization": f"Bearer {self.api_key}"}, self.timeout,
            )
            return str(response["choices"][0]["message"]["content"])
        if self.name == "anthropic":
            response = post_json(
                "https://api.anthropic.com/v1/messages",
                {"model": self.model, "max_tokens": 800, "system": system_prompt, "messages": [{"role": "user", "content": user_prompt}]},
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout,
            )
            return str(response["content"][0]["text"])
        response = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}",
            {"system_instruction": {"parts": [{"text": system_prompt}]}, "contents": [{"parts": [{"text": user_prompt}]}]},
            {}, self.timeout,
        )
        return str(response["candidates"][0]["content"]["parts"][0]["text"])

    def structured_generate(self, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        suffix = f"\nReturn only JSON matching: {schema}" if schema else "\nReturn only one JSON object."
        return parse_json_object(self.generate(system_prompt + suffix, user_prompt))


def hosted_from_environment(name: str, model: str, timeout: int) -> HostedProvider:
    return HostedProvider(name, model, os.getenv(f"{name.upper()}_API_KEY", ""), timeout)

