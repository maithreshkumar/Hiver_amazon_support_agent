from __future__ import annotations

import json
from typing import Any

from hiver_support.llm.base import LLMProvider
from hiver_support.llm.http import parse_json_object, post_json


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 180,
        num_predict: int = 400,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.num_predict = num_predict

    def _chat(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }
        if json_mode:
            payload["format"] = "json"
        response = post_json(f"{self.base_url}/api/chat", payload, {}, self.timeout)
        content = str(response.get("message", {}).get("content", "")).strip()
        if not content:
            from hiver_support.errors import ProviderUnavailable

            raise ProviderUnavailable(
                f"Ollama model {self.model} completed without response content "
                f"(done_reason={response.get('done_reason', 'unknown')})."
            )
        return content

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._chat(system_prompt, user_prompt, False)

    def structured_generate(self, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        schema_note = (
            "\nReturn exactly one compact JSON object matching this schema: "
            + json.dumps(schema, separators=(",", ":"))
            if schema
            else ""
        )
        return parse_json_object(self._chat(system_prompt + schema_note, user_prompt, True))
