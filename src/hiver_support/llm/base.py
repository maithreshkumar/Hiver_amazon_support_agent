from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def structured_generate(
        self, system_prompt: str, user_prompt: str, schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError

