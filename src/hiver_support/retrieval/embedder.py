from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from hiver_support.llm.http import post_json


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: int = 180) -> None:
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", model)
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        response = post_json(
            f"{self.base_url}/api/embed",
            {"model": self.model, "input": texts, "truncate": True},
            {}, self.timeout,
        )
        vectors = np.asarray(response.get("embeddings", []), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError("Ollama returned an unexpected embedding shape")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

