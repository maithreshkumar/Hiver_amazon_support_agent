from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hiver_support.config import load_yaml
from hiver_support.errors import ConfigurationError
from hiver_support.retrieval.embedder import OllamaEmbedder


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    thread_id: str
    customer_text: str
    amazon_response: str
    similarity: float
    intent: str | None = None


class LocalVectorIndex:
    def __init__(self, directory: str | Path, embedder: OllamaEmbedder) -> None:
        directory = Path(directory)
        vectors_path, metadata_path = directory / "vectors.npy", directory / "metadata.parquet"
        if not vectors_path.exists() or not metadata_path.exists():
            raise ConfigurationError(f"Retrieval index is missing at {directory}; run scripts/build_retrieval_index.py")
        self.vectors = np.load(vectors_path, mmap_mode="r")
        self.metadata = pd.read_parquet(metadata_path)
        if len(self.vectors) != len(self.metadata):
            raise ConfigurationError("Retrieval vectors and metadata have different lengths")
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5, intent: str | None = None) -> list[RetrievalMatch]:
        query_vector = self.embedder.embed([query])[0]
        return self.search_vector(query_vector, top_k, intent)

    def search_vector(
        self, query_vector: np.ndarray, top_k: int = 5, intent: str | None = None
    ) -> list[RetrievalMatch]:
        """Search with an already-normalized vector, enabling batched query embedding."""
        scores = np.asarray(self.vectors @ query_vector)
        candidate_indices = np.argsort(scores)[::-1]
        results: list[RetrievalMatch] = []
        seen_responses: set[str] = set()
        for index in candidate_indices:
            row = self.metadata.iloc[int(index)]
            row_intent = row.get("intent") if "intent" in row else None
            if intent and row_intent and row_intent != intent:
                continue
            response_key = " ".join(str(row["amazon_response"]).casefold().split())
            if response_key in seen_responses:
                continue
            seen_responses.add(response_key)
            results.append(
                RetrievalMatch(
                    thread_id=str(row["thread_id"]), customer_text=str(row["customer_text"]),
                    amazon_response=str(row["amazon_response"]), similarity=float(scores[index]),
                    intent=str(row_intent) if row_intent else None,
                )
            )
            if len(results) >= top_k:
                break
        return results


def build_index(config_path: str | Path = "configs/retrieval.yaml") -> dict[str, object]:
    config = load_yaml(config_path)
    corpus = pd.read_parquet(config["corpus_path"])
    corpus = corpus.drop_duplicates("thread_id").dropna(subset=["customer_text", "amazon_response"])
    if "language" in corpus:
        corpus = corpus.loc[corpus["language"] == "en"]
    limit = min(int(config["max_records"]), len(corpus))
    corpus = corpus.sample(limit, random_state=int(config["random_seed"])).reset_index(drop=True)
    embedder = OllamaEmbedder(str(config["model"]), str(config["base_url"]))
    batches: list[np.ndarray] = []
    size = int(config["batch_size"])
    for start in range(0, len(corpus), size):
        batches.append(embedder.embed(corpus["customer_text"].iloc[start : start + size].astype(str).tolist()))
        print(f"Embedded {min(start + size, len(corpus)):,}/{len(corpus):,}", flush=True)
    vectors = np.vstack(batches).astype(np.float32)
    directory = Path(config["index_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors.npy", vectors)
    keep = [column for column in ["thread_id", "customer_text", "amazon_response", "created_at", "language", "intent"] if column in corpus]
    corpus[keep].to_parquet(directory / "metadata.parquet", index=False)
    manifest = {"provider": "ollama", "model": embedder.model, "records": len(corpus), "dimensions": int(vectors.shape[1]), "source": str(config["corpus_path"]), "training_only": True}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
