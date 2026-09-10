from pathlib import Path

import pandas as pd
import pytest


def test_built_index_contains_only_training_retrieval_threads() -> None:
    metadata_path = Path("indexes/amazon_support/metadata.parquet")
    if not metadata_path.exists():
        pytest.skip("Local retrieval index has not been built")
    indexed = set(pd.read_parquet(metadata_path, columns=["thread_id"])["thread_id"].astype(str))
    retrieval = set(
        pd.read_parquet("data/processed/retrieval_corpus.parquet", columns=["thread_id"])["thread_id"].astype(str)
    )
    test = set(pd.read_parquet("data/processed/test.parquet", columns=["thread_id"])["thread_id"].astype(str))
    golden = set(
        pd.read_csv("data/golden/golden_candidates.csv", dtype={"thread_id": str})["thread_id"].astype(str)
    )
    assert indexed <= retrieval
    assert not indexed & test
    assert not indexed & golden

