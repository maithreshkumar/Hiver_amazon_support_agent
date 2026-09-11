import hashlib
from pathlib import Path

import pandas as pd
import pytest


PROTECTED_HUMAN_ARTIFACTS = {
    "data/golden/golden_set.csv": "9273e9c34ffa28579b898af20f8954c76fc5d386c39ed9a3a259d168cd36cd0f",
    "data/golden/reply_quality_human_ratings.csv": "2dd1545f8322f5b6364138ed9dd533cfc04b4d3c8a59d472e6226d9bd59d732a",
    "data/golden/reply_quality_human_ratings_post_repair.csv": "b06709c781980fe56866f201c9342ea5c912e047a9ca1b6a43c6c43373bf5745",
}


def test_frozen_human_evaluation_artifacts_are_byte_unchanged() -> None:
    for path, expected_hash in PROTECTED_HUMAN_ARTIFACTS.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash


def test_built_index_contains_only_training_retrieval_threads() -> None:
    metadata_path = Path("indexes/amazon_support/metadata.parquet")
    if not metadata_path.exists():
        pytest.skip("Local retrieval index has not been built")
    compact = pd.read_parquet("data/evaluation/leakage_membership.parquet")
    golden_hashes = set(compact.loc[compact["in_golden"], "thread_id_sha256"])
    assert len(golden_hashes) == 200
    assert not golden_hashes & set(
        compact.loc[compact["in_retrieval_index"], "thread_id_sha256"]
    )

    local_sources = [
        Path("data/processed/retrieval_corpus.parquet"),
        Path("data/processed/test.parquet"),
        Path("data/golden/golden_candidates.csv"),
    ]
    if not all(path.exists() for path in local_sources):
        return
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
