from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hiver_support.data.normalize import detect_language, text_metadata
from hiver_support.data.schema import REQUIRED_COLUMNS, parse_bool, parse_id_list, validate_columns
from hiver_support.data.threads import component_map

LOGGER = logging.getLogger("hiver_support.prepare")

RAW_TABLE = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id TEXT PRIMARY KEY,
    author_id TEXT NOT NULL,
    inbound INTEGER,
    created_at TEXT,
    text TEXT,
    response_tweet_id TEXT,
    in_response_to_tweet_id TEXT
)
"""


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _require_raw(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download Kaggle dataset "
            "thoughtvector/customer-support-on-twitter, extract twcs.csv, and place it there."
        )


def stage_csv(raw_path: Path, db_path: Path, batch_size: int) -> dict[str, Any]:
    """Stage immutable source data and graph edges in SQLite without loading it all in RAM."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(RAW_TABLE)
    connection.execute("CREATE TABLE edges (left_id TEXT NOT NULL, right_id TEXT NOT NULL)")

    stats: Counter[str] = Counter()
    seen: set[str] = set()
    tweet_batch: list[tuple[object, ...]] = []
    edge_batch: list[tuple[str, str]] = []

    with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames or [])
        for row in reader:
            stats["raw_rows"] += 1
            tweet_id = (row.get("tweet_id") or "").strip()
            author_id = (row.get("author_id") or "").strip()
            inbound = parse_bool(row.get("inbound"))
            if not tweet_id or not author_id or inbound is None:
                stats["malformed_rows"] += 1
                continue
            if tweet_id in seen:
                stats["duplicate_tweet_ids"] += 1
                continue
            seen.add(tweet_id)
            stats["valid_rows"] += 1
            if "\ufffd" in (row.get("text") or ""):
                stats["encoding_replacement_rows"] += 1
            tweet_batch.append(
                (
                    tweet_id,
                    author_id,
                    int(inbound),
                    row.get("created_at") or "",
                    row.get("text") or "",
                    row.get("response_tweet_id") or "",
                    row.get("in_response_to_tweet_id") or "",
                )
            )
            for linked_id in parse_id_list(row.get("response_tweet_id")):
                if linked_id != tweet_id:
                    edge_batch.append((tweet_id, linked_id))
            for linked_id in parse_id_list(row.get("in_response_to_tweet_id")):
                if linked_id != tweet_id:
                    edge_batch.append((tweet_id, linked_id))
            if len(tweet_batch) >= batch_size:
                connection.executemany("INSERT INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?)", tweet_batch)
                connection.executemany("INSERT INTO edges VALUES (?, ?)", edge_batch)
                connection.commit()
                tweet_batch.clear()
                edge_batch.clear()
                if stats["valid_rows"] % (batch_size * 10) == 0:
                    LOGGER.info("Staged %s valid rows", f"{stats['valid_rows']:,}")

    if tweet_batch:
        connection.executemany("INSERT INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?)", tweet_batch)
        connection.executemany("INSERT INTO edges VALUES (?, ?)", edge_batch)
        connection.commit()
    connection.execute("CREATE INDEX edges_left_idx ON edges(left_id)")
    connection.execute("CREATE INDEX edges_right_idx ON edges(right_id)")
    connection.execute("CREATE INDEX tweets_author_idx ON tweets(author_id)")
    connection.commit()
    stats["missing_reference_edges"] = connection.execute(
        "SELECT COUNT(*) FROM edges e LEFT JOIN tweets t ON e.right_id=t.tweet_id WHERE t.tweet_id IS NULL"
    ).fetchone()[0]
    connection.close()
    for key in (
        "raw_rows",
        "valid_rows",
        "malformed_rows",
        "duplicate_tweet_ids",
        "encoding_replacement_rows",
        "missing_reference_edges",
    ):
        stats.setdefault(key, 0)
    return dict(stats)


def find_connected_amazon_ids(connection: sqlite3.Connection, support_account: str) -> tuple[set[str], int]:
    connection.execute("DROP TABLE IF EXISTS connected")
    connection.execute(
        """CREATE TEMP TABLE connected AS
           WITH RECURSIVE connected_ids(tweet_id) AS (
               SELECT tweet_id FROM tweets WHERE lower(author_id)=lower(?)
               UNION
               SELECT e.right_id FROM edges e
               JOIN connected_ids c ON e.left_id=c.tweet_id
               UNION
               SELECT e.left_id FROM edges e
               JOIN connected_ids c ON e.right_id=c.tweet_id
           )
           SELECT tweet_id FROM connected_ids""",
        (support_account,),
    )
    connection.execute("CREATE UNIQUE INDEX connected_tweet_idx ON connected(tweet_id)")
    ids = {row[0] for row in connection.execute("SELECT tweet_id FROM connected")}
    LOGGER.info("Recovered %s tweets connected to %s", f"{len(ids):,}", support_account)
    return ids, 1


def _read_connected(connection: sqlite3.Connection) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    rows = connection.execute(
        """SELECT t.tweet_id,t.author_id,t.inbound,t.created_at,t.text,
                  t.response_tweet_id,t.in_response_to_tweet_id
           FROM tweets t JOIN connected c ON t.tweet_id=c.tweet_id"""
    ).fetchall()
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    edges = connection.execute(
        """SELECT e.left_id,e.right_id FROM edges e
           JOIN connected a ON e.left_id=a.tweet_id
           JOIN connected b ON e.right_id=b.tweet_id"""
    ).fetchall()
    return frame, [(str(a), str(b)) for a, b in edges]


def _safe_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def enrich_threads(frame: pd.DataFrame, edges: list[tuple[str, str]], support_account: str) -> pd.DataFrame:
    frame = frame.copy()
    ids = frame["tweet_id"].astype(str).tolist()
    mapping = component_map(ids, edges)
    frame["thread_id"] = frame["tweet_id"].astype(str).map(mapping)
    frame["parent_tweet_id"] = frame["in_response_to_tweet_id"].fillna("").astype(str).str.split(",").str[0]
    frame["created_at_parsed"] = _safe_timestamp(frame["created_at"])
    frame["speaker_role"] = frame["author_id"].where(
        frame["author_id"].str.casefold() == support_account.casefold(), "CUSTOMER"
    )
    frame.loc[frame["author_id"].str.casefold() == support_account.casefold(), "speaker_role"] = "AMAZON"

    metadata = pd.DataFrame([text_metadata(value) for value in frame["text"]], index=frame.index)
    frame = pd.concat([frame.drop(columns=["text"]), metadata], axis=1)
    frame = frame.sort_values(["thread_id", "created_at_parsed", "tweet_id"], kind="stable")
    frame["turn_number"] = frame.groupby("thread_id").cumcount() + 1
    frame["branching_thread"] = frame["tweet_id"].isin(
        frame["parent_tweet_id"].value_counts().loc[lambda values: values > 1].index
    ).groupby(frame["thread_id"]).transform("any")
    id_set = set(ids)
    frame["missing_parent"] = frame["parent_tweet_id"].map(lambda value: bool(value) and value not in id_set)
    return frame


def drop_contaminated_threads(frame: pd.DataFrame, support_account: str) -> tuple[pd.DataFrame, int]:
    other_support = frame.loc[
        (frame["inbound"] == 0) & (frame["author_id"].str.casefold() != support_account.casefold()),
        "thread_id",
    ].unique()
    clean = frame.loc[~frame["thread_id"].isin(other_support)].copy()
    return clean, len(other_support)


def make_conversations(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for thread_id, group in frame.groupby("thread_id", sort=False):
        customers = group.loc[group["speaker_role"] == "CUSTOMER", "raw_text"].tolist()
        amazon = group.loc[group["speaker_role"] == "AMAZON", "raw_text"].tolist()
        first_customer_message = customers[0] if customers else ""
        ordered_times = group["created_at_parsed"].dropna().sort_values()
        gaps = ordered_times.diff().dt.total_seconds().div(86400)
        maximum_gap_days = float(gaps.max()) if gaps.notna().any() else 0.0
        records.append(
            {
                "thread_id": thread_id,
                "first_customer_message": first_customer_message,
                "language": detect_language(first_customer_message),
                "all_customer_messages": customers,
                "all_amazon_responses": amazon,
                "turn_count": len(group),
                "start_time": group["created_at_parsed"].min(),
                "end_time": group["created_at_parsed"].max(),
                "split_time": group["created_at_parsed"].max(),
                "maximum_interturn_gap_days": maximum_gap_days,
                "temporal_gap_anomaly": maximum_gap_days > 30,
                "terminal_speaker": group.iloc[-1]["speaker_role"],
                "resolution_status": "UNKNOWN",
                "branching_thread": bool(group["branching_thread"].any()),
                "incomplete_thread": bool(group["missing_parent"].any()),
            }
        )
    return pd.DataFrame.from_records(records)


def make_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    by_id = frame.set_index("tweet_id", drop=False)
    pairs: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        if row.speaker_role != "AMAZON" or not row.parent_tweet_id or row.parent_tweet_id not in by_id.index:
            continue
        parent = by_id.loc[row.parent_tweet_id]
        if isinstance(parent, pd.DataFrame) or parent["speaker_role"] != "CUSTOMER":
            continue
        pairs.append(
            {
                "thread_id": row.thread_id,
                "customer_tweet_id": parent["tweet_id"],
                "amazon_tweet_id": row.tweet_id,
                "customer_raw_text": parent["raw_text"],
                "customer_text": parent["model_text"],
                "amazon_response": row.raw_text,
                "created_at": row.created_at_parsed,
                "resolution_status": "UNKNOWN",
            }
        )
    return pd.DataFrame.from_records(pairs)


def temporal_split(conversations: pd.DataFrame, ratios: dict[str, float]) -> dict[str, set[str]]:
    ordered = conversations.sort_values(["split_time", "thread_id"], kind="stable")
    total = len(ordered)
    train_end = int(total * float(ratios["train"]))
    dev_end = train_end + int(total * float(ratios["dev"]))
    return {
        "train": set(ordered.iloc[:train_end]["thread_id"]),
        "dev": set(ordered.iloc[train_end:dev_end]["thread_id"]),
        "test": set(ordered.iloc[dev_end:]["thread_id"]),
    }


def assert_disjoint(splits: dict[str, set[str]]) -> None:
    assert not (splits["train"] & splits["dev"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["dev"] & splits["test"])


def run(config_path: Path, rebuild_staging: bool = False) -> dict[str, Any]:
    config = _load_config(config_path)
    raw_path = Path(config["paths"]["raw_csv"])
    db_path = Path(config["paths"]["staging_db"])
    processed_dir = Path(config["paths"]["processed_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    account = config["brand"]["support_account"]
    batch_size = int(config["processing"]["batch_size"])
    _require_raw(raw_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if rebuild_staging or not db_path.exists():
        stats = stage_csv(raw_path, db_path, batch_size)
    else:
        stats_path = reports_dir / "staging_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    for key in ("malformed_rows", "duplicate_tweet_ids"):
        stats.setdefault(key, 0)
    (reports_dir / "staging_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    connection = sqlite3.connect(db_path)
    edge_rows = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    unique_edges = connection.execute(
        "SELECT COUNT(*) FROM (SELECT 1 FROM edges GROUP BY left_id,right_id)"
    ).fetchone()[0]
    missing_reference_targets = connection.execute(
        """SELECT COUNT(DISTINCT e.right_id) FROM edges e
           LEFT JOIN tweets t ON e.right_id=t.tweet_id WHERE t.tweet_id IS NULL"""
    ).fetchone()[0]
    connected_ids, rounds = find_connected_amazon_ids(connection, account)
    frame, edges = _read_connected(connection)
    amazon_rows = int((frame["author_id"].str.casefold() == account.casefold()).sum())
    connection.close()

    threads = enrich_threads(frame, edges, account)
    contaminated_count = 0
    if config["processing"].get("exclude_contaminated_threads", True):
        threads, contaminated_count = drop_contaminated_threads(threads, account)
    conversations = make_conversations(threads)
    pairs = make_pairs(threads)
    thread_languages = conversations.set_index("thread_id")["language"]
    threads["language"] = threads["thread_id"].map(thread_languages).fillna("und")
    if not pairs.empty:
        pairs["language"] = pairs["thread_id"].map(thread_languages).fillna("und")
    splits = temporal_split(conversations, config["processing"]["split_ratios"])
    assert_disjoint(splits)

    threads.to_parquet(processed_dir / "amazon_threads.parquet", index=False)
    threads[[
        "thread_id", "tweet_id", "author_id", "inbound", "created_at", "raw_text",
        "response_tweet_id", "in_response_to_tweet_id"
    ]].to_parquet(processed_dir / "amazon_raw.parquet", index=False)
    threads.to_parquet(processed_dir / "amazon_clean.parquet", index=False)
    conversations.to_parquet(processed_dir / "amazon_conversations.parquet", index=False)
    pairs.to_parquet(processed_dir / "amazon_pairs.parquet", index=False)
    for name, ids in splits.items():
        conversations.loc[conversations["thread_id"].isin(ids)].to_parquet(processed_dir / f"{name}.parquet", index=False)
    retrieval = pairs.loc[pairs["thread_id"].isin(splits["train"])].copy()
    retrieval.to_parquet(processed_dir / "retrieval_corpus.parquet", index=False)

    report = {
        **stats,
        "reply_edge_rows": edge_rows,
        "unique_reply_edges": unique_edges,
        "duplicate_reply_edge_rows": edge_rows - unique_edges,
        "unique_missing_reference_targets": missing_reference_targets,
        "amazonhelp_authored_rows": amazon_rows,
        "connected_tweets_before_contamination_filter": len(connected_ids),
        "graph_traversal_passes": rounds,
        "graph_traversal_strategy": "indexed SQLite recursive CTE over undirected reply edges",
        "contaminated_threads_excluded": contaminated_count,
        "final_usable_tweets": len(threads),
        "tweets_in_english_threads": int((threads["language"] == "en").sum()),
        "tweets_in_undetermined_language_threads": int((threads["language"] == "und").sum()),
        "invalid_connected_timestamps": int(frame["created_at"].pipe(_safe_timestamp).isna().sum()),
        "invalid_final_timestamps": int(threads["created_at_parsed"].isna().sum()),
        "amazonhelp_inbound_inconsistencies": int(
            ((threads["speaker_role"] == "AMAZON") & (threads["inbound"] != 0)).sum()
        ),
        "recovered_threads": len(conversations),
        "english_threads": int((conversations["language"] == "en").sum()),
        "incomplete_threads": int(conversations["incomplete_thread"].sum()),
        "branching_threads": int(conversations["branching_thread"].sum()),
        "extreme_temporal_gap_threads": int(conversations["temporal_gap_anomaly"].sum()),
        "customer_amazon_pairs": len(pairs),
        "train_threads": len(splits["train"]),
        "dev_threads": len(splits["dev"]),
        "test_threads": len(splits["test"]),
        "split_anchor": "thread terminal timestamp (robust to anomalous years-old linked ancestors)",
        "train_time_start": conversations.loc[conversations["thread_id"].isin(splits["train"]), "split_time"].min().isoformat(),
        "train_time_end": conversations.loc[conversations["thread_id"].isin(splits["train"]), "split_time"].max().isoformat(),
        "dev_time_start": conversations.loc[conversations["thread_id"].isin(splits["dev"]), "split_time"].min().isoformat(),
        "dev_time_end": conversations.loc[conversations["thread_id"].isin(splits["dev"]), "split_time"].max().isoformat(),
        "test_time_start": conversations.loc[conversations["thread_id"].isin(splits["test"]), "split_time"].min().isoformat(),
        "test_time_end": conversations.loc[conversations["thread_id"].isin(splits["test"]), "split_time"].max().isoformat(),
        "retrieval_threads": int(retrieval["thread_id"].nunique()) if not retrieval.empty else 0,
        "raw_sha256_not_computed": "Use scripts/hash_raw.py when provenance hashing is required; it is intentionally separate from fast runs.",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (reports_dir / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Preparation complete: %s", json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe AmazonHelp conversation artifacts.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--rebuild-staging", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args.config, rebuild_staging=args.rebuild_staging)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
