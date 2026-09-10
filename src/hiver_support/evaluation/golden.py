from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

_RISK = re.compile(
    r"\b(?:fraud|hacked|unauthori[sz]ed|stolen|lawyer|legal action|injur(?:y|ed)|dangerous|chargeback)\b",
    re.IGNORECASE,
)
_NOISY = re.compile(r"https?://|[^\w\s]{3,}|[\U0001F300-\U0001FAFF]")


def _bucket(text: str) -> str:
    if _RISK.search(text):
        return "escalation_risk"
    if len(text.split()) <= 5:
        return "ambiguous_or_short"
    if _NOISY.search(text):
        return "noisy_or_context_missing"
    return "common"


def create_candidates(config_path: Path, size: int = 200) -> Path:
    if not 150 <= size <= 250:
        raise ValueError("Golden candidate size must be between 150 and 250")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    processed = Path(config["paths"]["processed_dir"])
    pairs = pd.read_parquet(processed / "amazon_pairs.parquet")
    test_ids = set(pd.read_parquet(processed / "test.parquet", columns=["thread_id"])["thread_id"].astype(str))
    candidates = pairs.loc[pairs["thread_id"].astype(str).isin(test_ids)].drop_duplicates("thread_id").copy()
    candidates = candidates.loc[candidates["language"] == "en"].copy()
    candidates["sampling_bucket"] = candidates["customer_text"].map(_bucket)

    targets = {
        "common": round(size * 0.60),
        "ambiguous_or_short": round(size * 0.15),
        "escalation_risk": round(size * 0.15),
    }
    targets["noisy_or_context_missing"] = size - sum(targets.values())
    selected: list[pd.DataFrame] = []
    used: set[str] = set()
    seed = int(config["processing"]["random_seed"])
    for bucket, target in targets.items():
        pool = candidates.loc[candidates["sampling_bucket"] == bucket]
        take = min(target, len(pool))
        sample = pool.sample(take, random_state=seed + len(selected)) if take else pool.head(0)
        selected.append(sample)
        used.update(sample["thread_id"].astype(str))
    result = pd.concat(selected, ignore_index=True)
    if len(result) < size:
        remainder = candidates.loc[~candidates["thread_id"].astype(str).isin(used)]
        result = pd.concat(
            [result, remainder.sample(size - len(result), random_state=seed + 99)], ignore_index=True
        )

    output = pd.DataFrame(
        {
            "example_id": [f"gold-{index:03d}" for index in range(1, len(result) + 1)],
            "thread_id": result["thread_id"].astype(str),
            "customer_message": result["customer_raw_text"],
            "sampling_bucket": result["sampling_bucket"],
            "ai_suggested_intent": "PENDING_APPROVED_TAXONOMY",
            "ai_suggested_escalation": result["customer_text"].map(
                lambda text: "ESCALATE" if _RISK.search(text) else "PENDING_POLICY_REVIEW"
            ),
            "ai_suggested_reason": result["customer_text"].map(
                lambda text: "Risk keyword requires human review" if _RISK.search(text) else ""
            ),
            "human_intent": "",
            "human_escalation": "",
            "human_reason": "",
            "human_label_complete": False,
        }
    )
    golden_dir = Path("data/golden")
    golden_dir.mkdir(parents=True, exist_ok=True)
    output_path = golden_dir / "golden_candidates.csv"
    output.to_csv(output_path, index=False)
    return output_path

