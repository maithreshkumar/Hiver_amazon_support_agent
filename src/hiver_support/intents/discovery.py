from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd
import yaml
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
_SPACE = re.compile(r"\s+")

_CANDIDATE_INTENTS = [
    {
        "name": "delivery_tracking_or_delay",
        "pattern": r"\b(?:deliver(?:y|ed|ing)?|package|parcel|tracking)\b",
        "definition": "Order location, estimated arrival, late delivery, or tracking-status questions.",
        "include": "Tracking and ordinary late-arrival questions.",
        "exclude": "A package explicitly marked delivered but missing; paid/Prime delivery-promise complaints.",
        "confusions": "delivered_but_missing, shipping_promise_failure",
    },
    {
        "name": "delivered_but_missing",
        "pattern": r"(?:says|marked|showing).{0,25}delivered.{0,45}(?:not|never|missing|can.?t find)|delivered.{0,35}(?:not here|never received|missing)",
        "definition": "Carrier or Amazon marks an order delivered, but the customer cannot locate it.",
        "include": "Delivered status plus explicit non-receipt.",
        "exclude": "Orders that are only late or still in transit.",
        "confusions": "delivery_tracking_or_delay",
    },
    {
        "name": "shipping_promise_failure",
        "pattern": r"\b(?:one|1|two|2)[ -]?day (?:shipping|delivery)|\bguaranteed delivery|\bprime shipping",
        "definition": "A paid, guaranteed, or Prime delivery-speed promise was not met.",
        "include": "One-day/two-day, guaranteed, or Prime shipping failures.",
        "exclude": "Ordinary delivery-status questions with no service-level promise.",
        "confusions": "delivery_tracking_or_delay, prime_membership_or_video",
    },
    {
        "name": "return_request",
        "pattern": r"\breturn(?:ed|ing|s)?\b",
        "definition": "Customer wants to send an item back or asks about return logistics.",
        "include": "Return eligibility, labels, pickup, or return status.",
        "exclude": "Refund-only questions that do not involve returning an item.",
        "confusions": "refund_or_charge, damaged_or_defective_item",
    },
    {
        "name": "refund_or_charge",
        "pattern": r"\brefund(?:ed|ing|s)?\b|money back|\bcharged?\b|\bpayment\b|credit card|debit card",
        "definition": "Refund status, duplicate/unknown charge, or payment problem.",
        "include": "Refunds, charges, failed payments, and payment-method issues.",
        "exclude": "Fraud/account takeover requiring a separate high-risk route.",
        "confusions": "return_request, account_access_or_security",
    },
    {
        "name": "order_cancellation",
        "pattern": r"\bcancel(?:led|ing|ation|s)?\b",
        "definition": "Customer wants to cancel an order or asks why it was cancelled.",
        "include": "Cancellation requests and unexpected cancellation.",
        "exclude": "Returns after fulfilment.",
        "confusions": "return_request, refund_or_charge",
    },
    {
        "name": "damaged_or_defective_item",
        "pattern": r"\b(?:damag(?:ed|e)|broken|defective|doesn.?t work|not working)\b",
        "definition": "Received item is physically damaged, defective, or unusable.",
        "include": "Broken packaging when product condition may be affected; defective products.",
        "exclude": "A different or incomplete item with no defect.",
        "confusions": "wrong_or_incomplete_item, return_request",
    },
    {
        "name": "wrong_or_incomplete_item",
        "pattern": r"wrong (?:item|product)|not what i ordered|ordered .{0,30} (?:got|received)|missing (?:item|part)",
        "definition": "Customer received the wrong product, quantity, or an incomplete order.",
        "include": "Wrong item, missing units, or missing parts.",
        "exclude": "Correct product that is damaged or entirely missing delivery.",
        "confusions": "damaged_or_defective_item, delivered_but_missing",
    },
    {
        "name": "account_access_or_security",
        "pattern": r"account.{0,35}(?:locked|login|log in|password|access|hack|unauthori[sz]ed)|(?:login|log in|password).{0,35}account|\bphishing\b|\bscam\b",
        "definition": "Login, account access, suspicious message, compromise, or security concern.",
        "include": "Locked accounts, password/login trouble, phishing, scams, and compromise signals.",
        "exclude": "Ordinary payment failures without an account-security concern.",
        "confusions": "refund_or_charge",
    },
    {
        "name": "prime_membership_or_video",
        "pattern": r"\bprime\b",
        "definition": "Prime membership, billing/benefits, or Prime Video availability/playback.",
        "include": "Membership questions and Prime Video issues.",
        "exclude": "A complaint specifically about the promised Prime shipping speed.",
        "confusions": "shipping_promise_failure",
    },
    {
        "name": "seller_or_marketplace_issue",
        "pattern": r"\b(?:seller|vendor|marketplace|third[ -]?party)\b",
        "definition": "Issue primarily involving a marketplace or third-party seller.",
        "include": "Seller conduct, seller contact, or marketplace fulfilment disputes.",
        "exclude": "Amazon-fulfilled delivery issues with no seller involvement.",
        "confusions": "delivery_tracking_or_delay, refund_or_charge",
    },
    {
        "name": "general_or_context_missing",
        "pattern": r"^\s*(?:MENTION\s*)*(?:help|hi|hey|hello|URL|\?|please help)?\s*$",
        "definition": "Too little text or missing linked-media context to infer an operational intent safely.",
        "include": "Greeting-only, help-only, mention-only, and link/image-dependent messages.",
        "exclude": "Messages with enough content for another intent.",
        "confusions": "all intents",
    },
]


def _propose_name(top_terms: list[str]) -> tuple[str, str]:
    joined = " ".join(top_terms)
    if "mention mention" in joined:
        return "low_context_general_support", "Candidate for OTHER/CLARIFICATION; merge overlapping low-context clusters."
    if "amazon pay" in joined or "pay balance" in joined:
        return "amazon_pay_or_account", "Likely needs a payment/account split after inspecting more examples."
    if "membership" in joined or "prime video" in joined:
        return "prime_membership_or_video", "Review whether membership billing and Prime Video availability should be split."
    if "shipping" in joined:
        return "shipping_sla_delay", "Separate from generic delivery status if examples consistently mention paid/Prime delivery promises."
    if "customer service" in joined or "worst" in joined:
        return "service_complaint", "Often lacks a concrete operational request; consider routing separately from fulfilment intents."
    if "order" in joined and ("phone" in joined or " id" in joined):
        return "order_status_with_identifier", "Review PII masking and merge with delivery status where the request is otherwise identical."
    if "package" in joined:
        return "missing_or_delayed_package", "Split delivered-but-missing from merely late when enough examples support both."
    if "delivery" in joined:
        return "delivery_status_or_driver_issue", "Review a split between tracking/delay and delivery-driver conduct."
    if "ordered" in joined or "received" in joined or "item" in joined:
        return "wrong_missing_or_unsatisfactory_item", "Review wrong-item, damaged-item, and refund subgroups."
    if "help" in joined or "hey" in joined or "mention mention" in joined:
        return "low_context_general_support", "Candidate for OTHER/CLARIFICATION; merge overlapping low-context clusters."
    if "url" in joined or "packaging" in joined:
        return "image_or_link_context_required", "Usually unsafe to auto-handle without the linked media context."
    return "mixed_long_tail", "Inspect and merge or split based on representative messages."


def _topic_text(value: object) -> str:
    text = html.unescape(str(value))
    text = _URL.sub(" URL ", text)
    text = _MENTION.sub(" MENTION ", text)
    return _SPACE.sub(" ", text).strip()


def discover(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["intent_discovery"]
    pairs_path = Path(config["paths"]["processed_dir"]) / "amazon_pairs.parquet"
    train_path = Path(config["paths"]["processed_dir"]) / "train.parquet"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Run scripts/prepare_data.py first; missing {pairs_path}")
    pairs = pd.read_parquet(pairs_path, columns=["thread_id", "customer_text", "language"])
    train_ids = set(pd.read_parquet(train_path, columns=["thread_id"])["thread_id"].astype(str))
    pairs = pairs.loc[pairs["thread_id"].astype(str).isin(train_ids)]
    pairs = pairs.dropna(subset=["customer_text"]).drop_duplicates("thread_id")
    pairs = pairs.loc[pairs["language"] == "en"].copy()
    pairs["topic_text"] = pairs["customer_text"].map(_topic_text)
    candidate_evidence: list[dict[str, object]] = []
    for candidate in _CANDIDATE_INTENTS:
        matches = pairs["topic_text"].str.contains(candidate["pattern"], case=False, regex=True, na=False)
        examples = pairs.loc[matches, ["thread_id", "customer_text"]].head(5).to_dict("records")
        candidate_evidence.append(
            {
                **candidate,
                "keyword_signal_count": int(matches.sum()),
                "representative_examples": examples,
                "status": "pending_human_approval",
            }
        )
    if len(pairs) > int(settings["max_messages"]):
        pairs = pairs.sample(int(settings["max_messages"]), random_state=config["processing"]["random_seed"])
    cluster_count = min(int(settings["candidate_clusters"]), max(2, len(pairs) // 20))
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=int(settings["min_df"]),
        max_df=0.92,
        max_features=int(settings["max_features"]),
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(pairs["topic_text"])
    model = NMF(
        n_components=cluster_count,
        random_state=config["processing"]["random_seed"],
        init="nndsvda",
        max_iter=300,
    )
    topic_weights = model.fit_transform(matrix)
    pairs["cluster"] = topic_weights.argmax(axis=1)
    terms = vectorizer.get_feature_names_out()

    proposals: list[dict[str, object]] = []
    markdown = [
        "# AmazonHelp intent taxonomy proposal",
        "",
        "> Status: **pending human approval**. These are unsupervised evidence clusters, not final intents.",
        "",
        "Both the signal analysis and topic model are restricted to **English training threads**. "
        "Counts below are overlapping keyword signals, not labels or evaluation results. A human must "
        "approve names, definitions, merges, and splits before weak labeling begins.",
        "",
        "## Candidate taxonomy (AI-proposed, human approval required)",
        "",
        "| Candidate | Keyword-signal threads | Definition | Common confusions |",
        "|---|---:|---|---|",
    ]
    for candidate in candidate_evidence:
        markdown.append(
            f"| `{candidate['name']}` | {candidate['keyword_signal_count']} | "
            f"{candidate['definition']} | {candidate['confusions']} |"
        )
    markdown.extend(
        [
            "",
            "Recommended review: keep the taxonomy near 8–15 intents; merge categories that cannot be "
            "labelled consistently, but retain `general_or_context_missing` so vague messages are not "
            "forced into a confident operational class.",
            "",
            "## Unsupervised topic evidence",
            "",
        ]
    )
    for cluster in range(cluster_count):
        member_positions = [position for position, value in enumerate(pairs["cluster"].tolist()) if value == cluster]
        nearest_positions = sorted(
            member_positions, key=lambda position: topic_weights[position, cluster], reverse=True
        )[:8]
        top_term_indices = model.components_[cluster].argsort()[-12:][::-1]
        top_terms = [str(terms[index]) for index in top_term_indices]
        proposed_name, review_note = _propose_name(top_terms)
        examples = pairs.iloc[nearest_positions][["thread_id", "customer_text"]].to_dict("records")
        item = {
            "cluster_id": int(cluster),
            "proposed_name": proposed_name,
            "count_in_sample": len(member_positions),
            "top_terms": top_terms,
            "representative_examples": examples,
            "definition": "",
            "inclusion_criteria": [],
            "exclusion_criteria": [],
            "common_confusions": [],
            "decision": "keep_merge_or_split",
            "review_note": review_note,
        }
        proposals.append(item)
        markdown.extend(
            [
                f"## Cluster {cluster} — proposed `{proposed_name}`",
                "",
                f"Sample count: **{len(member_positions)}**",
                "",
                "Top terms: " + ", ".join(f"`{term}`" for term in top_terms),
                "",
                f"AI review note: {review_note}",
                "",
                "Representative messages:",
                "",
            ]
        )
        for example in examples:
            clean = str(example["customer_text"]).replace("\n", " ")
            markdown.append(f"- `{example['thread_id']}` — {clean}")
        markdown.extend(["", "Human decision: _keep / merge / split / discard_", ""])

    report_dir = Path(config["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "intent_taxonomy_proposal.md"
    output.write_text("\n".join(markdown), encoding="utf-8")
    proposal_path = report_dir / "intent_taxonomy_proposal.yaml"
    proposal_path.write_text(
        yaml.safe_dump(
            {
                "status": "pending_human_approval",
                "method": "English-only TF-IDF (1-2 grams) + non-negative matrix factorization",
                "sample_size": len(pairs),
                "candidate_intents": candidate_evidence,
                "clusters": proposals,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    pairs[["thread_id", "customer_text", "language", "cluster"]].to_parquet(
        report_dir / "intent_cluster_assignments.parquet", index=False
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose data-derived AmazonHelp intent clusters.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    output = discover(args.config)
    print(f"Wrote human-review proposal to {output}")


if __name__ == "__main__":
    main()
