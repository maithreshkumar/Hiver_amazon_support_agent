from __future__ import annotations

import json
from pathlib import Path

from hiver_support.evaluation.reply_ratings import validate_reply_ratings


def read(name: str) -> dict[str, object]:
    return json.loads(Path("data/reports", name).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    intent = read("final_intent_metrics.json")["models"]
    confidence = read("confidence_analysis.json")
    routing = read("routing_metrics.json")
    retrieval = read("retrieval_analysis.json")
    latency = read("latency_analysis.json")
    responses = read("response_evaluation_summary.json")
    duplicates = read("near_duplicate_audit.json")
    failures = read("failure_analysis.json")["top_five"]
    provenance = json.loads(Path("data/golden/import_provenance.json").read_text(encoding="utf-8"))
    judge_path = Path("data/reports/reply_quality_judge.json")
    judge = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else None
    agreement_path = Path("data/reports/human_judge_agreement.json")
    agreement = json.loads(agreement_path.read_text(encoding="utf-8")) if agreement_path.exists() else None
    safety_outlier = (
        next(
            (
                item for item in agreement["largest_disagreements"]
                if item["example_id"] == "gold-147" and item["dimension"] == "safety"
            ),
            None,
        )
        if agreement else None
    )
    ratings_path = Path("data/golden/reply_quality_human_ratings.csv")
    rating_status = validate_reply_ratings(ratings_path) if ratings_path.exists() else {"rows": 0, "completed": 0, "remaining": 0}
    configured = next(row for row in confidence["thresholds"] if row["threshold"] == 0.72)

    lines = [
        "# Phase 3 evaluation report",
        "",
        "## Status",
        "",
        (
            f"**Phase 3 is complete.** Human reply ratings: **{rating_status['completed']}/{rating_status['rows']}**. "
            "The original frozen judge was not rerun or tuned after human ratings were observed."
            if agreement and rating_status["completed"] == rating_status["rows"]
            else f"Automated evaluation is complete through the human reply-rating checkpoint. Human reply ratings: **{rating_status['completed']}/{rating_status['rows']}**. Judge-agreement metrics remain pending."
        ),
        "",
        "## Golden-set integrity and provenance",
        "",
        "The canonical set contains **200 rows, 200 unique example IDs, and 200 unique thread IDs**. All rows have an approved intent, a human route, a non-empty reason, `human_gold` provenance, and a complete Boolean label. Overlap with training, weak labels, and retrieval is zero. The examples remain in their sealed order.",
        "",
        f"Imported source SHA-256: `{provenance['source_sha256']}`. Canonical normalized SHA-256: `{provenance['canonical_sha256']}`.",
        "",
        "Annotation methodology: **Human-reviewed golden set where the human made the final intent and routing decisions after reviewing the examples and correcting disagreements during assisted annotation.** This is not represented as independent blind annotation.",
        "",
        "## Final human-gold intent classification",
        "",
        "| System | Correct | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("most_frequent_trivial", "Most-frequent"),
        ("tfidf_logistic_regression", "TF-IDF + logistic regression"),
        ("distilroberta", "DistilRoBERTa"),
    ):
        value = intent[key]
        lines.append(
            f"| {label} | {value['correct']}/200 | {value['accuracy']:.4f} | {value['macro_precision']:.4f} | {value['macro_recall']:.4f} | {value['macro_f1']:.4f} | {value['weighted_f1']:.4f} |"
        )
    lines += [
        "",
        "Full per-intent precision/recall/F1/support and confusion matrices are stored in `data/reports/final_intent_metrics.json`; weak classes are not omitted.",
        "",
        "## Confidence and selective classification",
        "",
        f"The frozen Phase 2 threshold remains **0.72**; it was not tuned on gold. Full-coverage accuracy is {intent['distilroberta']['accuracy']:.4f}. At 0.72, coverage is **{configured['covered']}/200 ({pct(configured['coverage'])})**, with covered accuracy **{configured['accuracy_on_covered']:.4f}**, covered macro-F1 **{configured['macro_f1_on_covered']:.4f}**, and **{configured['low_confidence_escalations']}** low-confidence cases. Threshold sweeps are final-test observations only, not a new selection procedure.",
        "",
        "## End-to-end RAG and routing",
        "",
        f"The real classifier → qwen3 embedding retrieval → qwen3:4b generation → deterministic safety → routing path was run on a deterministic {routing['rows']}-example stratified subset. This subset was chosen before response-quality inspection to fit the required 40–50 human-rating range and the measured CPU generation cost. Intent headline metrics remain on all 200. The routing figures below must not be described as 200-row routing results.",
        "",
        f"Routing accuracy was **{routing['correct']}/{routing['rows']} ({pct(routing['accuracy'])})**. Escalation precision/recall/F1 were **{routing['escalation_precision']:.4f}/{routing['escalation_recall']:.4f}/{routing['escalation_f1']:.4f}**. There were **{routing['false_auto_handle_count']} false auto-handles** ({pct(routing['false_auto_handle_rate_among_human_escalations'])} of human escalations) and **{routing['false_escalation_count']} false escalations**. The system auto-handled {responses['auto_handle']} and escalated {responses['escalate']} cases; {responses['safety_overrides']} drafts triggered the deterministic safety backstop and {responses['provider_failures']} provider/structured-output failures safely escalated.",
        "",
        "Disagreements were primarily attributable to frozen-classifier errors, low confidence, unsupported-claim/safety overrides, provider failures, and conservative-policy mismatch. False auto-handle remains the higher-severity error even though false escalation is more common.",
        "",
        "## Retrieval evidence-usefulness diagnostics",
        "",
        f"Top-1 cosine similarity median/mean/p95 were **{retrieval['top1_similarity']['median']:.4f}/{retrieval['top1_similarity']['mean']:.4f}/{retrieval['top1_similarity']['p95']:.4f}**. No sampled case fell below the frozen 0.55 weak-evidence threshold. However, **{retrieval['high_similarity_classifier_mismatch_proxy_count']}** cases paired high similarity (≥0.65) with a wrong classifier intent, showing that high cosine similarity is not sufficient evidence relevance. All {retrieval['legacy_social_format_evidence_count']} retrieved replies contained at least one historical social-format marker (mention, agent initials, or link).",
        "",
        "There is no independently labelled retrieval-relevance standard and the index metadata has no trusted intent labels. Therefore these are similarity and evidence-usefulness diagnostics—not retrieval accuracy.",
        "",
        "## Reply baselines and independent LLM judge",
        "",
    ]
    if judge:
        lines += [
            f"A separate local `{judge['judge_model']}` judge scored blinded candidates from the trivial reply, TF-IDF nearest historical reply, and RAG+safety systems. The generator was `{judge['generator_model']}`. Prompt `{judge['prompt_version']}`, rubric `{judge['rubric_version']}`; human ratings were not visible.",
            "",
            "| Reply system | Overall | Grounded | Helpful | Actionable | Safe | Historical consistency | Avoids unsupported claims |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for system in ("trivial", "tfidf_retrieval_only", "rag_with_safety"):
            score = judge["mean_scores"][system]
            lines.append(
                f"| `{system}` | {score['overall']:.3f} | {score['groundedness']:.3f} | {score['helpfulness']:.3f} | {score['actionability']:.3f} | {score['safety']:.3f} | {score['historical_consistency']:.3f} | {score['unsupported_claim_avoidance']:.3f} |"
            )
        lines += ["", "These judge scores are automated measurements. Human validation is reported separately below."]
    else:
        lines.append("Judge run pending.")
    if agreement:
        pooled = agreement["overall_pooled_dimensions"]
        aggregate = agreement["per_response_aggregate"]
        lines += [
            "",
            "## Human–LLM judge agreement",
            "",
            f"Across **{agreement['paired_dimension_ratings']} paired dimension scores** from {agreement['human_rating_rows']} responses, exact agreement was **{pct(pooled['exact_agreement'])}** and agreement within ±1 was **{pct(pooled['agreement_within_one'])}**. Linear/quadratic weighted Cohen's kappa were **{pooled['linear_weighted_cohen_kappa']:.4f}/{pooled['quadratic_weighted_cohen_kappa']:.4f}**, indicating no useful agreement beyond chance in this sample. Pooled Pearson/Spearman correlations were **{pooled['pearson_correlation']:.4f}/{pooled['spearman_rank_correlation']:.4f}**.",
            "",
            f"Per-response aggregate human and judge means were **{aggregate['human_mean']:.3f}** and **{aggregate['judge_mean']:.3f}**. The judge's mean bias was **{aggregate['judge_minus_human_mean_bias']:.3f}** points, with Pearson/Spearman correlations **{aggregate['pearson_correlation']:.3f}/{aggregate['spearman_rank_correlation']:.3f}**.",
            "",
            "| Dimension | Exact | Within ±1 | Linear κ | Quadratic κ | Human mean | Judge mean | Judge−human bias |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for dimension in agreement["rubric_dimensions"]:
            metric = agreement["per_dimension"][dimension]
            lines.append(
                f"| `{dimension}` | {pct(metric['exact_agreement'])} | {pct(metric['agreement_within_one'])} | {metric['linear_weighted_cohen_kappa']:.3f} | {metric['quadratic_weighted_cohen_kappa']:.3f} | {metric['human_distribution']['mean']:.3f} | {metric['judge_distribution']['mean']:.3f} | {metric['judge_minus_human_mean_bias']:+.3f} |"
            )
        lines += [
            "",
            f"The largest systematic bias was Safety under-scoring: human mean {agreement['per_dimension']['safety']['human_distribution']['mean']:.3f} versus judge mean {agreement['per_dimension']['safety']['judge_distribution']['mean']:.3f} ({agreement['per_dimension']['safety']['judge_minus_human_mean_bias']:+.3f}). The judge also under-scored unsupported-claim avoidance ({agreement['per_dimension']['unsupported_claim_avoidance']['judge_minus_human_mean_bias']:+.3f}), historical consistency ({agreement['per_dimension']['historical_consistency']['judge_minus_human_mean_bias']:+.3f}), and groundedness ({agreement['per_dimension']['groundedness']['judge_minus_human_mean_bias']:+.3f}), while over-scoring helpfulness ({agreement['per_dimension']['helpfulness']['judge_minus_human_mean_bias']:+.3f}). Actionability bias was smaller ({agreement['per_dimension']['actionability']['judge_minus_human_mean_bias']:+.3f}), but agreement remained weak.",
            "",
            f"The deliberate privacy failure `gold-147` received human Safety={safety_outlier['human_score'] if safety_outlier else 'n.a.'} and judge Safety={safety_outlier['judge_score'] if safety_outlier else 'n.a.'}. The response repeated the customer's phone number publicly; the judge rationale incorrectly described the response as providing a secure contact method. This is direct evidence that the small local judge is not reliable enough to replace human safety review.",
            "",
            "Human rating preparation used semi-automation, with the human making every final score decision. Score distributions, 1–5 confusion matrices, correlations, and the 15 largest paired disagreements are preserved in `data/reports/human_judge_agreement.json` and companion CSVs.",
        ]
    total = latency["measurements_seconds"]["total_component_sum"]
    generation = latency["measurements_seconds"]["generation"]
    lines += [
        "",
        "## Latency and operating environment",
        "",
        f"On Windows x64 with a 12th Gen Intel Core i7-12650HX (20 logical processors), Ollama `qwen3:4b` generation median/p95 was **{generation['median']:.2f}s/{generation['p95']:.2f}s**. Total component-sum median/p95 was **{total['median']:.2f}s/{total['p95']:.2f}s**. Intent classification and retrieval medians were **{latency['measurements_seconds']['intent']['median']:.4f}s** and **{latency['measurements_seconds']['embedding_and_retrieval']['median']:.4f}s**. Embeddings were batched and amortized; generation remained real per-request wall time. CPU inference is too slow for production without acceleration or a hosted provider.",
        "",
        "## Top five observed failures",
        "",
    ]
    for index, failure in enumerate(failures, 1):
        lines += [
            f"### {index}. `{failure['category']}` — {failure['example_id']}",
            "",
            f"Customer: {failure['customer_message']}",
            "",
            f"Expected `{failure['expected_intent']}` / `{failure['expected_route']}`; predicted `{failure['predicted_intent']}` / `{failure['predicted_route']}` at confidence {failure['intent_confidence']:.4f}, top-1 similarity {failure['top1_similarity']:.4f}. Root cause category: `{failure['why']}`. The concrete response and remediation are preserved in `data/reports/failure_analysis.json`.",
            "",
        ]
    lines += [
        "## Taxonomy gaps",
        "",
        "Human review found legitimate support messages that do not cleanly fit the frozen 12 labels: pricing/product-price questions, delivery instructions, staff or driver conduct, packaging/environmental complaints, Amazon Pantry/cart questions, support availability, and device/application failures outside Prime Video. They remain honestly mapped to the closest approved label—often `general_or_context_missing`—because changing taxonomy after observing test data would compromise evaluation integrity.",
        "",
        "## Near-duplicate and sampling audit",
        "",
        f"Character TF-IDF (3–5 character n-grams), after lowercasing and removing URLs/mentions, found **{duplicates['exact_normalized_duplicate_pairs']} exact duplicate pairs** and **{duplicates['near_duplicate_pairs']} pairs across {duplicates['examples_in_near_duplicate_pairs']} examples** at cosine ≥ {duplicates['threshold']}. Human review also noted the accidental-Kindle-cancellation paraphrases `gold-004`/`gold-117`, which this lexical threshold misses. Nothing was deleted after labels were observed; these themes can modestly overweight behavior in a 200-example set.",
        "",
        "## What is misleading about my headline number?",
        "",
        f"The {intent['distilroberta']['accuracy']:.4f} DistilRoBERTa accuracy is for one brand and only {intent['distilroberta']['rows']} unevenly distributed examples. The set includes near duplicates, and weak-supervision training can favor lexical patterns. The taxonomy has known gaps; historical Twitter data is noisy; many messages depend on unavailable images or links. Conservative escalation can improve safety while reducing automation, and strong routing safety does not establish reply quality. Retrieved historical support behavior is neither current Amazon policy nor proof that an issue was resolved. Local CPU inference differs from production. The {routing['rows']}-response routing/RAG sample is smaller than the intent test. Human–judge agreement is weak, so automated judge scores are not a dependable quality proxy. Finally, both label and rating annotation used assisted review with human final decisions, not a fully independent multi-annotator protocol.",
        "",
        "## What I would do with one more week",
        "",
        "Add a second independent annotator and adjudication; expand weak classes and context-missing hard negatives using only training/development evidence; prototype link/image context handling; add intent-aware retrieval reranking and broader training-only evidence; calibrate confidence on development data; evaluate a taxonomy extension prospectively; benchmark GPU/hosted inference; add a current-policy knowledge source with timestamps; and monitor false-auto-handle plus unsupported-claim rates in production-like traffic.",
        "",
        "## Decision log",
        "",
        "| Decision | Alternatives considered | Reason | Trade-off |",
        "|---|---|---|---|",
        "| Lock to Amazon/AmazonHelp | Multi-brand classifier | Company-specific evidence and coherent behavior | Narrower external validity |",
        "| Keep 12 intents | Fine-grained long tail | Learnable, auditable routing units | Known support topics collapse into general |",
        "| Merge shipping-promise failure | Separate SLA label | Same operational delay/tracking path | Less SLA-specific analysis |",
        "| Split Prime membership/video | Combined Prime label | Different requests and evidence | Smaller per-class supports |",
        "| Use DistilRoBERTa primary classifier | Prompt an LLM | Fast, deterministic, locally persisted | Weak labels cap quality |",
        "| Semantic retrieval | LLM memory or keyword-only | Traceable historical evidence | Similarity is not relevance |",
        "| Describe historical behavior only | Claim current policy | Source does not prove present policy or outcomes | Replies must remain cautious |",
        "| Conservative escalation | Maximize automation | False auto-handle has higher safety cost | Many false escalations |",
        "| Training-only retrieval corpus | Index all data | Prevent test leakage | Less evidence coverage |",
        "| Sealed 200-example gold | Iterate on test examples | Honest final measurement | Small, uneven test set |",
        "| Weak supervision | Hand-label training set | Feasible within take-home constraints | Lexical/model bias |",
        "| Local Ollama | Hosted-only dependency | Reproducible without credentials | Slow CPU latency |",
        "| Deterministic safety backstop | Trust generator self-report | Catches stale links/sensitive or unsupported text | Over-blocking |",
        "| Freeze 0.72 threshold from development | Optimize on gold | Avoid test-set tuning | Low final coverage |",
        "| Stratified 48-reply quality sample | Generate/judge all 200 | Covers required human-rating range under measured CPU cost | Routing/RAG metrics are subset estimates |",
        "",
        "## Reproduction",
        "",
        "With dependencies, local models, and persisted response/judge outputs already present:",
        "",
        "```powershell",
        "python scripts\\validate_golden.py",
        "python scripts\\evaluate_final_intents.py",
        "python scripts\\audit_golden_duplicates.py --threshold 0.35",
        "python scripts\\analyze_phase3_outputs.py",
        "python scripts\\evaluate_human_judge_agreement.py",
        "python scripts\\build_phase3_report.py",
        "python -m pytest -q",
        "```",
        "",
        "Regenerating the qwen3:4b responses or independent judge is intentionally separate because CPU inference can exceed the approximately 15-minute headline-metric reproduction target:",
        "",
        "```powershell",
        "python scripts\\run_golden_responses.py --size 48",
        "python scripts\\judge_replies.py",
        "```",
        "",
        "## Human-rating completion",
        "",
        f"The canonical blinded file contains {rating_status['rows']} rows and {rating_status['completed']} completed ratings. The imported CSV was preserved byte-for-byte with SHA-256 `{agreement['ratings_sha256'] if agreement else 'pending'}`. The original judge was not regenerated after import. No further human checkpoint is required for this Phase 3 evaluation.",
    ]
    Path("PHASE_3_EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
