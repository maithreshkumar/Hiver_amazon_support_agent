# Phase 3 evaluation report

## Status

**Phase 3 is complete.** Human reply ratings: **48/48**. The original frozen judge was not rerun or tuned after human ratings were observed.

## Golden-set integrity and provenance

The canonical set contains **200 rows, 200 unique example IDs, and 200 unique thread IDs**. All rows have an approved intent, a human route, a non-empty reason, `human_gold` provenance, and a complete Boolean label. Overlap with training, weak labels, and retrieval is zero. The examples remain in their sealed order.

Imported source SHA-256: `1adea385334692171f637cb86a20d38c30af6a77e65dc095a4daf4bb13757a17`. Canonical normalized SHA-256: `9273e9c34ffa28579b898af20f8954c76fc5d386c39ed9a3a259d168cd36cd0f`.

Annotation methodology: **Human-reviewed golden set where the human made the final intent and routing decisions after reviewing the examples and correcting disagreements during assisted annotation.** This is not represented as independent blind annotation.

## Final human-gold intent classification

| System | Correct | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Most-frequent | 47/200 | 0.2350 | 0.0196 | 0.0833 | 0.0317 | 0.0894 |
| TF-IDF + logistic regression | 69/200 | 0.3450 | 0.3679 | 0.5413 | 0.3538 | 0.2885 |
| DistilRoBERTa | 82/200 | 0.4100 | 0.4116 | 0.6299 | 0.4325 | 0.4070 |

Full per-intent precision/recall/F1/support and confusion matrices are stored in `data/reports/final_intent_metrics.json`; weak classes are not omitted.

## Confidence and selective classification

The frozen Phase 2 threshold remains **0.72**; it was not tuned on gold. Full-coverage accuracy is 0.4100. At 0.72, coverage is **73/200 (36.50%)**, with covered accuracy **0.5342**, covered macro-F1 **0.4907**, and **127** low-confidence cases. Threshold sweeps are final-test observations only, not a new selection procedure.

## End-to-end RAG and routing

The real classifier → qwen3 embedding retrieval → qwen3:4b generation → deterministic safety → routing path was run on a deterministic 48-example stratified subset. This subset was chosen before response-quality inspection to fit the required 40–50 human-rating range and the measured CPU generation cost. Intent headline metrics remain on all 200. The routing figures below must not be described as 200-row routing results.

Routing accuracy was **30/48 (62.50%)**. Escalation precision/recall/F1 were **0.6364/0.9333/0.7568**. There were **2 false auto-handles** (6.67% of human escalations) and **16 false escalations**. The system auto-handled 4 and escalated 44 cases; 29 drafts triggered the deterministic safety backstop and 5 provider/structured-output failures safely escalated.

Disagreements were primarily attributable to frozen-classifier errors, low confidence, unsupported-claim/safety overrides, provider failures, and conservative-policy mismatch. False auto-handle remains the higher-severity error even though false escalation is more common.

## Retrieval evidence-usefulness diagnostics

Top-1 cosine similarity median/mean/p95 were **0.7777/0.7670/0.8467**. No sampled case fell below the frozen 0.55 weak-evidence threshold. However, **21** cases paired high similarity (≥0.65) with a wrong classifier intent, showing that high cosine similarity is not sufficient evidence relevance. All 240 retrieved replies contained at least one historical social-format marker (mention, agent initials, or link).

There is no independently labelled retrieval-relevance standard and the index metadata has no trusted intent labels. Therefore these are similarity and evidence-usefulness diagnostics—not retrieval accuracy.

## Reply baselines and independent LLM judge

A separate local `qwen2.5:1.5b` judge scored blinded candidates from the trivial reply, TF-IDF nearest historical reply, and RAG+safety systems. The generator was `qwen3:4b`. Prompt `reply-judge-v1`, rubric `support-quality-v1`; human ratings were not visible.

| Reply system | Overall | Grounded | Helpful | Actionable | Safe | Historical consistency | Avoids unsupported claims |
|---|---:|---:|---:|---:|---:|---:|---:|
| `trivial` | 3.590 | 3.583 | 3.562 | 3.500 | 3.625 | 3.646 | 3.625 |
| `tfidf_retrieval_only` | 3.608 | 3.646 | 3.562 | 3.500 | 3.625 | 3.646 | 3.667 |
| `rag_with_safety` | 3.462 | 3.500 | 3.417 | 3.354 | 3.479 | 3.500 | 3.521 |

These judge scores are automated measurements. Human validation is reported separately below.

## Human–LLM judge agreement

Across **288 paired dimension scores** from 48 responses, exact agreement was **21.18%** and agreement within ±1 was **62.50%**. Linear/quadratic weighted Cohen's kappa were **-0.0532/-0.0635**, indicating no useful agreement beyond chance in this sample. Pooled Pearson/Spearman correlations were **-0.0699/-0.0669**.

Per-response aggregate human and judge means were **3.948** and **3.462**. The judge's mean bias was **-0.486** points, with Pearson/Spearman correlations **-0.178/-0.180**.

| Dimension | Exact | Within ±1 | Linear κ | Quadratic κ | Human mean | Judge mean | Judge−human bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| `groundedness` | 12.50% | 62.50% | -0.128 | -0.102 | 3.833 | 3.500 | -0.333 |
| `helpfulness` | 29.17% | 64.58% | 0.027 | 0.051 | 2.917 | 3.417 | +0.500 |
| `actionability` | 29.17% | 70.83% | -0.082 | -0.130 | 3.458 | 3.354 | -0.104 |
| `safety` | 14.58% | 52.08% | -0.041 | -0.070 | 4.917 | 3.479 | -1.438 |
| `historical_consistency` | 27.08% | 66.67% | -0.086 | -0.155 | 4.146 | 3.500 | -0.646 |
| `unsupported_claim_avoidance` | 14.58% | 58.33% | -0.064 | -0.101 | 4.417 | 3.521 | -0.896 |

The largest systematic bias was Safety under-scoring: human mean 4.917 versus judge mean 3.479 (-1.438). The judge also under-scored unsupported-claim avoidance (-0.896), historical consistency (-0.646), and groundedness (-0.333), while over-scoring helpfulness (+0.500). Actionability bias was smaller (-0.104), but agreement remained weak.

The deliberate privacy failure `gold-147` received human Safety=1 and judge Safety=5. The response repeated the customer's phone number publicly; the judge rationale incorrectly described the response as providing a secure contact method. This is direct evidence that the small local judge is not reliable enough to replace human safety review.

Human rating preparation used semi-automation, with the human making every final score decision. Score distributions, 1–5 confusion matrices, correlations, and the 15 largest paired disagreements are preserved in `data/reports/human_judge_agreement.json` and companion CSVs.

## Latency and operating environment

On Windows x64 with a 12th Gen Intel Core i7-12650HX (20 logical processors), Ollama `qwen3:4b` generation median/p95 was **20.28s/71.21s**. Total component-sum median/p95 was **20.59s/71.52s**. Intent classification and retrieval medians were **0.0246s** and **0.2824s**. Embeddings were batched and amortized; generation remained real per-request wall time. CPU inference is too slow for production without acceleration or a hosted provider.

## Top five observed failures

### 1. `unsafe_false_auto_handle` — gold-101

Customer: @AmazonHelp so why would amazon cancel my order when all I asked for was an update because it wasn’t in. Now you want me to pay an extra 60 bucks to reorder my items because the sales over? #baitandswitch

Expected `order_cancellation` / `ESCALATE`; predicted `order_cancellation` / `AUTO_HANDLE` at confidence 0.8595, top-1 similarity 0.7236. Root cause category: `overly_conservative_or_policy_mismatch`. The concrete response and remediation are preserved in `data/reports/failure_analysis.json`.

### 2. `high_confidence_classifier_error` — gold-045

Customer: @115850 someone is placing orders in my number continuously. I am getting calls 4m ur delivery agents regarding this everyday. But I dnt knw the bastard. Can u pls cl me. Hw can I provide my contact number ?

Expected `account_access_or_security` / `ESCALATE`; predicted `delivery_tracking_or_delay` / `ESCALATE` at confidence 0.8265, top-1 similarity 0.7992. Root cause category: `classifier_error`. The concrete response and remediation are preserved in `data/reports/failure_analysis.json`.

### 3. `context_missing_or_taxonomy_gap` — gold-028

Customer: @115850 Are your call center not working 24/7?

Expected `general_or_context_missing` / `AUTO_HANDLE`; predicted `damaged_or_defective_item` / `ESCALATE` at confidence 0.6461, top-1 similarity 0.7532. Root cause category: `classifier_error`. The concrete response and remediation are preserved in `data/reports/failure_analysis.json`.

### 4. `false_escalation` — gold-011

Customer: @AmazonHelp grrrr trying to watch Vikings today on prime but it keeps saying band with to low.

A load of b.s. everything else is running fine

Expected `prime_video_issue` / `AUTO_HANDLE`; predicted `prime_video_issue` / `ESCALATE` at confidence 0.9280, top-1 similarity 0.7459. Root cause category: `unsupported_claim`. The concrete response and remediation are preserved in `data/reports/failure_analysis.json`.

### 5. `additional_material_disagreement` — gold-117

Customer: @AmazonHelp hi, I have accidentally purchased a kindle book (missclick), can I somehow cancel the order?

Expected `order_cancellation` / `AUTO_HANDLE`; predicted `order_cancellation` / `ESCALATE` at confidence 0.8614, top-1 similarity 0.8198. Root cause category: `unsupported_claim`. The concrete response and remediation are preserved in `data/reports/failure_analysis.json`.

## Taxonomy gaps

Human review found legitimate support messages that do not cleanly fit the frozen 12 labels: pricing/product-price questions, delivery instructions, staff or driver conduct, packaging/environmental complaints, Amazon Pantry/cart questions, support availability, and device/application failures outside Prime Video. They remain honestly mapped to the closest approved label—often `general_or_context_missing`—because changing taxonomy after observing test data would compromise evaluation integrity.

## Near-duplicate and sampling audit

Character TF-IDF (3–5 character n-grams), after lowercasing and removing URLs/mentions, found **0 exact duplicate pairs** and **8 pairs across 9 examples** at cosine ≥ 0.35. Human review also noted the accidental-Kindle-cancellation paraphrases `gold-004`/`gold-117`, which this lexical threshold misses. Nothing was deleted after labels were observed; these themes can modestly overweight behavior in a 200-example set.

## What is misleading about my headline number?

The 0.4100 DistilRoBERTa accuracy is for one brand and only 200 unevenly distributed examples. The set includes near duplicates, and weak-supervision training can favor lexical patterns. The taxonomy has known gaps; historical Twitter data is noisy; many messages depend on unavailable images or links. Conservative escalation can improve safety while reducing automation, and strong routing safety does not establish reply quality. Retrieved historical support behavior is neither current Amazon policy nor proof that an issue was resolved. Local CPU inference differs from production. The 48-response routing/RAG sample is smaller than the intent test. Human–judge agreement is weak, so automated judge scores are not a dependable quality proxy. Finally, both label and rating annotation used assisted review with human final decisions, not a fully independent multi-annotator protocol.

## What I would do with one more week

Add a second independent annotator and adjudication; expand weak classes and context-missing hard negatives using only training/development evidence; prototype link/image context handling; add intent-aware retrieval reranking and broader training-only evidence; calibrate confidence on development data; evaluate a taxonomy extension prospectively; benchmark GPU/hosted inference; add a current-policy knowledge source with timestamps; and monitor false-auto-handle plus unsupported-claim rates in production-like traffic.

## Decision log

| Decision | Alternatives considered | Reason | Trade-off |
|---|---|---|---|
| Lock to Amazon/AmazonHelp | Multi-brand classifier | Company-specific evidence and coherent behavior | Narrower external validity |
| Keep 12 intents | Fine-grained long tail | Learnable, auditable routing units | Known support topics collapse into general |
| Merge shipping-promise failure | Separate SLA label | Same operational delay/tracking path | Less SLA-specific analysis |
| Split Prime membership/video | Combined Prime label | Different requests and evidence | Smaller per-class supports |
| Use DistilRoBERTa primary classifier | Prompt an LLM | Fast, deterministic, locally persisted | Weak labels cap quality |
| Semantic retrieval | LLM memory or keyword-only | Traceable historical evidence | Similarity is not relevance |
| Describe historical behavior only | Claim current policy | Source does not prove present policy or outcomes | Replies must remain cautious |
| Conservative escalation | Maximize automation | False auto-handle has higher safety cost | Many false escalations |
| Training-only retrieval corpus | Index all data | Prevent test leakage | Less evidence coverage |
| Sealed 200-example gold | Iterate on test examples | Honest final measurement | Small, uneven test set |
| Weak supervision | Hand-label training set | Feasible within take-home constraints | Lexical/model bias |
| Local Ollama | Hosted-only dependency | Reproducible without credentials | Slow CPU latency |
| Deterministic safety backstop | Trust generator self-report | Catches stale links/sensitive or unsupported text | Over-blocking |
| Freeze 0.72 threshold from development | Optimize on gold | Avoid test-set tuning | Low final coverage |
| Stratified 48-reply quality sample | Generate/judge all 200 | Covers required human-rating range under measured CPU cost | Routing/RAG metrics are subset estimates |

## Reproduction

With dependencies, local models, and persisted response/judge outputs already present:

```powershell
python scripts\validate_golden.py
python scripts\evaluate_final_intents.py
python scripts\audit_golden_duplicates.py --threshold 0.35
python scripts\analyze_phase3_outputs.py
python scripts\evaluate_human_judge_agreement.py
python scripts\build_phase3_report.py
python -m pytest -q
```

Regenerating the qwen3:4b responses or independent judge is intentionally separate because CPU inference can exceed the approximately 15-minute headline-metric reproduction target:

```powershell
python scripts\run_golden_responses.py --size 48
python scripts\judge_replies.py
```

## Human-rating completion

The canonical blinded file contains 48 rows and 48 completed ratings. The imported CSV was preserved byte-for-byte with SHA-256 `2dd1545f8322f5b6364138ed9dd533cfc04b4d3c8a59d472e6226d9bd59d732a`. The original judge was not regenerated after import. No further human checkpoint is required for this Phase 3 evaluation.
