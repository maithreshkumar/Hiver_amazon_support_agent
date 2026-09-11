# Phase 3 evaluation report

## Status

**Phase 3 is complete.** Final post-repair human reply ratings: **48/48**. The historical pre-repair judge remains unchanged. The fixed-rubric post-repair judge comparison is explicitly post-hoc and was not tuned against the human scores.

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

After manually observed runtime defects, the saved raw drafts were passed through the repaired deterministic validator, router, and canonical final-response builder without rerunning or tuning the LLM. This changed 3 routes and 33 final replies. These are the final runtime routing metrics.

Routing accuracy was **31/48 (64.58%)**. Escalation precision/recall/F1 were **0.6383/1.0000/0.7792**. There were **0 false auto-handles** (0.00% of human escalations) and **17 false escalations**. The system auto-handled 1 and escalated 47 cases; 47 drafts triggered the deterministic safety backstop and 5 provider/structured-output failures safely escalated.

Disagreements were primarily attributable to frozen-classifier errors, low confidence, unsupported-claim/safety overrides, provider failures, and conservative-policy mismatch. False auto-handle remains the higher-severity error even though false escalation is more common.

## Retrieval evidence-usefulness diagnostics

Top-1 cosine similarity median/mean/p95 were **0.7777/0.7670/0.8467**. No sampled case fell below the frozen 0.55 weak-evidence threshold. However, **21** cases paired high similarity (≥0.65) with a wrong classifier intent, showing that high cosine similarity is not sufficient evidence relevance. All 240 retrieved replies contained at least one historical social-format marker (mention, agent initials, or link).

There is no independently labelled retrieval-relevance standard and the index metadata has no trusted intent labels. Therefore these are similarity and evidence-usefulness diagnostics—not retrieval accuracy.

## Reply baselines and independent LLM judge

A separate local `qwen2.5:1.5b` judge scored blinded candidates from the trivial reply, TF-IDF nearest historical reply, and RAG+safety systems. The generator was `qwen3:4b`. Prompt `reply-judge-v1`, rubric `support-quality-v1`; human ratings were not visible.

| Reply system | Overall | Grounded | Helpful | Actionable | Safe | Historical consistency | Avoids unsupported claims |
|---|---:|---:|---:|---:|---:|---:|---:|
| `trivial` | 3.417 | 3.417 | 3.375 | 3.375 | 3.438 | 3.458 | 3.438 |
| `tfidf_retrieval_only` | 3.351 | 3.396 | 3.292 | 3.292 | 3.354 | 3.375 | 3.396 |
| `rag_with_safety` | 3.267 | 3.312 | 3.208 | 3.208 | 3.271 | 3.292 | 3.312 |

These judge scores are automated measurements. Human validation is reported separately below.

## Human–LLM judge agreement

These metrics describe the final post-repair responses. The human personally ranked every score without seeing judge scores. The post-repair judge pass retained the original model, rubric, prompt, and temperature; it reused 15 byte-identical judgments and evaluated 33 changed replies. Because it occurred after the behavioral repair and human-rating checkpoint, it is explicitly post-hoc and does not replace the preserved pre-repair evaluation.

Across **288 paired dimension scores** from 48 responses, exact agreement was **16.67%** and agreement within ±1 was **52.08%**. Linear/quadratic weighted Cohen's kappa were **-0.1081/-0.1158**, indicating no useful agreement beyond chance in this sample. Pooled Pearson/Spearman correlations were **-0.1303/-0.1468**.

Per-response aggregate human and judge means were **3.854** and **3.267**. The judge's mean bias was **-0.587** points, with Pearson/Spearman correlations **-0.247/-0.295**.

| Dimension | Exact | Within ±1 | Linear κ | Quadratic κ | Human mean | Judge mean | Judge−human bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| `groundedness` | 14.58% | 52.08% | -0.200 | -0.220 | 3.646 | 3.312 | -0.333 |
| `helpfulness` | 14.58% | 54.17% | -0.149 | -0.208 | 2.708 | 3.208 | +0.500 |
| `actionability` | 25.00% | 66.67% | -0.096 | -0.087 | 3.417 | 3.208 | -0.208 |
| `safety` | 12.50% | 50.00% | 0.000 | 0.000 | 5.000 | 3.271 | -1.729 |
| `historical_consistency` | 18.75% | 47.92% | -0.208 | -0.243 | 3.854 | 3.292 | -0.562 |
| `unsupported_claim_avoidance` | 14.58% | 41.67% | -0.066 | -0.100 | 4.500 | 3.312 | -1.188 |

The largest systematic bias was Safety under-scoring: human mean 5.000 versus judge mean 3.271 (-1.729). The judge also under-scored unsupported-claim avoidance (-1.188), historical consistency (-0.562), and groundedness (-0.333), while over-scoring helpfulness (+0.500). Actionability bias was smaller (-0.208), but agreement remained weak.

The original pre-repair `gold-147` response repeated a phone number and received human Safety=1. The repaired final response no longer echoes that PII; the post-repair human file assigns Safety=5 to all 48 final responses. The historical rating and judge artifacts remain packaged separately so this repair is auditable rather than erased.

Automation was used to prepare and enter the rating CSV, but the human personally ranked and approved every score. Score distributions, 1–5 confusion matrices, correlations, and the 15 largest paired disagreements are preserved in `data/reports/post_repair/human_judge_agreement_post_repair.json` and companion CSVs.

## Latency and operating environment

On Windows x64 with a 12th Gen Intel Core i7-12650HX (20 logical processors), Ollama `qwen3:4b` generation median/p95 was **20.28s/71.21s**. Total component-sum median/p95 was **20.59s/71.52s**. Intent classification and retrieval medians were **0.0246s** and **0.2824s**. Embeddings were batched and amortized; generation remained real per-request wall time. CPU inference is too slow for production without acceleration or a hosted provider.

## Top five observed failures

### 1. `high_confidence_classifier_error` — gold-045

Customer: @115850 someone is placing orders in my number continuously. I am getting calls 4m ur delivery agents regarding this everyday. But I dnt knw the bastard. Can u pls cl me. Hw can I provide my contact number ?

Expected `account_access_or_security` / `ESCALATE`; predicted `delivery_tracking_or_delay` / `ESCALATE` at confidence 0.8265, top-1 similarity 0.7992. Root cause category: `classifier_error`. The concrete response and remediation are preserved in `data/reports/post_repair/failure_analysis.json`.

### 2. `context_missing_or_taxonomy_gap` — gold-028

Customer: @115850 Are your call center not working 24/7?

Expected `general_or_context_missing` / `AUTO_HANDLE`; predicted `damaged_or_defective_item` / `ESCALATE` at confidence 0.6461, top-1 similarity 0.7532. Root cause category: `classifier_error`. The concrete response and remediation are preserved in `data/reports/post_repair/failure_analysis.json`.

### 3. `false_escalation` — gold-011

Customer: @AmazonHelp grrrr trying to watch Vikings today on prime but it keeps saying band with to low.

A load of b.s. everything else is running fine

Expected `prime_video_issue` / `AUTO_HANDLE`; predicted `prime_video_issue` / `ESCALATE` at confidence 0.9280, top-1 similarity 0.7459. Root cause category: `unsupported_claim`. The concrete response and remediation are preserved in `data/reports/post_repair/failure_analysis.json`.

### 4. `additional_material_disagreement` — gold-117

Customer: @AmazonHelp hi, I have accidentally purchased a kindle book (missclick), can I somehow cancel the order?

Expected `order_cancellation` / `AUTO_HANDLE`; predicted `order_cancellation` / `ESCALATE` at confidence 0.8614, top-1 similarity 0.8198. Root cause category: `unsupported_claim`. The concrete response and remediation are preserved in `data/reports/post_repair/failure_analysis.json`.

### 5. `additional_material_disagreement` — gold-004

Customer: Dear @115830 how the hell do you cancel a kindle book ordered in error? There is no option to cancel!!

Expected `order_cancellation` / `AUTO_HANDLE`; predicted `order_cancellation` / `ESCALATE` at confidence 0.8457, top-1 similarity 0.8085. Root cause category: `unsupported_claim`. The concrete response and remediation are preserved in `data/reports/post_repair/failure_analysis.json`.

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

From an artifact-ready clone, the single reviewer command is:

```powershell
python scripts\reproduce_headline.py
```

Artifact fetch and explicit verification, when needed:

```powershell
python scripts\fetch_submission_artifacts.py
python scripts\verify_submission_artifacts.py
python scripts\preflight_submission.py
```

## Human-rating completion

The final post-repair file contains 48 rows and 48 completed ratings. It is preserved byte-for-byte with SHA-256 `b06709c781980fe56866f201c9342ea5c912e047a9ca1b6a43c6c43373bf5745`. The fixed post-repair judge artifact is used only as a comparison against these human scores; the historical pre-repair evaluation is retained separately. No further human checkpoint is required.
