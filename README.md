# Amazon/AmazonHelp AI Support Agent

An evidence-first support agent built for the Hiver SDE Intern assignment. It classifies Amazon support intent, retrieves training-only examples of historical AmazonHelp behavior, drafts a grounded response, applies deterministic safety validation, and chooses either `AUTO_HANDLE` or `ESCALATE`.

## Assignment objective

The system performs three connected tasks:

1. Classify each customer message into one of 12 human-approved intents.
2. Draft a response grounded in similar historical AmazonHelp conversations.
3. Route the case to `AUTO_HANDLE` or `ESCALATE`, treating unsafe automation as the higher-cost error.

The final runtime is:

```text
Customer message
    → DistilRoBERTa intent classification
    → qwen3-embedding:0.6b semantic retrieval
    → qwen3:4b grounded draft
    → deterministic validation
    → conservative risk/routing policy
    → final response
```

The deterministic layer prevents PII echoing, requests for sensitive data, stale historical URLs, unresolved placeholders, unsupported promises, and unsafe auto-handling. Escalations use context-aware safe templates and preserve explicit reason codes.

## What was deliberately not built

- Multi-brand support or a general customer-service classifier.
- Live Amazon account, order, payment, or fulfilment integrations.
- Autonomous cancellation, refund, account, or payment actions.
- A guarantee that historical Twitter replies represent current Amazon policy.
- Image, video, or linked-page understanding for tweets whose meaning is outside the text.
- A claim that an AmazonHelp reply proves the customer’s issue was resolved.

## Dataset

The source is the Kaggle **Customer Support on Twitter** dataset. The full source has 2,811,774 rows. AmazonHelp-seeded graph reconstruction produced 371,417 usable tweets, 82,246 reconstructed support threads, 61,421 English threads, and 168,216 direct customer-to-Amazon reply pairs.

Splits are chronological and thread-disjoint, using each thread’s terminal timestamp to avoid years-old linked ancestors distorting the boundary:

| Split | Threads |
|---|---:|
| Train | 57,572 |
| Development | 12,336 |
| Test | 12,338 |

The classifier used 2,514 accepted weak labels: 2,136 training rows and 378 internal validation rows. The deployed retrieval index contains 3,000 randomly seeded, training-only records. Final evaluation uses a sealed 200-example human golden set. Reply and routing evaluation uses a predetermined stratified subset of 48 examples.

The raw 492.6 MiB CSV, processed training corpus, SQLite staging database, and optimizer checkpoints are intentionally excluded from the submission. Compact hashed membership evidence independently verifies zero golden overlap with train, weak-label, and retrieval-index thread sets.

## Intent taxonomy

The human-approved taxonomy is frozen at version 1.0:

| Intent | Scope |
|---|---|
| `delivery_tracking_or_delay` | Tracking, arrival estimates, late delivery, and failed delivery-speed promises |
| `delivered_but_missing` | Marked delivered but not received or found |
| `return_request` | Return eligibility, logistics, label, pickup, or status |
| `refund_or_payment_issue` | Refund status, disputed charges, or payment-method failures |
| `order_cancellation` | Cancellation request or unexpected cancellation |
| `damaged_or_defective_item` | Damaged, broken, defective, or unusable product |
| `wrong_or_incomplete_item` | Wrong product, quantity, component, or incomplete order |
| `account_access_or_security` | Login, compromise, phishing, fraud, or unauthorized access |
| `prime_membership` | Prime enrollment, billing, cancellation, eligibility, or benefits |
| `prime_video_issue` | Prime Video playback, availability, subtitle, or application problems |
| `seller_or_marketplace_issue` | Third-party seller or marketplace disputes |
| `general_or_context_missing` | Insufficient text, link-only context, or no safe primary intent |

The approval and pre-training refinements are recorded in `data/governance/HUMAN_APPROVAL.md`.

## Training methodology

Training used weak supervision over training-only Amazon messages. High-precision taxonomy signals supplied silver candidates, with limited local-model adjudication for ambiguous training examples. DistilRoBERTa (`distilroberta-base`) was fine-tuned for two epochs with seed 42. The 200 golden examples were never used for training, weak labeling, threshold selection, or retrieval.

Two required baselines are persisted alongside the neural model: a most-frequent intent predictor and TF-IDF plus logistic regression. The 0.72 intent-confidence threshold was frozen from development-time policy before final golden evaluation.

## Retrieval and RAG

`qwen3-embedding:0.6b` embeds the customer message and the 3,000-record training-only retrieval corpus. Cosine similarity selects the nearest historical customer/AmazonHelp pairs. `qwen3:4b` receives at most five examples and is instructed to describe historical support behavior rather than claim current policy or completed account actions.

High cosine similarity is not treated as proof of relevance. The 48-example top-1 similarity median was 0.7777, but high-similarity classifier mismatches remain a known failure mode.

## Safety and routing

The router defaults to escalation. Auto-handling requires sufficient intent confidence, retrieval evidence, groundedness, and no safety or policy violation. Account compromise, suspected fraud, sensitive payment disputes, low confidence, weak retrieval, provider failures, unsupported claims, PII leakage, stale URLs, placeholders, and sensitive-information requests are conservatively escalated.

The final response is built only after deterministic validation and routing. Reason codes and the user-visible response are derived from the same final decision, preventing contradictory provenance.

## Quick start from a fresh clone

Tested on Windows 11 with Python 3.14.3 and CPU-only PyTorch. Python 3.11 or newer is required.

```powershell
git lfs install
git lfs pull
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python scripts\fetch_submission_artifacts.py
python scripts\preflight_submission.py
```

`fetch_submission_artifacts.py` retrieves only missing Git LFS objects and verifies every required file. It never downloads the raw Twitter dataset and requires no Kaggle credentials.

## Reproduce headline results

After setup, the primary reviewer command is:

```powershell
python scripts\reproduce_headline.py
```

This verifies all artifact hashes and structures, validates the golden data and compact leakage evidence, runs the three intent systems on all 200 examples, calculates confidence and routing metrics, loads frozen post-repair responses and judge outputs, recomputes human–judge agreement, and writes `data/reproduction/latest/headline_results.json`.

It does **not** retrain, rebuild the dataset, call Ollama, regenerate the 48 responses, or rerun the judge. The final isolated clean-copy rehearsal completed the reproduction in **18.1 seconds** on the documented Windows CPU environment (dependency installation excluded), comfortably below the 15-minute target. Runtime varies by disk and CPU.

Artifact verification can also be run separately:

```powershell
python scripts\verify_submission_artifacts.py
```

## Run the live agent

Live generation and retrieval require Ollama. In one terminal:

```powershell
ollama serve
```

Then install the local models and run the CLI:

```powershell
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
python scripts\demo.py
```

Type `exit` to stop. The live-agent path is intentionally separate from headline reproduction because local generation is much slower and already-frozen outputs are the scientifically relevant evaluation inputs.

## Final results

### Intent classification on 200 human-gold examples

| System | Correct | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Most-frequent baseline | 47/200 | 0.2350 | 0.0317 | 0.0894 |
| TF-IDF + logistic regression | 69/200 | 0.3450 | 0.3538 | 0.2885 |
| DistilRoBERTa | 82/200 | **0.4100** | **0.4325** | **0.4070** |

At the frozen 0.72 threshold, coverage is 73/200 (36.5%), covered accuracy is 0.5342, and 127 messages are routed as low-confidence cases.

### Final post-repair routing on 48 responses

| Metric | Result |
|---|---:|
| Accuracy | 31/48 (64.58%) |
| Escalation precision | 0.6383 |
| Escalation recall | **1.0000** |
| Escalation F1 | 0.7792 |
| False auto-handles | **0/30 human escalations (0.00%)** |
| False escalations | 17 |
| Final decisions | 1 auto-handle, 47 escalations |

The repair changed 3 routes and 33 displayed replies. Deterministic safety handling triggered on 47/48 cases. This eliminates false auto-handles in this small sample but demonstrates substantial over-escalation.

### Human versus fixed LLM judge

The human personally ranked all 48 final post-repair replies without seeing judge scores. A fixed `qwen2.5:1.5b` judge used the original rubric and temperature: 15 byte-identical judgments were reused and 33 changed replies were judged post-repair. This additional comparison is explicitly post-hoc and does not replace the preserved pre-repair evaluation.

| Metric | Result |
|---|---:|
| Exact agreement across 288 dimension scores | 16.67% |
| Agreement within ±1 | 52.08% |
| Linear weighted Cohen’s κ | -0.1081 |
| Quadratic weighted Cohen’s κ | -0.1158 |
| Pooled Pearson correlation | -0.1303 |
| Pooled Spearman correlation | -0.1468 |
| Human aggregate mean | 3.8542 |
| Judge aggregate mean | 3.2674 |

Agreement is poor. The judge strongly underrates safety and unsupported-claim avoidance, sometimes misreading “do not post sensitive information” as a request to provide it. Automated judge scores are therefore reported as diagnostics, not substitutes for human safety review.

### Reply-quality comparison

| Reply system | Fixed-judge overall mean |
|---|---:|
| Trivial safe baseline | 3.4167 |
| TF-IDF historical reply | 3.3507 |
| Final RAG + safety | 3.2674 |

The human mean for final RAG+safety replies is 3.8542. The judge’s contrary ranking and negative agreement statistics are material limitations, not results to conceal or tune away.

## Top five observed failure modes

1. **Security message classified as delivery (`gold-045`)** — the model focused on order/delivery wording even though someone else was placing orders using the customer’s number. Routing still escalated safely.
2. **Taxonomy gap for support availability (`gold-028`)** — “Are your call center not working 24/7?” was labelled context-missing but classified as defective item.
3. **Prime Video false escalation (`gold-011`)** — the intent was correct, but an unsupported-claim validator rejected the draft and escalated an otherwise auto-handleable playback issue.
4. **Kindle cancellation over-escalation (`gold-117`)** — correct cancellation intent, but the conservative unsupported-claim policy suppressed useful historical cancellation guidance.
5. **Duplicate Kindle cancellation pattern (`gold-004`)** — a close semantic repeat of the prior error, illustrating both conservative over-routing and sample dependence.

## Taxonomy gaps

Observed messages include pricing questions, delivery instructions, driver/staff conduct, packaging complaints, Pantry/cart issues, call-center availability, and non-Prime device/application failures. They remain mapped to the closest frozen label—often `general_or_context_missing`—because changing the taxonomy after seeing final test data would invalidate the evaluation.

## What is misleading about my headline number?

The 0.4100 accuracy is based on one brand and only 200 unevenly distributed examples. Some intents have one to five examples, and the set contains lexical and semantic near-duplicates. Weak supervision can reward lexical patterns rather than robust intent understanding. Historical Twitter text is noisy and frequently depends on unavailable links or media. The frozen taxonomy has known gaps.

The 48-response routing and reply-quality sample is smaller than the intent test. Conservative escalation achieved zero false auto-handles in that sample by escalating 47/48 cases, including 17 unnecessary escalations. Historical AmazonHelp behavior is not current policy or proof of resolution. Human annotation used assisted file preparation with one human making final decisions, not independent double annotation. Human–judge agreement is negative, so the local judge cannot validate quality reliably. The fast reproduction path appropriately uses frozen LLM outputs, but it does not demonstrate that a new Ollama run will reproduce identical text.

## What I would do with one more week

- Add a second independent annotator, measure inter-annotator agreement, and adjudicate disagreements.
- Expand rare intents and context-missing hard negatives using only training/development evidence.
- Add prospective labels for the observed taxonomy gaps without changing this sealed evaluation.
- Build intent-aware retrieval reranking with hard negatives and human relevance judgments.
- Calibrate confidence and routing thresholds on development data.
- Add current, timestamped policy knowledge instead of relying only on historical social replies.
- Adversarially test PII, fraud, sarcasm, multi-intent requests, stale links, and missing media.
- Benchmark GPU and hosted generation for latency and compare quality without tuning on gold.

## Decision log

| Decision | Reason and trade-off |
|---|---|
| Lock to Amazon/AmazonHelp | Coherent company-specific evidence; narrower external validity |
| Reconstruct reply graphs | Preserves conversational structure; more processing complexity |
| Split by terminal thread time | Prevents thread leakage and anomalous ancestor dates from moving conversations |
| Keep raw text immutable | Auditability; requires separate normalized/model-facing fields |
| Human-approve 12 intents before training | Prevents post-test taxonomy tuning; leaves known long-tail gaps |
| Merge shipping-promise failures with delay/tracking | Same operational path; less SLA-specific analysis |
| Split Prime membership from Prime Video | Different evidence and handling; smaller class support |
| Use weak supervision | Feasible training volume; label noise caps performance |
| Use DistilRoBERTa as primary classifier | Fast deterministic inference; local model artifact is large |
| Freeze the 0.72 threshold on development policy | Avoids test tuning; lowers coverage |
| Restrict retrieval to training threads | Prevents leakage; reduces evidence coverage |
| Treat retrieval similarity as evidence, not truth | Avoids overstating relevance; requires conservative handling |
| Use local Ollama generation | Credential-free live demo; slow CPU latency |
| Add deterministic validation after generation | Enforces non-negotiable safety; can over-escalate |
| Prefer false escalation over false auto-handle | Lower safety risk; poor automation coverage |
| Freeze LLM outputs for reproduction | Fast, stable headline metrics; not a live-generation replication |
| Use Git LFS for large inference artifacts | Keeps Git history reasonable; requires LFS objects to be pushed with the submission |

## Limitations

- Single-brand, historical 2017 Twitter data.
- No current Amazon policy or transactional backend.
- Weakly supervised classifier with low final accuracy and uneven class support.
- Missing linked-media context and incomplete conversations.
- Retrieval relevance lacks independent human labels.
- One human labeler and one human reply-quality rater.
- Poor LLM-judge agreement.
- Conservative validation produces many false escalations.
- CPU `qwen3:4b` generation median/p95 latency was 20.28/71.21 seconds; total component-sum median/p95 was 20.59/71.52 seconds.

## Repository structure

```text
configs/                 Frozen taxonomy, training, retrieval, judge, runtime, and routing policy
src/hiver_support/       Data, classifier, retrieval, generation, safety, routing, and evaluation code
models/intent/           Inference-only DistilRoBERTa model and tokenizer
models/baselines/        Persisted trivial and TF-IDF/logistic baselines
indexes/amazon_support/  Frozen 3,000-record semantic index
data/golden/             Sealed 200 gold labels plus historical and post-repair human ratings
data/evaluation/         Compact hashed leakage-membership evidence
data/reports/            Frozen metrics and historical pre-repair artifacts
data/reports/post_repair Final repaired responses, metrics, judge output, and agreement
artifacts/manifest.json  Required artifact paths, versions, sizes, hashes, and delivery method
scripts/                 Setup, verification, reproduction, evaluation, and live-demo commands
tests/                   Automated behavior, leakage, schema, and submission checks
```

## Tests

```powershell
python -m pytest -q
```

The suite covers schema handling, data splitting, intent taxonomy, trained classifier loading, safety validation, routing, fallback behavior, artifact integrity, missing/corrupted artifacts, no old-output fallback, immutable gold/ratings, and compact leakage verification.

## Reproducibility and environment

- Tested OS: Windows 11 x64.
- Tested Python: 3.14.3; project minimum is 3.11.
- CPU: Intel Core i7-12650HX, 20 logical processors.
- PyTorch: 2.14.0 CPU build.
- Transformers: 5.17.0.
- Ollama generator: `qwen3:4b`, Q4_K_M.
- Ollama embeddings: `qwen3-embedding:0.6b`, Q8_0.
- Independent judge: `qwen2.5:1.5b`, Q4_K_M.
- Training/retrieval seeds: 42.
- Intent threshold: 0.72; weak-retrieval threshold: 0.55.
- Locked dependencies: `requirements-lock.txt`.
- Artifact set: `hiver-submission-v1`, verified by SHA-256 and semantic structure checks.

## Optional full data rebuild

This is not required for headline reproduction. To rebuild from the original dataset, place Kaggle’s `twcs.csv` under `data/raw/` and run:

```powershell
python scripts\prepare_data.py --rebuild-staging
python scripts\discover_intents.py
python scripts\weak_label.py
python scripts\train_intent.py
python scripts\train_baselines.py
python scripts\build_retrieval_index.py
```

Do not regenerate the sealed 200-example golden set or tune against it.

## Historical reports

`PHASE_2_COMPLETION_REPORT.md` is a historical Phase 2 snapshot. It is superseded for final metrics by this README, `PHASE_3_EVALUATION_REPORT.md`, and `FINAL_BEHAVIORAL_REPAIR_REPORT.md`. Historical pre-repair responses, ratings, judge outputs, and agreement artifacts remain packaged for auditability and must not be presented as final repaired-system results.
