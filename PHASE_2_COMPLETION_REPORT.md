# Phase 2 completion report

> **Historical snapshot:** This report records Phase 2 at completion and is superseded for submission metrics by `README.md`, `PHASE_3_EVALUATION_REPORT.md`, and `FINAL_BEHAVIORAL_REPAIR_REPORT.md`. Do not use Phase 2 numbers as final repaired-system results.

## Status

**YES — Phase 2 is complete. Ready to begin the separately authorized Phase 3 human-labeling workflow.**

Completed: 2026-09-10. Phase 3 golden-set labeling and headline evaluation have **not** started.

The system now runs the complete local path:

`DistilRoBERTa intent classification → qwen3 semantic retrieval → qwen3:4b grounded drafting → deterministic safety checks → conservative routing`

## Human approval and provenance

- Taxonomy status: `approved_by_human`, version `1.0`.
- Approval record: `data/governance/HUMAN_APPROVAL.md`.
- Canonical taxonomy: `configs/intents.yaml`.
- The original data-derived proposal is preserved and marked superseded rather than rewritten as if it had been the approved taxonomy.
- `shipping_promise_failure` was merged into `delivery_tracking_or_delay`.
- `prime_membership_or_video` was split into `prime_membership` and `prime_video_issue`.
- `refund_or_charge` was renamed `refund_or_payment_issue`.
- The approved escalation policy remains conservative for security/account compromise, fraud, payment disputes, low intent confidence, ambiguous classifications, weak retrieval, weak grounding, and unsupported claims.

## Approved 12-intent taxonomy

1. `delivery_tracking_or_delay`
2. `delivered_but_missing`
3. `return_request`
4. `refund_or_payment_issue`
5. `order_cancellation`
6. `damaged_or_defective_item`
7. `wrong_or_incomplete_item`
8. `account_access_or_security`
9. `prime_membership`
10. `prime_video_issue`
11. `seller_or_marketplace_issue`
12. `general_or_context_missing`

## Weak-label training data

Weak labeling was restricted to English, training-only AmazonHelp threads. Labels retain the label source, method, provider/model, embedding model, confidence, rationale, generation timestamp, row ID, and thread ID.

- Candidates: **3,160**
- Accepted at confidence ≥ 0.75: **2,514**
- Quarantined: **646**
- Unique accepted threads: **2,514**
- Mean accepted confidence: **0.8759**
- Confidence range: **0.7501–0.9200**
- Embedding model: local Ollama `qwen3-embedding:0.6b`
- Structured adjudication model: local Ollama `qwen2.5:1.5b`

| Intent | Accepted labels |
|---|---:|
| `delivery_tracking_or_delay` | 293 |
| `delivered_but_missing` | 210 |
| `return_request` | 194 |
| `refund_or_payment_issue` | 191 |
| `order_cancellation` | 201 |
| `damaged_or_defective_item` | 193 |
| `wrong_or_incomplete_item` | 190 |
| `account_access_or_security` | 265 |
| `prime_membership` | 266 |
| `prime_video_issue` | 280 |
| `seller_or_marketplace_issue` | 162 |
| `general_or_context_missing` | 69 |

Label methods:

- Rule plus embedding agreement: **1,503**
- Rule or embedding provisional label: **996**
- Structured local-LLM adjudication: **15**

Artifacts:

- `data/processed/weak_labels.parquet`
- `data/reports/weak_label_quarantine.parquet`

## DistilRoBERTa training

- Base model: `distilroberta-base`
- Output: `models/intent/`
- Weak labels supplied: **2,514**
- Internal stratified weak-training rows: **2,136**
- Internal held-out weak-validation rows: **378**
- Epochs: **2**
- Batch size: **16**
- Maximum sequence length: **192**
- CPU training runtime: **530 seconds**

Final internal validation against held-out weak labels:

| Metric | Result |
|---|---:|
| Accuracy | 0.8810 |
| Macro-F1 | 0.8837 |
| Weighted-F1 | 0.8793 |
| Loss | 0.5029 |

These internal numbers measure agreement with weak supervision; they are not human-ground-truth results.

## Chronological development evaluation and baselines

Evaluation used **1,126 English messages from the separate chronological development split**. The reference labels are evaluation-only silver labels selected by single high-precision taxonomy signals; they were never used for training. The sealed test/golden data was not used.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| DistilRoBERTa | **0.8552** | **0.8325** | **0.8489** |
| TF-IDF + logistic regression | 0.8277 | 0.8052 | 0.7901 |
| Most-frequent trivial baseline | 0.1066 | 0.0161 | 0.0205 |

The transformer improves over TF-IDF by **2.74 percentage points accuracy** and **2.74 points macro-F1** on this diagnostic set.

DistilRoBERTa per-intent development F1:

| Intent | Support | F1 |
|---|---:|---:|
| `delivery_tracking_or_delay` | 120 | 0.8117 |
| `delivered_but_missing` | 37 | 0.7609 |
| `return_request` | 120 | 0.9874 |
| `refund_or_payment_issue` | 120 | 0.8734 |
| `order_cancellation` | 120 | 0.9496 |
| `damaged_or_defective_item` | 120 | 0.8538 |
| `wrong_or_incomplete_item` | 39 | 0.5618 |
| `account_access_or_security` | 75 | 0.9068 |
| `prime_membership` | 115 | 0.9084 |
| `prime_video_issue` | 69 | 0.8725 |
| `seller_or_marketplace_issue` | 71 | 0.9078 |
| `general_or_context_missing` | 120 | 0.5965 |

At the configured intent threshold of **0.72**, the model covers **741/1,126 (65.8%)** development examples and reaches **97.84% agreement** on covered examples. Lower-confidence examples are escalated.

Artifacts:

- `models/baselines/trivial.joblib`
- `models/baselines/simple.joblib`
- `data/processed/dev_silver_labels.parquet`
- `data/reports/development_metrics.json`

## Semantic retrieval

- Status: **built and queryable**
- Source: `data/processed/retrieval_corpus.parquet`
- Scope: training-only
- Indexed records: **3,000 unique English threads**
- Model: local Ollama `qwen3-embedding:0.6b`
- Vector dimensions: **1,024**
- Search: normalized cosine similarity over persisted NumPy vectors
- Index directory: `indexes/amazon_support/`

For `My package says delivered but I have not received it`, the top historical similarities were **0.821, 0.813, and 0.798**, from thread IDs `315506`, `879213`, and `828295`.

## Generation and routing

- Default local generator: Ollama `qwen3:4b`
- Prompt version: `amazon-grounded-v1`
- Evidence is explicitly presented as historical handling behavior, not current policy or proof of resolution.
- Retrieved evidence IDs are filtered so the generator cannot cite invented thread IDs.
- The deterministic safety backstop removes drafts containing unsupported promises, stale historical URLs, unqualified sensitive-information requests, or unsupported refund/replacement/return instructions.
- Any safety override, model-flagged unsupported claim, risk intent/signal, low classifier confidence, high ambiguity, weak retrieval, or weak grounding produces `ESCALATE` with a reason.
- Ollama/provider failures now produce a safe structured escalation rather than crashing the CLI.

## End-to-end unseen-message results

Seven hand-written messages were verified as exact non-duplicates of the training retrieval corpus. Every case ran the classifier, semantic retrieval, `qwen3:4b` drafting, safety backstop, and router.

| Case | Predicted intent | Confidence | Top similarity | Safety override | Route |
|---|---|---:|---:|---|---|
| Late/stalled parcel | `delivery_tracking_or_delay` | 0.6434 | 0.7847 | Yes | `ESCALATE` |
| Marked delivered but absent | `delivered_but_missing` | 0.6786 | 0.7966 | No | `ESCALATE` |
| Return unsuitable headphones | `return_request` | 0.3652 | 0.7037 | Yes | `ESCALATE` |
| Unknown charge/account access | `account_access_or_security` | 0.7515 | 0.8052 | Yes | `ESCALATE` |
| Prime Video playback error | `prime_video_issue` | 0.9411 | 0.6865 | No | `AUTO_HANDLE` |
| Unresponsive seller/wrong color | `seller_or_marketplace_issue` | 0.8259 | 0.7535 | Yes | `ESCALATE` |
| Context-free request for help | `account_access_or_security` | 0.2511 | 0.8695 | No | `ESCALATE` |

The safe auto-handled Prime Video reply was:

> I'm sorry to hear you're experiencing issues playing Prime Video episodes. Could you share the specific error message you see when trying to play the next episode? This will help us troubleshoot the problem more effectively.

Unsupported or sensitive cases received the safe handoff:

> I'm sorry this needs further investigation. Please contact Amazon through a secure support channel, and do not post order, address, account, or payment details publicly.

The first audit exposed a stale historical URL, a public request for account context, and an unsupported refund/replacement instruction. After expanding the deterministic safety rules, the three affected cases were rerun through the full stack and all were safely overridden and escalated.

The interactive CLI was also run with the separate unseen message `My Prime Video app freezes after I press play on every movie.` It predicted `prime_video_issue` at **0.9372**, retrieved relevant evidence at **0.7169** top similarity, returned a grounded diagnostic question, routed `AUTO_HANDLE`, and exited cleanly.

Full audit artifact: `data/reports/phase2_e2e_outputs.json`.

## Leakage verification

- Indexed thread IDs not in training retrieval corpus: **0**
- Indexed thread IDs overlapping chronological test split: **0**
- Indexed thread IDs overlapping golden candidates: **0**
- End-to-end messages exactly duplicated in training corpus: **0/7**
- Golden candidates: **200 unique test threads**
- `human_intent` remains blank for all **200** candidates.
- `human_label_complete` remains `False` for all **200** candidates.
- No Phase 3 golden-set metric was calculated.

## Automated tests

Final command: `python -m pytest -q`

Result: **20 passed in 12.08 seconds**.

Coverage includes schema and normalization, thread-disjoint temporal splits, approved taxonomy enforcement, weak-label parsing, trained-model loading and batched inference, provider configuration, provider-outage fallback, evidence filtering, deterministic generation safety, conservative routing, and built-index leakage protection.

## Known limitations

1. Development labels are silver rules, not human ground truth. Lexical selection makes this diagnostic easier and narrower than natural traffic.
2. Weak supervision can reproduce rule, embedding, or local-LLM biases. `wrong_or_incomplete_item` and `general_or_context_missing` are the weakest observed development classes.
3. The vague greeting was safely escalated but misclassified as account/security at low confidence; this validates the confidence gate but shows the classifier needs more diverse context-missing examples.
4. The 3,000-record index is a reproducible laptop-sized sample, not the complete training corpus.
5. Historical AmazonHelp tweets include stale links and old behavior. They are not current Amazon policy or evidence that an issue was resolved.
6. Local `qwen3:4b` CPU generation was approximately **102–109 seconds** for the measured rerun cases. This is safe for an offline demo but too slow for production latency targets without acceleration or a hosted provider.
7. Ollama returned one transient HTTP 500 after a long batch. The CLI now safely escalates provider failures, and a subsequent live CLI run succeeded.
8. Hosted OpenAI, Gemini, and Anthropic adapters exist but were not live-tested because no credentials were supplied.
9. The deterministic safety layer is intentionally conservative; false escalations are preferred over unsupported or sensitive auto-handling.

## Phase 3 readiness

**Ready, with the intended human dependency.** Phase 2 artifacts, local models, retrieval, routing, CLI, provenance, tests, and leakage controls are complete. Phase 3 should begin only after explicit authorization to label the sealed 200-example golden set. Its human labels must remain independent of weak labels and model predictions, and only Phase 3 may produce headline test metrics.
