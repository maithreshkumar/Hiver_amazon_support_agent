# Hiver Amazon Support Agent

An evidence-first implementation of the Hiver SDE Intern take-home: reconstruct AmazonHelp support conversations from the Customer Support on Twitter dataset, derive an intent taxonomy from Amazon data, ground replies in training-only historical behavior, and evaluate intent, routing, and reply quality without fabricating resolution evidence or human labels.

## Current stage

Phase 3 is complete. The 200-example human-reviewed golden set is sealed and validated with zero training/retrieval leakage. Final intent, confidence, routing, RAG, retrieval, latency, baseline, independent-judge, human–judge agreement, duplicate, and failure artifacts are present. The 48 human reply ratings were imported byte-for-byte after validation; the original judge was not rerun or tuned after seeing them.

## Setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place the Kaggle `thoughtvector/customer-support-on-twitter` file at:

```text
data/raw/twcs.csv
```

The raw directory is ignored by Git and the source file is never modified.

## Full data foundation rebuild

```bash
python scripts/prepare_data.py --rebuild-staging
python scripts/discover_intents.py
python scripts/create_golden_candidates.py --size 200
pytest
```

The first command stages the full 500+ MB CSV in SQLite so graph traversal is reproducible without holding the whole dataset in memory. Runtime depends on disk and CPU and is measured from the actual run rather than promised in advance.

## Outputs

- `data/processed/amazon_raw.parquet` — immutable-text Amazon-connected subset.
- `data/processed/amazon_clean.parquet` — enriched subset with normalized/model-facing text and language metadata.
- `data/processed/amazon_threads.parquet` — ordered tweet-level turns with raw, normalized, and PII-masked model text.
- `data/processed/amazon_conversations.parquet` — conversation-level representation.
- `data/processed/amazon_pairs.parquet` — direct customer→Amazon historical response pairs.
- `data/processed/{train,dev,test}.parquet` — temporally ordered, thread-disjoint splits.
- `data/processed/retrieval_corpus.parquet` — training-only retrieval evidence.
- `data/reports/data_quality.json` — observed counts; never placeholders.
- `data/reports/intent_taxonomy_proposal.md` — human-review cluster evidence.
- `data/golden/golden_candidates.csv` — 200 held-out candidates with empty human-label fields.

## Important limitation

An AmazonHelp reply is evidence of historical support behavior, not proof of successful resolution. Until a later customer turn supplies clear evidence, `resolution_status` is conservatively `UNKNOWN`.

## Human checkpoint

Completed on 2026-09-09. The exact decision and refinements are recorded in `data/governance/HUMAN_APPROVAL.md`; `configs/intents.yaml` is the canonical 12-intent taxonomy.

## Phase 2 build and run commands

```powershell
python scripts\weak_label.py
python -m pip install -e ".[phase2,dev]"
python scripts\train_intent.py
python scripts\train_baselines.py
python scripts\evaluate_dev.py
python scripts\build_retrieval_index.py
python scripts\run_phase2_e2e.py
python scripts\demo.py
```

The local defaults are `qwen3:4b` for grounded generation and `qwen3-embedding:0.6b` for semantic retrieval. Start Ollama before running the index, end-to-end, or interactive CLI commands. Type `exit` to stop the CLI.

## Phase 3 evaluation and reproduction

Reproduce the current headline evaluation from the sealed gold labels and persisted generation/judge outputs:

```powershell
python scripts\run_final_evaluation.py
```

This reruns golden validation, all-200 intent/baseline metrics, confidence analysis, duplicate audit, response-derived routing/retrieval/latency analysis, human–judge agreement, and report generation while verifying that the human-label file remains byte-for-byte unchanged. It uses the persisted generation and original frozen judge outputs, so it does not regenerate slow LLM results. On the measured machine it runs well under 15 minutes, excluding dependency/model downloads.

Run the complete automated test suite separately:

```powershell
python -m pytest -q
```

The canonical 48-row rating file is `data/golden/reply_quality_human_ratings.csv`. Human final decisions were prepared with semi-automation and imported without rewriting values. The deliberate `gold-147` Safety=1 score records a public phone-number repetition that the original judge missed. Full agreement distributions and confusion matrices are in `data/reports/human_judge_agreement.json`.

The detailed current results and limitations are in `PHASE_3_EVALUATION_REPORT.md`.

## Architecture

```text
immutable twcs.csv
       |
       v
schema validation + SQLite graph staging
       |
       v
AmazonHelp-seeded connected components
       |
       v
thread ordering + conservative normalization
       |
       +--> temporal train/dev/test split
       |          |
       |          +--> training-only retrieval corpus
       |
       +--> English-only TF-IDF + NMF evidence topics
                         |
                         v
                  human taxonomy approval
```

## Decision log (living)

1. Amazon/AmazonHelp is the single locked brand.
2. Raw Twitter text is immutable evidence; normalization is stored separately.
3. Reply links form the conversation graph; no LLM guesses thread membership.
4. Components are seeded by AmazonHelp and expanded in both graph directions.
5. Threads containing another outbound support account are excluded by default.
6. Splits are temporal and performed at thread level, never tweet level; terminal time is the split anchor because the source contains a small number of years-old linked ancestors.
7. Retrieval evidence is restricted to training threads.
8. Historical replies are not labelled successful resolutions without evidence.
9. Obvious PII is masked only in model-facing copies.
10. Intent discovery is data-derived; taxonomy v1.0 was human-approved before weak labeling or training.
11. SQLite is used for the full graph staging step to keep memory bounded.
12. Parquet is used for derived artifacts; the original CSV remains untouched.

## What is misleading about my headline number?

The final intent result is human-grounded but covers one brand, a small uneven test set, known taxonomy gaps, unavailable link/image context, and some near duplicates. Training used weak supervision over historical, noisy Twitter support language. Routing/RAG metrics use a smaller deterministic response subset, and conservative escalation reduces unsafe automation at the cost of many false escalations. Human–judge agreement is weak, so automated judge scores are not a reliable substitute for human review. Historical AmazonHelp behavior is neither current policy nor proof of resolution. Exact computed values are generated into `PHASE_3_EVALUATION_REPORT.md`.

## What I'd do with one more week

Double-annotate a larger golden set; measure inter-annotator agreement; add hard-negative retrieval and reranking; adversarially test sarcasm, ambiguity, PII, fraud, and multi-intent cases; validate routing policy with domain reviewers; and compare local Ollama generation against configured hosted providers on quality, latency, and estimated cost.
