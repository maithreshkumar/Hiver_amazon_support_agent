# Final behavioral repair report

## Outcome

The runtime now follows one deterministic order: classify, retrieve, draft, validate, route, and then build the only user-facing reply. An `ESCALATE` decision always uses a context-aware safe escalation response; an unsafe or unsupported raw draft cannot be returned unchanged. The approved 0.72 intent threshold, 12-intent taxonomy, DistilRoBERTa artifact, retrieval system, and Ollama models were not changed or retrained.

## Root causes and fixes

- The agent previously returned the generator's pre-routing `draft_reply`, so a later low-confidence escalation could still expose a normal auto-handled draft. `routing/final_response.py` is now the single post-routing response builder.
- The generator previously mixed raw generation, safety validation, and reply replacement. It now preserves `raw_draft` and returns structured validation flags; final replacement happens only after routing.
- Routing returned on the first trigger. It now aggregates security, payment, investigation, ambiguity, confidence, retrieval, grounding, provider, and draft-safety codes into provenance and an accurate reason.
- Deterministic validation now blocks public sensitive-data requests, customer PII echoing, unverified URLs, unresolved placeholders, unsupported operational promises/policy claims, and redundant delivery-date questions. It permits semantically supported paraphrases without lexical matching.
- Security/fraud, payment/refund, delivery, and general escalations now use separate concise templates. Provider failures use the same final-response path and retain all simultaneously active reason codes.
- Delivered-but-missing, courier/staff-conduct complaints, and involuntary cancellation complaints receive order/account-specific investigation signals. Safe, grounded, high-confidence responses can still be auto-handled.

## Files changed

- `configs/escalation_policy.yaml`
- `src/hiver_support/generation/reply.py`
- `src/hiver_support/routing/policy.py`
- `src/hiver_support/routing/final_response.py`
- `src/hiver_support/agent.py`
- `src/hiver_support/evaluation/responses.py`
- `scripts/validate_runtime_behavior.py`
- `scripts/reprocess_phase3_runtime.py`
- `scripts/analyze_phase3_outputs.py`
- `scripts/run_final_evaluation.py`
- `scripts/build_phase3_report.py`
- `tests/test_generation.py`
- `tests/test_routing.py`
- `tests/test_runtime_safety.py`
- `tests/test_agent_fallback.py`
- `tests/test_artifact_leakage.py`
- `PHASE_3_EVALUATION_REPORT.md`

## Observed end-to-end regression cases

| Case | Intent / confidence | Top retrieval | Final behavior |
|---|---|---:|---|
| A: package three days late | `delivery_tracking_or_delay` / 0.7101 | 0.7722 | `ESCALATE`; low-confidence code; delivery-safe reply replaced the normal draft. |
| B: hacked account | `account_access_or_security` / 0.9395 | Provider fallback | `ESCALATE`; security-specific reply; no public sensitive-data request. |
| C: due yesterday | `delivery_tracking_or_delay` / 0.5591 | 0.8010 | `ESCALATE`; raw draft redundantly asked whether the date had passed, but the final reply did not. The final validator now flags this exact indirect wording. |
| D: marked delivered but missing | `delivered_but_missing` / 0.7614 | Provider fallback | Correct distinct intent; `ESCALATE`; delivery investigation template; no stale URL. |
| E: simple headphone return | `return_request` / 0.7790 | 0.7312 | `AUTO_HANDLE`; safe grounded Amazon-versus-third-party question preserved unchanged. |
| F: hacked account with phone | `account_access_or_security` / 0.9395 | Provider fallback | `ESCALATE`; security template did not echo `+1 415-555-0199`. |
| G: `[link]` draft | Deterministic injected regression | n/a | `PLACEHOLDER_OUTPUT`; placeholder rejected and replaced before output. |

The broader real-model run covered 16 messages across all requested themes. All 16 final replies passed deterministic final validation; two safe cases auto-handled. Ten cases exercised the fail-closed provider-timeout path, confirming safety but also demonstrating that local CPU/Ollama reliability and latency remain material limitations.

## Final Phase 3 metrics

The classifier was not changed. Re-running the canonical 200-row evaluation produced DistilRoBERTa accuracy **0.4100**, macro-F1 **0.4325**, and 73/200 coverage at the frozen 0.72 threshold, with covered accuracy **0.5342**.

The 48 saved raw RAG drafts were reprocessed through the final deterministic runtime without rerunning or tuning the LLM. Three routes and 33 displayed final replies changed. Final routing accuracy is **31/48 (64.58%)**; escalation precision/recall/F1 are **0.6383/1.0000/0.7792**; false auto-handles are **0/30 (0.00%)**; false escalations are **17**. Only one of these frozen 48 examples remains auto-handled, so the final policy is safe but has low automation coverage on this sample.

The existing human-versus-judge exact and within-±1 agreement remain **21.18%** and **62.50%**. Those ratings describe the original displayed responses. They were deliberately not changed or rescored; because runtime hardening rewrites some responses, the frozen reply-quality scores are historical pre-repair measurements, not scores for the rewritten replies.

## Verification and integrity

- Targeted safety/routing suite: **30 passed in 0.69s** after final provenance aggregation.
- Complete suite: **58 passed in 12.79s**.
- Golden set: 200/200 complete; zero train, retrieval, or weak-label overlap.
- Canonical golden SHA-256: `9273e9c34ffa28579b898af20f8954c76fc5d386c39ed9a3a259d168cd36cd0f`.
- Canonical human-rating SHA-256: `2dd1545f8322f5b6364138ed9dd533cfc04b4d3c8a59d472e6226d9bd59d732a`.
- The 200 human golden labels, 48 human ratings, and frozen judge output were not modified.
- `PHASE_3_EVALUATION_REPORT.md` was regenerated to use the post-repair routing metrics and explicitly separate historical human reply ratings from current runtime behavior.

## Reproduction

```powershell
python scripts\validate_runtime_behavior.py --classification-only
python scripts\validate_runtime_behavior.py
python scripts\reprocess_phase3_runtime.py
python scripts\run_final_evaluation.py
python -m pytest -q
```
