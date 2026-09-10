# Human labeling instructions

The candidate file is sampled only from held-out **test threads**. Do not copy these rows into training prompts, weak-label training data, classifier fitting, or the retrieval index.

For each row:

1. Read `customer_message` without looking up the historical Amazon reply.
2. Choose exactly one intent from the **human-approved** `configs/intents.yaml` taxonomy.
3. Choose `AUTO_HANDLE` or `ESCALATE` under the approved escalation policy.
4. If escalating, write a concise, case-specific reason.
5. Set `human_label_complete` to `true` only after personally checking all three fields.

The `ai_suggested_*` columns are hints only. They are not human labels, may be wrong, and must not be copied blindly. The final `golden_set.csv` should contain 150–250 rows whose human fields were actually reviewed.

Later, independently score 40–50 generated replies with the same reply-quality rubric used by the LLM judge. Do not ask the generating model to impersonate those human ratings.

