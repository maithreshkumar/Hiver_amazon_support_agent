# Golden-set human labeling guide

This workflow creates Phase 3 ground truth. Read only the customer message shown by the CLI. Do not inspect model predictions, confidence, retrieval results, weak labels, historical replies, or generated answers while labeling.

## Start or resume

```powershell
python scripts\label_golden.py --labeler YOUR_NAME_OR_ID
```

Progress is saved after every confirmed example. Enter `q` at an intent or route prompt to stop safely. Resume with the same command. To correct a prior label:

```powershell
python scripts\label_golden.py --labeler YOUR_NAME_OR_ID --example gold-057
```

Check progress with `python scripts\label_golden.py --status`. After reaching 200/200, run `python scripts\validate_golden.py`.

## Routing standard

- `AUTO_HANDLE`: a low-risk, sufficiently clear request where a generic evidence-grounded answer can safely help without accessing private account/order/payment state or promising an outcome.
- `ESCALATE`: security or compromise, suspected fraud, sensitive payment dispute, account-specific investigation, unclear/missing context, multiple unresolved intents, legal/safety concern, or any case where an unsupported answer would be risky.

When uncertain, choose `ESCALATE` and briefly state why. The reason must describe your human judgment, not a model score.

## Approved intent definitions

- `delivery_tracking_or_delay`: tracking, arrival status, late/rescheduled delivery, or failed delivery-speed promise.
- `delivered_but_missing`: explicitly marked delivered but not received or locatable.
- `return_request`: return eligibility, label, pickup, logistics, or return status.
- `refund_or_payment_issue`: refund status, duplicate/unknown charge, failed payment, or payment-method issue without a stronger account-compromise signal.
- `order_cancellation`: cancel an order or an unexpected/failed cancellation.
- `damaged_or_defective_item`: damaged, broken, defective, or unusable received item.
- `wrong_or_incomplete_item`: wrong product/quantity, missing unit, part, or accessory.
- `account_access_or_security`: login/access failure, suspicious message, phishing, fraud, compromise, or unauthorized account access.
- `prime_membership`: Prime enrollment, billing, cancellation, eligibility, renewal, or membership benefits excluding video.
- `prime_video_issue`: Prime Video playback, app, subtitle, episode/title, streaming, or availability issue.
- `seller_or_marketplace_issue`: issue primarily involving a marketplace or third-party seller.
- `general_or_context_missing`: greeting, help-only, link/image-dependent, severely ambiguous, or insufficient-context message.

## Provenance and isolation

The canonical file is `data/golden/golden_set.csv`. The CLI sets `label_source=human_gold`, records the human labeler ID and timestamps, and appends an audit event after each save. AI suggestion fields from the candidate file are intentionally absent from the canonical labeling file and never displayed.
