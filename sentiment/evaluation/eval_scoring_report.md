# Review-level model — Eval Scoring Report

**Date:** 2026-08-10
**Eval sample:** `eval_sample_labeled_1000_scoring.csv` — 1,000 rows, 100% labeled, validated against `WORKSHEET_COLUMNS` schema (see the data note below).
**Checkpoints scored:**
- `review_level_baseline_20260730/final` — baseline (2026-07-30)
- `review_level/final` — retrained (2026-08-06)

Inference run locally (GTX 1650 Ti) via `run_eval_inference.py`: `AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")`, `truncation=True`, `max_length=512`, argmax + full softmax per row per checkpoint. Scoring computed from `inference_raw_baseline.csv` / `inference_raw_retrained.csv` via `score_checkpoints.py`.

## Data note — one discrepancy

16 `ambiguous=True` rows were expected in the 1,000-row cut; only 15 are present. Traced: one review (rating 3, resolved `neutral_mixed` by a native Korean speaker) has a resolved label upstream in `eval_sample_merged_1120_v3.csv`, but is also listed in `rejected_review_ids.csv` and was dropped before the `_1119_final` merge that the scoring cut derives from — a deliberate exclusion, not a labeling gap, so it stays excluded. All downstream "ambiguous" numbers below reflect n=15, not 16.

## Headline — all 1,000 labeled rows (ambiguous rows included)

| Metric | Baseline (07-30) | Retrained (08-06) |
|---|---|---|
| Macro-F1 | 0.7492 | **0.7808** |
| Accuracy | 0.7900 | **0.8160** |

Per-class F1:

| Class | Baseline P / R / F1 | Retrained P / R / F1 |
|---|---|---|
| frustrated | 0.9935 / 0.5709 / 0.7251 | 0.9670 / 0.6567 / **0.7822** |
| high_satisfaction | 0.8564 / 0.9632 / 0.9066 | 0.8697 / 0.9693 / **0.9168** |
| neutral_mixed | 0.5608 / 0.6831 / 0.6160 | 0.6081 / 0.6831 / **0.6434** |

All three classes improve for the retrained checkpoint — this is a clean sweep on the headline number, not a mixed bag. The main driver: retrained recovers 20 more true `frustrated` reviews that baseline was mis-routing into `neutral_mixed` (confusion matrix below), lifting `frustrated` recall from 0.571 to 0.657 while precision stays ~0.97.

**Confusion matrices** (rows = true label, cols = predicted; order: frustrated / high_satisfaction / neutral_mixed):

Baseline:
```
frustrated          [153,   3, 112]
high_satisfaction    [0, 471,  18]
neutral_mixed        [1,  76, 166]
```

Retrained:
```
frustrated          [176,   0,  92]
high_satisfaction    [0, 474,  15]
neutral_mixed        [6,  71, 166]
```

## Ambiguous rows only (n=15, hard cases escalated to a native Korean speaker)

| Metric | Baseline | Retrained |
|---|---|---|
| Macro-F1 | **0.4762** | 0.4091 |
| Accuracy | **0.667** | 0.600 |

**This is the one place retrained regresses.** On the specific 15 rows that were hard enough to escalate during labeling, baseline outperforms retrained on both macro-F1 and accuracy. n=15 is small and this is noisy — a couple of rows flipping changes the numbers a lot — but it's a real, directional signal that retraining did not improve (and mildly hurt) performance on the hardest cases, even while lifting the aggregate.

## Confidence routing (max-prob < 60% → needs-review)

| | Baseline | Retrained |
|---|---|---|
| Needs-review bucket size | 116/1000 (11.6%) | **62/1000 (6.2%)** |
| Accuracy *within* needs-review bucket | 0.457 | **0.597** |
| Macro-F1 on confident-only subset | 0.784 | **0.790** |

Retrained is more decisive (roughly half as many low-confidence predictions) and more accurate on the ones it does flag — a practical win if the needs-review bucket ever becomes a human-review queue.

## Rating-3 / neutral_mixed breakout

- Rows with `rating == 3` (proxy stratum): n=227
- Rows with ground-truth `label == neutral_mixed` (all ratings): n=243 — these are different sets; rating-3 is a proxy, not identical to the neutral_mixed label.

| | Baseline | Retrained |
|---|---|---|
| Macro-F1 on rating==3 rows | 0.5265 | **0.5975** |
| Accuracy on rating==3 rows | 0.630 | **0.670** |
| neutral_mixed F1 within rating==3 | 0.7255 | **0.7475** |
| Recall on true neutral_mixed (all ratings) | 0.6831 | 0.6831 (identical — both models catch exactly 166/243) |

Retrained improves on the documented rating-3 blind spot. The identical neutral_mixed recall (166/243 for both) is coincidental in raw count but the underlying precision differs (0.561 baseline vs 0.608 retrained) — retrained produces fewer false-positive neutral_mixed predictions, mainly because it's better at correctly calling true frustrated reviews frustrated instead of neutral_mixed.

## Negation / hedging spot-check

The three literal example patterns chosen in advance mostly don't occur verbatim in this 1,000-row sample:
- `나쁘지 않아요` (exact) — 2 matches
- `좋다고 하기는 어렵네요` (exact) — 0 matches
- `그럭저럭 괜찮아요` (exact) — 0 matches

Broadened to substring components for real coverage: `나쁘지 않` (23 matches), `괜찮아요` (14 matches); `좋다고 하기` and `그럭저럭` still had zero hits at any granularity — these specific hedge constructions just don't appear in this sample.

On the 23 `나쁘지 않` ("not bad") rows — the most useful negation-hedging cut available:

| | Baseline | Retrained |
|---|---|---|
| Accuracy | 65.2% (15/23) | 65.2% (15/23) |

Same aggregate accuracy, different composition: 6 shared errors, 1 new error introduced by retrained (key `r0827`, true `high_satisfaction` → retrained called `neutral_mixed`, baseline got it right), 1 error fixed by retrained (key `r0997`, true `high_satisfaction` → baseline called `neutral_mixed`, retrained got it right).

**Consistent failure mode in both checkpoints:** hedged negative reviews (rating 2-3, "나쁘지 않아요"-style, true label `frustrated`) get systematically pulled toward `neutral_mixed` — 4 of the 5 true-`frustrated` rows in this pattern set are mis-classified this way by both models (keys `r0625`, `r0649`, `r0658`, `r0679`). Hedged neutral reviews occasionally get pulled the other way, toward `high_satisfaction` (keys `r0279`, `r0523`). Retraining did not fix this; it's a shared blind spot, not something the 08-06 retrain addressed.

Full match list: `negation_spotcheck_matches.csv`.

## Summary — tradeoffs to weigh

Retrained wins clearly and consistently on every headline/aggregate cut: macro-F1, accuracy, all three per-class F1 scores, rating-3 performance, and confidence calibration (fewer low-confidence predictions, and more accurate on the ones it flags).

Two places it does **not** win:
1. **Ambiguous/escalated rows (n=15):** baseline macro-F1 0.476 vs retrained 0.409. Small n, but it's the one clean regression.
2. **Negation/hedging (n=23, "나쁘지 않" pattern):** flat at 65.2% accuracy for both — retraining didn't move this blind spot, and both models share the same systematic hedged-frustration → neutral_mixed failure.

Neither of these overturns the aggregate case for retrained, but both indicate the retrain improved the model on the bulk of "normal" cases without meaningfully improving (or, on the escalated set, while slightly hurting) performance on the genuinely hard cases that a real deployment would still need a human-review safety net for.
