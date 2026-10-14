# Segment-level model — Eval Scoring Report

**Date:** 2026-08-28
**Eval sample:** `segment_eval_worksheet_final_1189.csv` — 1,189 rows, 100% labeled, 4-class label space (positive_benefit / minor_complaint / strong_complaint / usage_tip).
**Checkpoints scored:**
- `segment_level_baseline_20260730/final` — baseline (2026-07-30)
- `segment_level/final` — retrained (2026-08-06)

Inference run locally via `run_segment_eval_inference.py`: `AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")`, `truncation=True`, `max_length=256` (segment text is much shorter than full reviews — deliberately not the review-level 512), argmax + full softmax per row per checkpoint. Scoring computed from `segment_inference_raw_baseline.csv` / `segment_inference_raw_retrained.csv` via `score_segment_checkpoints.py`.

## How the 1,200-row sample became 1,189

- Started from 1,200 rows; 5 dropped 2026-08-20 as uncodable (empty/placeholder text, off-topic content) → `dropped_ambiguous_segments_2026-08-20.csv`. Trimmed worksheet: 1,195 rows.
- 103 rows flagged `ambiguous=True` by the primary labeler, plus 20 blind IAA plant rows (sampled from the confidently-labeled pool), sent to a native-speaker reviewer as a shuffled 123-row batch with all prior labels stripped.
- Her labels merged back in 2026-08-27, replacing the primary labeler's guess on the 103 ambiguous rows. 7 unrelated orphan rows were discovered with no label and no `labeled_by` at all (a pre-existing data gap, not part of the ambiguous/native-review mechanism); the primary labeler hand-labeled 6.
- 3 rows remained genuinely unresolved (2 from the ambiguous pool where even the native speaker left no label, 1 orphan) — dropped 2026-08-27, logged in `dropped_still_ambiguous_segments_2026-08-27.csv`. Worksheet: 1,192 rows.
- IAA: blind Cohen's kappa on the 20-row spot check = **0.593** (moderate agreement, Landis-Koch scale), computed from the untouched blind reads and preserved in `iaa_kappa_result_BLIND_ORIGINAL_do_not_edit.csv` — this number does not change based on anything below. The primary labeler then arbitrated the 20 IAA rows row-by-row (agreeing with her read or keeping his own); 17 resolved and their final label applied to the scoring worksheet, 3 stayed contested even after arbitration and were dropped (`dropped_iaa_contested_2026-08-27.csv`). Worksheet: **1,189 rows**, the final scoring sample.
- Net: 11 rows dropped total across all three rounds (5 + 3 + 3) out of the original 1,200 — none of it silently absorbed into a label; every drop is logged with a reason.

Two provenance flags carried in the final worksheet distinguish rows that went through this process from ordinary confidently-labeled rows: `original_ambiguous_pool` (n=101) and `still_ambiguous_after_two_reads` (n=11, a subset). Both are excluded from the headline number below and reported as their own cuts.

## Headline — confident-labeled rows only (n=1,088; original ambiguous pool excluded)

| Metric | Baseline (07-30) | Retrained (08-06) |
|---|---|---|
| Macro-F1 | **0.8298** | 0.8148 |
| Accuracy | **0.8401** | 0.8263 |

Per-class F1:

| Class | Baseline P / R / F1 | Retrained P / R / F1 |
|---|---|---|
| minor_complaint | 0.7702 / 0.8357 / **0.8016** | 0.7382 / 0.8385 / 0.7851 |
| positive_benefit | 0.9763 / 0.8164 / **0.8892** | 0.9620 / 0.8164 / 0.8832 |
| strong_complaint | 0.8319 / 0.9061 / **0.8674** | 0.8251 / 0.8638 / 0.8440 |
| usage_tip | 0.7132 / 0.8151 / **0.7608** | 0.7377 / 0.7563 / 0.7469 |

**Baseline wins the aggregate and every single class.** This is not a mixed bag — retrained does not win a single per-class F1 on the headline cut. The clearest single number: strong_complaint F1 drops from 0.8674 to 0.8440. This confirms, on real held-out eval data, the exact regression flagged as a watch-item from the training-time proxy (0.648 → 0.629 after the retrain — the only segment class that dipped there too).

**Confusion matrices** (rows = true label, cols = predicted; order: minor_complaint / positive_benefit / strong_complaint / usage_tip):

Baseline:
```
minor_complaint      [295,   6,  38,  14]
positive_benefit      [53, 329,   1,  20]
strong_complaint      [15,   0, 193,   5]
usage_tip              [20,   2,   0,  97]
```

Retrained:
```
minor_complaint      [296,  10,  39,   8]
positive_benefit      [54, 329,   0,  20]
strong_complaint      [24,   1, 184,   4]
usage_tip              [27,   2,   0,  90]
```

The strong_complaint regression is visible directly: baseline misses 15 true strong_complaint rows (mostly to minor_complaint), retrained misses 24 — 9 more, almost entirely the same failure direction (strong_complaint read as minor_complaint).

## Ambiguous-only cut (n=101, original ambiguous pool — not folded into headline)

| Metric | Baseline | Retrained |
|---|---|---|
| Macro-F1 | 0.4336 | 0.4252 |
| Accuracy | 0.4752 | **0.4950** |

Both checkpoints crater roughly 40 macro-F1 points off the headline number — confirms these rows are genuinely hard, not just hard for a human. Proportionally this cut is ~9.3% of the scoring sample, far bigger than review-level's equivalent n=15 (1.5%) — and it does look meaningfully different from the aggregate, which is exactly why it's excluded from the headline rather than folded in. Between checkpoints it's close to a wash: retrained edges accuracy, baseline edges macro-F1, neither difference is large relative to n=101.

## Still-ambiguous-after-two-reads cut (n=11, subset of the above)

| Metric | Baseline | Retrained |
|---|---|---|
| Macro-F1 | 0.3917 | 0.4167 |
| Accuracy | 0.4545 | 0.4545 |

The hardest of the hard — rows where even the native speaker's independent read didn't resolve to confident agreement. n=11 is too small to treat either direction as signal; reported here for completeness, not as a factor in the decision below.

## Confidence routing (max-prob < 60% → needs-review, on the headline population n=1,088)

| | Baseline | Retrained |
|---|---|---|
| Needs-review bucket size | **9/1,088 (0.8%)** | 20/1,088 (1.8%) |
| Accuracy *within* needs-review bucket | 0.556 | 0.500 |
| Macro-F1 on confident-only subset | **0.8329** | 0.8228 |

Unlike review-level (where the retrain meaningfully improved calibration), retrained is *worse* here on every routing metric — bigger needs-review bucket, lower accuracy within it, lower macro-F1 on the confident remainder. That said, treat this cut cautiously: median max-softmax-probability is ~0.99 for both checkpoints across all 1,189 rows — the model is severely overconfident regardless of which checkpoint, so a 60% threshold barely triggers (well under 2% of rows either way) and isn't doing much discriminating work. This is a calibration problem worth flagging as a limitation on its own, separate from the baseline-vs-retrained comparison.

## rating3_cons breakout (n=196, zero training exposure to this stratum)

| | Baseline | Retrained |
|---|---|---|
| Macro-F1 | **0.4587** | 0.4330 |
| Accuracy | **0.7041** | 0.6531 |

Both checkpoints are weak here, and the accuracy/macro-F1 gap is explained by a real directional bias rather than chance-level spread: predictions default hard toward `minor_complaint` — 69.9% of baseline's predictions on this stratum, 62.8% of retrained's (chance for 4 classes would be 25% each). Retrained spreads slightly more toward strong_complaint (24.0% vs baseline's 17.3%) but is not meaningfully more accurate for it — baseline still wins both metrics on this cut.

## IAA (reference only — not a model-performance metric)

Blind Cohen's kappa, native speaker vs. primary labeler on a 20-row spot check sampled from the *confidently*-labeled pool: **0.593** (moderate agreement). This is a statement about labeler reliability across the whole 1,189-row ground truth, not about these 20 rows specifically — they were sampled from calls the primary labeler felt confident about, so the ~30% disagreement rate is the honest caveat to carry into any read of the "true" labels above, not something a per-row flag can localize.

## Decision

**Baseline (`segment_level_baseline_20260730/final`) wins every cut that has a clear winner:** the headline aggregate, every one of the four per-class F1 scores, the rating3_cons breakout, and confidence routing. Retrained does not win a single cut outright — its best showing is a near-tie on the ambiguous-only and still-ambiguous cuts, both too close (and too small, in the still-ambiguous case) to call a win.

This is the mirror image of the review-level evaluation, where the retrain won cleanly. Here the retrain looks like a net regression on real eval data, and the training-time proxy's isolated strong_complaint dip (0.648 → 0.629) turned out to be a real, generalizing signal rather than training noise — it's the single largest per-class gap on the headline cut (F1 0.8674 → 0.8440), and the confusion matrices show the same failure direction getting worse (15 → 24 true strong_complaint rows misread, mostly as minor_complaint).

**Recommendation: keep the baseline checkpoint as the active segment-level model; do not promote the 2026-08-06 retrain.** Nothing in the ambiguous, still-ambiguous, or IAA numbers changes this — none of them favor retrained either, they're just noisier.

**Decision: baseline (`segment_level_baseline_20260730/final`) stays active for segment-level. The 2026-08-06 retrain is not promoted.**
