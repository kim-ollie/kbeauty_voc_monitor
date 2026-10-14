"""
Full-corpus inference, KAGGLE VARIANT.

KAGGLE VARIANT of full_corpus_inference.py, adapted for a Kaggle Notebook
with a GPU accelerator, following the same before/after pattern already
used going from finetune_kcelectra_sentiment_v2.py to
finetune_kcelectra_sentiment_kaggle.py. Changes from the local script:

  1. CUDA_VISIBLE_DEVICES pinned to a single GPU, set before torch is
     imported. Same reasoning as the training Kaggle variant: Kaggle's
     "GPU T4 x2" option attaches two T4s, and without the pin, whatever
     multi-GPU behavior Trainer/model.to(device) does by default can
     silently change throughput vs. whatever the small-batch timing test
     (kaggle_timing_test.py, run first) measured
     on. Remove the pin later if you deliberately want to try both T4s.

  2. Paths and what to upload. INPUT_CSV and both checkpoint paths point
     at /kaggle/input/datasets/{KAGGLE_DATASET_SLUG}/..., output paths
     point at /kaggle/working/.... The datasets/ segment in there is
     deliberate, not a typo: assuming the more commonly documented
     /kaggle/input/{slug}/... form gives a FileNotFoundError -- Kaggle actually
     mounts attached datasets at
     /kaggle/input/datasets/<owner-username>/<dataset-slug>/. Set
     KAGGLE_DATASET_SLUG below to "<your-kaggle-username>/<dataset-slug>"
     not a bare slug -- the datasets/ prefix is already baked into the path
     templates so you only need owner/slug. Worth reconfirming this is
     still Kaggle's current mount convention when you add the dataset to
     the notebook (its file browser panel shows the real mounted path).

     Upload these five things as ONE Kaggle Dataset, preserving the
     folder structure exactly as it sits locally:
       phase1_scoring_corpus.csv                     (~789 MB)
       full_corpus_inference_kaggle.py                (this file)
       kaggle_timing_test.py                          (needs to sit next
                                                         to this file --
                                                         see point 4)
       review_level/final/                            (~489 MB)
       segment_level_baseline_20260730/final/         (~489 MB)
     ~1.8 GB total -- well inside Kaggle's private-dataset quota.
     The CSV alone is a slow upload on a home connection, so start it
     before you need the notebook.

  3. Added resumable, incrementally-flushed scoring. This is the one
     structural change beyond a path/GPU swap. Neither training run hit
     Kaggle's session cap (review_level finished in ~70 min, segment_level
     in ~2h40m, inside the documented ~12h limit), but this run's
     workload shape is different and untested: a single forward-pass-only
     scoring pass over 814,452 reviews plus roughly 2.6x that many
     non-empty pros/cons/tip segments (per the local dry run's ratio),
     vs. training's much smaller row counts (174K/421K capped rows) run
     for 6 epochs each. Whether that nets out faster or slower than the
     training runs isn't obvious without measuring it, which is exactly
     what kaggle_timing_test.py is for. Given the real uncertainty, and
     that the original script holds every prediction in memory and writes
     once at the very end -- a cutoff one minute before that final write
     would lose 100% of the run, not just the unsaved tail -- resumability
     is cheap insurance against a real platform limit.

     So this variant flushes review-level and segment-level predictions
     to /kaggle/working/*_partial.csv every FLUSH_EVERY_N_ROWS rows
     (fsynced, not just buffered in the OS page cache -- a killed session
     doesn't get to rely on a clean shutdown flushing anything). Re-
     running this same command after Kaggle reconnects skips review_ids /
     segment rows already present in those partial files rather than
     re-scoring them, and skips loading a checkpoint entirely if that
     head's work is already fully done. Nothing about the scoring logic
     itself changed -- same segment extraction (no rating filter), same
     id2label-keyed column mapping, same rating-3-cons coverage guard --
     this only changes when results hit disk and what happens on a
     second run.

  4. kaggle_timing_test.py does `import full_corpus_inference_kaggle as
     inf` to reuse this file's functions instead of duplicating them.
     That import resolves via plain Python behavior, not anything
     Kaggle-specific: running a script with `python /some/dir/foo.py`
     adds foo.py's own containing directory to sys.path automatically, so
     as long as both .py files sit in the same flat directory inside the
     uploaded dataset (per point 2), `!python
     /kaggle/input/datasets/{KAGGLE_DATASET_SLUG}/kaggle_timing_test.py`
     finds this file next to it with no sys.path.append or Kaggle
     "utility script" mechanism needed.

Input:
    phase1_scoring_corpus.csv -- same as the local script, uploaded as
    part of the Kaggle dataset.

Output (in /kaggle/working/):
    phase1_review_level_probs_partial.csv    -- review-level long-format
                                                 progress file, appended to
                                                 incrementally
    phase1_segment_level_probs_partial.csv   -- segment-level long-format
                                                 progress file, same idea
    phase1_review_sentiment_scores.csv       -- final wide-format table,
                                                 written only once both
                                                 heads report zero rows
                                                 remaining -- same schema
                                                 as the local script's
                                                 output, ready to load into MySQL
    phase1_skipped_reviews.csv               -- same as the local script

Usage (Kaggle notebook cell, or !python from within one):
    python full_corpus_inference_kaggle.py              # dry run, ~30 rows
    python full_corpus_inference_kaggle.py --full        # the real run
    (re-run the same command after a session cutoff to resume)
"""
import os

# Pin to a single visible GPU before torch is imported -- see point 1 above.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================================================================
# CONFIGURATION -- edit before running
# =============================================================================

# "<your-kaggle-username>/<dataset-slug>". Do NOT set this to a bare slug;
# see point 2 in the module docstring for why.
KAGGLE_DATASET_SLUG = "YOUR-USERNAME/YOUR-DATASET-SLUG"

INPUT_CSV = f"/kaggle/input/datasets/{KAGGLE_DATASET_SLUG}/phase1_scoring_corpus.csv"
REVIEW_CHECKPOINT = f"/kaggle/input/datasets/{KAGGLE_DATASET_SLUG}/review_level/final"
SEGMENT_CHECKPOINT = f"/kaggle/input/datasets/{KAGGLE_DATASET_SLUG}/segment_level_baseline_20260730/final"

WORKING_DIR = Path("/kaggle/working")
REVIEW_PARTIAL_PATH = WORKING_DIR / "phase1_review_level_probs_partial.csv"
SEGMENT_PARTIAL_PATH = WORKING_DIR / "phase1_segment_level_probs_partial.csv"
FINAL_PATH = WORKING_DIR / "phase1_review_sentiment_scores.csv"
FINAL_TEMP_PATH = FINAL_PATH.with_suffix(".csv.part")
SKIPPED_PATH = WORKING_DIR / "phase1_skipped_reviews.csv"

TOKENIZER_NAME = "beomi/KcELECTRA-base-v2022"
BATCH_SIZE = 16

# How often to flush + fsync scored rows to the partial CSVs. Tune down if
# a session keeps dying before even one flush lands; tune up if flushing
# itself becomes a measurable fraction of the timing-test throughput
# (unlikely at this size -- fsyncing a few thousand short rows is cheap
# relative to a transformer forward pass, but the constant is here to
# adjust rather than hardcoded assuming that stays true).
FLUSH_EVERY_N_ROWS = 20_000

REVIEW_MAX_LENGTH = 512
EXPECTED_REVIEW_LABELS = {"frustrated", "neutral_mixed", "high_satisfaction"}
REVIEW_LABEL_TO_SCHEMA_SUFFIX = {
    "frustrated": "negative",
    "neutral_mixed": "neutral",
    "high_satisfaction": "positive",
}

SEGMENT_MAX_LENGTH = 256
EXPECTED_SEGMENT_LABELS = {
    "positive_benefit",
    "minor_complaint",
    "strong_complaint",
    "usage_tip",
}
SEGMENT_FIELDS = ("pros", "cons", "tip")

MODEL_VERSION = "review:retrained-20260806|segment:baseline-20260730"


# =============================================================================
# SEGMENT EXTRACTION -- unchanged from full_corpus_inference.py
# =============================================================================


def build_segment_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per non-empty pros/cons/tip segment, every rating included.
    Same as the local script -- see that file's header for why this is
    NOT build_training_data_final.build_segment_rows()."""
    rows = []
    for review_id, rating, pros, cons, tip in df[
        ["review_id", "rating", "pros_ko", "cons_ko", "tip_ko"]
    ].itertuples(index=False, name=None):
        for field_name, text in (("pros", pros), ("cons", cons), ("tip", tip)):
            if isinstance(text, str) and text.strip():
                rows.append((review_id, field_name, text))
    return pd.DataFrame.from_records(rows, columns=["review_id", "segment", "text"])


def check_rating3_cons_coverage(corpus_df: pd.DataFrame, segment_df: pd.DataFrame) -> None:
    """Same hard-stop regression guard as the local script -- runs against
    the full candidate segment frame regardless of resume state, since
    resuming only changes what's already been *scored*, not what's a
    valid scoring candidate in the first place."""
    rating3_with_cons = corpus_df[
        (corpus_df["rating"] == 3)
        & corpus_df["cons_ko"].notna()
        & (corpus_df["cons_ko"].astype(str).str.strip() != "")
    ]
    scored_rating3_cons = segment_df[
        (segment_df["segment"] == "cons")
        & segment_df["review_id"].isin(rating3_with_cons["review_id"])
    ]
    expected = len(rating3_with_cons)
    actual = len(scored_rating3_cons)
    print(
        f"  rating-3 cons coverage check: {actual:,}/{expected:,} "
        f"rating-3 reviews with cons text are in the scoring input"
    )
    if actual != expected:
        raise SystemExit(
            f"rating-3 cons coverage check FAILED -- {expected - actual:,} rating-3 "
            "cons segments are missing from the scoring input. Stopping rather "
            "than silently shipping a re-introduced gap."
        )


# =============================================================================
# RESUME / INCREMENTAL-FLUSH HELPERS -- new in the Kaggle variant, no torch
# dependency, so these are the part covered by pure unit tests (no model
# stubbing needed).
# =============================================================================


def load_already_scored_review_ids(path: Path) -> set:
    """review_ids already present in the review-level partial file, or an
    empty set if it doesn't exist yet (first run)."""
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["review_id"])["review_id"])


def load_already_scored_segment_keys(path: Path) -> pd.DataFrame:
    """(review_id, segment) pairs already present in the segment-level
    partial file, as a DataFrame suitable for an anti-join -- returns an
    empty-but-correctly-columned frame if the file doesn't exist yet."""
    if not path.exists():
        return pd.DataFrame(columns=["review_id", "segment"])
    return pd.read_csv(path, usecols=["review_id", "segment"])


def filter_unscored_segments(
    segment_df: pd.DataFrame, already_scored: pd.DataFrame
) -> pd.DataFrame:
    """Anti-join segment_df against already-scored (review_id, segment)
    keys. Vectorized (merge-based) rather than a row-wise membership test
    -- this runs against ~2M segment rows at full-corpus scale, and a
    Python-level loop over that many rows on every resume would itself
    become a meaningful chunk of runtime."""
    if already_scored.empty:
        return segment_df.reset_index(drop=True)
    merged = segment_df.merge(
        already_scored.assign(_already_scored=True),
        on=["review_id", "segment"],
        how="left",
    )
    remaining = merged[merged["_already_scored"].isna()].drop(columns=["_already_scored"])
    return remaining.reset_index(drop=True)


def flush_rows_to_csv(rows: list, path: Path) -> None:
    """Append buffered result rows to a partial CSV, fsynced immediately --
    a killed Kaggle session doesn't get to rely on a clean shutdown to
    flush the OS page cache. No-op on an empty buffer (so callers can call
    this unconditionally at loop end without a length check first)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = pd.DataFrame(rows)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        chunk.to_csv(f, index=False, header=write_header)
        f.flush()
        os.fsync(f.fileno())


# =============================================================================
# INFERENCE
# =============================================================================


def load_and_check_model(path: str, expected_labels: set, device: str):
    print(f"\n--- Loading checkpoint: {path} ---")
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    label_set = set(id2label.values())
    print(f"  id2label: {id2label}")

    if label_set != expected_labels:
        raise SystemExit(
            f"id2label mismatch for checkpoint '{path}': got {sorted(label_set)}, "
            f"expected {sorted(expected_labels)}. Stopping -- do not score against "
            "a checkpoint whose label space doesn't match what's expected."
        )

    return model, id2label


def score_review_level_resumable(df, tokenizer, device) -> pd.DataFrame:
    """Resumable version of the local script's score_review_level(). Same
    scoreable/skipped split; the difference is everything after that:
    already-scored review_ids are filtered out before any model call, and
    results are flushed to REVIEW_PARTIAL_PATH incrementally rather than
    collected in memory and returned as one frame."""
    scoreable = df[
        df["review_text_ko_combined"].notna()
        & (df["review_text_ko_combined"].astype(str).str.strip() != "")
    ].copy()
    skipped = df[~df["review_id"].isin(scoreable["review_id"])][["review_id"]].copy()
    skipped["reason"] = "no Korean text (review_text_ko_combined empty)"

    already_scored = load_already_scored_review_ids(REVIEW_PARTIAL_PATH)
    remaining = scoreable[~scoreable["review_id"].isin(already_scored)]

    print(
        f"\nReview-level: {len(scoreable):,} scoreable, {len(skipped):,} skipped "
        f"(no Korean text), {len(already_scored):,} already scored (resumed), "
        f"{len(remaining):,} remaining"
    )

    if len(remaining) == 0:
        print("  Nothing left to score for this head -- skipping checkpoint load.")
        return skipped

    model, id2label = load_and_check_model(REVIEW_CHECKPOINT, EXPECTED_REVIEW_LABELS, device)
    label_order = [id2label[i] for i in range(len(id2label))]

    buffer = []
    t0 = time.time()
    remaining_ids = remaining["review_id"].tolist()
    remaining_texts = remaining["review_text_ko_combined"].astype(str).tolist()

    with torch.no_grad():
        for start in range(0, len(remaining_texts), BATCH_SIZE):
            batch_ids = remaining_ids[start : start + BATCH_SIZE]
            batch_texts = remaining_texts[start : start + BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=REVIEW_MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for review_id, pr in zip(batch_ids, probs):
                row = {"review_id": review_id}
                for i, label in enumerate(label_order):
                    suffix = REVIEW_LABEL_TO_SCHEMA_SUFFIX[label]
                    row[f"review_p_{suffix}"] = float(pr[i])
                buffer.append(row)

            done = min(start + BATCH_SIZE, len(remaining_texts))
            is_last = done >= len(remaining_texts)
            if len(buffer) >= FLUSH_EVERY_N_ROWS or is_last:
                flush_rows_to_csv(buffer, REVIEW_PARTIAL_PATH)
                buffer = []
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  {done:,}/{len(remaining_texts):,}  ({rate:.1f} rows/s)", end="\r")

    print(f"\n  done in {time.time() - t0:.1f}s")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return skipped


def score_segment_level_resumable(segment_df, tokenizer, device) -> None:
    """Resumable version of the local script's score_segment_level(), with
    the incremental-flush pattern applied the same way as the review-level
    head above. Writes long-format (review_id, segment, p_*) rows to
    SEGMENT_PARTIAL_PATH -- pivot_segments_to_wide() reads this back in
    main() once both heads report zero remaining."""
    already_scored = load_already_scored_segment_keys(SEGMENT_PARTIAL_PATH)
    remaining = filter_unscored_segments(segment_df, already_scored)

    print(
        f"Segment-level: {len(segment_df):,} non-empty segments, "
        f"{len(already_scored):,} already scored (resumed), "
        f"{len(remaining):,} remaining"
    )

    if len(remaining) == 0:
        print("  Nothing left to score for this head -- skipping checkpoint load.")
        return

    model, id2label = load_and_check_model(SEGMENT_CHECKPOINT, EXPECTED_SEGMENT_LABELS, device)
    label_order = [id2label[i] for i in range(len(id2label))]

    buffer = []
    t0 = time.time()
    remaining_keys = list(
        remaining[["review_id", "segment"]].itertuples(index=False, name=None)
    )
    remaining_texts = remaining["text"].astype(str).tolist()

    with torch.no_grad():
        for start in range(0, len(remaining_texts), BATCH_SIZE):
            batch_keys = remaining_keys[start : start + BATCH_SIZE]
            batch_texts = remaining_texts[start : start + BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=SEGMENT_MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for (review_id, segment), pr in zip(batch_keys, probs):
                row = {"review_id": review_id, "segment": segment}
                for i, label in enumerate(label_order):
                    row[f"p_{label}"] = float(pr[i])
                buffer.append(row)

            done = min(start + BATCH_SIZE, len(remaining_texts))
            is_last = done >= len(remaining_texts)
            if len(buffer) >= FLUSH_EVERY_N_ROWS or is_last:
                flush_rows_to_csv(buffer, SEGMENT_PARTIAL_PATH)
                buffer = []
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  {done:,}/{len(remaining_texts):,}  ({rate:.1f} rows/s)", end="\r")

    print(f"\n  done in {time.time() - t0:.1f}s")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def pivot_segments_to_wide(segment_scores: pd.DataFrame) -> pd.DataFrame:
    """Unchanged from the local script."""
    prob_cols = ["p_positive_benefit", "p_minor_complaint", "p_strong_complaint", "p_usage_tip"]
    pieces = []
    for field_name in SEGMENT_FIELDS:
        sub = segment_scores[segment_scores["segment"] == field_name][
            ["review_id"] + prob_cols
        ].copy()
        sub = sub.rename(columns={c: f"{field_name}_{c}" for c in prob_cols})
        sub = sub.set_index("review_id")
        pieces.append(sub)
    wide = pd.concat(pieces, axis=1)
    return wide.reset_index()


# =============================================================================
# MAIN
# =============================================================================


def build_dry_run_sample(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Unchanged from the local script."""
    sample = df.head(n).copy()
    rating3_cons_present = (
        (sample["rating"] == 3)
        & sample["cons_ko"].notna()
        & (sample["cons_ko"].astype(str).str.strip() != "")
    ).any()
    if not rating3_cons_present:
        candidates = df[
            (df["rating"] == 3)
            & df["cons_ko"].notna()
            & (df["cons_ko"].astype(str).str.strip() != "")
            & ~df["review_id"].isin(sample["review_id"])
        ]
        if len(candidates) > 0:
            sample = pd.concat([sample, candidates.head(1)], ignore_index=True)
            print(
                f"  (added review_id={candidates.iloc[0]['review_id']} to the dry-run "
                "sample -- a rating-3 review with cons text, to exercise that path)"
            )
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run all 814,452 rows (default: small dry run)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print(
            "  WARNING: no CUDA device found. Confirm the notebook's accelerator "
            "is set to a GPU option before running --full -- this will be "
            "extremely slow on CPU at 814K-row scale."
        )

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"loaded {len(df):,} rows from {INPUT_CSV}")

    if not args.full:
        df = build_dry_run_sample(df)
        print(f"DRY RUN -- using {len(df)} rows")
    else:
        if len(df) != 814_452:
            print(
                f"  WARNING: expected 814,452 rows from the export, got {len(df):,}. "
                "Not stopping automatically, but confirm this is expected."
            )

    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # --- review-level (resumable) ---
    skipped = score_review_level_resumable(df, tokenizer, device)

    # --- segment-level (resumable) ---
    segment_df = build_segment_frame(df)
    print(f"\n{len(segment_df):,} non-empty segments in the scoring input")
    check_rating3_cons_coverage(df, segment_df)
    score_segment_level_resumable(segment_df, tokenizer, device)

    # --- are both heads actually done? ---
    review_ids_scored = load_already_scored_review_ids(REVIEW_PARTIAL_PATH)
    scoreable_ids = set(
        df.loc[
            df["review_text_ko_combined"].notna()
            & (df["review_text_ko_combined"].astype(str).str.strip() != ""),
            "review_id",
        ]
    )
    review_remaining = len(scoreable_ids - review_ids_scored)

    already_segment_keys = load_already_scored_segment_keys(SEGMENT_PARTIAL_PATH)
    segment_remaining = len(filter_unscored_segments(segment_df, already_segment_keys))

    if review_remaining > 0 or segment_remaining > 0:
        print(
            f"\nSession ended with work still remaining "
            f"(review: {review_remaining:,}, segment: {segment_remaining:,}). "
            f"Partial progress is saved in {REVIEW_PARTIAL_PATH.name} and "
            f"{SEGMENT_PARTIAL_PATH.name} -- re-run this same command to resume. "
            f"{FINAL_PATH.name} is NOT written until both heads report zero remaining."
        )
        sys.exit(0)

    # --- both heads complete: assemble the final wide-format table ---
    print("\nBoth heads complete -- assembling final output...")
    review_scores = pd.read_csv(REVIEW_PARTIAL_PATH)
    segment_scores_long = pd.read_csv(SEGMENT_PARTIAL_PATH)
    segment_scores_wide = pivot_segments_to_wide(segment_scores_long)

    final = df[["review_id", "product_id", "review_month"]].merge(
        review_scores, on="review_id", how="inner"  # inner: drops the ~9 skipped rows
    )
    final = final.merge(segment_scores_wide, on="review_id", how="left")
    final["model_version"] = MODEL_VERSION

    FINAL_TEMP_PATH.unlink(missing_ok=True)
    final.to_csv(FINAL_TEMP_PATH, index=False)
    FINAL_TEMP_PATH.replace(FINAL_PATH)
    skipped.to_csv(SKIPPED_PATH, index=False)

    print(f"\nwrote {FINAL_PATH} ({len(final):,} rows)")
    print(f"wrote {SKIPPED_PATH} ({len(skipped):,} rows)")

    if not args.full:
        print("\n" + final.head(10).to_string())
        print(
            "\nDry run complete. Confirm: (1) column names match "
            "review_sentiment_scores_v1's schema exactly, (2) row count = "
            "scoreable input rows, (3) re-running this same command now "
            "(still no --full) reports everything already scored, 0 remaining "
            "-- that's the resume path working, not a bug. Delete the "
            "*_partial.csv files before starting the real --full run so it "
            "doesn't think dry-run rows are already done."
        )
    else:
        print("Full run complete. Load phase1_review_sentiment_scores.csv into MySQL next.")
        print(
            "Also run qa_segment_field_distribution.py against it "
            "before loading it."
        )


if __name__ == "__main__":
    main()
