"""
Build sentiment-model training CSV files from `hwahae_reviews_clean_v3`.

This script does not scrape Hwahae. It reads already cleaned review data
from MySQL, creates review-level and segment-level heuristic labels, separates
rows that need manual/LLM review, and writes the resulting CSV files.

EVAL_SAMPLE_REVIEW_IDS is loaded from two CSV files and unioned --
eval_sample_review_ids.csv (the 1,000-id base draw) and freed_review_ids_v2.csv
(99 ids freed from training so the eval-sample rebuild could backfill ratings
1/2/3 toward target -- 12 rating-1, 11 rating-2, 76 rating-3, all drawn from the
validation split only, so this cost zero actual training signal). With a
hardcoded set instead, the 99 freed ids would silently leak back into
training -- use the loader below, not a literal set.

Required packages:
    pip install pandas mysql-connector-python python-dotenv

Database credentials are read from a .env file:
    DB_HOST=localhost
    DB_USER=root
    DB_PASSWORD=your_password
    DB_NAME=hwahae_project
"""

from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path
from typing import Any

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from mysql.connector import Error


# ============================================================
# USER SETTINGS
# ============================================================

# MySQL source view created by the SQL preprocessing stage.
SOURCE_VIEW = "hwahae_reviews_clean_v3"

# Use a number for a quick test.
# Use None to load the complete view.
MAX_REVIEWS_THIS_RUN: int | None = None

# False:
#   SELECT ... LIMIT N
#   Fast smoke test, but not a representative sample.
#
# True:
#   SELECT ... ORDER BY RAND(RANDOM_SEED) LIMIT N
#   Representative random sample, but MySQL must scan and sort the view.
#   Use only with a reasonable MAX_REVIEWS_THIS_RUN value.
USE_RANDOM_SAMPLE = False
RANDOM_SEED = 42

# Output folder. Existing files with the same names are overwritten.
OUTPUT_DIR = Path("training_data_out")

# Reviews with platform_status='pending_validation' or 'blind_hidden'
# are excluded by default.
INCLUDE_PENDING_OR_BLIND = False

# Any manually labeled evaluation review IDs must be excluded from training.
# Loaded from file rather than hand-populated inline. Base draw:
# eval_sample_review_ids.csv (1,000 ids). Plus freed_review_ids_v2.csv (99 ids freed from training
# so ratings 1/2/3 can be backfilled toward target in the eval-sample rebuild
# -- 12 rating-1 + 11 rating-2 + 76 rating-3, all drawn from the validation
# split only, so this costs zero actual training signal).
def _load_review_id_csv(path: str) -> set[int]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return {int(row[0]) for row in reader if row}

EVAL_SAMPLE_REVIEW_IDS: set[int] = (
    _load_review_id_csv("eval_sample_review_ids.csv")
    | _load_review_id_csv("freed_review_ids_v2.csv")
)

# Print transformation progress after this many source reviews.
# Use None to disable intermediate progress messages.
PROGRESS_EVERY_REVIEWS: int | None = 100_000

# Print label counts for every generated CSV.
PRINT_LABEL_COUNTS = True

# CSV export settings. QUOTE_ALL plus an explicit escape character safely
# handles commas, quotation marks, backslashes, and embedded line breaks.
CSV_ENCODING = "utf-8-sig"
CSV_QUOTING = csv.QUOTE_ALL
CSV_QUOTECHAR = '"'
CSV_ESCAPECHAR = "\\"
CSV_DOUBLEQUOTE = True
CSV_LINE_TERMINATOR = "\n"
CSV_CHUNK_SIZE: int | None = 100_000

# Write each CSV to a temporary .part file first, then replace the final file
# only after the export finishes successfully.
WRITE_TO_TEMP_FILE_FIRST = True


# ============================================================
# MYSQL CONFIG
# ============================================================

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": "utf8mb4",
    "use_unicode": True,
}


# ============================================================
# OUTPUT FILES
# ============================================================

REVIEW_LEVEL_KO_FILE = "review_level_ko_train.csv"
SEGMENT_LEVEL_KO_TRAIN_FILE = "segment_level_ko_train.csv"
SEGMENT_LEVEL_KO_REVIEW_FILE = "segment_level_ko_needs_review.csv"
SEGMENT_LEVEL_EN_TRAIN_FILE = "segment_level_en_train.csv"
SEGMENT_LEVEL_EN_REVIEW_FILE = "segment_level_en_needs_review.csv"

OUTPUT_FILE_ORDER = (
    REVIEW_LEVEL_KO_FILE,
    SEGMENT_LEVEL_KO_TRAIN_FILE,
    SEGMENT_LEVEL_KO_REVIEW_FILE,
    SEGMENT_LEVEL_EN_TRAIN_FILE,
    SEGMENT_LEVEL_EN_REVIEW_FILE,
)


# ============================================================
# REQUIRED SOURCE COLUMNS
# ============================================================

VIEW_BOOL_COLS = (
    "lang_routing_mismatch",
    "cons_ko_empty_complaint_risk",
    "cons_en_empty_complaint_risk",
)

SOURCE_COLUMNS = (
    "review_id",
    "rating",
    "platform_status",
    "has_korean_detail",
    "lang_routing_mismatch",
    "review_text_ko_combined",
    "pros_ko",
    "cons_ko",
    "tip_ko",
    "pros_en",
    "cons_en",
    "tip_en",
    "cons_ko_empty_complaint_risk",
    "cons_en_empty_complaint_risk",
)

REQUIRED_SOURCE_COLS = set(SOURCE_COLUMNS)

SEGMENT_OUTPUT_COLS = [
    "review_id",
    "segment",
    "lang",
    "text",
    "label",
    "needs_review",
]


# ============================================================
# CONTAMINATION-RISK MARKERS
# ============================================================

# These are triage flags only. They do not automatically change labels.
WARNING_MARKER_KO = re.compile(
    r"(자극|트러블|화끈|따가|붉어|뒤집어|올라왔|안맞|맞지\s*않|피하세요|주의)"
)

NEGATED_WARNING_KO = re.compile(
    r"(자극|트러블|화끈|따가|붉어|뒤집어|올라왔|안맞|맞지\s*않|피하세요|주의)"
    r"(?:이|가|은|는)?\s*(?:없|덜|안\s)"
)

WARNING_MARKER_EN = re.compile(
    r"\b(avoid|don't use|do not use|broke me out|broke out|irritat|"
    r"sting|burn(?:ing|ed)?|allerg|breakout|clog|redness|itch)\b",
    re.IGNORECASE,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def validate_settings() -> None:
    """Fail early when a user setting is invalid."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", SOURCE_VIEW):
        raise ValueError(
            "SOURCE_VIEW must be a plain MySQL table/view name containing "
            "only letters, numbers, and underscores."
        )

    if MAX_REVIEWS_THIS_RUN is not None and MAX_REVIEWS_THIS_RUN <= 0:
        raise ValueError("MAX_REVIEWS_THIS_RUN must be a positive integer or None.")

    if USE_RANDOM_SAMPLE and MAX_REVIEWS_THIS_RUN is None:
        raise ValueError(
            "USE_RANDOM_SAMPLE=True requires MAX_REVIEWS_THIS_RUN to be a number."
        )

    if PROGRESS_EVERY_REVIEWS is not None and PROGRESS_EVERY_REVIEWS <= 0:
        raise ValueError("PROGRESS_EVERY_REVIEWS must be a positive integer or None.")

    missing_db_settings = [
        key
        for key in ("user", "password", "database")
        if not DB_CONFIG.get(key)
    ]

    if missing_db_settings:
        env_names = {
            "user": "DB_USER",
            "password": "DB_PASSWORD",
            "database": "DB_NAME",
        }
        missing_env_names = ", ".join(env_names[key] for key in missing_db_settings)
        raise ValueError(
            f"Missing database settings: {missing_env_names}. "
            "Add them to the .env file."
        )


def validate_source_columns(df: pd.DataFrame) -> None:
    """Check that the SQL preprocessing view has the expected schema."""
    missing_columns = sorted(REQUIRED_SOURCE_COLS - set(df.columns))

    if missing_columns:
        raise ValueError(
            "The source view is missing required columns:\n  - "
            + "\n  - ".join(missing_columns)
        )


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """Convert MySQL 0/1/NULL values into reliable Python booleans."""
    normalized = series.replace(
        {
            True: 1,
            False: 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "TRUE": 1,
            "FALSE": 0,
        }
    )
    return pd.to_numeric(normalized, errors="coerce").fillna(0).astype(bool)


def print_label_distribution(df: pd.DataFrame) -> str:
    """Return a compact label-count string for progress output."""
    if not PRINT_LABEL_COUNTS or df.empty or "label" not in df.columns:
        return ""

    counts = df["label"].value_counts(dropna=False).to_dict()
    return " | " + ", ".join(
        f"{label}={count:,}"
        for label, count in counts.items()
    )


# ============================================================
# LABELING HELPERS
# ============================================================

def flag_contamination_risk(text: str | None, lang: str) -> bool:
    """
    Flag tip/pros text that may contain embedded negative sentiment.

    This is only a review-priority signal. A flagged row is sent to a
    separate needs-review CSV instead of being silently relabeled.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    if lang == "ko":
        has_warning = bool(WARNING_MARKER_KO.search(text))
        has_negated_warning = bool(NEGATED_WARNING_KO.search(text))
        return has_warning and not has_negated_warning

    if lang == "en":
        return bool(WARNING_MARKER_EN.search(text))

    raise ValueError(f"Unsupported language: {lang}")


def rating_to_review_label(rating: Any) -> str | None:
    """Convert a 1-5 Hwahae rating into a three-class review label."""
    if pd.isna(rating):
        return None

    rating_int = int(rating)

    if rating_int <= 2:
        return "frustrated"

    if rating_int == 3:
        return "neutral_mixed"

    if rating_int >= 4:
        return "high_satisfaction"

    return None


def build_segment_rows(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Create one row per non-empty pros/cons/tip segment.

    Labels:
        tip  -> usage_tip
        pros -> positive_benefit
        cons -> strong_complaint for ratings 1-2
                minor_complaint for ratings 4-5
                skipped for rating 3 because it is ambiguous
    """
    if lang not in {"ko", "en"}:
        raise ValueError("lang must be 'ko' or 'en'.")

    suffix = f"_{lang}"
    column_positions = {
        column_name: position
        for position, column_name in enumerate(df.columns)
    }

    review_id_position = column_positions["review_id"]
    rating_position = column_positions["rating"]
    tip_position = column_positions[f"tip{suffix}"]
    pros_position = column_positions[f"pros{suffix}"]
    cons_position = column_positions[f"cons{suffix}"]
    empty_complaint_position = column_positions[
        f"cons_{lang}_empty_complaint_risk"
    ]

    rows: list[tuple[Any, str, str, str, str, bool]] = []
    total_reviews = len(df)

    print(f"  Building {lang.upper()} segment rows from {total_reviews:,} reviews...")

    for position, row in enumerate(
        df.itertuples(index=False, name=None),
        start=1,
    ):
        review_id = row[review_id_position]
        rating = row[rating_position]

        tip = row[tip_position]
        if isinstance(tip, str) and tip.strip():
            rows.append(
                (
                    review_id,
                    "tip",
                    lang,
                    tip,
                    "usage_tip",
                    flag_contamination_risk(tip, lang),
                )
            )

        pros = row[pros_position]
        if isinstance(pros, str) and pros.strip():
            rows.append(
                (
                    review_id,
                    "pros",
                    lang,
                    pros,
                    "positive_benefit",
                    flag_contamination_risk(pros, lang),
                )
            )

        cons = row[cons_position]
        if isinstance(cons, str) and cons.strip() and pd.notna(rating):
            rating_int = int(rating)

            if rating_int != 3:
                label = (
                    "strong_complaint"
                    if rating_int <= 2
                    else "minor_complaint"
                )
                rows.append(
                    (
                        review_id,
                        "cons",
                        lang,
                        cons,
                        label,
                        bool(row[empty_complaint_position]),
                    )
                )

        if (
            PROGRESS_EVERY_REVIEWS is not None
            and (
                position % PROGRESS_EVERY_REVIEWS == 0
                or position == total_reviews
            )
        ):
            print(
                f"    {position:,}/{total_reviews:,} reviews processed | "
                f"{len(rows):,} segment rows created"
            )

    return pd.DataFrame.from_records(
        rows,
        columns=SEGMENT_OUTPUT_COLS,
    )


# ============================================================
# TRAINING-DATA ASSEMBLY
# ============================================================

def build_all(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply exclusions and create all five output DataFrames."""
    step_start = time.perf_counter()
    df = df.copy()

    print("\n[2/4] Preparing and filtering source rows...")

    validate_source_columns(df)

    # Make ratings numeric once so later int conversions are predictable.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    for column_name in VIEW_BOOL_COLS:
        df[column_name] = normalize_bool_series(df[column_name])

    df["has_korean_detail"] = normalize_bool_series(df["has_korean_detail"])

    loaded_count = len(df)

    if EVAL_SAMPLE_REVIEW_IDS:
        df = df[~df["review_id"].isin(EVAL_SAMPLE_REVIEW_IDS)].copy()
    after_eval_count = len(df)

    if not INCLUDE_PENDING_OR_BLIND:
        df = df[df["platform_status"].isna()].copy()
    after_status_count = len(df)

    language_mismatch_count = int(df["lang_routing_mismatch"].sum())

    print(f"  Loaded rows:                    {loaded_count:>12,}")
    print(
        f"  After eval-sample exclusion:    {after_eval_count:>12,} "
        f"({loaded_count - after_eval_count:,} excluded)"
    )
    print(
        f"  After platform-status filter:   {after_status_count:>12,} "
        f"({after_eval_count - after_status_count:,} excluded)"
    )
    print(
        f"  Language mismatches flagged:    {language_mismatch_count:>12,} "
        "(excluded from Korean outputs)"
    )

    outputs: dict[str, pd.DataFrame] = {}

    print("\n[3/4] Creating labels and output datasets...")

    # Review-level Korean labels.
    ko_review_eligible = df[
        df["has_korean_detail"]
        & ~df["lang_routing_mismatch"]
    ].copy()

    ko_review_eligible["label"] = ko_review_eligible["rating"].apply(
        rating_to_review_label
    )

    ko_review = ko_review_eligible[
        ko_review_eligible["review_text_ko_combined"].notna()
        & ko_review_eligible["label"].notna()
    ][
        ["review_id", "review_text_ko_combined", "label"]
    ].rename(
        columns={"review_text_ko_combined": "text"}
    )

    outputs[REVIEW_LEVEL_KO_FILE] = ko_review.reset_index(drop=True)

    print(
        f"  {REVIEW_LEVEL_KO_FILE}: "
        f"{len(outputs[REVIEW_LEVEL_KO_FILE]):,} rows"
        f"{print_label_distribution(outputs[REVIEW_LEVEL_KO_FILE])}"
    )

    # Segment-level Korean labels.
    ko_segment_source = df[~df["lang_routing_mismatch"]].copy()
    ko_segments = build_segment_rows(ko_segment_source, "ko")

    outputs[SEGMENT_LEVEL_KO_TRAIN_FILE] = ko_segments.loc[
        ~ko_segments["needs_review"],
        ["review_id", "segment", "text", "label"],
    ].reset_index(drop=True)

    outputs[SEGMENT_LEVEL_KO_REVIEW_FILE] = ko_segments.loc[
        ko_segments["needs_review"],
        ["review_id", "segment", "text", "label"],
    ].reset_index(drop=True)

    print(
        f"  {SEGMENT_LEVEL_KO_TRAIN_FILE}: "
        f"{len(outputs[SEGMENT_LEVEL_KO_TRAIN_FILE]):,} rows"
        f"{print_label_distribution(outputs[SEGMENT_LEVEL_KO_TRAIN_FILE])}"
    )
    print(
        f"  {SEGMENT_LEVEL_KO_REVIEW_FILE}: "
        f"{len(outputs[SEGMENT_LEVEL_KO_REVIEW_FILE]):,} rows"
        f"{print_label_distribution(outputs[SEGMENT_LEVEL_KO_REVIEW_FILE])}"
    )

    # Segment-level English labels.
    en_segments = build_segment_rows(df, "en")

    outputs[SEGMENT_LEVEL_EN_TRAIN_FILE] = en_segments.loc[
        ~en_segments["needs_review"],
        ["review_id", "segment", "text", "label"],
    ].reset_index(drop=True)

    outputs[SEGMENT_LEVEL_EN_REVIEW_FILE] = en_segments.loc[
        en_segments["needs_review"],
        ["review_id", "segment", "text", "label"],
    ].reset_index(drop=True)

    print(
        f"  {SEGMENT_LEVEL_EN_TRAIN_FILE}: "
        f"{len(outputs[SEGMENT_LEVEL_EN_TRAIN_FILE]):,} rows"
        f"{print_label_distribution(outputs[SEGMENT_LEVEL_EN_TRAIN_FILE])}"
    )
    print(
        f"  {SEGMENT_LEVEL_EN_REVIEW_FILE}: "
        f"{len(outputs[SEGMENT_LEVEL_EN_REVIEW_FILE]):,} rows"
        f"{print_label_distribution(outputs[SEGMENT_LEVEL_EN_REVIEW_FILE])}"
    )

    print(
        f"  Labeling completed in "
        f"{format_elapsed(time.perf_counter() - step_start)}."
    )

    return outputs


# ============================================================
# DATABASE ACCESS
# ============================================================

def get_connection():
    """Open a MySQL connection using DB_CONFIG."""
    return mysql.connector.connect(**DB_CONFIG)


def build_source_query() -> str:
    """Build the source SELECT query from the user settings."""
    selected_columns = ",\n    ".join(SOURCE_COLUMNS)
    query = f"SELECT\n    {selected_columns}\nFROM {SOURCE_VIEW}"

    if MAX_REVIEWS_THIS_RUN is None:
        return query

    if USE_RANDOM_SAMPLE:
        query += f" ORDER BY RAND({int(RANDOM_SEED)})"

    query += f" LIMIT {int(MAX_REVIEWS_THIS_RUN)}"
    return query


def load_reviews(connection) -> pd.DataFrame:
    """Load review rows from the configured MySQL source view."""
    query = build_source_query()
    return pd.read_sql(query, connection)


# ============================================================
# OUTPUT
# ============================================================

def save_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """Write generated DataFrames safely and print file-by-file progress."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[4/4] Writing CSV files...")

    for position, file_name in enumerate(OUTPUT_FILE_ORDER, start=1):
        if file_name not in outputs:
            raise KeyError(f"Missing expected output DataFrame: {file_name}")

        output_df = outputs[file_name]
        output_path = OUTPUT_DIR / file_name
        temporary_path = OUTPUT_DIR / f"{file_name}.part"
        write_path = temporary_path if WRITE_TO_TEMP_FILE_FIRST else output_path

        if temporary_path.exists():
            temporary_path.unlink()

        file_start = time.perf_counter()

        print(
            f"  [{position}/{len(OUTPUT_FILE_ORDER)}] "
            f"Writing {file_name}: {len(output_df):,} rows..."
        )

        try:
            output_df.to_csv(
                write_path,
                index=False,
                encoding=CSV_ENCODING,
                quoting=CSV_QUOTING,
                quotechar=CSV_QUOTECHAR,
                escapechar=CSV_ESCAPECHAR,
                doublequote=CSV_DOUBLEQUOTE,
                lineterminator=CSV_LINE_TERMINATOR,
                chunksize=CSV_CHUNK_SIZE,
            )
        except csv.Error as error:
            if temporary_path.exists():
                temporary_path.unlink()
            raise RuntimeError(
                f"CSV writer failed while exporting {file_name}. "
                "The generated DataFrames are still available in LAST_OUTPUTS "
                "when this code is run in the same notebook kernel."
            ) from error
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

        if WRITE_TO_TEMP_FILE_FIRST:
            temporary_path.replace(output_path)

        elapsed = time.perf_counter() - file_start
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(
            f"      Saved in {format_elapsed(elapsed)} | "
            f"{file_size_mb:,.1f} MB | {output_path}"
        )


def print_run_header() -> None:
    """Print the active settings before the database work starts."""
    print("=" * 72)
    print("HWAHAE TRAINING-DATA ASSEMBLY")
    print("=" * 72)
    print(f"Source view:                  {SOURCE_VIEW}")

    if MAX_REVIEWS_THIS_RUN is None:
        print("Review limit:                 None (full view)")
    else:
        print(f"Review limit:                 {MAX_REVIEWS_THIS_RUN:,}")

    print(f"Random sample:                {USE_RANDOM_SAMPLE}")
    if USE_RANDOM_SAMPLE:
        print(f"Random seed:                  {RANDOM_SEED}")

    print(f"Include pending/blind rows:   {INCLUDE_PENDING_OR_BLIND}")
    print(f"Eval review IDs excluded:     {len(EVAL_SAMPLE_REVIEW_IDS):,}")
    print(f"Output directory:             {OUTPUT_DIR.resolve()}")
    print("=" * 72)


# Keep the latest generated DataFrames accessible in a notebook.
# If export fails after labeling, fix the export settings and run:
#     save_outputs(LAST_OUTPUTS)
# without loading MySQL or rebuilding labels again.
LAST_OUTPUTS: dict[str, pd.DataFrame] | None = None


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global LAST_OUTPUTS

    validate_settings()
    print_run_header()

    run_start = time.perf_counter()
    connection = None

    try:
        print("\n[1/4] Connecting to MySQL and loading reviews...")

        connection = get_connection()

        if connection.is_connected():
            print("  Connected to MySQL.")

        load_start = time.perf_counter()
        df = load_reviews(connection)
        load_elapsed = time.perf_counter() - load_start

        print(
            f"  Loaded {len(df):,} rows from {SOURCE_VIEW} "
            f"in {format_elapsed(load_elapsed)}."
        )

        if df.empty:
            print("\nNo rows matched the current settings. Nothing was written.")
            return

        LAST_OUTPUTS = build_all(df)
        save_outputs(LAST_OUTPUTS)

        total_output_rows = sum(
            len(output_df) for output_df in LAST_OUTPUTS.values()
        )

        print("\n" + "=" * 72)
        print("RUN SUMMARY")
        print("=" * 72)
        print(f"Source reviews loaded:        {len(df):>12,}")
        print(f"CSV files written:            {len(LAST_OUTPUTS):>12,}")
        print(f"Total output rows:            {total_output_rows:>12,}")
        print(
            f"Needs-review rows:            "
            f"{len(LAST_OUTPUTS[SEGMENT_LEVEL_KO_REVIEW_FILE]) + len(LAST_OUTPUTS[SEGMENT_LEVEL_EN_REVIEW_FILE]):>12,}"
        )
        print(
            f"Total run time:               "
            f"{format_elapsed(time.perf_counter() - run_start):>12}"
        )
        print(f"Output directory:             {OUTPUT_DIR.resolve()}")
        print("=" * 72)
        print(
            "\nReminder: *_needs_review.csv files are triage queues, "
            "not final training data."
        )

    except Error as error:
        print("\nMySQL error:")
        print(error)
        raise

    except Exception as error:
        print("\nGeneral error:")
        print(error)
        raise

    finally:
        if connection and connection.is_connected():
            connection.close()
            print("\nMySQL connection closed.")


if __name__ == "__main__":
    main()