"""
Populate voc_decline_risk_v1 (CUSUM-based decline flags).

WHAT THIS SCRIPT DOES, end to end:
  1. Reads product_monthly_sentiment_v1 (a VIEW, always live).
  2. Computes, per product, the CUSUM series (S_t) and derives both
     h-tier flags from it, plus the volume-adjusted notable_month_dip
     signal, as unit-testable pure functions (see "PURE COMPUTATION"
     section below).
  3. Fully replaces voc_decline_risk_v1's contents with the result.

Database settings come from the DB_HOST/DB_USER/DB_PASSWORD/DB_NAME
environment variables (.env). The write is a parameterized INSERT via
cursor.executemany(), wrapped in try/except Error: rollback(); raise,
finally: cursor.close() -- with no ON DUPLICATE KEY UPDATE, no
multi-batch chunking, and no separate progress table (see below).

WHY THIS IS A FULL RECOMPUTE, NOT AN INCREMENTAL APPEND:
  - Size: voc_decline_risk_v1 is one row per product-month that exists in
    product_monthly_sentiment_v1 -- ~4,180 rows (103
    products), not 814K. Recomputing the whole thing every run costs
    nothing worth optimizing away.
  - Correctness: CUSUM's S_t is defined chronologically forward from each
    product's full gate-passing history, using a trailing baseline. If a
    review_sentiment_scores_v1 row were ever corrected after the fact
    (a re-score, a fixed model_version load), an *older* month's
    aggregate net_sentiment could change retroactively -- which would
    change that month's baseline contribution and therefore every later
    S_t in the same product's series. An incremental "just insert new
    months" loader would silently miss that ripple effect and leave
    stale S_t values in place. A full recompute from
    product_monthly_sentiment_v1 (the source of truth) every run has no
    such blind spot, and at this row count there's no cost to paying for
    correctness this way.

WHY DELETE, NOT TRUNCATE:
  - TRUNCATE TABLE is DDL in MySQL/MariaDB/InnoDB -- it carries an
    implicit commit and CANNOT be rolled back as part of a transaction.
    Using it here would mean a failed INSERT leaves the table
    empty/truncated with no way back except re-running this script
    successfully -- exactly the "silently missing every product's
    current data" failure mode this design is trying to avoid.
    DELETE FROM is ordinary DML, fully transactional under InnoDB: if the
    INSERT that follows it fails, ROLLBACK undoes the DELETE too, and
    voc_decline_risk_v1 is left exactly as it was before this run started.

WHY ONE INSERT, NOT BATCHED: at ~4,180 rows, one executemany() call and one commit is well
within any reasonable statement-size limit and keeps the whole
DELETE+INSERT as a single all-or-nothing unit -- there is no partial-
progress benefit to chunking a table this small, and chunking would
reintroduce the exact partial-reload risk the DELETE-not-TRUNCATE choice
above is trying to eliminate (an error in batch 3 of 5 would leave
batches 1-2's rows committed against a table that's supposed to be a
complete, consistent snapshot every time it's read).

Usage:
    python populate_voc_decline_risk.py
    python populate_voc_decline_risk.py --dry-run    # compute + print summary, no DB write
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

SOURCE_VIEW = "product_monthly_sentiment_v1"
TABLE_NAME = "voc_decline_risk_v1"

# CUSUM parameters. sigma_monthly = STDDEV(net_sentiment), gate-passing
# product-months. sigma_review = STDDEV(review_net_sentiment), full corpus
# -- used for notable_month_dip's per-pair SE, not for CUSUM itself.
SIGMA_MONTHLY = 0.07841667478303109
SIGMA_REVIEW = 0.37752813558355647

CUSUM_K = SIGMA_MONTHLY / 2                 # ~0.0392 -- allowance
BASELINE_WINDOW = 6                          # trailing months, gate-passing only
MIN_BASELINE_MONTHS = 3                      # + current month = 4, the minimum-history rule
H_BROAD = 4 * SIGMA_MONTHLY                  # ~0.3137 -- voc_decline_broad_flag
H_TIGHT = 5 * SIGMA_MONTHLY                  # ~0.3921 -- voc_decline_flag (the tight tier)

# notable_month_dip (a second, informational signal; a ~1.5-2x range was
# considered, 2.0 chosen).
DIP_SE_MULTIPLIER = 2.0

INSERT_COLUMNS = (
    "product_id",
    "as_of_month",
    "net_sentiment",
    "meets_review_count_gate",  # see build_output_rows()
    "review_count",             # see build_output_rows()
    "baseline_net_sentiment",
    "baseline_months_available",
    "cusum_score",
    "decline_since",
    "voc_decline_broad_flag",
    "voc_decline_flag",
    "notable_month_dip",
    "notable_month_dip_drop",
    "notable_month_dip_se",
)


# =============================================================================
# DATABASE
# =============================================================================


def get_db_config() -> dict:
    """Build the MySQL connection settings from environment variables, with
    the same connection-level collation the rest of this project's tooling
    uses against the same database."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
        "collation": "utf8mb4_unicode_ci",
        "use_unicode": True,
    }


def validate_db_config(db_config: dict) -> None:
    """Fail early if required database credentials are missing."""
    required_settings = {
        "user": "DB_USER",
        "password": "DB_PASSWORD",
        "database": "DB_NAME",
    }
    missing = [
        env_name
        for config_key, env_name in required_settings.items()
        if not db_config.get(config_key)
    ]
    if missing:
        raise ValueError(
            "Missing database settings: "
            + ", ".join(missing)
            + ". Add them to the .env file."
        )


def get_connection(db_config: dict):
    """Create a MySQL connection."""
    return mysql.connector.connect(**db_config)


def fetch_monthly_sentiment(connection) -> pd.DataFrame:
    """Read product_monthly_sentiment_v1 in full -- ~4,180 rows, no
    chunking needed (unlike the 814K-row review-level table)."""
    query = f"""
        SELECT product_id, review_month, review_count,
               meets_review_count_gate, net_sentiment
        FROM {SOURCE_VIEW}
        ORDER BY product_id, review_month
    """
    df = pd.read_sql_query(query, connection, parse_dates=["review_month"])
    # mysql-connector-python returns a TINYINT(1)/BOOLEAN column as a
    # plain Python int (0/1) via read_sql_query -- cast explicitly rather
    # than relying on truthiness, so a future connector version returning
    # something else (e.g. a literal bool already) doesn't silently change
    # behavior.
    df["meets_review_count_gate"] = df["meets_review_count_gate"].astype(bool)
    return df


# =============================================================================
# PURE COMPUTATION -- no DB/connection needed, so this is the part
# that can be unit-tested without a real MySQL instance.
# =============================================================================


def compute_cusum(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product, walk gate-passing months in calendar order and compute
    the CUSUM series. A gap or gate-failing month is simply absent from
    the sequence; it doesn't reset S_t.

    S_t itself does not depend on h (h only decides which side of the line
    counts as "flagged"), so this computes S_t once and derives BOTH
    voc_decline_broad_flag (h=H_BROAD) and voc_decline_flag (h=H_TIGHT)
    from that single value. Same numbers, half the computation, and no
    risk of the two passes' S_t series silently drifting apart from each
    other due to a future edit to one but not the other.

    Returns one row per (product_id, review_month) for every
    gate-passing month, including the first MIN_BASELINE_MONTHS months of
    each product's history where baseline/cusum are NaN but
    baseline_months_available is still populated (0, 1, or 2) -- useful
    on its own for explaining "why isn't this product flagged yet."
    Gate-failing months are NOT included here; build_output_rows() below
    is what restores full row-per-product-month coverage.
    """
    rows = []
    for product_id, group in monthly_df.groupby("product_id"):
        gated = group[group["meets_review_count_gate"]].sort_values("review_month").reset_index(drop=True)
        s = 0.0
        decline_since = None
        for i, row in gated.iterrows():
            history = gated["net_sentiment"].iloc[max(0, i - BASELINE_WINDOW):i]
            if len(history) < MIN_BASELINE_MONTHS:
                rows.append({
                    "product_id": product_id,
                    "review_month": row["review_month"],
                    "baseline_net_sentiment": np.nan,
                    "baseline_months_available": len(history),
                    "cusum_score": np.nan,
                    "decline_since": None,
                    "voc_decline_broad_flag": False,
                    "voc_decline_flag": False,
                })
                continue

            baseline = history.mean()
            s = max(0.0, s + (baseline - row["net_sentiment"] - CUSUM_K))
            if s > 0 and decline_since is None:
                decline_since = row["review_month"]
            elif s == 0.0:
                decline_since = None

            rows.append({
                "product_id": product_id,
                "review_month": row["review_month"],
                "baseline_net_sentiment": baseline,
                "baseline_months_available": len(history),
                "cusum_score": s,
                "decline_since": decline_since,
                "voc_decline_broad_flag": s > H_BROAD,
                "voc_decline_flag": s > H_TIGHT,
            })
    return pd.DataFrame(rows, columns=[
        "product_id", "review_month", "baseline_net_sentiment",
        "baseline_months_available", "cusum_score", "decline_since",
        "voc_decline_broad_flag", "voc_decline_flag",
    ])


def compute_notable_month_dip(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product, walk calendar-adjacent gate-passing month pairs and flag a pair whose drop exceeds DIP_SE_MULTIPLIER times the
    pair's own combined standard error. Only true calendar-adjacent pairs
    are evaluated -- a gap breaks the pair, the same rule used everywhere
    else in this mart.

    Returns one row per evaluated pair, keyed at the LATER month
    (review_month = "cur"). notable_month_dip_drop/se are only populated
    when the pair is actually flagged -- an evaluated-but-not-flagged
    pair still gets a row here (notable_month_dip = False) so
    build_output_rows() can distinguish "evaluated, not notable" from
    "no valid adjacent pair at all" (e.g. a product's first gate-passing
    month, or right after a gap).
    """
    rows = []
    for product_id, group in monthly_df.groupby("product_id"):
        gated = group[group["meets_review_count_gate"]].sort_values("review_month").reset_index(drop=True)
        for i in range(1, len(gated)):
            prev, cur = gated.iloc[i - 1], gated.iloc[i]
            if (cur["review_month"].to_period("M") - prev["review_month"].to_period("M")).n != 1:
                continue
            drop = prev["net_sentiment"] - cur["net_sentiment"]
            se = SIGMA_REVIEW * math.sqrt(1 / cur["review_count"] + 1 / prev["review_count"])
            flagged = se > 0 and drop >= DIP_SE_MULTIPLIER * se
            rows.append({
                "product_id": product_id,
                "review_month": cur["review_month"],
                "notable_month_dip": bool(flagged),
                "notable_month_dip_drop": drop if flagged else np.nan,
                "notable_month_dip_se": se if flagged else np.nan,
            })
    return pd.DataFrame(rows, columns=[
        "product_id", "review_month", "notable_month_dip",
        "notable_month_dip_drop", "notable_month_dip_se",
    ])


def build_output_rows(monthly_df: pd.DataFrame, cusum_df: pd.DataFrame, dip_df: pd.DataFrame) -> pd.DataFrame:
    """
    LEFT JOIN monthly_df (every product-month, gate-passing or not)
    against
    cusum_df and dip_df (gate-passing-only subsets). A gate-failing month
    ends up with net_sentiment populated and every CUSUM/dip column NULL
    or FALSE.
    """
    # meets_review_count_gate/review_count are carried through unchanged,
    # same as net_sentiment -- no new computation, just denormalizing two
    # already-known columns so every NULL baseline/cusum row below is
    # self-explanatory.
    out = monthly_df.rename(columns={"review_month": "as_of_month"})[
        ["product_id", "as_of_month", "net_sentiment", "meets_review_count_gate", "review_count"]
    ]

    cusum_renamed = cusum_df.rename(columns={"review_month": "as_of_month"})
    out = out.merge(cusum_renamed, on=["product_id", "as_of_month"], how="left")

    dip_renamed = dip_df.rename(columns={"review_month": "as_of_month"})[
        ["product_id", "as_of_month", "notable_month_dip", "notable_month_dip_drop", "notable_month_dip_se"]
    ]
    out = out.merge(dip_renamed, on=["product_id", "as_of_month"], how="left")

    # Columns that must never be NULL in the schema (NOT NULL DEFAULT
    # FALSE) -- a gate-failing month, or a month with no valid adjacent
    # pair, comes back from the LEFT JOIN as NaN here and needs an
    # explicit fillna, not a silent NULL insert (which would fail the
    # schema's NOT NULL constraint).
    for col in ("voc_decline_broad_flag", "voc_decline_flag", "notable_month_dip"):
        # A plain equality check against True, not fillna(False).astype(bool):
        # this LEFT JOIN leaves an object-dtype column of True/False/NaN for
        # a row with no match on the right side, and pandas' fillna+astype
        # on an object column emits a FutureWarning about silent downcasting
        # in a future version. `== True` sidesteps that path entirely --
        # NaN == True is always False under ordinary comparison semantics,
        # so this produces the exact same True/False/False result with no
        # warning and no dtype-inference step to depend on.
        out[col] = out[col] == True  # noqa: E712 (deliberate -- see above)

    return out


# =============================================================================
# LOAD
# =============================================================================


def build_insert_query() -> str:
    """Plain parameterized INSERT -- no ON DUPLICATE KEY UPDATE. This table is DELETEd immediately before
    this INSERT in the same transaction (see repopulate() and the module
    docstring's "why DELETE, not TRUNCATE" note) -- there is never an
    existing row for this INSERT to collide with."""
    columns_sql = ",\n            ".join(INSERT_COLUMNS)
    placeholders_sql = ", ".join(["%s"] * len(INSERT_COLUMNS))
    return f"""
        INSERT INTO {TABLE_NAME}
            (
            {columns_sql}
            )
        VALUES
            ({placeholders_sql})
    """


def _none_if_nan(value):
    """NaN/NaT -> None; a pandas Timestamp -> a plain date; everything
    else passed through unchanged. Applied per-value at row-build time
    rather than column-wide, since this table mixes float, bool, date,
    and int-ish columns that each need slightly different NaN handling.

    pd.isna() is checked FIRST, ahead of every type-specific branch --
    pd.NaT is NOT a pd.Timestamp instance (it's its own NaTType, a
    well-known pandas gotcha), so a NaT value would fall through every
    branch below unconverted and reach mysql-connector-python raw, which
    mis-serializes it as the literal date '0001-01-01' instead of NULL.
    A blanket pd.isna() catches NaT, plain NaN, and np.nan uniformly."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def rows_for_insert(output_df: pd.DataFrame) -> list[tuple]:
    """Build the list of value-tuples for executemany(), in
    INSERT_COLUMNS order, with NaN/NaT->None and numpy scalar types
    converted to plain Python types (mysql-connector-python is stricter
    about numpy dtypes as bind parameters than pandas is internally)."""
    records = output_df[list(INSERT_COLUMNS)].to_dict("records")
    return [tuple(_none_if_nan(record[col]) for col in INSERT_COLUMNS) for record in records]


def repopulate(connection, output_df: pd.DataFrame) -> int:
    """DELETE + single INSERT, one transaction, one commit. See module
    docstring for why DELETE (not TRUNCATE) and why unbatched are both deliberate here."""
    cursor = connection.cursor()
    rows = rows_for_insert(output_df)
    insert_query = build_insert_query()
    try:
        cursor.execute(f"DELETE FROM {TABLE_NAME}")
        if rows:
            cursor.executemany(insert_query, rows)
        connection.commit()
    except Error:
        connection.rollback()
        print(
            "\nERROR during repopulate -- rolled back. The DELETE above was "
            "undone along with it (DELETE, unlike TRUNCATE, is transactional "
            "under InnoDB), so voc_decline_risk_v1 is unchanged from before "
            "this run. Safe to fix the underlying issue and re-run."
        )
        raise
    finally:
        cursor.close()
    return len(rows)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print a summary; do not write to the database.",
    )
    args = parser.parse_args()

    db_config = get_db_config()
    validate_db_config(db_config)

    connection = None
    try:
        connection = get_connection(db_config)
        if connection.is_connected():
            print("Connected to MySQL.")

        print(f"Reading {SOURCE_VIEW} ...")
        monthly_df = fetch_monthly_sentiment(connection)
        print(f"  {len(monthly_df):,} product-months across {monthly_df['product_id'].nunique():,} products.")

        cusum_df = compute_cusum(monthly_df)
        dip_df = compute_notable_month_dip(monthly_df)
        output_df = build_output_rows(monthly_df, cusum_df, dip_df)

        broad_flagged = int(output_df["voc_decline_broad_flag"].sum())
        tight_flagged = int(output_df["voc_decline_flag"].sum())
        dip_flagged = int(output_df["notable_month_dip"].sum())
        print(f"  voc_decline_broad_flag TRUE: {broad_flagged:,} product-months")
        print(f"  voc_decline_flag TRUE:       {tight_flagged:,} product-months")
        print(f"  notable_month_dip TRUE:      {dip_flagged:,} product-months")

        latest = output_df.sort_values("as_of_month").groupby("product_id").tail(1)
        currently_flagged = latest[latest["voc_decline_flag"]]
        if len(currently_flagged):
            print("\n  Currently flagged (most recent month per product):")
            print(currently_flagged[["product_id", "as_of_month", "cusum_score", "decline_since"]].to_string(index=False))
        else:
            print("\n  No product is currently flagged (voc_decline_flag) as of its most recent month.")

        if args.dry_run:
            print("\n--dry-run: no database write performed.")
            return

        print(f"\nWriting {len(output_df):,} rows to {TABLE_NAME} (DELETE + single INSERT, one transaction) ...")
        written = repopulate(connection, output_df)
        print(f"Done -- {written:,} rows written.")
        print("\nRun the QA queries next.")

    except Error as e:
        print(f"\nMySQL error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("MySQL connection closed.")


if __name__ == "__main__":
    main()
