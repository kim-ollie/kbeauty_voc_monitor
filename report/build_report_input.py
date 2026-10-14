"""
As-of-date-parameterised aggregation layer for the monthly report.

    build_report_input(as_of_month: date) -> dict

Produces ONE validated, self-describing JSON payload per as-of month: every
fact the monthly LLM report is allowed to state, and nothing else. The
deterministic layer computes every number; the model only narrates.

Design in one paragraph
-----------------------
The SQL is deliberately dull: four small parameter-free SELECTs that pull the
mart tables whole (products 103 rows, monthly sentiment ~4.2K, decline-risk
~4.2K, complaint mentions 26,624). ALL windowing / aggregation happens in
pandas, so every block is a pure function of four DataFrames and is unit-
testable on synthetic data without a database. Pandas over SQL because of
nested output, a callable for the replay loop, and a validation layer that is
awkward as SQL.

Block map (block letter -> payload key)
    A header            -> "header"
    B watchlist         -> "watchlist"      (C, month-over-month, nests inside B)
    D category shift    -> "category_shift"
    E brand vs category -> "brand_vs_category"   (the unit is the product)
    F complaint move    -> "complaint_movement"
    G evidence phrases  -> "evidence_phrases"
    H coverage/caveats  -> "coverage"

Every number is emitted twice: a raw rounded value and a `<key>_display`
string the model must copy verbatim. Formatting lives in ONE place
(_KINDS below) so the renderer's number checks can reuse it.

Credentials come from .env via python-dotenv (get_db_config /
validate_db_config / get_connection).

Usage
    python build_report_input.py --as-of 2026-06            # one month
    python build_report_input.py --all                      # 2025-07 .. 2026-06
    python build_report_input.py --as-of 2026-06 --selftest # live reconciliation
    python build_report_input.py --all --force              # overwrite existing

Output
    report_inputs/report_input_YYYY-MM.json  (UTF-8, ensure_ascii=False)

KNOBS -- the block F window and the presentation caps below are methodological or
presentation choices; changing one is a one-line edit. Each is echoed into the payload's
`envelope.knobs` so a saved JSON records the values it was built with.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "phase4-4.1-v4"

# --- Window ------------------------------------------------------------------
WINDOW_MONTHS = 12
LAST_ALLOWED_AS_OF = date(2026, 6, 1)  # hard-fail, never clamp
ANCHOR_WINDOW = (date(2025, 7, 1), date(2026, 6, 1))  # the dashboard's window
REPLAY_FIRST_AS_OF = date(2025, 7, 1)  # 12 as-of months, 2025-07 .. 2026-06

# --- Complaint mart coverage --------------------------------------------------
COMPLAINT_DATA_START = date(2025, 7, 1)
COMPLAINT_DATA_END = date(2026, 6, 1)

# --- Fixed rules -------------------------------------------------------------
CATEGORY_MIN_PRODUCTS = 3  # category comparison gate, n >= 3 (catalogue-level)
CAT_SHIFT_MONTHS = 3  # block D: first 3 vs last 3 months of window (as on the dashboard)

# --- Block F -------------------------------------------------------------------
# Trailing N months vs the N before. The dashboard's Complaint Mix Shift uses
# N=6 (H1 vs H2 of the window). Same metric, different N -- so this is a
# parameter, and the live self-test reproduces the dashboard's number at N=6
# (see run_selftest).
MOVEMENT_HALF_MONTHS = 3

# --- Presentation caps ---------------------------------------------------------
MAX_WATCHLIST = 10          # cap on watchlist rows; total + truncated flag always emitted
F_REPORT_TOP_N = 3          # complaint theme groups reported as "rising" (at most this many ...
F_MATERIALITY_PP = 1.0      # ... AND only those whose share shift is >= this many percentage points.
                            # A salience floor, NOT a significance test: counting noise
                            # is ~0.3 pp, review clustering makes it larger, and season moves themes by more.
PHRASE_DISPLAY_MAX_CHARS = 140  # evidence phrase shown on one line, cut here (the raw phrase is kept as well)
BRAND_WINDOW_MONTHS = 6       # block E compares over the LAST N months of the window; the other blocks keep 12
BRAND_MIN_PRODUCT_MONTHS = 3  # a product needs >= this many gate-passing months in that window
BRAND_TOP_N = 3               # products reported below / above their category mean (each side)

# --- Static caveat facts (source-cited; rendered deterministically) ----------
SHARE_POSITIVE_CORPUS = "85.6%"  # predicted class mix, full corpus
CONS_TIP_UNDERCOUNT = "3.4%"     # measured, accepted limitation
TEMPORAL_FIDELITY_TEXT = (
    "Replays historical data through the current monitor. Risk thresholds, "
    "complaint themes and their groupings were all calibrated on data through "
    "June 2026 — this is what the monitor sees looking back, not what an "
    "analyst would have seen at the time."
)

OUTPUT_DIR = Path("report_inputs")

# --- SQL: four single statements, pulled whole (tables are small) ------------
# Deviation from the export scripts' "SQL lives in a .sql file" habit: those
# run ONE query; this pulls four. Kept inline, each a single statement.
SQL_PRODUCTS = """
SELECT product_id, product_name_ko, brand_name_ko, category_id, category_name
FROM products_v3
"""
SQL_MONTHLY = """
SELECT product_id, review_month, review_count, meets_review_count_gate, net_sentiment
FROM product_monthly_sentiment_v1
"""
SQL_RISK = """
SELECT product_id, as_of_month, net_sentiment, meets_review_count_gate, review_count,
       baseline_months_available, cusum_score, decline_since,
       voc_decline_broad_flag, voc_decline_flag,
       notable_month_dip, notable_month_dip_drop, computed_at
FROM voc_decline_risk_v1
"""
SQL_MENTIONS = """
SELECT m.topic_id, m.product_id, m.review_month, m.mention_count,
       m.avg_segment_sentiment, m.representative_phrase,
       t.theme_group, t.is_excluded
FROM product_complaint_theme_mentions_v1 m
JOIN complaint_topics_v1 t ON t.topic_id = m.topic_id
"""

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class ReportInputError(Exception):
    """A validation gate failed, or the request is out of bounds. Hard-fail:
    never emit a payload that failed a gate."""


# =============================================================================
# MONTH / WINDOW ARITHMETIC
# =============================================================================


def add_months(d: date, n: int) -> date:
    idx = d.year * 12 + (d.month - 1) + n
    return date(idx // 12, idx % 12 + 1, 1)


def month_list(start: date, n: int) -> list[date]:
    return [add_months(start, i) for i in range(n)]


def month_display(d: date) -> str:
    return f"{MONTH_NAMES[d.month - 1]} {d.year}"


def span_display(a: date, b: date) -> str:
    """'April – June 2026' within a year, 'November 2025 – January 2026' across."""
    if a == b:
        return month_display(a)
    if a.year == b.year:
        return f"{MONTH_NAMES[a.month - 1]} – {MONTH_NAMES[b.month - 1]} {b.year}"
    return f"{month_display(a)} – {month_display(b)}"


class Window:
    """The 12-month window ending at as_of (the dashboard's 0..11 boundary math)."""

    def __init__(self, as_of: date):
        self.as_of = as_of
        self.start = add_months(as_of, -(WINDOW_MONTHS - 1))
        self.months = month_list(self.start, WINDOW_MONTHS)

    @property
    def recent_cat(self) -> list[date]:
        return self.months[-CAT_SHIFT_MONTHS:]

    @property
    def prior_cat(self) -> list[date]:
        return self.months[:CAT_SHIFT_MONTHS]

    @property
    def brand_months(self) -> list[date]:
        """Block E's window: the last BRAND_WINDOW_MONTHS months of the 12."""
        return self.months[-BRAND_WINDOW_MONTHS:]


def build_window(as_of_month: date) -> Window:
    if isinstance(as_of_month, datetime):
        as_of_month = as_of_month.date()
    if not isinstance(as_of_month, date):
        raise ReportInputError(f"as_of_month must be a date, got {type(as_of_month).__name__}")
    if as_of_month.day != 1:
        raise ReportInputError(f"as_of_month must be first-of-month, got {as_of_month}")
    if as_of_month > LAST_ALLOWED_AS_OF:
        raise ReportInputError(
            f"as_of_month {as_of_month} is past {LAST_ALLOWED_AS_OF}. Any later month "
            "re-introduces the partial-month problem the June-2026 anchor exists to "
            "avoid. Hard-fail, not clamp."
        )
    w = Window(as_of_month)
    if len(w.months) != WINDOW_MONTHS or w.months[-1] != as_of_month:
        raise ReportInputError("window construction is wrong (length / end month)")
    # Self-test on construction: the anchor month must reproduce the
    # dashboard's window exactly.
    if as_of_month == ANCHOR_WINDOW[1] and (w.months[0], w.months[-1]) != ANCHOR_WINDOW:
        raise ReportInputError(
            f"anchor window mismatch: got {w.months[0]}..{w.months[-1]}, "
            f"expected {ANCHOR_WINDOW[0]}..{ANCHOR_WINDOW[1]} -- boundary math is wrong"
        )
    return w


# =============================================================================
# NUMBER FORMATTING -- the ONE place display strings are defined
# =============================================================================


def _norm(r: float) -> float:
    return 0.0 if r == 0 else r  # never emit -0.0 / "-0.000"


_KINDS = {
    # kind: (decimals, formatter over the ALREADY-ROUNDED raw value)
    "sent": (3, lambda r: f"{r:.3f}"),          # a sentiment level        0.812
    "delta": (3, lambda r: f"{r:+.3f}"),        # a sentiment difference   -0.061
    "drop": (3, lambda r: f"{abs(r):.3f}"),     # the SIZE of a drop, unsigned: 0.141 (never "+0.141")
    "cusum": (3, lambda r: f"{r:.3f}"),         # a CUSUM score            0.412
    "share": (4, lambda r: f"{r * 100:.1f}%"),  # a share                  29.1%
    "pp": (4, lambda r: f"{r * 100:+.1f} pp"),  # a share difference       +5.5 pp
    "int": (0, lambda r: f"{int(r):,}"),        # a count                  271,252
}


def num(target: dict, key: str, value: Any, kind: str) -> None:
    """Write `key` (rounded raw) and `key_display` (pre-formatted) into target.
    None / NaN -> both None (a declared-optional field; see OPTIONAL_KEYS)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        target[key] = None
        target[f"{key}_display"] = None
        return
    nd, fmt = _KINDS[kind]
    if kind == "int":
        raw = int(round(float(value)))
        target[key] = raw
        target[f"{key}_display"] = fmt(raw)
        return
    raw = _norm(round(float(value), nd))
    target[key] = raw
    target[f"{key}_display"] = fmt(raw)


def phrase_display(s: str | None, max_chars: int | None = None) -> str | None:
    """A representative phrase as it may be shown in a report: every run of
    whitespace (newlines, tabs, ideographic spaces) collapsed to one space, and cut
    at `max_chars` (default PHRASE_DISPLAY_MAX_CHARS) with an ellipsis. Cuts at the
    last space when one falls in the final 40% of the allowed length, otherwise
    hard-cuts (Korean words are space-separated, but a very long unbroken token is
    possible). The result never exceeds max_chars. Digits in the result are a
    subset of the digits in the input, so the renderer's number checks can use
    either. The raw phrase stays in the payload; this is presentation only."""
    if s is None:
        return None
    limit = PHRASE_DISPLAY_MAX_CHARS if max_chars is None else max_chars
    flat = " ".join(s.split())
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1]  # leave one character for the ellipsis
    space = cut.rfind(" ")
    if space >= int((limit - 1) * 0.6):
        cut = cut[:space]
    return cut.rstrip() + "…"


def date_fields(target: dict, key: str, d: Any) -> None:
    if not isinstance(d, date):  # None / NaN / NaT -> declared-absent
        d = None
    target[key] = None if d is None else d.isoformat()
    target[f"{key}_display"] = None if d is None else month_display(d)


# =============================================================================
# TABLE PREPARATION
# =============================================================================


def _to_date(v: Any) -> date | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return pd.Timestamp(v).date()


def _to_float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        if pd.isna(v):
            return float("nan")
    except (TypeError, ValueError):
        pass
    return float(v)  # handles decimal.Decimal from mysql.connector


def _as_float(s: pd.Series) -> pd.Series:
    return pd.Series([_to_float(v) for v in s], index=s.index, dtype="float64")


def _as_bool(s: pd.Series) -> pd.Series:
    """0/1/NULL/True/False -> plain bool; NULL -> False."""
    return pd.Series(
        [False if (v is None or (not isinstance(v, str) and pd.isna(v))) else bool(int(v)) for v in s],
        index=s.index,
        dtype=bool,
    )


def _clean_text(v: Any) -> str | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).replace("\x00", "").strip()  # NUL bytes have been seen in this data
    return s if s else None


def prepare_tables(raw: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Normalise dtypes once. Months become datetime.date objects (object dtype)
    so comparisons are Series.isin(list_of_dates) / == date -- immune to
    Timestamp-vs-date mismatches across pandas versions."""
    products = raw["products"].copy()
    products["product_id"] = products["product_id"].astype("int64")
    products["category_id"] = products["category_id"].astype("int64")
    for c in ("product_name_ko", "brand_name_ko", "category_name"):
        products[c] = products[c].map(_clean_text)

    monthly = raw["monthly"].copy()
    monthly["product_id"] = monthly["product_id"].astype("int64")
    monthly["review_month"] = monthly["review_month"].map(_to_date)
    monthly["review_count"] = monthly["review_count"].map(lambda v: int(_to_float(v))).astype("int64")
    monthly["gate"] = _as_bool(monthly["meets_review_count_gate"])
    monthly["net_sentiment"] = _as_float(monthly["net_sentiment"])
    monthly = monthly[["product_id", "review_month", "review_count", "gate", "net_sentiment"]]

    risk = raw["risk"].copy()
    snapshot = _to_date(risk["computed_at"].dropna().max()) if len(risk) else None
    risk["product_id"] = risk["product_id"].astype("int64")
    risk["as_of_month"] = risk["as_of_month"].map(_to_date)
    risk["decline_since"] = risk["decline_since"].map(_to_date)
    risk["gate"] = _as_bool(risk["meets_review_count_gate"])
    risk["broad"] = _as_bool(risk["voc_decline_broad_flag"])
    risk["strict"] = _as_bool(risk["voc_decline_flag"])
    risk["dip"] = _as_bool(risk["notable_month_dip"])
    for c in ("net_sentiment", "cusum_score", "notable_month_dip_drop", "baseline_months_available", "review_count"):
        risk[c] = _as_float(risk[c])
    risk = risk[[
        "product_id", "as_of_month", "net_sentiment", "gate", "review_count",
        "baseline_months_available", "cusum_score", "decline_since",
        "broad", "strict", "dip", "notable_month_dip_drop",
    ]]

    mentions = raw["mentions"].copy()
    mentions["topic_id"] = mentions["topic_id"].astype("int64")
    mentions["product_id"] = mentions["product_id"].astype("int64")
    mentions["review_month"] = mentions["review_month"].map(_to_date)
    mentions["mention_count"] = mentions["mention_count"].map(lambda v: int(_to_float(v))).astype("int64")
    mentions["avg_segment_sentiment"] = _as_float(mentions["avg_segment_sentiment"])
    mentions["representative_phrase"] = mentions["representative_phrase"].map(_clean_text)
    mentions["theme_group"] = mentions["theme_group"].map(_clean_text)
    mentions["is_excluded"] = _as_bool(mentions["is_excluded"])

    per_cat = products.groupby("category_id")["product_id"].nunique()
    eligible = sorted(int(c) for c in per_cat[per_cat >= CATEGORY_MIN_PRODUCTS].index)

    monthly_idx = {
        (int(p), m): (int(rc), bool(g), float(ns))
        for p, m, rc, g, ns in zip(
            monthly["product_id"], monthly["review_month"], monthly["review_count"],
            monthly["gate"], monthly["net_sentiment"],
        )
    }
    return {
        "products": products, "monthly": monthly, "risk": risk, "mentions": mentions,
        "monthly_idx": monthly_idx, "eligible_category_ids": eligible, "snapshot_date": snapshot,
    }


# =============================================================================
# DATABASE (mirrors export_phase3_complaint_extract.py exactly)
# =============================================================================


def get_db_config() -> dict:
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
    required = {"user": "DB_USER", "password": "DB_PASSWORD", "database": "DB_NAME"}
    missing = [env for key, env in required.items() if not db_config.get(key)]
    if missing:
        raise ValueError("Missing database settings: " + ", ".join(missing) + ". Add them to the .env file.")


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Read the four mart tables through a live MySQL connection."""
    import mysql.connector  # lazy: unit tests never need a driver
    from dotenv import load_dotenv

    load_dotenv()
    cfg = get_db_config()
    validate_db_config(cfg)
    conn = mysql.connector.connect(**cfg)
    try:
        out = {}
        for name, sql in (
            ("products", SQL_PRODUCTS), ("monthly", SQL_MONTHLY),
            ("risk", SQL_RISK), ("mentions", SQL_MENTIONS),
        ):
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            out[name] = pd.DataFrame(cur.fetchall(), columns=cols)
            cur.close()
            print(f"  loaded {name}: {len(out[name]):,} rows")
        return out
    finally:
        conn.close()


# =============================================================================
# BLOCK A -- header
# =============================================================================


def _month_totals(t: dict, month: date) -> tuple[int, int, int]:
    """(reviews, products with >= 1 review, products that clear the review gate) in ONE month, from the
    sparse monthly view (a product with no review in the month has no row)."""
    m = t["monthly"]
    sub = m[m["review_month"] == month]
    return int(sub["review_count"].sum()), len(sub), int(sub["gate"].sum())


def build_header(t: dict, w: Window) -> dict:
    products, m = t["products"], t["monthly"]
    win = m[m["review_month"].isin(w.months)]
    reviewed_ids = set(int(x) for x in win["product_id"].unique())
    h: dict[str, Any] = {}
    # Both denominators are emitted (catalogue vs reviewed-in-window). Which one
    # the report states is the prompt's choice, not defaulted here.
    num(h, "products_in_catalogue", len(products), "int")
    num(h, "products_reviewed_in_window", len(reviewed_ids), "int")
    num(h, "categories_in_catalogue", products["category_id"].nunique(), "int")
    cats_reviewed = products[products["product_id"].isin(reviewed_ids)]["category_id"].nunique()
    num(h, "categories_reviewed_in_window", cats_reviewed, "int")
    num(h, "reviews_in_window", int(win["review_count"].sum()), "int")
    # BAN row: the month's OWN review count and the previous month's, from the
    # same monthly table (a replay's first month reads June 2025, which the mart holds). None = the
    # previous month has no rows at all, i.e. there is nothing to compare with.
    prior = add_months(w.as_of, -1)
    reviews, _, _ = _month_totals(t, w.as_of)
    p_reviews, p_with, _ = _month_totals(t, prior)
    num(h, "asof_reviews", reviews, "int")
    h["prior_month"] = prior.isoformat()
    h["prior_month_display"] = month_display(prior)
    num(h, "prior_month_reviews", p_reviews if p_with else None, "int")
    h["window_start"] = w.start.isoformat()
    h["window_end"] = w.as_of.isoformat()
    h["window_display"] = span_display(w.start, w.as_of)
    return h


# =============================================================================
# BLOCKS B + C -- watchlist, with month-over-month delta nested per product
# =============================================================================


def _month_over_month(t: dict, product_id: int, as_of: date) -> dict:
    """Block C. Requires BOTH months to exist and clear the 20-review gate:
    a delta between a thin month and anything is noise dressed as signal, and
    the project already treats <20 as unreliable. Reason vocabulary is fixed."""
    idx = t["monthly_idx"]
    prior_month = add_months(as_of, -1)
    cur = idx.get((product_id, as_of))
    pri = idx.get((product_id, prior_month))
    out: dict[str, Any] = {"prior_month": prior_month.isoformat(), "prior_month_display": month_display(prior_month)}
    reason = None
    if cur is None:
        reason = "current_month_has_no_reviews"
    elif pri is None:
        reason = "prior_month_has_no_reviews"
    elif not cur[1]:
        reason = "current_month_below_review_gate"
    elif not pri[1]:
        reason = "prior_month_below_review_gate"
    out["available"] = reason is None
    out["unavailable_reason"] = reason
    if reason is None:
        num(out, "prior_net_sentiment", pri[2], "sent")
        num(out, "delta", cur[2] - pri[2], "delta")
    else:
        num(out, "prior_net_sentiment", None, "sent")
        num(out, "delta", None, "delta")
    return out


def _flag_history(risk: pd.DataFrame, product_ids: list[int], as_of: date) -> dict:
    """{(product_id, month): highest tier} for the given products, months <= as_of
    ONLY (never look ahead: a replay for as-of M must not know about M+1). The
    streak walk below only ever steps backwards, so the `<= as_of` filter is belt
    and braces -- it keeps any future edit that walks forward honest. Tier is
    the watchlist rule: 1 strict, 2 broad, 3 dip; an unflagged month, or a month
    with no risk row (the product had no reviews), is simply absent."""
    wanted = set(int(p) for p in product_ids)
    keep = [(int(p) in wanted) and (m <= as_of) for p, m in zip(risk["product_id"], risk["as_of_month"])]
    sub = risk[keep]
    hist: dict = {}
    for p, m, s, b, d in zip(sub["product_id"], sub["as_of_month"], sub["strict"], sub["broad"], sub["dip"]):
        tier = 1 if s else (2 if b else (3 if d else None))
        if tier is not None:
            hist[(int(p), m)] = tier
    return hist


def _streak_tier_1_2(hist: dict, product_id: int, as_of: date) -> int:
    """Consecutive months ending at as_of (inclusive) in which the product was at
    tier 1 or 2. A month with no flag, or no risk row, ends the streak."""
    n, m = 0, as_of
    while True:
        tier = hist.get((product_id, m))
        if tier is None or tier > 2:
            return n
        n += 1
        m = add_months(m, -1)


def build_watchlist(t: dict, w: Window) -> dict:
    risk, products = t["risk"], t["products"]
    r = risk[risk["as_of_month"] == w.as_of]
    flagged = r[r["broad"] | r["strict"] | r["dip"]].merge(
        products[["product_id", "product_name_ko", "brand_name_ko", "category_name"]],
        on="product_id", how="left",
    )
    if flagged["product_name_ko"].isna().any():
        raise ReportInputError("watchlist product missing from products_v3")
    flagged = flagged.assign(_cusum_sort=flagged["cusum_score"].fillna(-1.0)).sort_values(
        ["strict", "broad", "_cusum_sort", "product_id"], ascending=[False, False, False, True],
    )
    # Deterministic ranking: the model never selects who is at risk.
    tier1 = int(flagged["strict"].sum())
    tier2_only = int((flagged["broad"] & ~flagged["strict"]).sum())
    tier3_only = int((flagged["dip"] & ~flagged["broad"]).sum())
    total = len(flagged)
    shown = flagged.head(MAX_WATCHLIST)

    summary: dict[str, Any] = {}
    num(summary, "tier1_strict_count", tier1, "int")
    num(summary, "tier2_broad_only_count", tier2_only, "int")
    num(summary, "tier3_dip_only_count", tier3_only, "int")
    num(summary, "total_flagged", total, "int")
    # The headline reports ONLY tiers 1-2 (CUSUM tripped);
    # tier-3 dips never lead. Two different "quiet"s, named so they cannot be confused:
    #   headline_is_quiet     -> nothing at tier 1 or 2 (the headline reads "nothing tripped")
    #   no_flags_at_any_tier  -> nothing at ANY tier, dips included
    num(summary, "tier12_flagged_count", tier1 + tier2_only, "int")
    summary["headline_is_quiet"] = (tier1 + tier2_only) == 0
    summary["no_flags_at_any_tier"] = total == 0
    summary["truncated"] = total > len(shown)

    prior_month = add_months(w.as_of, -1)
    hist = _flag_history(risk, [int(p) for p in shown["product_id"]], w.as_of)

    items = []
    for row in shown.itertuples(index=False):
        pid = int(row.product_id)
        highest = 1 if row.strict else (2 if row.broad else 3)
        it: dict[str, Any] = {
            "product_id": pid,
            "product_name_ko": row.product_name_ko,
            "brand_name_ko": row.brand_name_ko,
            "category_name": row.category_name,
            "highest_tier": highest,
            "voc_decline_flag": bool(row.strict),
            "voc_decline_broad_flag": bool(row.broad),
            "notable_month_dip": bool(row.dip),
            "meets_review_count_gate": bool(row.gate),
        }
        # Persistence: tier-1/2 products lead consecutive reports, so a report
        # must be able to say "new this month" vs "still flagged, month N". The
        # streak counts tier-1/2 months only and is null for a tier-3-only row.
        prior_tier = hist.get((pid, prior_month))
        it["prior_month_highest_tier"] = prior_tier            # null: not flagged last month
        it["newly_flagged_this_month"] = prior_tier is None    # not flagged at ANY tier last month
        num(it, "consecutive_months_flagged_tier_1_2",
            _streak_tier_1_2(hist, pid, w.as_of) if highest <= 2 else None, "int")
        num(it, "net_sentiment", row.net_sentiment, "sent")
        num(it, "review_count", row.review_count, "int")
        num(it, "cusum_score", row.cusum_score, "cusum")
        date_fields(it, "decline_since", row.decline_since)
        num(it, "notable_month_dip_drop", row.notable_month_dip_drop if row.dip else None, "drop")
        it["month_over_month"] = _month_over_month(t, pid, w.as_of)
        items.append(it)
    return {"summary": summary, "products": items}


# =============================================================================
# BLOCK D -- category shift (mirrors the dashboard's Category Sentiment Shift sheet)
# =============================================================================


def build_category_shift(t: dict, w: Window) -> list[dict]:
    """First-3 vs last-3 months of the window, gate-passing product-months only,
    categories with >= 3 products in the CATALOGUE (the dashboard's `Category Eligible`,
    9 of 26). Row-level (product-month weighted) mean -- same weighting as the
    dashboard sheet, so the report and dashboard cannot disagree by construction."""
    products, m = t["products"], t["monthly"]
    m = m[m["gate"]].merge(products[["product_id", "category_id"]], on="product_id", how="inner")
    cat_meta = products.drop_duplicates("category_id").set_index("category_id")["category_name"]
    per_cat = products.groupby("category_id")["product_id"].nunique()
    rows = []
    for cid in t["eligible_category_ids"]:
        sub = m[m["category_id"] == cid]
        pri = sub[sub["review_month"].isin(w.prior_cat)]
        rec = sub[sub["review_month"].isin(w.recent_cat)]
        available = len(pri) > 0 and len(rec) > 0
        r: dict[str, Any] = {"category_id": int(cid), "category_name": cat_meta[cid]}
        num(r, "products_in_category", int(per_cat[cid]), "int")
        r["shift_available"] = available
        num(r, "prior_mean", pri["net_sentiment"].mean() if len(pri) else None, "sent")
        num(r, "recent_mean", rec["net_sentiment"].mean() if len(rec) else None, "sent")
        num(r, "shift", (rec["net_sentiment"].mean() - pri["net_sentiment"].mean()) if available else None, "delta")
        num(r, "prior_product_months", len(pri), "int")
        num(r, "recent_product_months", len(rec), "int")
        rows.append(r)
    rows.sort(key=lambda r: (not r["shift_available"], r["shift"] if r["shift"] is not None else 0.0, r["category_name"]))
    rank = 0
    for r in rows:
        if r["shift_available"]:
            rank += 1
            r["rank_worst_first"] = rank
        else:
            r["rank_worst_first"] = None
    return rows


# =============================================================================
# BLOCK E -- brand vs category average (structured, no text mining)
# =============================================================================


def _none_if_missing(v: Any) -> Any:
    """A missing name arrives as None or NaN depending on the pandas version; the block must carry None
    (validation then refuses an unnamed row) and never a NaN, which cannot be sorted or written."""
    return None if pd.isna(v) else v


def build_brand_vs_category(t: dict, w: Window) -> dict:
    """One row per PRODUCT in its category, over the last BRAND_WINDOW_MONTHS of
    the window, gate-passing months only, eligible categories only. The category
    mean is the unweighted mean of every gate-passing product-month of the category
    in that window and INCLUDES the product itself,
    so a dominant product's gap is understated -- stated, not hidden. A product is
    REPORTABLE only with >= BRAND_MIN_PRODUCT_MONTHS such months and >= 2 products in
    the category (a lone product trivially equals its category). The key is
    'brand_vs_category'; the rows name brand and product."""
    products, m = t["products"], t["monthly"]
    months = w.brand_months
    m = m[m["gate"] & m["review_month"].isin(months)].merge(
        products[["product_id", "brand_name_ko", "product_name_ko", "category_id", "category_name"]],
        on="product_id", how="inner",
    )
    m = m[m["category_id"].isin(t["eligible_category_ids"])]
    evaluated, low_support, single_product = [], 0, 0
    for cid, sub in m.groupby("category_id"):
        cat_mean = float(sub["net_sentiment"].mean())
        n_products = int(sub["product_id"].nunique())
        for pid, psub in sub.groupby("product_id"):
            n_months = len(psub)
            if n_months < BRAND_MIN_PRODUCT_MONTHS:
                low_support += 1
                continue
            if n_products < 2:
                single_product += 1
                continue
            product_mean = float(psub["net_sentiment"].mean())
            evaluated.append({
                "product_id": int(pid), "brand_name_ko": _none_if_missing(psub["brand_name_ko"].iloc[0]),
                "product_name_ko": _none_if_missing(psub["product_name_ko"].iloc[0]),
                "category_name": sub["category_name"].iloc[0],
                "product_mean": product_mean, "category_mean": cat_mean, "gap": product_mean - cat_mean,
                "n_months": n_months, "products_in_category": n_products,
            })

    def shape(e: dict) -> dict:
        o: dict[str, Any] = {"product_id": e["product_id"], "brand_name_ko": e["brand_name_ko"],
                             "product_name_ko": e["product_name_ko"], "category_name": e["category_name"]}
        num(o, "product_mean", e["product_mean"], "sent")
        num(o, "category_mean", e["category_mean"], "sent")
        num(o, "gap", e["gap"], "delta")
        num(o, "n_months", e["n_months"], "int")
        return o

    def tie(e: dict) -> tuple:  # names may be None; None must never be compared with a string
        return (-e["n_months"], e["brand_name_ko"] or "", e["product_name_ko"] or "", e["product_id"])

    # Gaps are compared at 9 decimals: float noise at the last digit (two products with the same six
    # values summed in a different order) must not decide an order; the tie-break below does.
    below = sorted([e for e in evaluated if round(e["gap"], 3) < 0], key=lambda e: (round(e["gap"], 9),) + tie(e))[:BRAND_TOP_N]
    above = sorted([e for e in evaluated if round(e["gap"], 3) > 0], key=lambda e: (-round(e["gap"], 9),) + tie(e))[:BRAND_TOP_N]
    out: dict[str, Any] = {
        "window_start": months[0].isoformat(), "window_end": months[-1].isoformat(),
        "window_display": span_display(months[0], months[-1]),
        "largest_below_category": [shape(e) for e in below],
        "largest_above_category": [shape(e) for e in above],
    }
    num(out, "window_months", len(months), "int")
    num(out, "products_evaluated", len(evaluated), "int")
    num(out, "products_excluded_low_support", low_support, "int")
    num(out, "products_excluded_single_product_category", single_product, "int")
    return out


# =============================================================================
# BLOCKS F + G -- complaint movement and evidence phrases
# =============================================================================


def live_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    """The standing filters, non-negotiable: topic_id <> -1 AND
    is_excluded = FALSE. (theme_group non-null is implied for live topics.)"""
    return mentions[(mentions["topic_id"] != -1) & (~mentions["is_excluded"]) & mentions["theme_group"].notna()]


def complaint_shift_table(
    mentions: pd.DataFrame, as_of: date, half_months: int, product_id: int | None = None,
) -> pd.DataFrame:
    """Share-of-live-mentions shift per theme_group: recent N months minus the N
    months before. Same metric as the dashboard's Complaint Mix Shift, which is
    this function with half_months=6. Returns columns: theme_group,
    recent_mentions, prior_mentions, share_recent, share_prior, shift.
    Empty frame if either half has zero live mentions."""
    recent = month_list(add_months(as_of, -(half_months - 1)), half_months)
    prior = month_list(add_months(as_of, -(2 * half_months - 1)), half_months)
    lm = live_mentions(mentions)
    if product_id is not None:
        lm = lm[lm["product_id"] == product_id]
    rec = lm[lm["review_month"].isin(recent)].groupby("theme_group")["mention_count"].sum()
    pri = lm[lm["review_month"].isin(prior)].groupby("theme_group")["mention_count"].sum()
    if rec.sum() == 0 or pri.sum() == 0:
        return pd.DataFrame(columns=["theme_group", "recent_mentions", "prior_mentions",
                                     "share_recent", "share_prior", "shift"])
    groups = sorted(set(rec.index) | set(pri.index))
    df = pd.DataFrame({"theme_group": groups})
    df["recent_mentions"] = [int(rec.get(g, 0)) for g in groups]
    df["prior_mentions"] = [int(pri.get(g, 0)) for g in groups]
    df["share_recent"] = df["recent_mentions"] / float(rec.sum())
    df["share_prior"] = df["prior_mentions"] / float(pri.sum())
    df["shift"] = df["share_recent"] - df["share_prior"]
    return df


def complaint_months_in_window(t: dict, w: Window) -> list[date]:
    seen = set(t["mentions"]["review_month"].unique())
    return [m for m in w.months if m in seen]


def build_complaint_movement(t: dict, w: Window) -> tuple[dict, list[dict]]:
    """Returns (block F, block G). Trailing 3 months vs the prior 3,
    computed only where BOTH halves are fully covered by the complaint mart,
    equivalently >= 6 complaint months in the window (as-of 2025-12 onward)."""
    half = MOVEMENT_HALF_MONTHS
    recent = month_list(add_months(w.as_of, -(half - 1)), half)
    prior = month_list(add_months(w.as_of, -(2 * half - 1)), half)
    have = complaint_months_in_window(t, w)
    expected = [m for m in w.months if COMPLAINT_DATA_START <= m <= COMPLAINT_DATA_END]
    if have != expected:
        raise ReportInputError(
            f"complaint-mart month coverage anomaly: window has months {[m.isoformat() for m in have]} "
            f"but the taxonomy was fit on {COMPLAINT_DATA_START}..{COMPLAINT_DATA_END}, "
            f"expected {[m.isoformat() for m in expected]}"
        )
    available = all(m in have for m in recent + prior)
    f: dict[str, Any] = {"available": available, "half_months": half}
    if not available:
        # No comparison window is printed when there is no complaint data behind it --
        # a model could narrate it as a real comparison. Null, not a plausible-looking span.
        f["recent_window_display"] = None
        f["prior_window_display"] = None
        f["unavailable_reason"] = "fewer_than_six_complaint_months_in_window"
        f["theme_groups"] = []
        return f, []
    f["recent_window_display"] = span_display(recent[0], recent[-1])
    f["prior_window_display"] = span_display(prior[0], prior[-1])
    df = complaint_shift_table(t["mentions"], w.as_of, half)
    if df.empty:
        raise ReportInputError("complaint movement: a comparison half has zero live mentions")
    f["unavailable_reason"] = None
    num(f, "recent_live_mentions", int(df["recent_mentions"].sum()), "int")
    num(f, "prior_live_mentions", int(df["prior_mentions"].sum()), "int")
    df = df.sort_values(["shift", "theme_group"], ascending=[False, True]).reset_index(drop=True)
    groups, reported = [], []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        g: dict[str, Any] = {"theme_group": row.theme_group, "rank_rise": i}
        num(g, "recent_mentions", row.recent_mentions, "int")
        num(g, "prior_mentions", row.prior_mentions, "int")
        num(g, "share_recent", row.share_recent, "share")
        num(g, "share_prior", row.share_prior, "share")
        num(g, "shift", row.shift, "pp")
        # Reported as rising = in the top F_REPORT_TOP_N AND at or above the
        # materiality floor. The floor is compared at the DISPLAYED precision (the
        # payload's rounded raw value, in pp, to one decimal), so a theme printed as
        # "+1.0 pp" is never excluded by an unrounded 0.9996.
        shift_pp_shown = round(round(float(row.shift), 4) * 100, 1)
        g["reported_as_rising"] = bool(i <= F_REPORT_TOP_N and shift_pp_shown >= F_MATERIALITY_PP)
        if g["reported_as_rising"]:
            reported.append(row.theme_group)
        groups.append(g)
    f["theme_groups"] = groups
    f["materiality_floor_pp"] = F_MATERIALITY_PP
    f["materiality_floor_display"] = f"{F_MATERIALITY_PP:.1f} pp"
    num(f, "reported_rising_count", len(reported), "int")
    return f, build_evidence_phrases(t, recent, reported)


def build_evidence_phrases(t: dict, recent_months: list[date], reported_groups: list[str]) -> list[dict]:
    """Block G. One phrase per reported theme group. Selection rule (deterministic,
    stated): among live cells in the RECENT window with a non-empty phrase, take
    highest mention_count, then most-negative avg_segment_sentiment, then lowest
    product_id / topic_id / review_month. Preferring multi-mention cells matters:
    a singleton cell's phrase is unimprovable.
    NOTE: representative_phrase lives on product_complaint_theme_mentions_v1, NOT
    on complaint_topics_v1."""
    lm = live_mentions(t["mentions"])
    lm = lm[lm["review_month"].isin(recent_months) & lm["representative_phrase"].notna()]
    prod = t["products"].set_index("product_id")
    out = []
    for grp in reported_groups:
        cand = lm[lm["theme_group"] == grp].sort_values(
            ["mention_count", "avg_segment_sentiment", "product_id", "topic_id", "review_month"],
            ascending=[False, True, True, True, True],
        )
        e: dict[str, Any] = {"theme_group": grp}
        if cand.empty:
            e.update({"phrase_available": False, "representative_phrase": None,
                      "representative_phrase_display": None})
        else:
            c = cand.iloc[0]
            pid = int(c["product_id"])
            e.update({
                "phrase_available": True,
                "representative_phrase": c["representative_phrase"],
                # the phrase as it may be SHOWN (one line, <= PHRASE_DISPLAY_MAX_CHARS).
                # The raw phrase is multi-line, may carry emoji, and runs to ~290 chars.
                "representative_phrase_display": phrase_display(c["representative_phrase"]),
                "product_name_ko": prod.loc[pid, "product_name_ko"],
                "brand_name_ko": prod.loc[pid, "brand_name_ko"],
            })
            date_fields(e, "review_month", c["review_month"])
        out.append(e)
    return out


# =============================================================================
# BLOCK H -- coverage, confidence, caveats (rendered deterministically later)
# =============================================================================


def build_month_counts(t: dict, w: Window) -> dict:
    """BAN row: products EVALUATED (>= 20 reviews) this month and last, on the same basis (products with
    >= 1 review in the month). None = last month has no rows to compare with. A function of its own so the
    layout preview can build the same fields from a local extract."""
    _, _, asof_evaluated = _month_totals(t, w.as_of)
    _, p_with, p_evaluated = _month_totals(t, add_months(w.as_of, -1))
    h: dict[str, Any] = {}
    num(h, "asof_products_evaluated", asof_evaluated, "int")
    num(h, "prior_products_with_reviews", p_with if p_with else None, "int")
    num(h, "prior_products_evaluated", p_evaluated if p_with else None, "int")
    return h


def build_coverage(t: dict, w: Window, f_block: dict) -> dict:
    m = t["monthly"]
    win = m[m["review_month"].isin(w.months)]
    below = int((~win["gate"]).sum())
    have = complaint_months_in_window(t, w)
    ment = t["mentions"][t["mentions"]["review_month"].isin(have)]
    total_mentions = int(ment["mention_count"].sum())
    unclustered = int(ment.loc[ment["topic_id"] == -1, "mention_count"].sum())

    h: dict[str, Any] = {}
    num(h, "complaint_months_in_window", len(have), "int")
    h["complaint_comparison_available"] = bool(f_block["available"])
    num(h, "unclustered_share", (unclustered / total_mentions) if total_mentions else None, "share")
    # Gate-failure rate: denominator is product-months WITH >= 1 review (the sparse
    # view's rows). NOT the dashboard's insufficient-data share -- that is the
    # insufficient-data UNION. The 12-month figures below are kept for reference; the
    # basis the report uses is the as-of-month one.
    num(h, "product_months_with_reviews", len(win), "int")
    num(h, "product_months_below_gate", below, "int")
    num(h, "gate_failure_rate_among_reviewed", (below / len(win)) if len(win) else None, "share")
    # The AS-OF MONTH's own figures ("as-of month, reviewed products"): of the products with >= 1 review in the
    # month the report is about, how many had < 20 and so cannot be flagged. The
    # 12-month rate above understates a thin month (2026-06: 10.6% vs 21.4%).
    # The rendered review_gate caveat below prints the as-of month's count
    # (asof_products_below_gate). June 2026 is a complete month; its drop in volume
    # is real and broad. The wording is short on purpose -- no "of 103", no 12-month comparator. The count covers
    # products with 1-19 reviews; a product with NO review in the month is not in
    # the sparse view and is not counted (only matters for replay months before
    # 2026-05, where 83-101 of 103 products had any review).
    asof = win[win["review_month"] == w.as_of]
    asof_below = int((~asof["gate"]).sum())
    num(h, "asof_products_with_reviews", len(asof), "int")
    num(h, "asof_products_below_gate", asof_below, "int")
    num(h, "asof_gate_failure_rate", (asof_below / len(asof)) if len(asof) else None, "share")
    h.update(build_month_counts(t, w))

    caveats = [{"id": "temporal_fidelity", "text": TEMPORAL_FIDELITY_TEXT}]
    if h["unclustered_share"] is not None:
        pct = h["unclustered_share_display"]
        caveats.append({"id": "unclustered_mentions", "text": (
            f"{pct} of complaint mentions in the window could not be assigned to a theme "
            "and are left out of every theme comparison.")})
    if not f_block["available"]:
        caveats.append({"id": "complaint_movement_unavailable", "text": (
            "Complaint-theme movement is not shown for this month: the theme model covers "
            "July 2025 onward, and a like-for-like comparison needs six months of themed "
            "data (available from December 2025).")})
    # 0 -> no line (nothing was excluded); 1 -> singular grammar; otherwise plural.
    if asof_below == 1:
        caveats.append({"id": "review_gate", "text": (
            "1 product was excluded this month because it had fewer than 20 reviews, "
            "too thin to evaluate.")})
    elif asof_below > 1:
        n_txt = h["asof_products_below_gate_display"]
        caveats.append({"id": "review_gate", "text": (
            f"{n_txt} products were excluded this month because they had fewer than 20 "
            "reviews, too thin to evaluate.")})
    caveats.append({"id": "complaint_undercount", "text": (
        f"About {CONS_TIP_UNDERCOUNT} of genuine complaint text is misread as a usage tip, so "
        "complaint counts run slightly low.")})
    caveats.append({"id": "reviewer_base", "text": (
        "Reviews come from self-selected reviewers on one platform. A decline here is a "
        f"softening within an overwhelmingly satisfied base ({SHARE_POSITIVE_CORPUS} of reviews "
        "score positive), not a wave of negative reviews.")})
    h["caveats"] = caveats
    return h


# =============================================================================
# VALIDATION GATES -- hard-fail on any
# =============================================================================

# Fields that may legitimately be None (each has an availability flag or is
# structurally absent). Anything else being None is a gate failure.
OPTIONAL_KEYS = {
    "decline_since", "decline_since_display",
    "notable_month_dip_drop", "notable_month_dip_drop_display",
    "cusum_score", "cusum_score_display",
    "unavailable_reason",
    "prior_net_sentiment", "prior_net_sentiment_display", "delta", "delta_display",
    "prior_mean", "prior_mean_display", "recent_mean", "recent_mean_display",
    "shift", "shift_display", "rank_worst_first",
    "unclustered_share", "unclustered_share_display",
    "gate_failure_rate_among_reviewed", "gate_failure_rate_among_reviewed_display",
    "representative_phrase", "product_name_ko", "brand_name_ko",
    "review_month", "review_month_display",
    # each with an availability reason or a structural absence:
    "prior_month_highest_tier",                                   # null = not flagged last month
    "consecutive_months_flagged_tier_1_2", "consecutive_months_flagged_tier_1_2_display",  # null on tier-3-only rows
    "recent_window_display", "prior_window_display",              # null when complaint movement is unavailable
    "representative_phrase_display",                              # null when no phrase is available
    "asof_gate_failure_rate", "asof_gate_failure_rate_display",   # null only if the as-of month has no reviews
    # BAN row: null = the previous month has no rows, so there is nothing to compare with
    "prior_month_reviews", "prior_month_reviews_display",
    "prior_products_with_reviews", "prior_products_with_reviews_display",
    "prior_products_evaluated", "prior_products_evaluated_display",
}
NONNEG_KEYS = {"asof_reviews", "prior_month_reviews", "asof_products_evaluated", "prior_products_with_reviews",
               "prior_products_evaluated"}
# Hard ranges by key. net_sentiment is [-1, 1] STRUCTURALLY (monthly mean of
# review_p_positive - review_p_negative); [0,1] is empirically true over the dashboard's window but not
# guaranteed over the 2024-08.. history earlier as-of windows reach into. Values
# below 0 are therefore a WARNING, not a failure.
RANGES = {
    "net_sentiment": (-1.0, 1.0), "prior_net_sentiment": (-1.0, 1.0),
    "prior_mean": (-1.0, 1.0), "recent_mean": (-1.0, 1.0),
    "product_mean": (-1.0, 1.0), "category_mean": (-1.0, 1.0),
    "delta": (-2.0, 2.0), "shift": (-2.0, 2.0), "gap": (-2.0, 2.0),
    "share_recent": (0.0, 1.0), "share_prior": (0.0, 1.0),
    "unclustered_share": (0.0, 1.0), "gate_failure_rate_among_reviewed": (0.0, 1.0),
    "asof_gate_failure_rate": (0.0, 1.0),
    "cusum_score": (0.0, math.inf),
}
SENTIMENT_LEVEL_KEYS = {"net_sentiment", "prior_net_sentiment", "prior_mean", "recent_mean",
                        "product_mean", "category_mean"}


def _walk(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
            yield path, k, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _check_brand_rows(bvc: dict, t: dict, w: Window) -> list[str]:
    """Gate helper: recompute every block-E row from the monthly table by a SECOND path
    (plain loops over monthly_idx, not the DataFrame code that built the rows) and return
    the problems found; [] means every row agrees. Tolerance 0.0006: the payload's raw
    values are rounded to 3 decimals."""
    problems: list[str] = []
    products = t["products"].set_index("product_id")
    idx = t["monthly_idx"]
    months = w.brand_months
    below, above = bvc["largest_below_category"], bvc["largest_above_category"]
    if len(below) > BRAND_TOP_N or len(above) > BRAND_TOP_N:
        problems.append(f"more than BRAND_TOP_N={BRAND_TOP_N} rows on one side")
    if [r["gap"] for r in below] != sorted(r["gap"] for r in below):
        problems.append("below-category rows are not ordered lowest gap first")
    if [r["gap"] for r in above] != sorted((r["gap"] for r in above), reverse=True):
        problems.append("above-category rows are not ordered highest gap first")
    ids = [r["product_id"] for r in below + above]
    if len(ids) != len(set(ids)):
        problems.append("a product appears twice")
    for side, rows in (("below", below), ("above", above)):
        for r in rows:
            pid = r["product_id"]
            if pid not in products.index:
                problems.append(f"product {pid}: not in the catalogue")
                continue
            p = products.loc[pid]
            cid = int(p["category_id"])
            if not r["brand_name_ko"] or not r["product_name_ko"]:
                problems.append(f"product {pid}: brand or product name is empty")
            if (r["brand_name_ko"], r["product_name_ko"], r["category_name"]) != (p["brand_name_ko"], p["product_name_ko"], p["category_name"]):
                problems.append(f"product {pid}: names differ from the catalogue")
            if cid not in t["eligible_category_ids"]:
                problems.append(f"product {pid}: its category is not eligible")
            if (side == "below" and not r["gap"] < 0) or (side == "above" and not r["gap"] > 0):
                problems.append(f"product {pid}: gap {r['gap']} is on the wrong side")
            own = [idx[(pid, m)][2] for m in months if (pid, m) in idx and idx[(pid, m)][1]]
            cat_ids = [int(q) for q in products.index[products["category_id"] == cid]]
            cat = [idx[(q, m)][2] for q in cat_ids for m in months if (q, m) in idx and idx[(q, m)][1]]
            if len(own) < BRAND_MIN_PRODUCT_MONTHS or len(own) != r["n_months"]:
                problems.append(f"product {pid}: {len(own)} gate-passing months recomputed, row says {r['n_months']} (minimum {BRAND_MIN_PRODUCT_MONTHS})")
                continue
            if any(math.isnan(v) for v in own + cat):
                problems.append(f"product {pid}: a gate-passing month has no net sentiment")
                continue
            pm, cm = sum(own) / len(own), sum(cat) / len(cat)
            for key, want in (("product_mean", pm), ("category_mean", cm), ("gap", pm - cm)):
                if abs(r[key] - want) > 0.0006:
                    problems.append(f"product {pid}: {key} {r[key]} but recomputed {want:.4f}")
    return problems


def _check_month_counts(payload: dict, t: dict, w: Window) -> list[str]:
    """Gate helper (BAN row): recompute this month's and last month's review count and product counts by a
    second path (plain loops over monthly_idx) and return the problems; [] means all agree."""
    problems: list[str] = []
    hd, cv = payload["header"], payload["coverage"]
    idx = t["monthly_idx"]

    def totals(month):
        rows = [v for (pid, m), v in idx.items() if m == month]
        return sum(r[0] for r in rows), len(rows), sum(1 for r in rows if r[1])

    prior = add_months(w.as_of, -1)
    if hd["prior_month"] != prior.isoformat() or hd["prior_month_display"] != month_display(prior):
        problems.append(f"prior month is {hd['prior_month']} / {hd['prior_month_display']}, expected {prior.isoformat()}")
    reviews, with_rev, evaluated = totals(w.as_of)
    for got, want, name in ((hd["asof_reviews"], reviews, "asof_reviews"),
                            (cv["asof_products_with_reviews"], with_rev, "asof_products_with_reviews"),
                            (cv["asof_products_evaluated"], evaluated, "asof_products_evaluated"),
                            (cv["asof_products_below_gate"], with_rev - evaluated, "asof_products_below_gate")):
        if got != want:
            problems.append(f"{name} is {got}, recomputed {want}")
    p_reviews, p_with, p_evaluated = totals(prior)
    if p_with == 0:
        for name, got in (("prior_month_reviews", hd["prior_month_reviews"]), ("prior_products_with_reviews", cv["prior_products_with_reviews"]),
                          ("prior_products_evaluated", cv["prior_products_evaluated"])):
            if got is not None:
                problems.append(f"{name} is {got} but the previous month has no rows")
    else:
        for got, want, name in ((hd["prior_month_reviews"], p_reviews, "prior_month_reviews"),
                                (cv["prior_products_with_reviews"], p_with, "prior_products_with_reviews"),
                                (cv["prior_products_evaluated"], p_evaluated, "prior_products_evaluated")):
            if got != want:
                problems.append(f"{name} is {got}, recomputed {want}")
    if not (0 <= cv["asof_products_evaluated"] <= cv["asof_products_with_reviews"] <= hd["asof_reviews"]):
        problems.append("as-of counts are not 0 <= evaluated <= products with reviews <= reviews")
    if cv["prior_products_with_reviews"] is not None and not (
            0 <= cv["prior_products_evaluated"] <= cv["prior_products_with_reviews"] <= hd["prior_month_reviews"]):
        problems.append("prior-month counts are not 0 <= evaluated <= products with reviews <= reviews")
    return problems


def validate_payload(payload: dict, t: dict, w: Window) -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []

    def gate(name: str, ok: bool, detail: str = "") -> None:
        (passed if ok else failures).append(name if ok else f"{name}: {detail}")

    gate("window_length", len(w.months) == WINDOW_MONTHS and w.months[-1] == w.as_of,
         "window is not exactly 12 months ending at as_of")
    if w.as_of == ANCHOR_WINDOW[1]:
        gate("anchor_equality", (w.months[0], w.months[-1]) == ANCHOR_WINDOW, "anchor window != dashboard window")
    gate("header_non_empty", payload["header"]["products_reviewed_in_window"] > 0, "no products with reviews in window")
    gate("category_shift_non_empty", any(r["shift_available"] for r in payload["category_shift"]),
         "no eligible category has both halves populated")

    # flag consistency across the WHOLE as-of slice, not just watchlist rows
    r = t["risk"][t["risk"]["as_of_month"] == w.as_of]
    bad = r[r["strict"] & ~r["broad"]]
    gate("flag_consistency", bad.empty, f"{len(bad)} row(s) with voc_decline_flag but not voc_decline_broad_flag")

    # grain
    pids = [p["product_id"] for p in payload["watchlist"]["products"]]
    gate("grain_watchlist", len(pids) == len(set(pids)), "duplicate product_id in watchlist")
    tgs = [g["theme_group"] for g in payload["complaint_movement"]["theme_groups"]]
    gate("grain_theme_groups", len(tgs) == len(set(tgs)), "duplicate theme_group in complaint movement")

    # Invariants -- cheap, and each catches a wiring mistake that would otherwise
    # reach the prompt as a plausible-looking number.
    fm = payload["complaint_movement"]
    rising = [g for g in fm["theme_groups"] if g["reported_as_rising"]]
    below_floor = [g["theme_group"] for g in rising if round(g["shift"] * 100, 1) < F_MATERIALITY_PP]
    gate("reported_rising_respects_floor",
         not below_floor and len(rising) <= F_REPORT_TOP_N and
         (not fm["available"] or fm["reported_rising_count"] == len(rising)),
         f"reported as rising below the {F_MATERIALITY_PP} pp floor / over the cap / count mismatch: {below_floor}")
    bvc = payload["brand_vs_category"]
    gate("brand_window_valid",
         1 <= BRAND_MIN_PRODUCT_MONTHS <= BRAND_WINDOW_MONTHS
         and bvc["window_months"] == BRAND_WINDOW_MONTHS == len(w.brand_months)   # (a window longer than WINDOW_MONTHS cannot equal len(brand_months))
         and bvc["window_end"] == w.as_of.isoformat() and bvc["window_start"] == w.brand_months[0].isoformat(),
         "block E is not the last BRAND_WINDOW_MONTHS months ending at as_of, or the window knobs are inconsistent")
    brand_problems = _check_brand_rows(bvc, t, w)
    gate("brand_vs_category_recomputed", not brand_problems, "; ".join(brand_problems[:6]))
    month_problems = _check_month_counts(payload, t, w)
    gate("month_counts_recomputed", not month_problems, "; ".join(month_problems[:6]))
    cov = payload["coverage"]
    gate("asof_gate_counts_consistent",
         0 <= cov["asof_products_below_gate"] <= cov["asof_products_with_reviews"] <= payload["header"]["products_in_catalogue"],
         "as-of-month gate counts are not 0 <= below <= with_reviews <= catalogue")
    ws = payload["watchlist"]["summary"]
    gate("headline_quiet_consistent",
         ws["headline_is_quiet"] == (ws["tier12_flagged_count"] == 0)
         and ws["no_flags_at_any_tier"] == (ws["total_flagged"] == 0)
         and ws["tier12_flagged_count"] == ws["tier1_strict_count"] + ws["tier2_broad_only_count"],
         "watchlist summary quiet flags disagree with the tier counts")

    # nulls + ranges
    null_bad, range_bad = [], []
    for path, key, val in _walk(payload):
        if val is None and key not in OPTIONAL_KEYS:
            null_bad.append(f"{path}.{key}" if path else key)
        if key in RANGES and isinstance(val, (int, float)) and not isinstance(val, bool):
            lo, hi = RANGES[key]
            if not (lo <= val <= hi):
                range_bad.append(f"{path}.{key}={val}")
            if key in SENTIMENT_LEVEL_KEYS and val < 0:
                warnings.append(f"negative sentiment level at {path}.{key}={val} (structurally possible; usually >= 0)")
        if key.endswith("_count") or key.endswith("_mentions") or key.startswith("products_") \
                or key.startswith("categories_") or key == "reviews_in_window" or key in NONNEG_KEYS:
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val < 0:
                range_bad.append(f"{path}.{key}={val}")
    gate("no_nulls_in_printable_fields", not null_bad, "; ".join(null_bad[:8]))
    gate("ranges", not range_bad, "; ".join(range_bad[:8]))

    if failures:
        raise ReportInputError("validation gate(s) failed for as_of "
                               f"{w.as_of}:\n  - " + "\n  - ".join(failures))
    return {"gates_passed": passed, "warnings": sorted(set(warnings))}


# =============================================================================
# THE FUNCTION THE SPEC NAMES
# =============================================================================


def build_report_input(as_of_month: date, tables: dict | None = None) -> dict:
    """The payload for one as-of month. `tables` = prepare_tables(...) output; if omitted, loads from the
    live DB. The replay loop passes one preloaded `tables` for all 12 months."""
    w = build_window(as_of_month)
    t = tables if tables is not None else prepare_tables(load_raw_tables())
    if t["snapshot_date"] is None:
        raise ReportInputError("voc_decline_risk_v1 has no computed_at -- cannot stamp mart_snapshot_date")

    f_block, g_block = build_complaint_movement(t, w)
    payload: dict[str, Any] = {
        "envelope": {
            "script_version": SCRIPT_VERSION,
            "mart_snapshot_date": t["snapshot_date"].isoformat(),
            "as_of_month": w.as_of.isoformat(),
            "as_of_month_display": month_display(w.as_of),
            "window_start": w.start.isoformat(),
            "window_end": w.as_of.isoformat(),
            "knobs": {
                "movement_half_months": MOVEMENT_HALF_MONTHS, "max_watchlist": MAX_WATCHLIST,
                "f_report_top_n": F_REPORT_TOP_N, "f_materiality_pp": F_MATERIALITY_PP,
                "phrase_display_max_chars": PHRASE_DISPLAY_MAX_CHARS,
                "brand_window_months": BRAND_WINDOW_MONTHS,
                "brand_min_product_months": BRAND_MIN_PRODUCT_MONTHS,
                "brand_top_n": BRAND_TOP_N, "category_min_products": CATEGORY_MIN_PRODUCTS,
            },
        },
        "header": build_header(t, w),
        "watchlist": build_watchlist(t, w),
        "category_shift": build_category_shift(t, w),
        "brand_vs_category": build_brand_vs_category(t, w),
        "complaint_movement": f_block,
        "evidence_phrases": g_block,
    }
    payload["coverage"] = build_coverage(t, w, f_block)
    payload["validation"] = validate_payload(payload, t, w)
    return payload


# =============================================================================
# EMIT + REPLAY-FRIENDLY CLI
# =============================================================================


def _json_default(o: Any):
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dumps_payload(payload: dict) -> str:
    # ensure_ascii=False: product names / phrases are Korean.
    # allow_nan=False: a NaN sneaking through is a hard error, not "NaN" in JSON.
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)


def write_report_input(payload: dict, out_dir: Path = OUTPUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_input_{payload['envelope']['as_of_month'][:7]}.json"
    tmp = path.with_suffix(".json.part")
    tmp.write_text(dumps_payload(payload) + "\n", encoding="utf-8")
    tmp.replace(path)  # incremental save: a crash never leaves a half-written report_input
    return path


def parse_as_of(s: str) -> date:
    parts = s.split("-")
    if len(parts) == 2:
        parts.append("01")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def replay_months() -> list[date]:
    return month_list(REPLAY_FIRST_AS_OF, 12)  # 2025-07 .. 2026-06


def run_selftest(t: dict) -> bool:
    """LIVE reconciliation at the anchor month against numbers already verified
    elsewhere in this project. Every FAIL is a finding, not noise."""
    as_of = ANCHOR_WINDOW[1]
    p = build_report_input(as_of, t)
    w = build_window(as_of)
    hdr, ws, cov = p["header"], p["watchlist"]["summary"], p["coverage"]
    checks = []

    def chk(name, got, want, tol=None):
        ok = (abs(got - want) <= tol) if tol is not None else (got == want)
        checks.append((name, got, want, ok))

    chk("products_in_catalogue", hdr["products_in_catalogue"], 103)                     # dashboard KPI
    chk("reviews_in_window", hdr["reviews_in_window"], 271_252)                         # dashboard KPI
    chk("eligible_categories (>=3 products)", len(p["category_shift"]), 9)              # 9 of 26 categories
    chk("complaint_months_in_window", cov["complaint_months_in_window"], 12)
    # Strict is a subset of broad, so the anchor's one flagged product is counted broad-incl-strict.
    chk("flagged at anchor (tier1 + tier2-only)", ws["tier1_strict_count"] + ws["tier2_broad_only_count"], 1)
    chk("unclustered_share ~ 50,146/172,317", cov["unclustered_share"], 50_146 / 172_317, tol=0.001)
    # As-of-month gate figures vs a separate SQL count for 2026-06
    # (risk rows 103, gate-passing 81). Same source table family, different route.
    chk("as-of products with reviews (Q1 risk rows)", cov["asof_products_with_reviews"], 103)
    chk("as-of products below gate (Q1: 103 - 81)", cov["asof_products_below_gate"], 22)
    chk("tier12_flagged_count == flagged at anchor", ws["tier12_flagged_count"], 1)
    # Dashboard mix-shift reproduction: this pipeline's F function at N=6 for product 2008976,
    # whose dashboard numbers were verified separately.
    d = complaint_shift_table(t["mentions"], as_of, 6, product_id=2008976).set_index("theme_group")
    for grp, want in (("Adverse skin reaction", 0.0553), ("Scent", -0.0332)):
        chk(f"dashboard mix shift 2008976 {grp}", round(float(d.loc[grp, "shift"]), 4) if grp in d.index else float("nan"), want, tol=0.00006)
    chk("dashboard mix shift 2008976 H1 n", int(d["prior_mentions"].sum()), 914)
    chk("dashboard mix shift 2008976 H2 n", int(d["recent_mentions"].sum()), 300)

    width = max(len(c[0]) for c in checks)
    print(f"\nSelf-test at as-of {as_of}  (window {w.start} .. {w.as_of})")
    for name, got, want, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  got={got}  want={want}")
    print("  unclustered_share raw:", cov["unclustered_share"], "(ledger recorded 0.2909; 50,146/172,317 = %.5f)" % (50_146 / 172_317))
    return all(c[3] for c in checks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build report input JSON(s).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--as-of", help="YYYY-MM or YYYY-MM-DD (first of month), <= 2026-06")
    g.add_argument("--all", action="store_true", help="the 12 replay months, 2025-07 .. 2026-06")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--force", action="store_true", help="overwrite existing report_input files")
    ap.add_argument("--selftest", action="store_true", help="live reconciliation at 2026-06 (needs the DB)")
    args = ap.parse_args(argv)

    months = replay_months() if args.all else [parse_as_of(args.as_of)]
    for m in months:
        build_window(m)  # hard-fail on an out-of-bounds month BEFORE any DB work

    out_dir = Path(args.out_dir)
    todo = [m for m in months
            if args.force or not (out_dir / f"report_input_{m.isoformat()[:7]}.json").exists()]
    if not todo and not args.selftest:
        print("Nothing to do: every requested report_input already exists (use --force).")
        return 0

    print("Loading mart tables ...")
    t = prepare_tables(load_raw_tables())
    if args.selftest and not run_selftest(t):
        print("\nSelf-test FAILED -- stopping before writing anything. Every FAIL above is a finding "
              "to root-cause, not to waive.")
        return 2
    for m in todo:
        p = build_report_input(m, t)  # a failed gate raises: the loop STOPS, never skips silently
        path = write_report_input(p, out_dir)
        ws = p["watchlist"]["summary"]
        print(f"  {m.isoformat()[:7]}  watchlist total={ws['total_flagged']} "
              f"(t1={ws['tier1_strict_count']}, t2only={ws['tier2_broad_only_count']}, "
              f"t3only={ws['tier3_dip_only_count']})  headline_quiet={'yes' if ws['headline_is_quiet'] else 'no'}"
              f"  F={'yes' if p['complaint_movement']['available'] else 'unavailable'}"
              f"  warnings={len(p['validation']['warnings'])}  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
