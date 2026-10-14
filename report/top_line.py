"""The top of the monthly report: the BAN row and the count row.

Pure functions over the saved monthly payloads: no model call, no database, no network, no file writes.

The message opens with two big-number lines, each beside last month's value:
    Reviews: 14,220 (−27% vs May)
    Products evaluated (20+ reviews): 81 of 103 (May: 85 of 103)
and the count row keeps its three counts, each with last month's beside it:
    🔴 1 at risk (May: 0)   🟡 4 dips (May: 3)   🔺 3 rising complaint themes (May: 2)
The second BAN line also says how many products had too few reviews (under 20) to be evaluated.

Sources. Review counts and evaluated products, this month and last, come from the payload (
header.asof_reviews / prior_month_reviews, coverage.asof_products_evaluated / prior_products_*), which
the builder computes from the mart. The counts row's "last month" comes from LAST MONTH'S SAVED PAYLOAD
(what the previous report said), and only when that payload is the month right before; otherwise no
"(May: n)" is printed for it. A first replay month therefore has no comparison on the counts row.

Every number printed is a string copied from the payload; the one number made here is the percentage
change in reviews: (this month - last month) / last month, from the payload's raw counts, rounded half
away from zero. Nothing is printed for a percentage when last month had no reviews.
"""
import brand_block as bb
import mix_shift as ms

REVIEWS_LABEL = "Reviews:"
EVALUATED_LABEL = "Products evaluated (20+ reviews):"


def month_word(display):
    """'May 2026' -> 'May'."""
    return display.split()[0]


def _adjacent(payload, previous):
    return previous is not None and ms.month_index(ms.as_of(payload)) - ms.month_index(ms.as_of(previous)) == 1


def _plural(n, singular, plural):
    return singular if n == 1 else plural


def build_top(payload, previous=None):
    """The top rows as data. previous: the payload of the month before, or None."""
    hd, cv, ws, cm = payload["header"], payload["coverage"], payload["watchlist"]["summary"], payload["complaint_movement"]
    word = month_word(hd["prior_month_display"])
    has_prior = hd["prior_month_reviews"] is not None
    reviews = {"value": hd["asof_reviews_display"], "month": word,
               "prior": hd["prior_month_reviews_display"] if has_prior else None,
               "pct": bb.pct_text(hd["asof_reviews"] - hd["prior_month_reviews"], hd["prior_month_reviews"]) if has_prior else None}
    evaluated = {"value": cv["asof_products_evaluated_display"], "of": cv["asof_products_with_reviews_display"], "month": word,
                 "prior": cv["prior_products_evaluated_display"] if has_prior else None,
                 "prior_of": cv["prior_products_with_reviews_display"] if has_prior else None}
    adj = _adjacent(payload, previous)
    prev_ws = previous["watchlist"]["summary"] if adj else None
    prev_cm = previous["complaint_movement"] if adj else None
    themes_ok = bool(cm["available"])
    counts = [
        {"key": "at_risk", "emoji": "🔴", "n": ws["tier12_flagged_count_display"],
         "label": "at risk", "prior": prev_ws["tier12_flagged_count_display"] if adj else None},
        {"key": "dips", "emoji": "🟡", "n": ws["tier3_dip_only_count_display"],
         "label": _plural(ws["tier3_dip_only_count"], "dip", "dips"),
         "prior": prev_ws["tier3_dip_only_count_display"] if adj else None},
    ]
    if themes_ok:
        counts.append({"key": "themes", "emoji": ms.ARROW_UP, "n": cm["reported_rising_count_display"],
                       "label": _plural(cm["reported_rising_count"], "rising complaint theme", "rising complaint themes"),
                       "prior": prev_cm["reported_rising_count_display"] if (adj and prev_cm["available"]) else None})
    else:
        counts.append({"key": "themes", "emoji": "⚪", "n": None, "label": "complaint themes n/a", "prior": None})
    return {"as_of": ms.as_of(payload), "reviews": reviews, "evaluated": evaluated, "counts": counts, "prior_month_word": word,
            "counts_compared": adj}


def reviews_line(top):
    r = top["reviews"]
    if r["prior"] is None:
        return f"{REVIEWS_LABEL} {r['value']}"
    change = f"{r['pct']} vs {r['month']}" if r["pct"] else f"{r['month']}: {r['prior']}"
    return f"{REVIEWS_LABEL} {r['value']} ({change})"


def evaluated_line(top):
    e = top["evaluated"]
    base = f"{EVALUATED_LABEL} {e['value']} of {e['of']}"
    return base if e["prior"] is None else f"{base} ({e['month']}: {e['prior']} of {e['prior_of']})"


def count_text(c, word):
    if c["n"] is None:
        return f"{c['emoji']} {c['label']}"
    tail = f" ({word}: {c['prior']})" if c["prior"] is not None else ""
    return f"{c['emoji']} {c['n']} {c['label']}{tail}"


def counts_line(top):
    return "   ".join(count_text(c, top["prior_month_word"]) for c in top["counts"])


def top_lines(top):
    """The three lines of the top of the message, as plain text. Their word count is taken from these."""
    return [reviews_line(top), evaluated_line(top), counts_line(top)]


def word_count(lines):
    """Whitespace tokens, bullets not counted (the same convention as the other blocks)."""
    return sum(len([t for t in line.split() if t != "•"]) for line in lines)


def verify_top(top, payload, previous=None):
    """Problems with the top rows against their payloads; [] means all clear."""
    problems = []
    hd, cv, ws, cm = payload["header"], payload["coverage"], payload["watchlist"]["summary"], payload["complaint_movement"]
    r, e = top["reviews"], top["evaluated"]
    if r["value"] != hd["asof_reviews_display"] or e["value"] != cv["asof_products_evaluated_display"] or e["of"] != cv["asof_products_with_reviews_display"]:
        problems.append("a this-month figure is not the payload's string")
    has_prior = hd["prior_month_reviews"] is not None
    if has_prior != (r["prior"] is not None) or has_prior != (e["prior"] is not None):
        problems.append("last month's figures are shown or hidden against the payload")
    if has_prior:
        if r["prior"] != hd["prior_month_reviews_display"] or e["prior"] != cv["prior_products_evaluated_display"] or e["prior_of"] != cv["prior_products_with_reviews_display"]:
            problems.append("a last-month figure is not the payload's string")
        if r["pct"] != bb.pct_text(hd["asof_reviews"] - hd["prior_month_reviews"], hd["prior_month_reviews"]):
            problems.append("the reviews percentage is not (this month - last month) / last month")
        if r["month"] != month_word(hd["prior_month_display"]):
            problems.append("the comparison month is not the payload's prior month")
    counts = {c["key"]: c for c in top["counts"]}
    if counts["at_risk"]["n"] != ws["tier12_flagged_count_display"] or counts["dips"]["n"] != ws["tier3_dip_only_count_display"]:
        problems.append("a count is not the payload's string")
    if cm["available"]:
        if counts["themes"]["n"] != cm["reported_rising_count_display"]:
            problems.append("the themes count is not the payload's string")
        if cm["reported_rising_count"] != len(ms.shown_themes(payload)):
            problems.append("the payload's rising-theme count differs from the themes the section shows")
    elif counts["themes"]["n"] is not None:
        problems.append("themes shown as a count although the complaint comparison is unavailable")
    adj = _adjacent(payload, previous)
    if top["counts_compared"] != adj:
        problems.append("counts compared against a payload that is not the month before")
    if adj:
        want = {"at_risk": previous["watchlist"]["summary"]["tier12_flagged_count_display"],
                "dips": previous["watchlist"]["summary"]["tier3_dip_only_count_display"],
                "themes": previous["complaint_movement"]["reported_rising_count_display"] if (previous["complaint_movement"]["available"] and cm["available"]) else None}
        for k, v in want.items():
            if counts[k]["prior"] != v:
                problems.append(f"last month's {k} count is not the previous payload's")
    else:
        for k, c in counts.items():
            if c["prior"] is not None:
                problems.append(f"last month's {k} count printed without an adjacent previous payload")
    return problems
