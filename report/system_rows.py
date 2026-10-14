"""The system rows of the monthly report: the header, the at-risk rows and the dip rows.

Pure functions over one saved payload's watchlist: no model call, no database, no network, no file writes.

    VoC Risk Report · June 2026
    At risk  net sentiment, −1 to +1 · last month → this month
    🔴 마녀공장 · 퓨어 소이빈 클렌징 오일  Cleansing Oils
    0.584 → 0.472 🔻 −0.112 (−19%) · 66 reviews

Tier 1 is 🔴, tier 2 (the broader threshold) 🟠, tier 3 (a one-month dip, not a risk signal) 🟡. Tier 1 and 2 rows go under "At risk", tier 3
rows under "Dips". The unit note "net sentiment, −1 to +1" is printed once, on the first section that has numbers.

The % is the one number made here: the payload's own printed delta over the absolute value of last month's level, rounded
half away from zero, so the delta and the % on a row always agree; "n/a" when last month's level is under 0.05 (a % of a near-zero level says
nothing). A row whose month-over-month comparison is unavailable prints this month's level and the reason, and no delta.

Every other number is the payload's own display string; a hyphen-minus is printed as the typographic "−" the other rows use.
"""
import brand_block as bb

TIER_EMOJI = {1: "🔴", 2: "🟠", 3: "🟡"}
ARROW_DOWN, ARROW_UP, ARROW_FLAT = "🔻", "🟢", "⚪"
PCT_FLOOR = 0.05
NOTE = "net sentiment, −1 to +1"
AT_RISK = {"key": "at_risk", "title": "At risk", "context": "last month → this month", "color": "#e01e5a", "tiers": (1, 2)}
DIPS = {"key": "dips", "title": "Dips", "context": "one-month drops, not risk signals · last month → this month", "color": "#ecb22e", "tiers": (3,)}
QUIET_LINE = "No product tripped the risk signal, so no product review quotes are shown this month. The complaint mix shift is below."
NO_COMPARISON = {
    "prior_month_below_review_gate": "last month had fewer than 20 reviews",
    "prior_month_has_no_reviews": "last month had no reviews",
    "current_month_below_review_gate": "this month had fewer than 20 reviews",
    "current_month_has_no_reviews": "this month had no reviews",
}


def header_line(payload):
    return f"VoC Risk Report · {payload['envelope']['as_of_month_display']}"


def reviews_text(entry):
    return f"{entry['review_count_display']} review" if entry["review_count"] == 1 else f"{entry['review_count_display']} reviews"


def sentiment_pct(delta, before):
    """The % beside a net-sentiment delta: delta / |before|, "n/a" when |before| is under PCT_FLOOR."""
    if abs(before) < PCT_FLOOR:
        return "n/a"
    return bb.pct_text(delta, before)


def arrow(delta):
    return ARROW_UP if delta > 0 else ARROW_DOWN if delta < 0 else ARROW_FLAT


def build_row(entry):
    """One watchlist entry as a row (a dict of the strings that get printed)."""
    mom = entry["month_over_month"]
    row = {"product_id": entry["product_id"], "tier": entry["highest_tier"], "emoji": TIER_EMOJI[entry["highest_tier"]],
           "brand": entry["brand_name_ko"], "product": entry["product_name_ko"], "category": entry["category_name"],
           "now": entry["net_sentiment_display"], "reviews": reviews_text(entry), "available": bool(mom["available"]),
           "prior": None, "delta": None, "arrow": None, "pct": None, "no_comparison": None}
    if mom["available"]:
        row.update(prior=mom["prior_net_sentiment_display"], delta=bb.minus(mom["delta_display"]), arrow=arrow(mom["delta"]),
                   pct=sentiment_pct(mom["delta"], mom["prior_net_sentiment"]))
    else:
        row["no_comparison"] = NO_COMPARISON[mom["unavailable_reason"]]
    return row


def build_sections(payload):
    """[at-risk section, dips section] as data, each present only when it has rows. The payload's order is kept.
    The first section carries the unit note. `truncated` says how many flagged products the payload listed of how many there are."""
    ws = payload["watchlist"]
    sections = []
    for spec in (AT_RISK, DIPS):
        entries = [e for e in ws["products"] if e["highest_tier"] in spec["tiers"]]
        if entries:
            sections.append({**{k: spec[k] for k in ("key", "title", "context", "color")}, "note": False, "rows": [build_row(e) for e in entries]})
    if sections:
        sections[0]["note"] = True
    truncated = None
    if ws["summary"]["truncated"]:
        truncated = {"shown": len(ws["products"]), "total": ws["summary"]["total_flagged_display"]}
    return {"sections": sections, "truncated": truncated}


def context_text(section):
    return f"{NOTE} · {section['context']}" if section["note"] else section["context"]


def name_line(row):
    return f"{row['emoji']} {row['brand']} · {row['product']} {row['category']}"


def number_line(row):
    if not row["available"]:
        return f"{row['now']} · {row['reviews']} · no comparison: {row['no_comparison']}"
    pct = f" ({row['pct']})" if row["pct"] else ""
    return f"{row['prior']} → {row['now']} {row['arrow']} {row['delta']}{pct} · {row['reviews']}"


def truncated_line(truncated):
    return f"Showing {truncated['shown']} of {truncated['total']} flagged products."


def section_lines(section):
    """A section as plain lines (title and context, then a name line and a number line per row)."""
    lines = [f"{section['title']} {context_text(section)}"]
    for r in section["rows"]:
        lines.append(name_line(r))
        lines.append(number_line(r))
    return lines


def verify_sections(built, payload):
    """Problems with the built sections against the payload; [] means all clear."""
    problems = []
    ws = payload["watchlist"]
    want = {1: [], 2: [], 3: []}
    for e in ws["products"]:
        want[e["highest_tier"]].append(e)
    rows = {r["product_id"]: (s["key"], r) for s in built["sections"] for r in s["rows"]}
    if len(rows) != sum(len(s["rows"]) for s in built["sections"]):
        problems.append("a product is shown twice")
    if set(rows) != {e["product_id"] for e in ws["products"]}:
        problems.append("the rows are not the payload's watchlist products")
    order = [r["product_id"] for s in built["sections"] for r in s["rows"]]
    payload_order = [e["product_id"] for spec in (AT_RISK, DIPS) for e in ws["products"] if e["highest_tier"] in spec["tiers"]]
    if order != payload_order:
        problems.append("the rows are not in the payload's order, at risk first")
    for e in ws["products"]:
        got = rows.get(e["product_id"])
        if got is None:
            continue
        key, r = got
        mom = e["month_over_month"]
        if (key == "at_risk") != (e["highest_tier"] in (1, 2)):
            problems.append(f"product {e['product_id']}: in the wrong section")
        if r["emoji"] != TIER_EMOJI[e["highest_tier"]]:
            problems.append(f"product {e['product_id']}: the tier emoji is not the payload's tier")
        if (r["brand"], r["product"], r["category"], r["now"]) != (e["brand_name_ko"], e["product_name_ko"], e["category_name"], e["net_sentiment_display"]):
            problems.append(f"product {e['product_id']}: a printed string is not the payload's")
        if r["available"] != bool(mom["available"]):
            problems.append(f"product {e['product_id']}: comparison availability differs from the payload")
        elif mom["available"]:
            if (r["prior"], r["delta"]) != (mom["prior_net_sentiment_display"], bb.minus(mom["delta_display"])):
                problems.append(f"product {e['product_id']}: the prior level or the delta is not the payload's")
            if r["pct"] != sentiment_pct(mom["delta"], mom["prior_net_sentiment"]):
                problems.append(f"product {e['product_id']}: the % is not delta / |last month|")
            if r["arrow"] != arrow(mom["delta"]):
                problems.append(f"product {e['product_id']}: the arrow does not match the delta")
        elif r["no_comparison"] != NO_COMPARISON[mom["unavailable_reason"]]:
            problems.append(f"product {e['product_id']}: the no-comparison reason is not the payload's")
        if r["reviews"] != reviews_text(e):
            problems.append(f"product {e['product_id']}: the review count is not the payload's")
    notes = [s["note"] for s in built["sections"]]
    if notes and notes != [True] + [False] * (len(notes) - 1):
        problems.append("the unit note must be on the first section only")
    if bool(built["truncated"]) != bool(ws["summary"]["truncated"]):
        problems.append("the truncation line disagrees with the payload")
    return problems


def word_count(lines):
    """Whitespace tokens, bullets not counted (the same convention as the other blocks)."""
    return sum(len([t for t in line.split() if t != "•"]) for line in lines)
