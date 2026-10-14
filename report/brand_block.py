"""Products vs category section for the monthly report.

Pure functions over the payload's block E (payload key `brand_vs_category`): no model call,
no database, no network, no file writes.

What the section shows: products, not brands (a brand can have products in several
categories), over the last 6 months of the window, a full row every month (no "still below" compression),
with the percentage kept. Each row names brand, product and category, then the product's average net
sentiment against its category's average, the gap, and the gap as a percentage of the category average.

The percentage is the one number made here (the renderer computes the % of a change from the
payload's raw values): gap / |category mean|, rounded half away from zero. Every other number is the
payload's own display string. A minus sign is printed as the typographic "−" the other rows use.
"""
import math

SECTION_TITLE = "Products vs category"
ROWS_PER_SIDE = 2            # the payload keeps 3 per side; the message prints 2
ARROW_BELOW = "🔻"
ARROW_ABOVE = "🟢"
NONE_LINE = "No product sits clearly above or below its category average this month."


def minus(s):
    """The payload prints a hyphen-minus; the report prints the typographic minus."""
    return s.replace("-", "−")


def pct_text(gap, category_mean):
    """'-25%' style text for the gap as a share of the category average, or None when it is undefined.
    Rounded half away from zero (Python's round() is half-to-even and would print 0.5 as 0)."""
    if category_mean is None or category_mean == 0 or gap is None:
        return None
    x = gap / abs(category_mean) * 100
    n = int(math.floor(abs(x) + 0.5))
    if n == 0:
        return "0%"
    return f"{'+' if x > 0 else '−'}{n}%"


def context_line(block):
    return f"net sentiment, −1 to +1 · last {block['window_months_display']} months ({block['window_display']})"


def make_row(r, side):
    return {
        "side": side, "product_id": r["product_id"], "brand": r["brand_name_ko"], "product": r["product_name_ko"],
        "category": r["category_name"], "product_mean": r["product_mean_display"], "category_mean": r["category_mean_display"],
        "gap": minus(r["gap_display"]), "pct": pct_text(r["gap"], r["category_mean"]),
        "arrow": ARROW_BELOW if side == "below" else ARROW_ABOVE,
    }


def build_section(block, rows_per_side=None):
    """The section as data. block: the payload's brand_vs_category."""
    n = ROWS_PER_SIDE if rows_per_side is None else rows_per_side
    rows = ([make_row(r, "below") for r in block["largest_below_category"][:n]]
            + [make_row(r, "above") for r in block["largest_above_category"][:n]])
    return {"title": SECTION_TITLE, "context": context_line(block), "rows": rows, "none_shown": not rows}


def name_line(row):
    return f"{row['brand']} · {row['product']} {row['category']}"


def number_line(row):
    tail = f" ({row['pct']})" if row["pct"] else ""
    return f"{row['product_mean']} vs {row['category_mean']} {row['arrow']} {row['gap']}{tail}"


def section_lines(section):
    """The section as plain lines. Its word count is taken from these."""
    lines = [f"{section['title']} {section['context']}"]
    for r in section["rows"]:
        lines.append(name_line(r))
        lines.append(number_line(r))
    if section["none_shown"]:
        lines.append(NONE_LINE)
    return lines


def word_count(lines):
    """Whitespace tokens, bullets not counted (the same convention as the other blocks)."""
    return sum(len([t for t in line.split() if t != "•"]) for line in lines)


def verify_section(section, block, rows_per_side=None):
    """Problems with the section against its payload block; [] means all clear."""
    problems = []
    n = ROWS_PER_SIDE if rows_per_side is None else rows_per_side
    if section["title"] != SECTION_TITLE:
        problems.append("title changed")
    if section["context"] != context_line(block):
        problems.append("context line is not the payload's window")
    by = {("below", r["product_id"]): r for r in block["largest_below_category"]}
    by.update({("above", r["product_id"]): r for r in block["largest_above_category"]})
    ids = [r["product_id"] for r in section["rows"]]
    if len(ids) != len(set(ids)):
        problems.append("a product is shown twice")
    for side in ("below", "above"):
        if sum(1 for r in section["rows"] if r["side"] == side) > n:
            problems.append(f"more than {n} rows on the {side} side")
    for r in section["rows"]:
        p = by.get((r["side"], r["product_id"]))
        if p is None:
            problems.append(f"product {r['product_id']}: not in the payload's {r['side']} list")
            continue
        if not r["brand"] or not r["product"]:
            problems.append(f"product {r['product_id']}: brand or product name is empty")
        want = (p["brand_name_ko"], p["product_name_ko"], p["category_name"], p["product_mean_display"], p["category_mean_display"], minus(p["gap_display"]))
        got = (r["brand"], r["product"], r["category"], r["product_mean"], r["category_mean"], r["gap"])
        if got != want:
            problems.append(f"product {r['product_id']}: a printed string is not the payload's")
        if r["pct"] != pct_text(p["gap"], p["category_mean"]):
            problems.append(f"product {r['product_id']}: percentage is not gap / category mean")
        if (r["side"] == "below" and not p["gap"] < 0) or (r["side"] == "above" and not p["gap"] > 0):
            problems.append(f"product {r['product_id']}: on the wrong side")
    # what is shown must be the payload's first rows, in its order
    for side, key in (("below", "largest_below_category"), ("above", "largest_above_category")):
        shown = [r["product_id"] for r in section["rows"] if r["side"] == side]
        if shown != [r["product_id"] for r in block[key]][:len(shown)]:
            problems.append(f"the {side} rows are not the payload's first rows in its order")
    if section["none_shown"] != (not section["rows"]):
        problems.append("none_shown disagrees with the rows")
    return problems
