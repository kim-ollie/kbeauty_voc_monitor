"""The number gate.

WHAT IT DOES, IN ONE PARAGRAPH
    The finished message is a list of lines, and the renderer tags each line with what kind of line it is ("row_numbers", "flag",
    "block_label", ...) and, where it matters, which product or theme it belongs to. For every line this gate does three things:
      1. reads the numbers actually printed on the line (customers' quoted words are cut out first: they are not ours);
      2. builds, from the saved payloads and the saved model records only, the list of numbers that line SHOULD carry, in order;
      3. compares the two lists. Same numbers in the same order: the line passes. Anything else (a number that is not in the data, a
         number in the wrong place, a number missing): the report fails and nothing is posted.

WHY IT CAN BE TRUSTED (independence)
    This file imports no renderer code: not system_rows, top_line, brand_block, mix_shift, flag_line, complaint_block or message. It
    has its own tokenizer, its own percentage rounding (exact decimal arithmetic, where the renderer uses floats), its own ordinal
    and streak counting and its own recount of the complaint labels from the model's raw assignments. If the renderer and this gate
    both computed a number the same wrong way, it would be a coincidence of two separate pieces of code, not a shared bug.

WHERE EXPECTED NUMBERS COME FROM (each one carries its source, so the explanation table can show it)
      payload:<path>    a display string of this month's payload (or, for "last month" figures, the previous month's payload)
      recomputed:...    a number the renderer makes itself: the % changes, the "Nth month" ordinals, the block's counts. The gate
                        recomputes it from raw values (payload numbers, the case file, the model's label assignments)
      template:...      a number written into a fixed sentence ("−1 to +1", "20+ reviews" is NOT one: its 20 is read out of the payload)
      case:...          a segment id, which must belong to a this-month text of the case file
      config:...        a setting outside the data (the label of the optional dashboard-link line, report_config.DASHBOARD_LABEL)

WHAT IT DOES NOT DO
    It does not judge wording, and it does not check that a quote is a real quote (gates.py does that). It checks numbers, and a few
    counts that the message states about itself (the header count row against the rows below it).
"""
import re
from decimal import Decimal, ROUND_HALF_UP

QUOTE_OPEN, QUOTE_CLOSE = "“", "”"
MINUS = "−"

# A number as it is printed: optional sign, a run of digits (a comma is part of it only between two digits, so "18,1219" is one
# number and "3, then" is the number 3), optional decimals, and an optional ending (% or an ordinal suffix or "+", as in "20+").
# The "(?<![\d.,])" stops a match from starting in the middle of another number.
TOKEN = re.compile(r"(?<![\d.,])[+\-−]?\d(?:,?\d)*(?:\.\d+)?(?:%|st|nd|rd|th|\+)?")

QUOTE_KINDS = {"block_label_quote", "block_point", "mix_example"}     # lines that carry a customer's words between “ and ”
SENTIMENT_PCT_FLOOR = Decimal("0.05")                                # below this a "% of last month's level" says nothing: printed "n/a"
BELOW_GATE_REASONS = ("prior_month_below_review_gate", "current_month_below_review_gate")
UNIT_NOTE = ["-1", "+1"]                                             # "net sentiment, −1 to +1"
MIX_UNAVAILABLE_NUMBERS = ["2025", "6"]                              # the fixed sentence: "starts in July 2025 and needs 6 months of data"
READOUT_SECTIONS = ("what_reviewers_say", "what_looks_new", "in_unclustered_segments")


class Refuse(Exception):
    """The gate cannot build an expectation for a line (its product is not on the payload, no records, ...): the line fails."""


# ------------------------------------------------------------------ reading numbers
def tok(text):
    """The numbers in a text, in order, with the typographic minus turned into a hyphen: '−0.112 (−19%)' -> ['-0.112', '-19%']."""
    return [m.group().replace(MINUS, "-") for m in TOKEN.finditer(text or "")]


def outside_quotes(text):
    """(text without the quoted words, the quoted words). One quote per line, from its first “ to its last ”."""
    a, b = text.find(QUOTE_OPEN), text.rfind(QUOTE_CLOSE)
    if a < 0 or b < a:
        raise Refuse("a quote line has no complete “…” pair")
    return text[:a] + " " + text[b + 1:], text[a + 1:b]


# ------------------------------------------------------------------ making numbers (the gate's own arithmetic)
def pct_text(change, base):
    """change / |base| as a signed whole percent, half away from zero, in exact decimal arithmetic. None when base is 0."""
    base = Decimal(str(base))
    if base == 0:
        return None
    x = Decimal(str(change)) / abs(base) * 100
    n = int(abs(x).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return "0%" if n == 0 else ("+" if x > 0 else "-") + f"{n}%"


def sentiment_pct_text(change, prior):
    """The % beside a net-sentiment change: none (printed 'n/a') when last month's level is under 0.05."""
    return None if abs(Decimal(str(prior))) < SENTIMENT_PCT_FLOOR else pct_text(change, prior)


def ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def month_index(payload):
    y, m = payload["envelope"]["as_of_month"][:7].split("-")
    return int(y) * 12 + int(m)


def rising_themes(payload):
    cm = payload["complaint_movement"]
    return [g for g in cm["theme_groups"] if g["reported_as_rising"]] if cm["available"] else []


def streak(theme, payload, previous_payloads):
    """How many consecutive months, this one included, the theme was reported as rising (stops at a gap in the months)."""
    n, newer = 1, payload
    for older in reversed(previous_payloads):
        if month_index(older) != month_index(newer) - 1 or theme not in {g["theme_group"] for g in rising_themes(older)}:
            break
        n, newer = n + 1, older
    return n


def review_gate(payload):
    """The review-count threshold ('20'), read out of the payload's own review-gate caveat rather than written here."""
    for c in payload["coverage"]["caveats"]:
        if c["id"] == "review_gate":
            m = re.search(r"fewer than (\d+) reviews", c["text"])
            if m:
                return m.group(1)
    raise Refuse("the payload's review_gate caveat does not state the threshold")


# ------------------------------------------------------------------ recounting the complaint block from the model's raw answers
def current_ids(rec):
    return {s["segment_id"] for s in rec["case"]["segments"] if s["period"] == "current"}


def label_counts(rec):
    """{label name: number of this-month texts the model gave that label}, from the raw assignments (a text counted once per label)."""
    cur = current_ids(rec)
    names = [lab["name"] for lab in rec["labeling"]["labels"]]
    if len(names) != len(set(names)):
        raise Refuse("two labels of the model's answer share a name")
    by_id = {lab["label_id"]: lab["name"] for lab in rec["labeling"]["labels"]}
    seen = {name: set() for name in names}
    for a in rec["labeling"]["assignments"]:
        if a["segment_id"] in cur and a["label_id"] in by_id:
            seen[by_id[a["label_id"]]].add(a["segment_id"])
    return {name: len(ids) for name, ids in seen.items()}


def points_citing(rec, segment_id):
    """How many evidence texts each 'worth a look' point has, for the points whose first cited text is `segment_id`."""
    out = set()
    for sec in READOUT_SECTIONS:
        for p in rec["readout"].get(sec) or []:
            ev = p.get("evidence") or []
            if ev and ev[0]["segment_id"] == segment_id:
                out.add(len(ev))
    return out


# ------------------------------------------------------------------ the expected numbers of each kind of line
# Each function returns a list of (wanted, source). `wanted` is one exact token, or a frozenset when any of several tokens is right.
def E(text, source):
    return [(t, source) for t in tok(text)]


class Context:
    def __init__(self, payload, previous_payloads, records, dashboard=None):
        self.payload, self.previous_payloads, self.records = payload, list(previous_payloads), records or {}
        self.dashboard = dashboard          # {"url", "label"} from report_config, or None: the only thing on a link line that is not the payload's
        prev = self.previous_payloads[-1] if self.previous_payloads else None
        self.prev = prev if prev is not None and month_index(prev) == month_index(payload) - 1 else None
        self.entries = {e["product_id"]: e for e in payload["watchlist"]["products"]}
        bvc = payload["brand_vs_category"]
        self.brand_rows = {r["product_id"]: r for r in bvc["largest_below_category"] + bvc["largest_above_category"]}
        self.themes = {g["theme_group"]: g for g in payload["complaint_movement"]["theme_groups"]}
        self.section_titles_seen = 0
        self.label_names = {}       # product id -> names of the label lines of its block, in order

    def entry(self, line):
        e = self.entries.get(line.get("product_id"))
        if e is None:
            raise Refuse(f"product {line.get('product_id')} is not on this month's watchlist")
        return e

    def record(self, line):
        rec = self.records.get(line.get("product_id"))
        if rec is None:
            raise Refuse(f"no model records were loaded for product {line.get('product_id')}")
        return rec


def x_header(c, line):
    return E(c.payload["envelope"]["as_of_month_display"], "payload:envelope.as_of_month_display")


def x_dashboard_link(c, line):
    """The optional link line: its label is a setting, so the numbers it may carry (a year, say) are the setting's own."""
    if c.dashboard is None:
        raise Refuse("a dashboard link line is on the message but no dashboard is set in report_config")
    return E(c.dashboard["label"], "config:report_config.DASHBOARD_LABEL")


def x_top_reviews(c, line):
    h = c.payload["header"]
    out = E(h["asof_reviews_display"], "payload:header.asof_reviews_display")
    if h["prior_month_reviews"] is not None:
        pct = pct_text(h["asof_reviews"] - h["prior_month_reviews"], h["prior_month_reviews"])
        out += [(pct, "recomputed:(this month - last month) / last month, header.asof_reviews and prior_month_reviews")] if pct \
            else E(h["prior_month_reviews_display"], "payload:header.prior_month_reviews_display")
    return out


def x_top_evaluated(c, line):
    cv = c.payload["coverage"]
    out = [(review_gate(c.payload) + "+", "payload:coverage.caveats[review_gate] (the threshold in 'fewer than N reviews')")]
    out += E(cv["asof_products_evaluated_display"], "payload:coverage.asof_products_evaluated_display")
    out += E(cv["asof_products_with_reviews_display"], "payload:coverage.asof_products_with_reviews_display")
    if c.payload["header"]["prior_month_reviews"] is not None:
        out += E(cv["prior_products_evaluated_display"], "payload:coverage.prior_products_evaluated_display")
        out += E(cv["prior_products_with_reviews_display"], "payload:coverage.prior_products_with_reviews_display")
    return out


def x_top_counts(c, line):
    ws, cm = c.payload["watchlist"]["summary"], c.payload["complaint_movement"]
    pws = c.prev["watchlist"]["summary"] if c.prev else None
    pcm = c.prev["complaint_movement"] if c.prev else None
    out = E(ws["tier12_flagged_count_display"], "payload:watchlist.summary.tier12_flagged_count_display")
    if pws:
        out += E(pws["tier12_flagged_count_display"], "payload(last month):watchlist.summary.tier12_flagged_count_display")
    out += E(ws["tier3_dip_only_count_display"], "payload:watchlist.summary.tier3_dip_only_count_display")
    if pws:
        out += E(pws["tier3_dip_only_count_display"], "payload(last month):watchlist.summary.tier3_dip_only_count_display")
    if cm["available"]:
        out += E(cm["reported_rising_count_display"], "payload:complaint_movement.reported_rising_count_display")
        if pcm and pcm["available"]:
            out += E(pcm["reported_rising_count_display"], "payload(last month):complaint_movement.reported_rising_count_display")
    return out


def x_section_title(c, line):
    c.section_titles_seen += 1
    return [(t, "template:the unit note, printed on the first section only") for t in UNIT_NOTE] if c.section_titles_seen == 1 else []


def x_row_name(c, line):
    e = c.entry(line)
    for field in ("brand_name_ko", "product_name_ko", "category_name"):
        if e[field] not in line["text"]:
            raise Refuse(f"the line does not name product {e['product_id']}'s {field}")
    return E(e["brand_name_ko"], "payload:watchlist.brand_name_ko") + E(e["product_name_ko"], "payload:watchlist.product_name_ko") \
        + E(e["category_name"], "payload:watchlist.category_name")


def x_row_numbers(c, line):
    e = c.entry(line)
    m = e["month_over_month"]
    if not m["available"]:
        out = E(e["net_sentiment_display"], "payload:net_sentiment_display") + E(e["review_count_display"], "payload:review_count_display")
        if m["unavailable_reason"] in BELOW_GATE_REASONS:
            out.append((review_gate(c.payload), "payload:coverage.caveats[review_gate] (the threshold in 'fewer than N reviews')"))
        return out
    out = E(m["prior_net_sentiment_display"], "payload:month_over_month.prior_net_sentiment_display")
    out += E(e["net_sentiment_display"], "payload:net_sentiment_display")
    out += E(m["delta_display"], "payload:month_over_month.delta_display")
    pct = sentiment_pct_text(m["delta"], m["prior_net_sentiment"])
    if pct:
        out.append((pct, "recomputed:delta / |last month's level|, month_over_month.delta and prior_net_sentiment"))
    return out + E(e["review_count_display"], "payload:review_count_display")


def x_flag(c, line):
    e = c.entry(line)
    n = e["consecutive_months_flagged_tier_1_2"]
    out = [] if n == 1 else E(e["consecutive_months_flagged_tier_1_2_display"], "payload:consecutive_months_flagged_tier_1_2_display")
    return out + E(e["decline_since_display"], "payload:decline_since_display")


def x_block_head(c, line):
    rec = c.record(line)
    names = c.label_names.get(line["product_id"], [])
    counts = label_counts(rec)
    if any(n not in counts for n in names):
        raise Refuse("a label line names a label the model's answer does not have")
    shown = sorted((counts[n] for n in names), reverse=True)
    others = sorted((v for k, v in counts.items() if k not in names and v > 0), reverse=True)
    if others and shown and others[0] > shown[-1]:
        raise Refuse(f"'top {len(names)} labels' is wrong: a label that is not shown has {others[0]} texts, a shown one {shown[-1]}")
    return [(str(len(current_ids(rec))), "recomputed:this-month texts in the case file"),
            (str(len(names)), "recomputed:label lines printed under this head"),
            (str(sum(counts[n] for n in names)), "recomputed:sum of the shown labels' texts, from the raw assignments")]


def x_block_label(c, line):
    rec = c.record(line)
    counts = label_counts(rec)
    if line["label"] not in counts or not line["text"].startswith(f"• {line['label']} — "):
        raise Refuse("the line does not carry a label of the model's answer")
    return [(str(counts[line["label"]]), "recomputed:texts given this label, from the raw assignments"),
            (str(len(current_ids(rec))), "recomputed:this-month texts in the case file")]


def x_block_point(c, line):
    """A 'worth a look' quote line: the id of a this-month text, then, when the point cites other texts too, '+N more' (N = the texts cited beyond
    the one quoted, recomputed from the model's raw answer). A line that says nothing about others must have a point that cites only that text."""
    rec = c.record(line)
    counts = points_citing(rec, line["segment_id"])
    if not counts:
        raise Refuse("no 'worth a look' point in the model's answer starts from the quoted text")
    out = [(frozenset(str(i) for i in current_ids(rec)), "case:id of a this-month text")]
    if len(tok(outside_quotes(line["text"])[0])) > 1:
        more = frozenset(f"+{k - 1}" for k in counts if k > 1)
        if not more:
            raise Refuse("the line says 'more reviews' but no point starting from the quoted text cites more than one")
        out.append((more, "recomputed:evidence texts cited by the point beyond the quoted one, from the model's raw answer"))
    elif 1 not in counts:
        raise Refuse("the line says nothing about other reviews, but every point starting from the quoted text cites more than one")
    return out


def x_brand_title(c, line):
    b = c.payload["brand_vs_category"]
    return [(t, "template:the unit note") for t in UNIT_NOTE] + E(b["window_months_display"], "payload:brand_vs_category.window_months_display") \
        + E(b["window_display"], "payload:brand_vs_category.window_display")


def brand_row(c, line):
    r = c.brand_rows.get(line.get("product_id"))
    if r is None:
        raise Refuse(f"product {line.get('product_id')} is not in this month's Products vs category rows")
    return r


def x_brand_name(c, line):
    r = brand_row(c, line)
    for field in ("brand_name_ko", "product_name_ko", "category_name"):
        if r[field] not in line["text"]:
            raise Refuse(f"the line does not name product {r['product_id']}'s {field}")
    return E(r["brand_name_ko"], "payload:brand_vs_category.brand_name_ko") + E(r["product_name_ko"], "payload:brand_vs_category.product_name_ko") \
        + E(r["category_name"], "payload:brand_vs_category.category_name")


def x_brand_numbers(c, line):
    r = brand_row(c, line)
    out = E(r["product_mean_display"], "payload:brand_vs_category.product_mean_display")
    out += E(r["category_mean_display"], "payload:brand_vs_category.category_mean_display")
    out += E(r["gap_display"], "payload:brand_vs_category.gap_display")
    pct = pct_text(r["gap"], r["category_mean"])
    return out + ([(pct, "recomputed:gap / category average, brand_vs_category.gap and category_mean")] if pct else [])


def x_mix_title(c, line):
    cm = c.payload["complaint_movement"]
    if not cm["available"]:
        return []
    return E(str(cm["half_months"]), "payload:complaint_movement.half_months") + E(cm["recent_window_display"], "payload:complaint_movement.recent_window_display") \
        + E(cm["prior_window_display"], "payload:complaint_movement.prior_window_display")


def theme(c, line):
    g = c.themes.get(line.get("theme"))
    if g is None or not g["reported_as_rising"]:
        raise Refuse(f"'{line.get('theme')}' is not a theme the payload reports as rising")
    return g


def x_mix_row(c, line):
    g = theme(c, line)
    n = streak(g["theme_group"], c.payload, c.previous_payloads)
    out = E(g["theme_group"], "payload:theme_group") + E(g["share_prior_display"], "payload:share_prior_display") \
        + E(g["share_recent_display"], "payload:share_recent_display") + E(g["shift_display"], "payload:shift_display")
    if n >= 2:                                                  # "· 2nd month in a row": only for a theme that was rising last month too
        out.append((ordinal(n), "recomputed:consecutive months rising, counted over the saved payloads"))
    return out


def x_mix_example(c, line):
    g = theme(c, line)
    e = next((x for x in c.payload["evidence_phrases"] if x["theme_group"] == g["theme_group"] and x.get("phrase_available")), None)
    if e is None:
        raise Refuse(f"the payload has no example review for '{g['theme_group']}'")
    return E(e["brand_name_ko"], "payload:evidence_phrases.brand_name_ko") + E(e["product_name_ko"], "payload:evidence_phrases.product_name_ko") \
        + E(e["review_month_display"], "payload:evidence_phrases.review_month_display")


def x_mix_none(c, line):
    return E(c.payload["complaint_movement"]["materiality_floor_display"], "payload:complaint_movement.materiality_floor_display")


def x_mix_unavailable(c, line):
    return [(t, "template:the fixed 'not available' sentence") for t in MIX_UNAVAILABLE_NUMBERS]


def x_truncated(c, line):
    ws = c.payload["watchlist"]
    return [(str(len(ws["products"])), "recomputed:watchlist products listed")] + E(ws["summary"]["total_flagged_display"], "payload:watchlist.summary.total_flagged_display")


def none(c, line):
    return []


EXPECT = {"header": x_header, "dashboard_link": x_dashboard_link, "top_reviews": x_top_reviews, "top_evaluated": x_top_evaluated, "top_counts": x_top_counts,
          "quiet": none, "section_title": x_section_title, "row_name": x_row_name, "row_numbers": x_row_numbers, "flag": x_flag,
          "block_head": x_block_head, "block_label": x_block_label, "block_label_quote": none, "block_heading": none,
          "block_point": x_block_point, "brand_title": x_brand_title, "brand_name": x_brand_name,
          "brand_numbers": x_brand_numbers, "brand_none": none, "mix_title": x_mix_title, "mix_row": x_mix_row, "mix_example": x_mix_example,
          "mix_none": x_mix_none, "mix_unavailable": x_mix_unavailable, "truncated": x_truncated}


# ------------------------------------------------------------------ the gate
def matches(printed, wanted):
    return printed in wanted if isinstance(wanted, frozenset) else printed == wanted


def check_line(index, line, ctx):
    """One line's verdict: {"index", "kind", "printed", "expected", "ok", "problem"}."""
    kind = line["kind"]
    res = {"index": index, "kind": kind, "text": line["text"], "printed": [], "expected": [], "ok": False, "problem": None}
    try:
        if kind not in EXPECT:
            raise Refuse(f"the gate has no rule for a line of kind '{kind}'")
        text = outside_quotes(line["text"])[0] if kind in QUOTE_KINDS else line["text"]
        res["printed"] = tok(text)
        res["expected"] = EXPECT[kind](ctx, line)
    except Refuse as why:
        res["problem"] = str(why)
        return res
    if len(res["printed"]) != len(res["expected"]):
        res["problem"] = f"{len(res['printed'])} number(s) printed, {len(res['expected'])} expected"
    else:
        bad = [(p, w) for p, (w, _) in zip(res["printed"], res["expected"]) if not matches(p, w)]
        if bad:
            res["problem"] = "printed " + repr(bad[0][0]) + " where the data gives " + (
                "one of the this-month text ids" if isinstance(bad[0][1], frozenset) else repr(bad[0][1]))
    res["ok"] = res["problem"] is None
    return res


def check_message_counts(lines, payload):
    """What the top row says about the rows below it: the counts of at-risk products, dips and rising themes match the lines shown."""
    ws = payload["watchlist"]["summary"]
    at_risk = sum(1 for ln in lines if ln["kind"] == "row_name" and payload_tier(payload, ln["product_id"]) in (1, 2))
    dips = sum(1 for ln in lines if ln["kind"] == "row_name" and payload_tier(payload, ln["product_id"]) == 3)
    out = []
    if not ws["truncated"]:
        if at_risk != ws["tier12_flagged_count"]:
            out.append(f"the top row counts {ws['tier12_flagged_count']} at risk but {at_risk} are shown")
        if dips != ws["tier3_dip_only_count"]:
            out.append(f"the top row counts {ws['tier3_dip_only_count']} dips but {dips} are shown")
    cm = payload["complaint_movement"]
    if cm["available"]:
        shown = sum(1 for ln in lines if ln["kind"] == "mix_row")
        if shown != cm["reported_rising_count"]:
            out.append(f"the top row counts {cm['reported_rising_count']} rising themes but {shown} are shown")
    return out


def payload_tier(payload, product_id):
    for e in payload["watchlist"]["products"]:
        if e["product_id"] == product_id:
            return e["highest_tier"]
    raise Refuse(f"product {product_id} is not on this month's watchlist")


def check(lines, payload, previous_payloads=(), records=None, dashboard=None):
    """Run the gate. lines: the message's tagged lines. records: {product_id: {"case", "readout", "labeling"}} for the products with a block.
    dashboard: {"url", "label"} from report_config when a link line is expected, else None.
    Returns {"passed", "tokens", "lines": [...], "failures": [...]}. passed means every number of every line is explained."""
    ctx = Context(payload, previous_payloads, records, dashboard)
    for ln in lines:                                            # the block's label names, gathered before its head line is checked
        if ln["kind"] == "block_label":
            ctx.label_names.setdefault(ln["product_id"], []).append(ln["label"])
    results = [check_line(i, ln, ctx) for i, ln in enumerate(lines)]
    failures = [f"line {r['index']} ({r['kind']}): {r['problem']}" for r in results if not r["ok"]]
    try:
        failures += check_message_counts(lines, payload)
    except Refuse as why:
        failures.append(str(why))
    return {"passed": not failures, "tokens": sum(len(r["printed"]) for r in results), "lines": results, "failures": failures}


def explanation_table(result):
    """The gate's working as text: for every line that carries numbers, each printed number and where it came from."""
    out = []
    for r in result["lines"]:
        if not r["printed"] and not r["problem"]:
            continue
        head = f"{r['index']:>3} {r['kind']:<17} {'PASS' if r['ok'] else 'FAIL'}  {r['text'][:70]}"
        out.append(head)
        if r["ok"]:
            for p, (w, source) in zip(r["printed"], r["expected"]):
                out.append(f"      {p:>10}  <- {source}")
        else:
            out.append(f"      {r['problem']}")
            out.append(f"      printed:  {r['printed']}")
            out.append(f"      expected: {[w if not isinstance(w, frozenset) else 'a this-month text id' for w, _ in r['expected']]}")
    out += ["", f"{result['tokens']} numbers read; {'all explained' if result['passed'] else str(len(result['failures'])) + ' failure(s)'}"]
    out += [f"  FAILURE: {f}" for f in result["failures"]]
    return "\n".join(out)
