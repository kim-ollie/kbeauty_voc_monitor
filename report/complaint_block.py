"""Complaint block for the monthly report.

Pure functions: no model call, no database, no network, no file writes.

Inputs are the two saved model responses for one flagged product-month and the
case file that holds the texts the model read:

  readout  - Arm A, schema probe-readout-v0 (prose points with verbatim quotes)
  labeling - Arm B, schema probe-labeling-v0 (up to 12 labels, one label per text)
  case     - probe-input-v0 (segments: segment_id, period, unclustered, text ...)

The block shows, for one flagged product:
  * the top LABEL_ROWS_TOP_N labels by count (this month only), each with one
    quote, and how many of the month's texts those labels cover;
  * up to READOUT_POINTS_MAX "worth a look" items from Arm A's "what looks new"
    and "in unclustered segments" sections. Each is shown as the reviewer's own
    words: the first cited quote and its id, followed, when the point cites more reviews
    than the one quoted, by "+N more review(s)" (made here, counted from the ids). The model's claim and its English gloss stay in the
    saved record and are not shown (a claim on one review is
    only a paraphrase of it, and a comparison with earlier months cannot be checked
    by the manager).

Every number in the block is made here, by counting. The model wrote none, and
the only model-written words shown are the label names.
"""
from collections import Counter, OrderedDict

BLOCK_VERSION = "complaint-block-v4"

# Layout knobs. Echoed into every block so a saved
# block always records the settings that produced it.
KNOBS = {
    "READOUT_MAX_PRODUCTS": 2,       # used by the caller: blocks per message
    "LABEL_ROWS_TOP_N": 3,
    "READOUT_POINTS_MAX": 4,
    "QUOTE_DISPLAY_MAX_CHARS": 140,
    "POINT_RANK": "new_first",           # "new_first": what the model marked new first, then more cited texts, then model order
                                         # "label_count": strongest label first; "order": as the model wrote them
    "POINT_SKIP_SHOWN_LABELS": True,     # a point on a label that is already a row adds nothing new
    "POINT_ONE_PER_REVIEW": True,        # the same review is never shown twice: not by two items (different lines of one review), not as a row quote and an item
    "POINT_ONE_PER_LABEL": False,        # off: an item is one review, so two items on one label are two reviews, not the same claim twice
}

# Arm A's "what_reviewers_say" is not shown: the label rows now say that, with counts.
POINT_SECTIONS = ("what_looks_new", "in_unclustered_segments")
ALL_SECTIONS = ("what_reviewers_say", "what_looks_new", "in_unclustered_segments")
LINK_MAX_LABELS_SHOWN = 3


# ---------------------------------------------------------------- indexing

def index_case(case):
    """segment_id -> segment dict. Fails on a duplicate id."""
    by_id = {}
    for s in case["segments"]:
        sid = s["segment_id"]
        if sid in by_id:
            raise ValueError(f"duplicate segment_id {sid}")
        by_id[sid] = s
    return by_id


def current_label_counts(labeling, segments_by_id):
    """Counter: label_id -> number of THIS-MONTH texts given that label.

    An assignment that names a segment the case does not contain is an error
    (the production S2 check rejects such a response before this runs).
    """
    counts = Counter()
    for a in labeling["assignments"]:
        seg = segments_by_id.get(a["segment_id"])
        if seg is None:
            raise ValueError(f"assignment for unknown segment_id {a['segment_id']}")
        if seg["period"] == "current":
            counts[a["label_id"]] += 1
    return counts


# ---------------------------------------------------------------- quotes

def cut_quote(quote, max_chars):
    """Shorten for display. A cut quote ends with an ellipsis; the shown part is
    still a contiguous piece of the original, so it stays verbatim."""
    if len(quote) <= max_chars:
        return quote
    return quote[: max_chars - 1] + "…"


def first_line_excerpt(text, max_chars):
    """First non-empty line of a text, cut for display (contiguous, verbatim)."""
    for line in text.split("\n"):
        if line.strip():
            return cut_quote(line.strip(), max_chars)
    return ""


# ---------------------------------------------------------------- label rows

def top_label_rows(labeling, counts, segments_by_id, n, max_chars):
    """The n most common labels this month; ties go to the lower label_id.

    The quote is the label's own example when that example is a this-month text.
    The model often picks an example from the prior-period sample (in the probe:
    4 of 12 in case 1), which would put last months' words under a "this month"
    heading. In that case the block falls back to the first this-month text
    (lowest segment_id) given that label, and says so in quote_source.
    """
    by_id = {l["label_id"]: l for l in labeling["labels"]}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    current_by_label = {}
    for a in labeling["assignments"]:
        seg = segments_by_id[a["segment_id"]]
        if seg["period"] == "current":
            current_by_label.setdefault(a["label_id"], []).append(a["segment_id"])
    rows = []
    for label_id, count in ranked[:n]:
        label = by_id[label_id]
        ex_id = label["example_segment_id"]
        ex_seg = segments_by_id.get(ex_id)
        if ex_seg is not None and ex_seg["period"] == "current":
            quote = cut_quote(label["example_quote"], max_chars)
            quote_id, source = ex_id, "label_example"
        else:
            quote_id = min(current_by_label[label_id])
            quote = first_line_excerpt(segments_by_id[quote_id]["text"], max_chars)
            source = "code_pick"
        rows.append({
            "label_id": label_id, "name": label["name"], "count": count,
            "quote": quote, "quote_segment_id": quote_id, "quote_source": source,
        })
    return rows


# ---------------------------------------------------------------- points

SECTION_RANK = {sec: i for i, sec in enumerate(POINT_SECTIONS)}


def select_points(readout, labeling, counts, n_current, shown_label_ids, knobs, shown_review_ids=()):
    """Choose the points to show. Returns (chosen, n_eligible).

    Eligible: points in "what looks new" and "in unclustered segments" (the label
    rows already say what reviewers say). Then, by the knobs:
      * drop a point whose main label is already shown as a row;
      * rank. POINT_RANK "new_first" (the default): "what looks new" before
        "unclustered", then the point citing more texts, then the model's order.
        The size of a point's label plays no part: for a point resting on one review
        it says nothing about the point. "label_count" ranks by strength (the
        this-month count of the point's main label; ties as above); "order" keeps
        the model's order;
      * keep one point per label (off by default);
      * POINT_ONE_PER_REVIEW: drop a point whose quoted review (the first one it cites) is already shown, as a
        label row's quote (shown_review_ids) or by a better-ranked item. Two points can cite different lines of one
        review; showing it twice would look like two reviews;
      * cap at READOUT_POINTS_MAX.
    Each chosen item is {"sec", "p", "link", "order"}.
    """
    pool = []
    for sec in POINT_SECTIONS:
        for p in readout.get(sec, []):
            pool.append({"sec": sec, "p": p, "order": len(pool),
                         "link": link_point(p["evidence"], labeling, counts, n_current)})
    n_eligible = len(pool)

    def main_label(e):
        return e["link"]["labels"][0]["label_id"] if e["link"]["labels"] else None

    def strength(e):
        return e["link"]["labels"][0]["count"] if e["link"]["labels"] else 0

    if knobs["POINT_SKIP_SHOWN_LABELS"]:
        pool = [e for e in pool if main_label(e) not in shown_label_ids]
    rank = knobs["POINT_RANK"]
    if rank == "new_first":
        pool = sorted(pool, key=lambda e: (SECTION_RANK[e["sec"]], -len(e["p"]["evidence"]), e["order"]))
    elif rank == "label_count":
        pool = sorted(pool, key=lambda e: (-strength(e), SECTION_RANK[e["sec"]], -len(e["p"]["evidence"]), e["order"]))
    elif rank != "order":
        raise ValueError(f"unknown POINT_RANK {rank!r}")
    if knobs["POINT_ONE_PER_LABEL"]:
        seen, kept = set(), []
        for e in pool:
            lid = main_label(e)
            if lid is not None and lid in seen:
                continue
            seen.add(lid)
            kept.append(e)
        pool = kept
    if knobs["POINT_ONE_PER_REVIEW"]:
        seen, kept = set(shown_review_ids), []
        for e in pool:
            rid = e["p"]["evidence"][0]["segment_id"]
            if rid in seen:
                continue
            seen.add(rid)
            kept.append(e)
        pool = kept
    return pool[: knobs["READOUT_POINTS_MAX"]], n_eligible


def link_point(evidence, labeling, counts, n_current):
    """Attach Arm B label counts to one Arm A point, using the segment ids it cites.

    kind 'one'  - every cited text has the same label
    kind 'span' - the cited texts carry more than one label; up to
                  LINK_MAX_LABELS_SHOWN are listed, most-cited first (ties: first cited)
    kind 'none' - none of the cited ids has an assignment
    """
    by_id = {l["label_id"]: l for l in labeling["labels"]}
    assigned = {a["segment_id"]: a["label_id"] for a in labeling["assignments"]}
    cited_labels = [assigned[e["segment_id"]] for e in evidence if e["segment_id"] in assigned]
    order = OrderedDict()
    for lid in cited_labels:
        order[lid] = order.get(lid, 0) + 1
    ranked = sorted(order.items(), key=lambda kv: -kv[1])   # stable: ties keep first-cited order
    labels = [{"label_id": lid, "name": by_id[lid]["name"], "count": counts.get(lid, 0)}
              for lid, _ in ranked[:LINK_MAX_LABELS_SHOWN]]
    kind = "none" if not order else ("one" if len(order) == 1 else "span")
    return {"kind": kind, "n_labels": len(order), "labels": labels, "n_current": n_current}


# ---------------------------------------------------------------- the block

def build_block(case, readout, labeling, knobs=None):
    knobs = {**KNOBS, **(knobs or {})}
    seg = index_case(case)
    counts = current_label_counts(labeling, seg)
    n_current = sum(1 for s in seg.values() if s["period"] == "current")
    rows = top_label_rows(labeling, counts, seg, knobs["LABEL_ROWS_TOP_N"],
                          knobs["QUOTE_DISPLAY_MAX_CHARS"])
    chosen, n_eligible = select_points(readout, labeling, counts, n_current,
                                       {r["label_id"] for r in rows}, knobs,
                                       {r["quote_segment_id"] for r in rows})
    points = []
    for e in chosen:
        sec, p = e["sec"], e["p"]
        ev = p["evidence"]
        first = ev[0]
        points.append({
            "section": sec,
            "claim": p["claim"],
            "basis": p.get("basis"),
            "quote": cut_quote(first["quote"], knobs["QUOTE_DISPLAY_MAX_CHARS"]),
            "quote_segment_id": first["segment_id"],
            "gloss": first.get("gloss", ""),
            "cited_segment_ids": [e["segment_id"] for e in ev],
            "link": e["link"],
        })
    block = {
        "version": BLOCK_VERSION,
        "knobs": knobs,
        "n_current": n_current,
        "n_current_assigned": sum(counts.values()),
        "labels": rows,
        "top_covers": sum(r["count"] for r in rows),
        "points": points,
        "points_eligible": n_eligible,
        "points_not_shown": n_eligible - len(points),
    }
    block["words"] = word_count(block_lines(block))
    return block


# ---------------------------------------------------------------- text form

def id_note(segment_id, n_cited):
    """The id printed after a "worth a look" quote, and what the item says about its own support: "(#123)" when it cites only that review,
    "(#123, +1 more review)" when it cites others too, counted from the ids."""
    more = n_cited - 1
    if more < 1:
        return f"(#{segment_id})"
    return f"(#{segment_id}, +{more} more review{'s' if more > 1 else ''})"


def block_lines(block):
    """The block as plain lines. Its word count is taken from these."""
    n = block["n_current"]
    k = len(block["labels"])
    lines = [f"What reviewers say — {n} complaint texts this month · top {k} labels cover {block['top_covers']}"]
    for r in block["labels"]:
        lines.append(f"• {r['name']} — {r['count']} of {n}")
        lines.append(f"“{r['quote']}”")
    if block["points"]:
        lines.append("Worth a look")
        for p in block["points"]:
            # the reviewer's own words, no claim and no gloss (both kept in the record)
            lines.append(f"“{p['quote']}” {id_note(p['quote_segment_id'], len(p['cited_segment_ids']))}")
    return lines


def word_count(lines):
    """Whitespace tokens, bullets not counted; numbers and Korean names count as words
    (the same convention as the other blocks)."""
    return sum(len([t for t in line.split() if t != "•"]) for line in lines)


# ---------------------------------------------------------------- checks

def verify_shown(block, case):
    """Problems with what the block would show; [] means all clear.

    Every quote must be a contiguous piece of one line of the text it cites, the
    cited id must exist and be a this-month text. (This is the S2 idea applied to
    what is shown; the gate for whole responses is in gates.py.)
    """
    seg = index_case(case)
    problems = []

    def check(where, quote, sid):
        s = seg.get(sid)
        if s is None:
            problems.append(f"{where}: segment {sid} not in case")
            return
        if s["period"] != "current":
            problems.append(f"{where}: segment {sid} is not a this-month text")
        q = quote[:-1] if quote.endswith("…") else quote
        if not any(q in line for line in s["text"].split("\n")):
            problems.append(f"{where}: quote is not a piece of one line of segment {sid}")

    for r in block["labels"]:
        check(f"label {r['label_id']}", r["quote"], r["quote_segment_id"])
    for i, p in enumerate(block["points"]):
        check(f"point {i}", p["quote"], p["quote_segment_id"])
    return problems


def link_stats(readout, labeling, case, sections=ALL_SECTIONS):
    """How many points cite texts that share one label: Counter of kinds.
    Used to check the design measurement; not part of the message."""
    seg = index_case(case)
    counts = current_label_counts(labeling, seg)
    n_current = sum(1 for s in seg.values() if s["period"] == "current")
    kinds = Counter()
    for sec in sections:
        for p in readout.get(sec, []):
            kinds[link_point(p["evidence"], labeling, counts, n_current)["kind"]] += 1
    return kinds


if __name__ == "__main__":
    import json
    import sys
    for arg in sys.argv[1:] or ["1", "2", "3", "4"]:
        c = int(arg)
        case = json.load(open(f"probe_inputs/case_{c}.json", encoding="utf-8"))
        A = json.load(open(f"probe_outputs/probe_output_case{c}_A_prompt-v0_r1.json", encoding="utf-8"))["parsed"]
        B = json.load(open(f"probe_outputs/probe_output_case{c}_B_prompt-v0_r1.json", encoding="utf-8"))["parsed"]
        blk = build_block(case, A, B)
        print(f"=== case {c}: {blk['words']} words, {blk['points_not_shown']} eligible points not shown, "
              f"problems: {verify_shown(blk, case) or 'none'}")
        print("\n".join(block_lines(blk)))
        print()
