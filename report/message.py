"""The monthly report as one object, and its plain-text and Markdown forms.

`build_message` puts together everything the message holds, from the saved payloads and, for the complaint blocks, the saved model records:

    header · the two big-number lines and the count row · [the quiet-month line] ·
    At risk (rows, each with its flag line and, for the first READOUT_MAX_PRODUCTS, its complaint block) · Dips ·
    Products vs category · Complaint mix shift

Nothing here computes a metric. Each part comes from its own module (system_rows, top_line, flag_line, complaint_block, brand_block,
mix_shift), and each part's own verifier runs; their problems are collected in `message["checks"]["problems"]` for the gates (gates.py) to read.
The Slack Block Kit form (block_kit.py) is made from the same object, so the two outputs cannot say different things.

A flagged tier 1-2 product whose records are missing stops the build (records.MissingRecords): a silent gap in a replay is what the report is
built to avoid. `require_blocks=False` builds the message without blocks, for previews only, and says so in the checks.
The one exception is the block rule (MIN_COMPLAINT_MONTHS_FOR_BLOCK): a month whose window holds a single complaint month (2025-07) gets no
complaint blocks and needs no records; the checks' notes say so (the message does not).

No model call, no database, no network, no file writes.
"""
import re

import brand_block as bb
import complaint_block as cb
import flag_line as fl
import mix_shift as ms
import records as rc
import report_config as cfg
import system_rows as sr
import top_line as tl

READOUT_MAX_PRODUCTS = cb.KNOBS["READOUT_MAX_PRODUCTS"]
UNSET = object()          # "read the setting from report_config"; a test may pass its own (or None for no link)
# A complaint block says what the flagged product's reviewers complain about this month, and its readout says what looks new against the earlier
# complaint months. A month whose window holds fewer complaint months than this has nothing to compare with, so it shows no complaint block
# (the row and the flag line stay; the message says nothing about the gap) and needs no model records.
MIN_COMPLAINT_MONTHS_FOR_BLOCK = 2


def build_message(payload, previous_payloads=(), store=None, strict_records=True, require_blocks=True, dashboard=UNSET):
    """The message as data. previous_payloads: the saved payloads of the earlier months, oldest first (any number, or none).
    dashboard: {"url", "label"} for the optional link line under the header, None for none; by default report_config's (none until a URL is set there)."""
    dash = cfg.dashboard() if dashboard is UNSET else dashboard
    prev = previous_payloads[-1] if previous_payloads else None
    month = payload["envelope"]["as_of_month"][:7]
    problems, notes, stamps = [], [], {}
    parts = [{"type": "header", "text": sr.header_line(payload)}]
    if dash:
        parts.append({"type": "dashboard", "label": dash["label"], "url": dash["url"]})

    top = tl.build_top(payload, prev)
    problems += [f"top rows: {p}" for p in tl.verify_top(top, payload, prev)]
    parts.append({"type": "top", "top": top, "lines": tl.top_lines(top)})
    if payload["watchlist"]["summary"]["headline_is_quiet"]:
        parts.append({"type": "quiet", "text": sr.QUIET_LINE})

    built = sr.build_sections(payload)
    problems += [f"system rows: {p}" for p in sr.verify_sections(built, payload)]
    entries = {e["product_id"]: e for e in payload["watchlist"]["products"]}
    n_blocks = 0
    window_months = payload["coverage"]["complaint_months_in_window"]
    no_block_month = window_months < MIN_COMPLAINT_MONTHS_FOR_BLOCK
    for s in built["sections"]:
        rows = []
        for r in s["rows"]:
            e = entries[r["product_id"]]
            flag = block = None
            if r["tier"] in (1, 2):
                flag = fl.flag_line(e)
                problems += [f"flag line, product {r['product_id']}: {p}" for p in fl.verify_flag_line(flag, e)]
                if no_block_month:
                    notes.append(f"product {r['product_id']}: no complaint block (the window holds {window_months} complaint month(s); a block needs at least "
                                 f"{MIN_COMPLAINT_MONTHS_FOR_BLOCK} to compare)")
                elif n_blocks < READOUT_MAX_PRODUCTS:
                    n_blocks += 1
                    if store is None:
                        if require_blocks:
                            raise rc.MissingRecords(f"product {r['product_id']} in {month} is flagged and no record store was given")
                        notes.append(f"product {r['product_id']}: no complaint block (built without records)")
                    else:
                        loaded = store.load(month, r["product_id"], strict=strict_records)
                        block = cb.build_block(loaded["case"], loaded["readout"], loaded["labeling"])
                        problems += [f"complaint block, product {r['product_id']}: {p}" for p in cb.verify_shown(block, loaded["case"])]
                        problems += [f"records, product {r['product_id']}: {p}" for p in loaded["problems"]]
                        stamps[r["product_id"]] = rc.stamps(loaded)
                else:
                    notes.append(f"product {r['product_id']}: named without a block (more than {READOUT_MAX_PRODUCTS} flagged products)")
            rows.append({"row": r, "flag_line": flag, "block": block})
        parts.append({"type": "section", "key": s["key"], "color": s["color"], "title": s["title"], "context": sr.context_text(s), "rows": rows})
    if built["truncated"]:
        parts.append({"type": "truncated", "text": sr.truncated_line(built["truncated"])})

    brand = bb.build_section(payload["brand_vs_category"])
    problems += [f"products vs category: {p}" for p in bb.verify_section(brand, payload["brand_vs_category"])]
    parts.append({"type": "brand", "section": brand})

    mix = ms.build_section(payload, list(previous_payloads))
    problems += [f"complaint mix shift: {p}" for p in ms.verify_section(mix, payload)]
    parts.append({"type": "mix", "section": mix})
    return {"as_of": month, "parts": parts, "checks": {"problems": problems, "notes": notes}, "records": stamps}


# ---------------------------------------------------------------- plain text
def _line(kind, text, **meta):
    return {"kind": kind, "text": text, **meta}


def _checked(tagged, lines, what):
    """The tags describe the very lines each module prints; if a module's own text ever stops matching, stop here."""
    if [t["text"] for t in tagged] != list(lines):
        raise ValueError(f"the tagged lines of {what} differ from the module's own lines")
    return tagged


def block_tagged(block, product_id):
    """The complaint block's lines, each with its kind. Label lines carry the label's name; point lines the quoted segment's id."""
    lines = cb.block_lines(block)
    out = [_line("block_head", lines[0], product_id=product_id)]
    i = 1
    for r in block["labels"]:
        out += [_line("block_label", lines[i], product_id=product_id, label=r["name"]),
                _line("block_label_quote", lines[i + 1], product_id=product_id)]
        i += 2
    if block["points"]:
        out.append(_line("block_heading", lines[i], product_id=product_id))
        i += 1
        for p in block["points"]:
            out.append(_line("block_point", lines[i], product_id=product_id, segment_id=p["quote_segment_id"]))
            i += 1
    return _checked(out, lines, "a complaint block")


def brand_tagged(sec):
    lines = bb.section_lines(sec)
    out = [_line("brand_title", lines[0])]
    for r in sec["rows"]:
        out += [_line("brand_name", bb.name_line(r), product_id=r["product_id"]),
                _line("brand_numbers", bb.number_line(r), product_id=r["product_id"])]
    if sec["none_shown"]:
        out.append(_line("brand_none", bb.NONE_LINE))
    return _checked(out, lines, "Products vs category")


def mix_tagged(sec):
    lines = ms.section_lines(sec)
    if not sec["available"]:
        return _checked([_line("mix_title", lines[0]), _line("mix_unavailable", lines[1])], lines, "Complaint mix shift")
    out = [_line("mix_title", lines[0])]
    for r in sec["rows"]:
        out.append(_line("mix_row", ms.row_line(r), theme=r["theme"]))
        if r["example"]:
            out.append(_line("mix_example", ms.example_line(r), theme=r["theme"]))
    if sec["none_shown"]:
        out.append(_line("mix_none", sec["none_line"]))
    return _checked(out, lines, "Complaint mix shift")


def part_tagged(part):
    """The part's lines as dicts: {"kind", "text", ...}. `kind` says what the line is (the number gate checks each kind against its own
    source); product_id, label, segment_id, theme say which entry it belongs to. The plain text is these lines' texts and nothing else."""
    t = part["type"]
    if t in ("header", "quiet", "truncated"):
        return [_line(t, part["text"])]
    if t == "dashboard":
        return [_line("dashboard_link", part["label"])]              # the address is not a line of words: gates.py checks it against report_config
    if t == "top":
        assert len(part["lines"]) == 3
        return [_line(k, x) for k, x in zip(("top_reviews", "top_evaluated", "top_counts"), part["lines"])]
    if t == "section":
        out = [_line("section_title", f"{part['title']} {part['context']}", section=part["key"])]
        for item in part["rows"]:
            r = item["row"]
            pid = r["product_id"]
            out += [_line("row_name", sr.name_line(r), product_id=pid), _line("row_numbers", sr.number_line(r), product_id=pid)]
            if item["flag_line"]:
                out.append(_line("flag", item["flag_line"], product_id=pid))
            if item["block"]:
                out += block_tagged(item["block"], pid)
        return out
    if t == "brand":
        return brand_tagged(part["section"])
    if t == "mix":
        return mix_tagged(part["section"])
    raise ValueError(f"unknown part {t!r}")


def part_lines(part):
    return [t["text"] for t in part_tagged(part)]


def tagged_lines(message):
    return [t for part in message["parts"] for t in part_tagged(part)]


def message_lines(message):
    return [t["text"] for t in tagged_lines(message)]


def to_text(message):
    return "\n".join(message_lines(message)) + "\n"


def message_words(message):
    return sr.word_count(message_lines(message))


def block_words(message):
    """Words of each complaint block in the message, in order."""
    return [cb.word_count(cb.block_lines(item["block"])) for p in message["parts"] if p["type"] == "section" for item in p["rows"] if item["block"]]


# ---------------------------------------------------------------- Markdown
MD_SPECIAL = "\\`*_[]<>|~"


def md_escape(text):
    """Review text, label names and product names come from outside this code: their Markdown characters must not become markup."""
    return "".join("\\" + c if c in MD_SPECIAL else c for c in text)


def _bold_after(line, label, value):
    old = f"{label} {value}"
    assert line.count(old) == 1, f"cannot find {old!r} in {line!r}"
    return line.replace(old, f"{label} **{value}**", 1)


def md_top(part):
    top = part["top"]
    out = [_bold_after(tl.reviews_line(top), tl.REVIEWS_LABEL, top["reviews"]["value"]),
           _bold_after(tl.evaluated_line(top), tl.EVALUATED_LABEL, top["evaluated"]["value"])]
    counts = []
    for c in top["counts"]:
        if c["n"] is None:
            counts.append(f"{c['emoji']} {c['label']}")
        else:
            tail = f" ({top['prior_month_word']}: {c['prior']})" if c["prior"] is not None else ""
            counts.append(f"{c['emoji']} **{c['n']}** {c['label']}{tail}")
    out.append("&emsp;".join(counts))       # a wide space: the plain text separates the three counts with spaces
    return out


def md_row(item):
    r = item["row"]
    out = [f"{r['emoji']} **{md_escape(r['brand'])} · {md_escape(r['product'])}** *{md_escape(r['category'])}*"]
    if r["available"]:
        pct = f" ({r['pct']})" if r["pct"] else ""
        out.append(f"`{r['prior']}` → `{r['now']}` {r['arrow']} **{r['delta']}**{pct} · {r['reviews']}")
    else:
        out.append(f"`{r['now']}` · {r['reviews']} · no comparison: {r['no_comparison']}")
    if item["flag_line"]:
        out.append(f"*{item['flag_line']}*")
    return out


def md_block(b):
    n = b["n_current"]
    out = [f"**What reviewers say** — {n} complaint texts this month · top {len(b['labels'])} labels cover {b['top_covers']}", ""]
    for r in b["labels"]:
        out += [f"- **{md_escape(r['name'])}** — `{r['count']}` of {n}", f"  > “{md_escape(r['quote'])}”"]
    if b["points"]:
        out += ["", "**Worth a look**", ""]
        for p in b["points"]:
            out += [f"> “{md_escape(p['quote'])}” {cb.id_note(p['quote_segment_id'], len(p['cited_segment_ids']))}", ""]
    return out


def md_brand(sec):
    out = [f"### {sec['title']}", f"*{sec['context']}*", ""]
    for r in sec["rows"]:
        pct = f" ({r['pct']})" if r["pct"] else ""
        out += [f"**{md_escape(r['brand'])} · {md_escape(r['product'])}** *{md_escape(r['category'])}*  ", f"`{r['product_mean']}` vs `{r['category_mean']}` {r['arrow']} **{r['gap']}**{pct}", ""]
    if sec["none_shown"]:
        out += [bb.NONE_LINE, ""]
    return out


def md_mix(sec):
    out = [f"### {sec['title']}"]
    if not sec["available"]:
        return out + ["", sec["unavailable_line"], ""]
    out += [f"*{sec['header']}*", ""]
    for r in sec["rows"]:
        out += [f"**{md_escape(r['theme'])}** `{r['share_prior']}` → `{r['share_recent']}` {ms.ARROW_UP} **{r['shift']}**{ms.month_tag(r)}  "]
        if r["example"]:
            out += [f"> {md_escape(ms.example_line(r))}", ""]
    if sec["none_shown"]:
        out += [sec["none_line"], ""]
    return out


def to_markdown(message):
    out = []
    for part in message["parts"]:
        t = part["type"]
        if t == "header":
            out += [f"## {part['text']}", ""]
        elif t == "top":
            out += [line + "  " for line in md_top(part)[:-1]] + [md_top(part)[-1], ""]
        elif t in ("quiet", "truncated"):
            out += [part["text"], ""]
        elif t == "dashboard":
            out += [f"[{md_escape(part['label'])}]({part['url']})", ""]
        elif t == "section":
            out += [f"### {part['title']}", f"*{part['context']}*", ""]
            for item in part["rows"]:
                lines = md_row(item)
                out += [ln + "  " for ln in lines[:-1]] + [lines[-1], ""]
                if item["block"]:
                    out += md_block(item["block"])
        elif t == "brand":
            out += md_brand(part["section"])
        elif t == "mix":
            out += md_mix(part["section"])
        else:
            raise ValueError(f"unknown part {t!r}")
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def strip_markdown(text):
    """The Markdown with its markup removed and its escapes undone, for checking that it says what the plain text says."""
    keep = {c: chr(0xE000 + ord(c)) for c in MD_SPECIAL}
    text = re.sub("\\\\([" + re.escape(MD_SPECIAL) + "])", lambda m: keep[m.group(1)], text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)                 # a link is its label
    text = re.sub(r"(?m)^#{1,6} ", "", text)
    text = re.sub(r"(?m)^\s*> ?", "", text)
    text = re.sub(r"(?m)^\s*- ", "", text)
    text = text.replace("**", "").replace("*", "").replace("`", "").replace("&emsp;", " ")
    for c, private in keep.items():
        text = text.replace(private, c)
    return text
