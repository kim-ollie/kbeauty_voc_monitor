"""The Slack Block Kit form of the monthly message, with colour bars as `attachments`.

`to_block_kit(message)` turns the message object of message.py into the JSON an incoming webhook takes:

    {"text": ..., "blocks": [header, (dashboard link), top numbers, (quiet line)], "attachments": [{"color", "fallback", "blocks"}, ...]}

Top level: the header, the optional dashboard link (one small `context` line), the top lines (the two big numbers, then a blank line, then one count
per line) and, in a quiet month, the quiet line. One attachment per section, each with its colour bar
(at risk red, dips yellow, Products vs category grey, Complaint mix shift orange).

LAYOUT. Slack has no font sizes: a `header` block is the large bold text and a `context` block is the small
grey text, so each section's name is a header block and its caption a context block under it ("What reviewers say" and "Worth a look" likewise, and the flag
line under an at-risk row is a context block). `section_style="bold"` is the fallback if Slack does not show a header block inside an attachment: the name
is a bold line at body size and the captions are still small. The words are the same in every style; only their blocks differ. It is built from the same object as the plain text
and the Markdown, so the three cannot say different things; gates.py checks that the words of this payload equal the plain text's.
`color_bars=False` builds the same content with no attachments (every block top-level, a divider between sections): the fallback if
Slack's legacy attachments turn out not to be accepted.

CUSTOMERS' WORDS ARE NEVER PARSED AS MARKUP. Every review quote goes into a `plain_text` object. Slack's mrkdwn has no way to escape
`*`, `_`, `~` or a backtick, and it reads `<!channel>` or `<@user>` as a mention; a review that holds any of those (2.9% of the review texts
in the probe's cases hold a `~`, and 0.4% a `>`) would change how the message looks or ping people. Product and brand names, label names and theme
names are ours to display but not ours to write: they go into mrkdwn with `&`, `<` and `>` escaped as Slack requires, and gates.py flags any of
them that holds a markup character.

WHAT SLACK ALLOWS (checked against Slack's documentation on 2026-09-26; gates.py holds these numbers and enforces them)
    blocks per message ......... 50   docs.slack.dev/block-kit: "You can include up to 50 blocks in each message"
    section text ............... 1 to 3000 characters (10 fields of 2000: unused here)   docs.slack.dev/reference/block-kit/blocks/section-block
    header text ................ plain_text only, 150 characters   docs.slack.dev/reference/block-kit/blocks/header-block
    context elements ........... 10   docs.slack.dev/reference/block-kit/blocks/context-block
    escaping in mrkdwn ......... & < > become &amp; &lt; &gt;   docs.slack.dev/messaging/formatting-message-text
    `text` fallback ............ best kept under 4,000 characters; Slack truncates above 40,000   docs.slack.dev/reference/methods/chat.postMessage
    attachments ................ Slack calls them "a legacy part of messaging functionality"; the official Python SDK's BlockAttachment
                                 ("a bridge between legacy attachments and Block Kit") takes `blocks`, `color` (a hex code) and `fallback`;
                                 up to 100 attachments (third-party guide to the `too_many_attachments` error, not Slack's own page)
    posting rate ............... about 1 message per second per channel   (matters to post_report.py's posting loop, not here)
    NOT documented on the pages read: whether an incoming webhook accepts attachments, and whether the blocks inside attachments count
    towards the 50. The gate counts them (the cautious reading); Slack's Block Kit Builder and the first real post settle the rest.

No model call, no database, no network, no file writes.
"""
import json
import re

import brand_block as bb
import complaint_block as cb
import message as msg
import mix_shift as ms
import top_line as tl

BAR_BRAND = "#9aa0a6"
BAR_MIX = "#e8590c"


# ------------------------------------------------------------------ Slack objects
def esc(text):
    """Slack's three control characters, as its documentation requires for mrkdwn text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mrkdwn(text):
    return {"type": "mrkdwn", "text": text, "verbatim": True}          # verbatim: no auto-linking, no mention parsing


def plain(text):
    return {"type": "plain_text", "text": text, "emoji": False}        # emoji False: ":smile:" in a review stays as typed


def section(text_object):
    return {"type": "section", "text": text_object}


def header(text):
    return {"type": "header", "text": plain(text)}


def context(text_object):
    return {"type": "context", "elements": [text_object]}


def divider():
    return {"type": "divider"}


SECTION_STYLES = ("header", "bold")


def head_blocks(title, caption, style):
    """A name and its caption as blocks: a header block (or, in the "bold" style, a bold line at body size), then the caption as a small context line."""
    if style not in SECTION_STYLES:
        raise ValueError(f"section_style must be one of {SECTION_STYLES}, not {style!r}")
    out = [header(title) if style == "header" else section(mrkdwn(f"*{esc(title)}*"))]
    if caption:
        out.append(context(mrkdwn(esc(caption))))
    return out


def link_text(url, label):
    """Slack's link markup. The address and label pass report_config's checks, so neither holds a character that needs escaping."""
    return f"<{url}|{label}>"


# ------------------------------------------------------------------ the parts, as Slack text
def bold_once(line, label, value):
    """`line` with the first `label value` given a bold value: 'Reviews: 14,220 (...)' -> 'Reviews: *14,220* (...)'."""
    old = f"{label} {value}"
    assert line.count(old) == 1, f"cannot find {old!r} in {line!r}"
    return line.replace(old, f"{label} *{value}*", 1)


def top_section(part):
    """The top lines: the two big numbers, a blank line, then one count per line, each value in bold (the same emphasis as the Markdown form)."""
    top = part["top"]
    reviews = bold_once(esc(tl.reviews_line(top)), esc(tl.REVIEWS_LABEL), esc(top["reviews"]["value"]))
    evaluated = bold_once(esc(tl.evaluated_line(top)), esc(tl.EVALUATED_LABEL), esc(top["evaluated"]["value"]))
    counts = []
    for c in top["counts"]:
        if c["n"] is None:
            counts.append(f"{c['emoji']} {esc(c['label'])}")
        else:
            tail = f" ({esc(top['prior_month_word'])}: {esc(c['prior'])})" if c["prior"] is not None else ""
            counts.append(f"{c['emoji']} *{esc(c['n'])}* {esc(c['label'])}{tail}")
    return section(mrkdwn("\n".join([reviews, evaluated, ""] + counts)))


def row_text(item):
    """A product row's name line and number line. (Its flag line, if any, is a context block after it: see rows_blocks.)"""
    r = item["row"]
    lines = [f"{r['emoji']} *{esc(r['brand'])} · {esc(r['product'])}* _{esc(r['category'])}_"]
    if r["available"]:
        pct = f" ({esc(r['pct'])})" if r["pct"] else ""
        lines.append(f"{esc(r['prior'])} → {esc(r['now'])} {r['arrow']} *{esc(r['delta'])}*{pct} · {esc(r['reviews'])}")
    else:
        lines.append(f"{esc(r['now'])} · {esc(r['reviews'])} · no comparison: {esc(r['no_comparison'])}")
    return "\n".join(lines)


def complaint_blocks(b, style="header"):
    """The complaint block: label lines in mrkdwn (the label is the model's word), every quote in a plain_text section."""
    n = b["n_current"]
    out = head_blocks("What reviewers say", f"{n} complaint texts this month · top {len(b['labels'])} labels cover {b['top_covers']}", style)
    for r in b["labels"]:
        out.append(section(mrkdwn(f"• *{esc(r['name'])}* — {r['count']} of {n}")))
        out.append(section(plain(f"“{r['quote']}”")))
    if b["points"]:
        out += head_blocks("Worth a look", None, style)
        for p in b["points"]:
            out.append(section(plain(f"“{p['quote']}” {cb.id_note(p['quote_segment_id'], len(p['cited_segment_ids']))}")))
    return out


def rows_blocks(part, style="header"):
    """A section of product rows (at risk or dips). Rows without a flag line or a complaint block share one Slack block; a row with a flag line is its own
    block followed by the flag line as a small context line, and then by its complaint block."""
    out = head_blocks(part["title"], part["context"], style)
    pending = []

    def flush():
        if pending:
            out.append(section(mrkdwn("\n\n".join(pending))))
            pending.clear()
    for item in part["rows"]:
        if item["flag_line"]:
            flush()
            out.append(section(mrkdwn(row_text(item))))
            out.append(context(mrkdwn(esc(item["flag_line"]))))
        else:
            pending.append(row_text(item))
        if item["block"]:
            flush()
            out += complaint_blocks(item["block"], style)
    flush()
    return out


def brand_blocks(sec, style="header"):
    out = head_blocks(sec["title"], sec["context"], style)
    rows = []
    for r in sec["rows"]:
        pct = f" ({esc(r['pct'])})" if r["pct"] else ""
        rows.append(f"*{esc(r['brand'])} · {esc(r['product'])}* _{esc(r['category'])}_\n{esc(r['product_mean'])} vs {esc(r['category_mean'])} {r['arrow']} *{esc(r['gap'])}*{pct}")
    if rows:
        out.append(section(mrkdwn("\n\n".join(rows))))
    if sec["none_shown"]:
        out.append(section(mrkdwn(esc(bb.NONE_LINE))))
    return out


def mix_blocks(sec, style="header"):
    if not sec["available"]:
        return head_blocks(sec["title"], None, style) + [section(mrkdwn(esc(sec["unavailable_line"])))]
    out = head_blocks(sec["title"], sec["header"], style)
    for r in sec["rows"]:
        out.append(section(mrkdwn(f"*{esc(r['theme'])}* {esc(r['share_prior'])} → {esc(r['share_recent'])} {ms.ARROW_UP} *{esc(r['shift'])}*{esc(ms.month_tag(r))}")))
        if r["example"]:
            out.append(section(plain(ms.example_line(r))))                    # a customer's words with the product's names: all literal
    if sec["none_shown"]:
        out.append(section(mrkdwn(esc(sec["none_line"]))))
    return out


# ------------------------------------------------------------------ the message
def to_block_kit(message, color_bars=True, section_style="header"):
    """The webhook payload for a built message. Deterministic; the same message always gives the same JSON."""
    top, groups = [], []          # (an unknown kind of part is refused by msg.part_tagged, called below for each part)
    lead = []                                                        # the header and top lines, as plain lines: the notification text
    for part in message["parts"]:
        t = part["type"]
        first = msg.part_tagged(part)[0]["text"]
        if t == "header":
            top.append(header(part["text"]))
            lead.append(part["text"])
        elif t == "dashboard":
            top.append(context(mrkdwn(link_text(part["url"], part["label"]))))
        elif t == "top":
            top.append(top_section(part))
            lead += msg.part_lines(part)
        elif t == "quiet":
            top.append(section(mrkdwn(esc(part["text"]))))
        elif t == "section":
            groups.append({"color": part["color"], "fallback": first, "blocks": rows_blocks(part, section_style)})
        elif t == "truncated":
            groups[-1]["blocks"].append(context(mrkdwn(esc(part["text"]))))
        elif t == "brand":
            groups.append({"color": BAR_BRAND, "fallback": first, "blocks": brand_blocks(part["section"], section_style)})
        elif t == "mix":
            groups.append({"color": BAR_MIX, "fallback": first, "blocks": mix_blocks(part["section"], section_style)})
    payload = {"text": "\n".join(lead)}
    if color_bars:
        payload["blocks"] = top
        payload["attachments"] = [{"color": g["color"], "fallback": g["fallback"], "blocks": g["blocks"]} for g in groups]
    else:
        blocks = list(top)
        for g in groups:
            blocks.append(divider())
            blocks += g["blocks"]
        payload["blocks"] = blocks
    return payload


def to_json(payload):
    """The payload as text for a file or for Slack's Block Kit Builder: UTF-8, Korean kept as Korean, stable key order."""
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


# ------------------------------------------------------------------ reading a payload back
def all_blocks(payload):
    """Every block of the payload in reading order: the top-level blocks, then each attachment's."""
    out = list(payload.get("blocks", []))
    for a in payload.get("attachments", []):
        out += a["blocks"]
    return out


def text_objects(block):
    if block["type"] == "header":
        return [block["text"]]
    if block["type"] == "section":
        return [block["text"]]
    if block["type"] == "context":
        return list(block["elements"])
    return []


def unescape(text):
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


LINK = re.compile(r"<(https://[^<>|\s&]+)\|([^<>|&]+)>")   # Slack's <address|label>: an https address and a label with no & (report_config allows none), never a mention


def payload_lines(payload):
    """The words the payload shows, as lines in reading order, with mrkdwn escapes undone (a plain_text object is taken as it is) and a link shown as
    its label (the address is not words; gates.py checks it against report_config)."""
    lines = []
    for block in all_blocks(payload):
        for obj in text_objects(block):
            text = unescape(LINK.sub(r"\2", obj["text"])) if obj["type"] == "mrkdwn" else obj["text"]
            lines += text.split("\n")
    return lines
