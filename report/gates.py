"""The gates a finished message has to pass before it may be posted.

`run_gates(message, payload, previous_payloads, store)` reads the finished message (its tagged lines, from message.py) and returns one
verdict per gate. A gate is either a BLOCK (a failure stops the report: nothing is posted) or a FLAG (a hit is listed for a person to read and
does not stop the report).

    build_checks      BLOCK  the parts' own verifiers, run while the message was built, found nothing
    text_is_lines     BLOCK  the text that would be posted is exactly the tagged lines the other gates read
    records_stamped   BLOCK  every block on the page has its model records stamped (which prompt, which run) and none of them had a problem
    numbers           BLOCK  the number gate (number_gate.py): every number on every line is explained by the data
    quotes            BLOCK  every quote printed is real (S2 on the finished message)
    s3_label_names    BLOCK  a label name the model wrote holds no digit, number word or frequency word (the prompt's rule 1)
    s4_wording        FLAG   no causal, advice or banned wording (cause, should, trend, surge, significant, ...) outside customers' quotes
    size              FLAG   words per message about 550, per complaint block about 160
    slack_limits      BLOCK  the Block Kit payload is inside Slack's documented limits and holds only what the gate knows how to check
    blocks_match_text BLOCK  the Block Kit payload shows the same words, in the same order, as the plain text the other gates read
    json_round_trip   BLOCK  the payload survives JSON encoding and decoding unchanged, Korean included (both as written and \\u-escaped)
    mrkdwn_hygiene    BLOCK  no raw & < > in any mrkdwn text (they would be read as mentions or links; the one link of dashboard_link aside), every mrkdwn object skips auto-linking,
                             every plain_text object leaves ":shortcodes:" as typed
    markup_in_names   FLAG   a product, brand, label or theme name placed in mrkdwn holds a markup character (* _ ~ or a backtick)
    dashboard_link    BLOCK  the optional link line: none anywhere while report_config sets no dashboard; with one, exactly one link in the payload, in
                             a top-level context block, with exactly the configured address and label, and one such line in the message

The Block Kit payload is built here from the same message object (block_kit.to_block_kit), so the gates read exactly what would be posted.
The Slack numbers below were checked against Slack's documentation on 2026-09-26 (see block_kit.py's docstring for the pages).

S3 and S4 are the probe's word lists (probe_p4_checks.SCANS). For S3 a hit on a word the prompt itself lists blocks; a hit on a word the
probe added ("extended") is flagged. Customers' own words are never scanned: they are theirs, and "because" in a review is not a causal claim.

No model call, no database, no network, no file writes.
"""
import json
import re

import block_kit as bk
import message as msg
import number_gate as ng
import probe_p4_checks as P4
import records as rc
import report_config as cfg

MESSAGE_WORD_CAP = 550
BLOCK_WORD_CAP = 160
PASS, FAIL, FLAG = "pass", "fail", "flag"

SLACK = {"blocks": 50, "attachments": 100, "section_text": 3000, "header_text": 150, "context_elements": 10, "text_recommended": 4000, "text_truncated": 40000}
BLOCK_KEYS = {"header": {"type", "text"}, "section": {"type", "text"}, "context": {"type", "elements"}, "divider": {"type"}}
TEXT_KEYS = {"mrkdwn": {"type", "text", "verbatim"}, "plain_text": {"type", "text", "emoji"}}
PAYLOAD_KEYS = {"text", "blocks", "attachments"}
ATTACHMENT_KEYS = {"color", "fallback", "blocks"}
HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")
MARKUP = "*_~`"


def verdict(status, detail=()):
    return {"status": status, "detail": list(detail)}


# ------------------------------------------------------------------ text helpers
def not_customer_words(line):
    """The part of a line that is ours: for a line that carries a customer's quote, the text around the quote."""
    if line["kind"] in ng.QUOTE_KINDS:
        try:
            return ng.outside_quotes(line["text"])[0]
        except ng.Refuse:
            return ""
    return line["text"]


def quote_of(line):
    """The quoted words of a quote line, or None when the line has no complete pair of marks."""
    try:
        return ng.outside_quotes(line["text"])[1]
    except ng.Refuse:
        return None


def words(text):
    return len([t for t in text.split() if t != "•"])


# ------------------------------------------------------------------ the gates
def gate_build_checks(message):
    problems = message["checks"]["problems"]
    return verdict(FAIL if problems else PASS, problems)


def gate_text_is_lines(message, lines):
    same = msg.to_text(message) == "\n".join(t["text"] for t in lines) + "\n"
    return verdict(PASS if same else FAIL, [] if same else ["the posted text differs from the lines the gates read"])


def gate_records_stamped(message, lines):
    detail = []
    for pid in sorted({t["product_id"] for t in lines if t["kind"] == "block_head"}):
        st = message["records"].get(pid)
        if not st:
            detail.append(f"product {pid}: a complaint block is shown but no record stamp was kept")
        elif st.get("problems"):
            detail.append(f"product {pid}: the records had {len(st['problems'])} problem(s)")
    return verdict(FAIL if detail else PASS, detail)


def load_records(lines, payload, store):
    """{product_id: {"case", "readout", "labeling"}} for every product with a block, loaded afresh from the store (not taken from the renderer)."""
    month = payload["envelope"]["as_of_month"][:7]
    pids = sorted({t["product_id"] for t in lines if t["kind"] == "block_head"})
    if pids and store is None:
        raise ng.Refuse("the message has complaint blocks and no record store was given to the gates")
    return {pid: store.load(month, pid, strict=False) for pid in pids}


def gate_numbers(lines, payload, previous_payloads, records, dashboard=None):
    r = ng.check(lines, payload, previous_payloads, records, dashboard)
    return verdict(PASS if r["passed"] else FAIL, r["failures"]), r


def piece_of_a_line(quote, text):
    """S2: a quote (its closing "…" aside) is a contiguous piece of one line of the text."""
    q = quote[:-1] if quote.endswith("…") else quote
    return bool(q) and any(q in ln for ln in text.split("\n"))


def gate_quotes(lines, payload, records):
    """Every quote printed is real. Block quotes must be a piece of one line of a this-month text of the product's case (a "worth a look" quote
    of the very text whose id is printed after it); the mix-shift example must be the payload's own example phrase for that theme."""
    detail = []
    for i, t in enumerate(lines):
        if t["kind"] not in ng.QUOTE_KINDS:
            continue
        quote = quote_of(t)
        where = f"line {i} ({t['kind']})"
        if not quote:
            detail.append(f"{where}: no quote between the marks")
            continue
        if t["kind"] == "mix_example":
            phrases = {e["representative_phrase_display"] for e in payload["evidence_phrases"] if e["theme_group"] == t.get("theme")}
            if quote not in phrases:
                detail.append(f"{where}: not the payload's example phrase for '{t.get('theme')}'")
            continue
        rec = records.get(t.get("product_id"))
        if rec is None:
            detail.append(f"{where}: no records to check the quote against")
            continue
        current = {s["segment_id"]: s for s in rec["case"]["segments"] if s["period"] == "current"}
        if t["kind"] == "block_point":
            printed = ng.tok(ng.outside_quotes(t["text"])[0])
            seg = current.get(int(printed[0])) if printed and printed[0].isdigit() else None       # the id first; a "+N more" may follow
            if seg is None or not piece_of_a_line(quote, seg["text"]):
                detail.append(f"{where}: the quote is not a piece of one line of the this-month text whose id is printed")
        elif not any(piece_of_a_line(quote, s["text"]) for s in current.values()):
            detail.append(f"{where}: the quote is not a piece of one line of any this-month text")
    return verdict(FAIL if detail else PASS, detail)


def gate_s3_label_names(lines):
    """The model's label names hold no digit, number word or frequency word. A word the prompt lists blocks; a word the probe added is flagged."""
    blocked, flagged = [], []
    for i, t in enumerate(lines):
        if t["kind"] != "block_label":
            continue
        for h in P4.scan_text(f"line {i}", t["label"]):
            if h["group"] == "S3":
                (blocked if h["tier"] == "prompt" else flagged).append(f"line {i}: {h['category']} '{h['match']}' in a label name ({h['tier']} list)")
    status = FAIL if blocked else FLAG if flagged else PASS
    return verdict(status, blocked + flagged)


def gate_s4_wording(lines):
    """Causal, advice and banned wording outside customers' quotes, for a person to read. Never blocks."""
    hits = []
    for i, t in enumerate(lines):
        for h in P4.scan_text(f"line {i}", not_customer_words(t)):
            if h["group"] == "S4":
                hits.append(f"line {i} ({t['kind']}): {h['category']} '{h['match']}' in \"{h['context']}\"")
    return verdict(FLAG if hits else PASS, hits)


def gate_size(lines):
    total = sum(words(t["text"]) for t in lines)
    detail = [f"the message has {total} words, above about {MESSAGE_WORD_CAP}"] if total > MESSAGE_WORD_CAP else []
    per_block = {}
    for t in lines:
        if t["kind"].startswith("block_"):
            per_block[t["product_id"]] = per_block.get(t["product_id"], 0) + words(t["text"])
    detail += [f"product {pid}'s complaint block has {n} words, above about {BLOCK_WORD_CAP}" for pid, n in sorted(per_block.items()) if n > BLOCK_WORD_CAP]
    return verdict(FLAG if detail else PASS, detail), {"message_words": total, "block_words": per_block}


# ------------------------------------------------------------------ the Slack gates
def utf16_len(text):
    """Length in UTF-16 code units: the cautious way to count characters (an emoji counts 2)."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def check_text_object(where, obj, fails, limit):
    if not isinstance(obj, dict) or obj.get("type") not in TEXT_KEYS:
        fails.append(f"{where}: not a mrkdwn or plain_text object")
        return
    if set(obj) != TEXT_KEYS[obj["type"]]:
        fails.append(f"{where}: the text object has keys {sorted(obj)}, expected {sorted(TEXT_KEYS[obj['type']])}")
    text = obj.get("text")
    if not isinstance(text, str) or not text:
        fails.append(f"{where}: the text is empty or not a string")
    elif utf16_len(text) > limit:
        fails.append(f"{where}: {utf16_len(text)} characters, above {limit}")


def check_block(where, block, fails):
    kind = block.get("type") if isinstance(block, dict) else None
    if kind not in BLOCK_KEYS:
        fails.append(f"{where}: block type {kind!r} is not one the gate knows")
        return
    if set(block) != BLOCK_KEYS[kind]:
        fails.append(f"{where}: the {kind} block has keys {sorted(block)}, expected {sorted(BLOCK_KEYS[kind])}")
        return
    if kind == "header":
        check_text_object(where, block["text"], fails, SLACK["header_text"])
        if block["text"].get("type") != "plain_text":
            fails.append(f"{where}: a header must be plain_text")
    elif kind == "section":
        check_text_object(where, block["text"], fails, SLACK["section_text"])
    elif kind == "context":
        els = block["elements"]
        if not isinstance(els, list) or not 1 <= len(els) <= SLACK["context_elements"]:
            fails.append(f"{where}: a context block holds 1 to {SLACK['context_elements']} elements")
        else:
            for k, el in enumerate(els):
                check_text_object(f"{where} element {k}", el, fails, SLACK["section_text"])


def gate_slack_limits(payload):
    fails, flags = [], []
    if not isinstance(payload, dict) or set(payload) - PAYLOAD_KEYS:
        return verdict(FAIL, [f"the payload has keys {sorted(payload) if isinstance(payload, dict) else payload!r}, allowed {sorted(PAYLOAD_KEYS)}"])
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        fails.append("the notification text is missing or empty")
    elif utf16_len(text) > SLACK["text_truncated"]:
        fails.append(f"the notification text has {utf16_len(text)} characters; Slack truncates above {SLACK['text_truncated']}")
    elif utf16_len(text) > SLACK["text_recommended"]:
        flags.append(f"the notification text has {utf16_len(text)} characters; Slack recommends at most {SLACK['text_recommended']}")
    blocks, attachments = payload.get("blocks", []), payload.get("attachments", [])
    if not isinstance(blocks, list) or not isinstance(attachments, list):
        return verdict(FAIL, fails + ["blocks and attachments must be lists"])
    total = len(blocks)
    for i, b in enumerate(blocks):
        check_block(f"block {i}", b, fails)
    if len(attachments) > SLACK["attachments"]:
        fails.append(f"{len(attachments)} attachments, above {SLACK['attachments']}")
    for j, a in enumerate(attachments):
        where = f"attachment {j}"
        if not isinstance(a, dict) or set(a) != ATTACHMENT_KEYS:
            fails.append(f"{where}: keys {sorted(a) if isinstance(a, dict) else a!r}, expected {sorted(ATTACHMENT_KEYS)}")
            continue
        if not isinstance(a["color"], str) or not HEX_COLOUR.match(a["color"]):
            fails.append(f"{where}: the colour is not a #rrggbb code")
        if not isinstance(a["fallback"], str) or not a["fallback"].strip():
            fails.append(f"{where}: no fallback text")
        if not isinstance(a["blocks"], list) or not a["blocks"]:
            fails.append(f"{where}: no blocks")
            continue
        total += len(a["blocks"])
        for k, b in enumerate(a["blocks"]):
            check_block(f"{where} block {k}", b, fails)
    if total > SLACK["blocks"]:
        fails.append(f"{total} blocks in all (attachments' included), above {SLACK['blocks']}")
    return verdict(FAIL if fails else FLAG if flags else PASS, fails + flags)


SEPARATORS = ("•", "—")          # a bullet, and the dash between a heading and its caption: layout, not words (the caption is a context block of its own)


def payload_words(lines):
    """The words of some lines, markup characters, bullets and a lone dash ignored: the payload adds bold and italic marks the plain text does not have,
    and puts a heading's caption in a block of its own where the plain text has a dash."""
    return [w for w in re.sub(f"[{re.escape(MARKUP)}]", "", " ".join(lines)).split() if w not in SEPARATORS]


def gate_blocks_match_text(payload, lines):
    """The payload shows the same words in the same order as the plain text, and its fallback strings are lines of that text."""
    detail = []
    want, got = payload_words([t["text"] for t in lines]), payload_words(bk.payload_lines(payload))
    if want != got:
        k = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
        detail.append(f"the payload's words differ from the plain text's at word {k}: {want[k:k + 3]} against {got[k:k + 3]} ({len(got)} words against {len(want)})")
    lead = "\n".join(t["text"] for t in lines if t["kind"] in ("header", "top_reviews", "top_evaluated", "top_counts"))
    if payload.get("text") != lead:
        detail.append("the notification text is not the header and the three top lines")
    titles = {t["text"] for t in lines if t["kind"] in ("section_title", "brand_title", "mix_title")}
    for j, a in enumerate(payload.get("attachments", [])):
        if a.get("fallback") not in titles:
            detail.append(f"attachment {j}: the fallback text is not a section title of the plain text")
    return verdict(FAIL if detail else PASS, detail)


def gate_json_round_trip(payload):
    detail = []
    for ascii_only in (False, True):
        try:
            text = json.dumps(payload, ensure_ascii=ascii_only)
            back = json.loads(text.encode("utf-8").decode("utf-8"))
        except (TypeError, ValueError) as why:
            detail.append(f"encoding failed ({'escaped' if ascii_only else 'as written'}): {why}")
            continue
        if back != payload:
            detail.append(f"the payload changed in a JSON round trip ({'escaped' if ascii_only else 'as written'})")
    return verdict(FAIL if detail else PASS, detail)


ENTITY = re.compile(r"&(?:amp|lt|gt);")


def gate_mrkdwn_hygiene(payload):
    detail = []
    for i, b in enumerate(bk.all_blocks(payload)):
        for obj in bk.text_objects(b):
            if obj.get("type") == "mrkdwn":
                if any(c in ENTITY.sub("", bk.LINK.sub("", obj["text"])) for c in "<>&"):
                    detail.append(f"block {i}: a raw & < or > in mrkdwn text ({obj['text'][:40]!r})")
                if obj.get("verbatim") is not True:
                    detail.append(f"block {i}: a mrkdwn object without verbatim=true would auto-link and parse mentions")
            elif obj.get("type") == "plain_text" and obj.get("emoji") is not False:
                detail.append(f"block {i}: a plain_text object without emoji=false would turn :shortcodes: in a review into emoji")
    return verdict(FAIL if detail else PASS, detail)


def gate_dashboard_link(payload, lines, dashboard):
    """The optional link line. While no dashboard is set there is no link anywhere; with one set, exactly one link, in a top-level context block, with
    exactly the address and label of report_config (read here, not from the message), and one line for it in the plain text."""
    detail = []
    links = [m.groups() for b in bk.all_blocks(payload) for obj in bk.text_objects(b) if obj.get("type") == "mrkdwn" for m in bk.LINK.finditer(obj["text"])]
    line_labels = [t["text"] for t in lines if t["kind"] == "dashboard_link"]
    if dashboard is None:
        if links:
            detail.append(f"{len(links)} link(s) in the payload although no dashboard is set in report_config")
        if line_labels:
            detail.append("a dashboard line is in the message although no dashboard is set in report_config")
    else:
        if line_labels != [dashboard["label"]]:
            detail.append(f"the message must hold one dashboard line with the label {dashboard['label']!r}, and holds {line_labels!r}")
        if links != [(dashboard["url"], dashboard["label"])]:
            detail.append("the payload must hold exactly one link, with the configured address and label")
        else:
            top = [b for b in payload.get("blocks", []) if b.get("type") == "context" and any(bk.link_text(*links[0]) == e.get("text") for e in b.get("elements", []))]
            if len(top) != 1:
                detail.append("the link must be in a top-level context block")
    return verdict(FAIL if detail else PASS, detail)


def mrkdwn_strings(message):
    """The strings that reach mrkdwn from outside: product, brand and category names, the model's label names, theme names."""
    for part in message["parts"]:
        if part["type"] == "section":
            for item in part["rows"]:
                r = item["row"]
                for field in ("brand", "product", "category"):
                    yield f"product {r['product_id']} {field}", r[field]
                for lab in (item["block"] or {}).get("labels", []):
                    yield f"product {r['product_id']} label", lab["name"]
        elif part["type"] == "brand":
            for r in part["section"]["rows"]:
                for field in ("brand", "product", "category"):
                    yield f"product {r['product_id']} {field}", r[field]
        elif part["type"] == "mix" and part["section"]["available"]:
            for r in part["section"]["rows"]:
                yield "theme", r["theme"]


def gate_markup_in_names(message):
    detail = [f"{where}: '{text}' holds {sorted({c for c in text if c in MARKUP})}" for where, text in mrkdwn_strings(message) if any(c in MARKUP for c in text)]
    return verdict(FLAG if detail else PASS, detail)


def run_gates(message, payload, previous_payloads=(), store=None, color_bars=True, section_style="header", dashboard=msg.UNSET):
    """All the gates on one built message.

    Returns {"passed": True when no gate BLOCKED, "gates": {name: {"status", "detail"}}, "flags": [text for a person to read],
             "numbers": the number gate's full result (its explanation table), "sizes": word counts,
             "block_kit": the Slack payload the Slack gates read (the same one to post)}.
    section_style: how the Slack payload lays out section names (block_kit.SECTION_STYLES). dashboard: {"url", "label"} or None; by default report_config's."""
    dash = cfg.dashboard() if dashboard is msg.UNSET else dashboard
    lines = msg.tagged_lines(message)
    gates = {"build_checks": gate_build_checks(message), "text_is_lines": gate_text_is_lines(message, lines),
             "records_stamped": gate_records_stamped(message, lines)}
    try:
        records = load_records(lines, payload, store)
        gates["numbers"], numbers = gate_numbers(lines, payload, previous_payloads, records, dash)
        gates["quotes"] = gate_quotes(lines, payload, records)
    except (ng.Refuse, rc.MissingRecords, rc.RecordProblem) as why:
        records, numbers = {}, None
        gates["numbers"] = verdict(FAIL, [str(why)])
        gates["quotes"] = verdict(FAIL, [str(why)])
    gates["s3_label_names"] = gate_s3_label_names(lines)
    gates["s4_wording"] = gate_s4_wording(lines)
    gates["size"], sizes = gate_size(lines)
    kit = bk.to_block_kit(message, color_bars=color_bars, section_style=section_style)
    gates["slack_limits"] = gate_slack_limits(kit)
    gates["blocks_match_text"] = gate_blocks_match_text(kit, lines)
    gates["json_round_trip"] = gate_json_round_trip(kit)
    gates["mrkdwn_hygiene"] = gate_mrkdwn_hygiene(kit)
    gates["markup_in_names"] = gate_markup_in_names(message)
    gates["dashboard_link"] = gate_dashboard_link(kit, lines, dash)
    flags = [f"{name}: {d}" for name, g in gates.items() if g["status"] == FLAG for d in g["detail"]]
    return {"passed": all(g["status"] != FAIL for g in gates.values()), "gates": gates, "flags": flags, "numbers": numbers, "sizes": sizes, "block_kit": kit}


def summary_lines(result):
    """A short report of the gates, for the run log."""
    out = []
    for name, g in result["gates"].items():
        out.append(f"{g['status'].upper():<5} {name}" + (f"  ({len(g['detail'])})" if g["detail"] else ""))
    out.append(f"words: {result['sizes']['message_words']} in the message; blocks {result['sizes']['block_words'] or 'none'}")
    out.append("RESULT: " + ("all blocking gates passed" if result["passed"] else "BLOCKED, nothing may be posted"))
    if result["flags"]:
        out.append(f"{len(result['flags'])} flag(s) for a read:")
        out += [f"  - {f}" for f in result["flags"]]
    return out
