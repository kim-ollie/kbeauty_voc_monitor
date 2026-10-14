"""Complaint mix shift section for the monthly report.

Pure functions over the saved monthly payloads (report_inputs/report_input_YYYY-MM.json):
no model call, no database, no network, no file writes.

What the section shows, and why:
  * It shows the payload's complaint movement (block F: the share of all complaint mentions of each
    theme group, the last 3 months against the 3 before): the themes the payload reports as rising
    (top 3, at least 1.0 pp). It is a MIX section, so a theme is shown when its share rose, whatever
    its own mention count did. In June 2026 total complaint mentions fell 26% and all three themes had
    fewer mentions than the quarter before, and they are shown: a share-only rise is the measure this
    section is about.
  * Every theme shown gets the same full row: its share before and after, the change in percentage
    points and one example review with brand, product and month. A theme that was shown in the previous
    month's report too ends its row with its month count ("... · 3rd month in a row").
  * No mention counts and no percentage changes are printed: they read oddly beside a rising share.
The category shift block is not printed in the report (it stays on the dashboard).

Every number printed is a string copied from the payload; the only numbers made here are the
ordinals of the month counts.
"""
SECTION_TITLE = "Complaint mix shift"
UNAVAILABLE_LINE = "Not available for this month: the theme model starts in July 2025 and needs 6 months of data to compare."
ARROW_UP = "🔺"


def ordinal(n):
    """1 -> 1st, 2 -> 2nd, 11 -> 11th, 22 -> 22nd."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def none_line(payload):
    """Said when the payload reports no rising theme: the payload's own floor string ("1.0 pp") is the number."""
    return f"No complaint theme gained {payload['complaint_movement']['materiality_floor_display']} or more of share this quarter."


def as_of(payload):
    return payload["envelope"]["as_of_month"][:7]


def month_index(ym):
    y, m = ym.split("-")
    return int(y) * 12 + int(m) - 1


def shown_themes(payload):
    """The payload's themes reported as rising (top 3, at least 1.0 pp of share), best rise first."""
    cm = payload["complaint_movement"]
    if not cm["available"]:
        return []
    return sorted((g for g in cm["theme_groups"] if g["reported_as_rising"]), key=lambda g: g["rank_rise"])


def streaks(payloads):
    """{theme: consecutive months it was shown, up to and including the last payload}.

    payloads: the saved payloads of consecutive months in order, the reported month last. The count
    stops at a gap in the months, and at a month in which the theme was not shown."""
    last = payloads[-1]
    out = {}
    for g in shown_themes(last):
        n = 1
        i = len(payloads) - 2
        while i >= 0:
            newer, older = payloads[i + 1], payloads[i]
            if month_index(as_of(newer)) - month_index(as_of(older)) != 1:
                break
            if g["theme_group"] not in {x["theme_group"] for x in shown_themes(older)}:
                break
            n += 1
            i -= 1
        out[g["theme_group"]] = n
    return out


def evidence_for(payload, theme):
    for e in payload.get("evidence_phrases", []):
        if e["theme_group"] == theme and e.get("phrase_available"):
            return e
    return None


def build_section(payload, previous_payloads=()):
    """The section as data. previous_payloads: earlier months' payloads in order (any number, oldest first)."""
    cm = payload["complaint_movement"]
    sec = {"as_of": as_of(payload), "available": bool(cm["available"]), "title": SECTION_TITLE,
           "header": None, "rows": [], "none_shown": False, "none_line": None, "unavailable_line": None}
    if not cm["available"]:
        sec["unavailable_line"] = UNAVAILABLE_LINE
        return sec
    sec["header"] = (f"Last {cm['half_months']} months ({cm['recent_window_display']}) vs {cm['prior_window_display']}"
                     " · share of all complaint mentions")
    shown = shown_themes(payload)
    st = streaks(list(previous_payloads) + [payload])
    for g in shown:
        n = st[g["theme_group"]]
        e = evidence_for(payload, g["theme_group"])
        sec["rows"].append({
            "theme": g["theme_group"], "share_prior": g["share_prior_display"], "share_recent": g["share_recent_display"],
            "shift": g["shift_display"], "months": n, "months_text": ordinal(n) if n >= 2 else None,
            "example": None if e is None else {"quote": e["representative_phrase_display"], "brand": e["brand_name_ko"],
                                               "product": e["product_name_ko"], "month": e["review_month_display"]}})
    sec["none_shown"] = not shown
    if not shown:
        sec["none_line"] = none_line(payload)
    return sec


# ---------------------------------------------------------------- text form
def month_tag(row):
    """" · 3rd month in a row" for a theme that was rising last month too, "" for one that is new this month."""
    return f" · {row['months_text']} month in a row" if row["months_text"] else ""


def row_line(row):
    return f"{row['theme']} {row['share_prior']} → {row['share_recent']} {ARROW_UP} {row['shift']}{month_tag(row)}"


def example_line(row):
    e = row["example"]
    return f"“{e['quote']}” — {e['brand']} · {e['product']}, {e['month']}"


def section_lines(section):
    """The section as plain lines. Its word count is taken from these."""
    if not section["available"]:
        return [section["title"], section["unavailable_line"]]
    lines = [f"{section['title']} {section['header']}"]
    for r in section["rows"]:
        lines.append(row_line(r))
        if r["example"]:
            lines.append(example_line(r))
    if section["none_shown"]:
        lines.append(section["none_line"])
    return lines


def word_count(lines):
    """Whitespace tokens, bullets not counted (the same convention as the complaint block)."""
    return sum(len([t for t in line.split() if t != "•"]) for line in lines)


# ---------------------------------------------------------------- checks
def verify_section(section, payload):
    """Problems with the section against its payload; [] means all clear.
    Every string printed must be the payload's own; every theme shown must be one the payload reports as rising;
    nothing is shown twice; at most the payload's top-3 cap."""
    problems = []
    cm = payload["complaint_movement"]
    if not section["available"]:
        return [] if not cm["available"] else ["section is unavailable but the payload says complaint movement is available"]
    by_theme = {g["theme_group"]: g for g in cm["theme_groups"]}
    names = [r["theme"] for r in section["rows"]]
    if len(names) != len(set(names)):
        problems.append("a theme is shown twice")
    if len(names) > 3:
        problems.append(f"{len(names)} themes shown; the payload's cap is 3")
    for r in section["rows"]:
        g = by_theme.get(r["theme"])
        if g is None:
            problems.append(f"{r['theme']}: not in the payload")
            continue
        if not g["reported_as_rising"]:
            problems.append(f"{r['theme']}: not reported as rising")
        for key, pkey in (("share_prior", "share_prior_display"), ("share_recent", "share_recent_display"), ("shift", "shift_display")):
            if r[key] != g[pkey]:
                problems.append(f"{r['theme']}: {key} is not the payload's string")
        if r["months"] < 1 or r["months_text"] != (ordinal(r["months"]) if r["months"] >= 2 else None):
            problems.append(f"{r['theme']}: the month count and its text disagree")
        if r["example"] is not None:
            e = evidence_for(payload, r["theme"])
            if e is None or (r["example"]["quote"], r["example"]["brand"], r["example"]["product"], r["example"]["month"]) != (
                    e["representative_phrase_display"], e["brand_name_ko"], e["product_name_ko"], e["review_month_display"]):
                problems.append(f"{r['theme']}: example is not the payload's")
    if section["none_shown"] != (not names):
        problems.append("none_shown disagrees with the themes shown")
    if section["none_shown"] and section["none_line"] != none_line(payload):
        problems.append("the none-shown line is not built from the payload's floor string")
    return problems
