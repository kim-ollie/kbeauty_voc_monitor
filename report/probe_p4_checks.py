#!/usr/bin/env python3
"""
probe_p4_checks.py  --  the complaint-readout probe's automatic checks.

What this does
    Reads probe_inputs/ (the five case files, md5-checked against manifest.json) and every
    record in probe_outputs/, and measures, by code and without any model call:
        S2  every quote is a verbatim substring of the segment it cites, every cited id exists in
            the case and is a current-period text (recomputed here from the case file, then
            compared with what the runner stored)
        S3  no digits, number words or frequency words in the model's claims, summary, label
            names and label notes (the prompts' rule 1)
        S4  no causal wording, no advice, none of trend / surge / spike / significant (rules 2-4)
        S6  cost and time per call (from the records; not pass/fail)
        S7  Arm B run-to-run agreement: how well two runs group the same texts together
        --  a brand-name scan of the case texts (a quote that names a brand can unblind case 5)
    S3 and S4 are word scans: they list CANDIDATES with their context for a human to read. A
    scan hit is not a verdict ("reviewers report breakouts after use" is allowed; a scan for
    "because" cannot tell that from the model asserting a cause). Each hit is tagged "prompt"
    (the word the prompt itself lists) or "extended" (a word this script adds).

    S7 compares the PARTITIONS the two runs make, not the label names (Arm B's labels are chosen
    anew in every run): adjusted Rand index (chance-corrected), plain pairwise agreement (the
    share of text pairs that both runs treat the same way: together in both, or apart in both),
    and the pairwise agreement two unrelated partitions of the same group sizes would show by
    chance. Run 1 is the run of record; the pair is cases 1 and 3, run 1 against run 2.

What this does NOT do
    No call to any API, no database, no .env. It never prints model text or a segment's text,
    and it never prints a per-case S3/S4 count: S2 and S7 are printed per case; S3/S4 only as
    totals over all cases, so nothing in the log tells the reviewer which case is the control.
    The detail file it writes holds model text and per-case hits: do not open it before the blind review.
    It never reads or prints a case's product id or month, and never opens the sealed file.

Usage (from the project folder, next to probe_run_model.py)
    python probe_p4_checks.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import probe_run_model as R

CHECKS_VERSION = "probe-p4-checks-v0"
DEFAULT_PRODUCTS = "products_v3.csv"
DEFAULT_DETAIL = Path("probe_packet") / "probe_p4_checks_DETAIL_do_not_open_before_P5.json"
RECORD_RE = re.compile(r"^probe_output_case(\d)_([AB])_(.+)_r(\d+)\.json$")
S7_CASES = (1, 3)                       # the pair chosen for the repeat runs
QUOTE_OK = ("verbatim", "spans_line_break")   # exact substrings; the second breaks prompt rule 5


# ----------------------------------------------------------------------------- word scans
def _alt(words):
    return "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in sorted(words, key=len, reverse=True))


def _rx(words):
    return re.compile(r"(?<![A-Za-z])(?:" + _alt(words) + r")(?![A-Za-z])", re.IGNORECASE)


# (group, category, tier, regex). S3 = digit / number_word / frequency_word; S4 = the rest.
SCANS = [
    ("S3", "digit", "prompt", re.compile(r"\d")),
    ("S3", "number_word", "prompt", _rx(["one", "two", "twice", "first", "second", "half", "double", "dozen"])),
    ("S3", "number_word", "extended", _rx([
        "zero", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "once", "thrice", "third", "fourth", "fifth", "triple", "dozens", "hundred", "hundreds",
        "thousand", "thousands", "pair", "couple"])),
    ("S3", "frequency_word", "prompt", _rx([
        "many", "most", "majority", "several", "few", "often", "commonly", "common", "rare",
        "widespread", "a lot"])),
    ("S3", "frequency_word", "extended", _rx([
        "frequently", "usually", "typically", "generally", "numerous", "multiple", "some",
        "occasionally", "sometimes", "repeatedly", "regularly", "plenty", "minority", "handful",
        "rarely", "seldom", "widely", "commonplace"])),
    ("S4", "causal", "prompt", _rx([
        "cause", "causes", "caused", "causing", "led to", "leads to", "lead to", "leading to",
        "result in", "results in", "resulted in", "resulting in", "drove", "drives", "drive",
        "driven by", "driving"])),
    ("S4", "causal", "extended", _rx([
        "because", "due to", "owing to", "as a result", "thanks to", "attributable", "results from",
        "resulting from", "stems from", "responsible for"])),
    ("S4", "advice", "prompt", _rx([
        "recommend", "recommends", "recommended", "recommending", "advise", "advises", "advised",
        "should", "must", "ought", "need to", "needs to", "reformulate", "reformulation", "redesign",
        "repackage", "repackaging"])),
    ("S4", "advice", "extended", _rx([
        "consider", "suggest", "suggests", "suggested", "improve", "improves", "improved",
        "improvement", "fix", "address", "look into", "switch"])),
    ("S4", "banned_word", "prompt", _rx([
        "trend", "trends", "trending", "surge", "surges", "surging", "surged", "spike", "spikes",
        "spiking", "spiked", "significant", "significantly"])),
]
CONTEXT_CHARS = 40


def scan_text(field, text):
    """All scan hits in one string. A hit carries the matched word and its neighbourhood: that is
    model text, so hits go to the detail file and the packet, never to the log."""
    hits = []
    if not isinstance(text, str):
        return hits
    for group, cat, tier, rx in SCANS:
        for m in rx.finditer(text):
            a, b = m.span()
            hits.append({
                "field": field, "group": group, "category": cat, "tier": tier, "match": m.group(0),
                "start": a, "end": b,
                "context": text[max(0, a - CONTEXT_CHARS):b + CONTEXT_CHARS],
            })
    return hits


def scan_fields_readout(parsed):
    """The strings prompt rules 1-4 apply to: the summary and every claim (not the glosses,
    which the prompt says are not checked)."""
    out = []
    if not isinstance(parsed, dict):
        return out
    ov = parsed.get("overall")
    if isinstance(ov, dict):
        out.append(("overall.summary", ov.get("summary")))
    for sec in R.READOUT_SECTIONS:
        pts = parsed.get(sec)
        if isinstance(pts, list):
            for i, p in enumerate(pts):
                if isinstance(p, dict):
                    out.append((f"{sec}[{i}].claim", p.get("claim")))
    return out


def scan_fields_labeling(parsed):
    """Label names and notes (rule 1-4 of the labeling prompt)."""
    out = []
    labs = parsed.get("labels") if isinstance(parsed, dict) else None
    if isinstance(labs, list):
        for i, lab in enumerate(labs):
            if isinstance(lab, dict):
                out.append((f"labels[{i}].name", lab.get("name")))
                out.append((f"labels[{i}].note", lab.get("note")))
    return out


def scan_record(arm, parsed):
    return scan_record_counted(arm, parsed)[0]


def scan_record_counted(arm, parsed):
    """(hits, number of strings scanned). The count shows in the log, so a scan that found nothing
    because it looked at nothing cannot pass for a clean result."""
    fields = scan_fields_readout(parsed) if arm == "A" else scan_fields_labeling(parsed)
    hits, n = [], 0
    for field, text in fields:
        if isinstance(text, str):
            n += 1
        hits.extend(scan_text(field, text))
    return hits, n


# --------------------------------------------------------------------------------- S2
def locate_quote(quote, case):
    """Ids of every segment of the case whose text contains the quote exactly."""
    if not isinstance(quote, str) or not quote:
        return []
    return [s["segment_id"] for s in case["segments"] if quote in s["text"]]


def _s2_row(where, section, sid, quote, idx, case):
    seg = idx.get(sid) if isinstance(sid, int) and not isinstance(sid, bool) else None
    row = {"where": where, "segment_id": sid if isinstance(sid, int) and not isinstance(sid, bool) else None,
           "id_in_case": seg is not None, "period": None, "unclustered": None,
           "quote_chars": len(quote) if isinstance(quote, str) else None,
           "substring": None, "single_line": None, "status": None, "located_in": []}
    if not isinstance(quote, str):
        row["status"] = "quote_not_a_string"
        return row
    if seg is not None:
        row["period"] = seg["period"]
        row["unclustered"] = seg["unclustered"]
        row["substring"] = quote in seg["text"]
        row["single_line"] = ("\n" not in quote) and ("\r" not in quote)
        row["status"] = R.quote_status(quote, seg["text"])
    else:
        row["status"] = "segment_not_in_case"
    if seg is None or not row["substring"]:
        row["located_in"] = locate_quote(quote, case)
    if section == "in_unclustered_segments" and seg is not None:
        row["section_needs_unclustered"] = True
    return row


def s2_readout(parsed, case):
    """Arm A. Independent recomputation from the case file (not the runner's stored checks)."""
    idx = R._index(case)
    rows = []
    if isinstance(parsed, dict):
        for sec in R.READOUT_SECTIONS:
            pts = parsed.get(sec)
            if not isinstance(pts, list):
                continue
            for i, p in enumerate(pts):
                ev = p.get("evidence") if isinstance(p, dict) else None
                if not isinstance(ev, list):
                    continue
                for j, e in enumerate(ev):
                    if isinstance(e, dict):
                        rows.append(_s2_row(f"{sec}[{i}].evidence[{j}]", sec, e.get("segment_id"),
                                            e.get("quote"), idx, case))
    return _s2_summary(rows)


def s2_labeling(parsed, case):
    """Arm B: the example quote of every label, plus coverage (every text labelled exactly once)."""
    idx = R._index(case)
    rows = []
    labs = parsed.get("labels") if isinstance(parsed, dict) else None
    if isinstance(labs, list):
        for i, lab in enumerate(labs):
            if isinstance(lab, dict):
                rows.append(_s2_row(f"labels[{i}]", None, lab.get("example_segment_id"),
                                    lab.get("example_quote"), idx, case))
    out = _s2_summary(rows, require_current=False)      # an example may come from any text, prior ones included
    want = [s["segment_id"] for s in case["segments"]]
    asg = parsed.get("assignments") if isinstance(parsed, dict) else None
    got = [a.get("segment_id") for a in asg if isinstance(a, dict)] if isinstance(asg, list) else []
    out["coverage"] = {"texts": len(want), "assignments": len(got),
                       "missing": len(set(want) - set(got)), "extra": len(set(got) - set(want)),
                       "repeated": len(got) - len(set(got))}
    out["coverage_exact"] = out["coverage"]["missing"] == out["coverage"]["extra"] == out["coverage"]["repeated"] == 0
    labs_ids = {lab.get("label_id") for lab in labs if isinstance(lab, dict)} if isinstance(labs, list) else set()
    used = [a.get("label_id") for a in asg if isinstance(a, dict)] if isinstance(asg, list) else []
    out["assignments_with_undefined_label"] = sum(1 for x in used if x not in labs_ids)
    return out


def _s2_summary(rows, require_current=True):
    n = len(rows)
    ok_id = sum(1 for r in rows if r["id_in_case"])
    ok_cur = sum(1 for r in rows if r["period"] == "current")
    ok_sub = sum(1 for r in rows if r["substring"] is True)
    exact = sum(1 for r in rows if r["status"] == "verbatim")
    breach_lb = sum(1 for r in rows if r["status"] == "spans_line_break")
    bad_section = sum(1 for r in rows if r.get("section_needs_unclustered") and r["unclustered"] is False)
    strict = n > 0 and ok_id == n and ok_sub == n and (ok_cur == n or not require_current)
    return {"entries": n, "ids_in_case": ok_id, "current_period": ok_cur, "exact_substring": ok_sub,
            "verbatim_single_line": exact, "spans_line_break": breach_lb,
            "unclustered_section_cites_themed": bad_section,
            "strict_pass": strict, "rows": rows}


def s2_agrees_with_runner(s2_rows, stored_checks):
    """The runner stored one quote_checks entry per evidence entry whose id is in the case. Compare
    statuses row by row (keyed by `where`); returns (agree, n_compared)."""
    stored = {q.get("where"): q.get("status") for q in stored_checks if isinstance(q, dict)}
    n = agree = 0
    for r in s2_rows:
        if r["where"] in stored:
            n += 1
            agree += stored[r["where"]] == r["status"]
    return agree == n, n


# --------------------------------------------------------------------------------- S7
def _c2(n):
    return n * (n - 1) // 2


def _pair_sums(a, b):
    n = len(a)
    cont = Counter(zip(a, b))
    sum_ij = sum(_c2(v) for v in cont.values())
    sum_a = sum(_c2(v) for v in Counter(a).values())
    sum_b = sum(_c2(v) for v in Counter(b).values())
    return n, _c2(n), sum_ij, sum_a, sum_b


def adjusted_rand(a, b):
    """Adjusted Rand index of two labelings of the same items (1 = identical partitions, about 0 =
    what chance gives; can be negative). Both partitions trivial (or n < 2) -> 1.0 by convention."""
    if len(a) != len(b):
        raise ValueError("labelings differ in length")
    n, n2, sum_ij, sum_a, sum_b = _pair_sums(a, b)
    if n2 == 0:
        return 1.0
    expected = sum_a * sum_b / n2
    maximum = (sum_a + sum_b) / 2
    if maximum == expected:
        return 1.0
    return (sum_ij - expected) / (maximum - expected)


def pairwise_agreement(a, b):
    """Share of text pairs on which the two runs agree: together in both, or apart in both."""
    n, n2, sum_ij, sum_a, sum_b = _pair_sums(a, b)
    if n2 == 0:
        return 1.0
    return (n2 - sum_a - sum_b + 2 * sum_ij) / n2


def chance_agreement(a, b):
    """Pairwise agreement expected if the two runs were unrelated but kept their group sizes."""
    n, n2, sum_ij, sum_a, sum_b = _pair_sums(a, b)
    if n2 == 0:
        return 1.0
    pa, pb = sum_a / n2, sum_b / n2
    return pa * pb + (1 - pa) * (1 - pb)


def _partition(parsed, case):
    if not isinstance(parsed, dict) or not isinstance(parsed.get("assignments"), list):
        return None
    m = {}
    for a in parsed["assignments"]:
        if not isinstance(a, dict) or a.get("segment_id") in m:
            return None
        m[a.get("segment_id")] = a.get("label_id")
    return m if set(m) == {s["segment_id"] for s in case["segments"]} else None


def s7_pair(parsed1, parsed2, case):
    p1, p2 = _partition(parsed1, case), _partition(parsed2, case)
    if p1 is None or p2 is None:
        return {"comparable": False, "reason": "a run does not label every text exactly once"}

    def stats(ids):
        a, b = [p1[i] for i in ids], [p2[i] for i in ids]
        return {"texts": len(ids), "adjusted_rand": round(adjusted_rand(a, b), 4),
                "pairwise_agreement": round(pairwise_agreement(a, b), 4),
                "pairwise_agreement_by_chance": round(chance_agreement(a, b), 4),
                "labels_used_run1": len(set(a)), "labels_used_run2": len(set(b))}

    all_ids = [s["segment_id"] for s in case["segments"]]
    cur_ids = [s["segment_id"] for s in case["segments"] if s["period"] == "current"]
    return {"comparable": True, "all_texts": stats(all_ids), "current_month_texts": stats(cur_ids)}


# ------------------------------------------------------------------------------- brands
def load_brand_strings(products_csv):
    """Every brand string in the product table (Korean name, alias, full name, length >= 2) and
    every full product name. ALL of them, not the control's: this scan must not need to know
    which product case 5 is. A review can still name a product without its brand or full name;
    that residual risk is disclosed in the packet, not solved here."""
    out = set()
    with open(products_csv, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            for col in ("brand_name_ko", "brand_alias", "brand_full_name"):
                for part in re.split(r"[,;/|]", (row.get(col) or "")):
                    part = part.strip()
                    if len(part) >= 2:
                        out.add(part)
            name = (row.get("product_name_ko") or "").strip()
            if len(name) >= 2:
                out.add(name)
    return sorted(out, key=lambda s: (-len(s), s))


def brand_regex(brands):
    if not brands:
        return None
    parts = []
    for b in brands:
        e = re.escape(b)
        parts.append(rf"(?<![A-Za-z0-9]){e}(?![A-Za-z0-9])" if b.isascii() else e)
    return re.compile("|".join(parts), re.IGNORECASE)


def brand_spans(text, rx):
    if rx is None or not isinstance(text, str):
        return []
    return [m.span() for m in rx.finditer(text)]


# ------------------------------------------------------------------- loading and running
def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def load_records(out_dir):
    recs = []
    for p in sorted(Path(out_dir).glob("probe_output_case*_*_r*.json")):
        m = RECORD_RE.match(p.name)
        if not m:
            continue
        raw = p.read_bytes()
        rec = json.loads(raw.decode("utf-8"))
        cid, arm, label, run = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        if rec.get("case_id") != cid or rec.get("arm") != arm:
            raise R.InputRefused(f"{p.name}: case_id/arm inside the file do not match its name")
        recs.append({"file": p.name, "case": cid, "arm": arm, "prompt_label": label, "run": run,
                     "md5": md5_bytes(raw), "record": rec})
    recs.sort(key=lambda r: (r["case"], r["arm"], r["run"]))
    return recs


def check_all(input_dir, out_dir, products_csv):
    """Run every check. Returns the full result dict (holds model text: detail file only)."""
    manifest, manifest_md5 = R.load_manifest(input_dir)
    cases, case_md5 = {}, {}
    for cid in R.CASE_IDS:
        cases[cid], case_md5[cid] = R.load_case(input_dir, cid, manifest)
    records = load_records(out_dir)
    brands = load_brand_strings(products_csv)
    brx = brand_regex(brands)

    out = {"checks_version": CHECKS_VERSION, "runner_version": R.RUNNER_VERSION,
           "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "manifest_md5": manifest_md5, "case_file_md5": {str(k): v for k, v in case_md5.items()},
           "brand_strings": len(brands), "records": [], "s7": {}, "brand_scan": {}}

    by_key = {}
    for r in records:
        rec, cid, arm = r["record"], r["case"], r["arm"]
        case = cases[cid]
        item = {"file": r["file"], "md5": r["md5"], "case": cid, "arm": arm, "run": r["run"],
                "status": rec.get("status"), "case_file_md5_matches": rec.get("case_file_md5") == case_md5[cid],
                "structure_problems": rec.get("structure_problems") or [], "runner_flags": rec.get("flags") or [],
                "usage": rec.get("usage"), "est_cost_usd": rec.get("est_cost_usd"),
                "duration_s": rec.get("duration_s"), "served_model": rec.get("served_model"),
                "started_at_utc": rec.get("started_at_utc")}
        parsed = rec.get("parsed")
        if parsed is None:
            item.update({"s2": None, "scan_hits": []})
        else:
            if arm == "A":
                item["s2"] = s2_readout(parsed, case)
            else:
                item["s2"] = s2_labeling(parsed, case)
            agree, n_cmp = s2_agrees_with_runner(item["s2"]["rows"], rec.get("quote_checks") or [])
            item["s2_agrees_with_runner"] = agree
            item["s2_rows_compared"] = n_cmp
            item["scan_hits"], item["scan_strings"] = scan_record_counted(arm, parsed)
        out["records"].append(item)
        by_key[(cid, arm, r["run"])] = parsed

    for cid in S7_CASES:
        p1, p2 = by_key.get((cid, "B", 1)), by_key.get((cid, "B", 2))
        out["s7"][str(cid)] = (s7_pair(p1, p2, cases[cid]) if p1 is not None and p2 is not None
                               else {"comparable": False, "reason": "run 1 or run 2 missing"})

    for cid in R.CASE_IDS:
        segs = cases[cid]["segments"]
        cur = [s for s in segs if s["period"] == "current"]
        with_brand = [s["segment_id"] for s in segs if brand_spans(s["text"], brx)]
        cur_brand = [s["segment_id"] for s in cur if brand_spans(s["text"], brx)]
        out["brand_scan"][str(cid)] = {"texts": len(segs), "texts_with_brand": len(with_brand),
                                       "current_texts_with_brand": len(cur_brand),
                                       "ids_with_brand": with_brand}
    return out, cases, records


# ------------------------------------------------------------------------------- log output
def _fmt_s2(item):
    s2 = item["s2"]
    if s2 is None:
        return "no parsed answer"
    if item["arm"] == "A":
        return (f"S2 {'PASS' if s2['strict_pass'] else 'FAIL'}: {s2['exact_substring']}/{s2['entries']} exact quotes, "
                f"{s2['ids_in_case']}/{s2['entries']} ids in case, {s2['current_period']}/{s2['entries']} current-period")
    cov = s2["coverage"]
    return (f"S2 example quotes {'PASS' if s2['strict_pass'] else 'FAIL'}: {s2['exact_substring']}/{s2['entries']} exact, "
            f"{s2['ids_in_case']}/{s2['entries']} ids in case; coverage {'exact' if s2['coverage_exact'] else 'NOT exact'} "
            f"({cov['assignments']}/{cov['texts']} labelled)")


def log_lines(res):
    lines = ["probe_p4_checks " + CHECKS_VERSION + " | no model text, no per-case S3/S4 counts in this log", ""]
    lines.append("S2, S6 per record (case, arm, run):")
    for it in res["records"]:
        u = it["usage"] or {}
        lines.append(f"  case {it['case']} arm {it['arm']} r{it['run']}: {it['status']} | {_fmt_s2(it)} | "
                     f"runner agrees: {it.get('s2_agrees_with_runner')} | case file md5 ok: {it['case_file_md5_matches']} | "
                     f"in={u.get('prompt_tokens')} out={u.get('completion_tokens')} {it['duration_s']}s ~${it['est_cost_usd']}")
    tot = sum((it["est_cost_usd"] or 0) for it in res["records"])
    sec = sum((it["duration_s"] or 0) for it in res["records"])
    lines += ["", f"S6 total over {len(res['records'])} records: ~${tot:.3f}, {sec:.0f} s of model time", ""]
    lines.append("S3 / S4 word scans (candidates for a human read; totals over ALL records of each arm):")
    for arm in ("A", "B"):
        c = Counter()
        n = strings = 0
        for it in res["records"]:
            if it["arm"] != arm or it["s2"] is None:
                continue
            n += 1
            strings += it["scan_strings"]
            for h in it["scan_hits"]:
                c[(h["group"], h["tier"])] += 1
        lines.append(f"  Arm {arm} ({n} records, {strings} strings scanned): S3 prompt-tier {c[('S3','prompt')]}, extended {c[('S3','extended')]}; "
                     f"S4 prompt-tier {c[('S4','prompt')]}, extended {c[('S4','extended')]}")
    lines += ["", "S7 (Arm B, run 1 against run 2; run 1 is the run of record):"]
    for cid, r in res["s7"].items():
        if not r.get("comparable"):
            lines.append(f"  case {cid}: not comparable ({r.get('reason')})")
            continue
        for key, name in (("all_texts", "all texts"), ("current_month_texts", "this-month texts")):
            s = r[key]
            lines.append(f"  case {cid}, {name} (n={s['texts']}): adjusted Rand {s['adjusted_rand']}, pairwise agreement "
                         f"{s['pairwise_agreement']} (by chance {s['pairwise_agreement_by_chance']}), "
                         f"labels used {s['labels_used_run1']} / {s['labels_used_run2']}")
    lines += ["", "Brand scan of the case texts (all brand strings in the product table; the control is not singled out):"]
    for cid, b in res["brand_scan"].items():
        if int(cid) == R.CONTROL_CASE:
            lines.append(f"  case {cid}: any text names a brand: {b['texts_with_brand'] > 0}")
        else:
            lines.append(f"  case {cid}: {b['texts_with_brand']} of {b['texts']} texts name a brand "
                         f"({b['current_texts_with_brand']} this-month)")
    return lines


def build_parser():
    p = argparse.ArgumentParser(description="Probe automatic checks S2, S3, S4, S6, S7.")
    p.add_argument("--input-dir", default=R.DEFAULT_INPUT_DIR)
    p.add_argument("--out-dir", default=R.DEFAULT_OUT_DIR)
    p.add_argument("--products", default=DEFAULT_PRODUCTS)
    p.add_argument("--detail", default=str(DEFAULT_DETAIL))
    p.add_argument("--force", action="store_true", help="overwrite an existing detail file")
    return p


def main(argv=None):
    R._utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        detail = Path(args.detail)
        if detail.exists() and not args.force:
            parser.error(f"{detail.name} already exists; pass --force to regenerate")
        res, _cases, _records = check_all(args.input_dir, args.out_dir, args.products)
        detail.parent.mkdir(parents=True, exist_ok=True)
        with open(detail, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        for line in log_lines(res):
            print(line)
        print(f"\nwrote {detail} ({detail.stat().st_size} bytes). It holds model text and per-case hits: "
              f"do not open it before the blind review; the packet shows the same by case code.")
        return 0
    except R.InputRefused as exc:
        parser.error(str(exc))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - a traceback could carry case data
        print(f"\nUNEXPECTED ERROR ({type(exc).__name__}): details withheld so nothing from a case leaks into "
              f"this log. Set PROBE_DEBUG=1 and run again to see a traceback.")
        if os.environ.get("PROBE_DEBUG") == "1":
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
