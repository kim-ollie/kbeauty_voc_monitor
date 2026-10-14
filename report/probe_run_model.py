#!/usr/bin/env python3
"""
probe_run_model.py  --  the complaint-readout probe's runner.

What this does
    Reads probe_inputs/case_N.json (built by probe_p1_extract.py), sends the model ONLY the
    four fields it is allowed to see (segment_id, period, unclustered, text) plus one of two
    prompt files, and saves the raw answer and its provenance to probe_outputs/.
        Arm A  = prose readout (probe_readout_prompt_v0.txt), answer schema probe-readout-v0
        Arm B  = one label per segment (probe_labeling_prompt_v0.txt), schema probe-labeling-v0
    This script reads OPENAI_API_KEY from .env only when it
    is about to call the API.

What this does NOT do (that is probe_p4_checks.py)
    No verdict on wording, numbers, causes or advice; no scoring of usefulness. "structure_ok"
    means only: the JSON has the agreed shape and every segment id it cites exists. Quote
    checks and rule flags are measured and recorded, not enforced. Read every output yourself.

Blindness rules kept here (case 5 is the sealed negative control)
    * The model never sees the score, theme names, product id or month.
    * The log prints STATUS LINES ONLY: never model text, never a segment's text.
    * Nothing the runner writes to a log names a product or a month, and a case file's
      identity fields are never copied into a record.
    * --print-request withholds case 5's user message entirely.
    * An unexpected error prints its class name only; set PROBE_DEBUG=1 to see a traceback.
    * The runner never opens the sealed selection file.
    * Do not open the probe_outputs/ records for case 5 before the blind review: the review
      packet (probe_p4_checks.py) is built for that. Model answers can quote reviews that name a brand.

Design rules carried over from run_report_model.py
    * No temperature. reasoning_effort defaults to "high" (override with --reasoning-effort).
    * No retry on our side: an API error, a refusal or a parse failure is data.
    * Never overwrites: each call writes a new ..._r1.json, ..._r2.json, ...
    * A prompt file, once used, is immutable: edit it and save it under a new name
      (probe_readout_prompt_v1.txt) or the run is refused.
    * More than 6 calls in one command needs --allow-many (dry-run before full run).

Usage (from the project folder)
    python probe_run_model.py --case 1 --arm A --print-request     # no API call
    python probe_run_model.py --case 1 --arm A                     # one real call
    python probe_run_model.py --case 1 2 3 4 5 --arm A B --allow-many
    python probe_run_model.py --case 1 3 --arm B --repeat 2        # S7 repeats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

RUNNER_VERSION = "probe-p2-run-v0"
RECORD_SCHEMA_VERSION = "probe-p2-record-v0"
INPUT_SCHEMA_VERSION = "probe-input-v0"
READOUT_SCHEMA_VERSION = "probe-readout-v0"
LABELING_SCHEMA_VERSION = "probe-labeling-v0"

# Placeholder-grade: same model and prices as run_report_model.py (OpenAI model page,
# 2026-09-24). The first real call confirms it (records store requested and served model).
DEFAULT_MODEL = "gpt-5.6-sol"
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_REASONING_EFFORT = "high"
LIST_PRICE_USD_PER_M = {"input": 4.00, "cached_input": 0.40, "output": 20.00}
PRICE_BASIS = "OpenAI model page 2026-09-24 list prices; estimate only"

DEFAULT_INPUT_DIR = "probe_inputs"
DEFAULT_OUT_DIR = "probe_outputs"
ARMS = ("A", "B")
ARM_NAMES = {"A": "readout", "B": "labeling"}
OUTPUT_SCHEMA_BY_ARM = {"A": READOUT_SCHEMA_VERSION, "B": LABELING_SCHEMA_VERSION}
DEFAULT_PROMPT_FILES = {"A": "probe_readout_prompt_v0.txt", "B": "probe_labeling_prompt_v0.txt"}
PROMPT_FILE_PREFIXES = {"A": "probe_readout_prompt_", "B": "probe_labeling_prompt_"}

CASE_IDS = (1, 2, 3, 4, 5)
CONTROL_CASE = 5
MAX_REPEAT = 3
MAX_CALLS_WITHOUT_ALLOW_MANY = 6

# ---------------------------------------------------------------- case-file layout
CASE_KEYS = (
    "schema", "case_id", "product_id", "as_of_month", "prior_months",
    "counts", "prior_sampling", "segments",
)
SEGMENT_KEYS = (
    "segment_id", "month", "period", "score", "unclustered", "theme_groups", "topic_ids", "text",
)
# The model sees exactly these four fields and nothing else.
MODEL_FACING = ("segment_id", "period", "unclustered", "text")
PERIODS = ("current", "prior")

# Caps the prompts state. Over-cap is flagged, not failed.
CAPS = {
    "claim_words": 35,
    "summary_words": 40,
    "gloss_words": 25,
    "points_per_section": 3,
    "evidence_min": 1,
    "evidence_max": 3,
    "quote_min_chars": 4,
    "quote_max_chars": 120,
    "labels_max": 12,
    "label_name_words": 4,
    "label_note_words": 20,
}

READOUT_SECTIONS = ("what_reviewers_say", "what_looks_new", "in_unclustered_segments")
READOUT_TOP_KEYS = ("schema", "overall", "what_reviewers_say", "what_looks_new", "in_unclustered_segments")
POINT_KEYS = ("claim", "basis", "evidence")
EVIDENCE_KEYS = ("segment_id", "quote", "gloss")
BASIS_VALUES = ("single_review", "multiple_reviews")
STANDS_OUT_VALUES = ("yes", "no")
LABELING_TOP_KEYS = ("schema", "labels", "assignments")
LABEL_KEYS = ("label_id", "name", "note", "example_segment_id", "example_quote")
ASSIGNMENT_KEYS = ("segment_id", "label_id")


class InputRefused(Exception):
    """A file or setting the runner will not send. Messages never carry case identity."""


# ------------------------------------------------------------------- small helpers
def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_str(x):
    return isinstance(x, str) and x.strip() != ""


def _words(s):
    return len(s.split())


def _md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def _md5_text(s):
    return _md5_bytes(s.encode("utf-8"))


_IDENT_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")


def _safe_key(k):
    """A model-supplied key name is echoed only if it looks like a plain identifier."""
    return k if isinstance(k, str) and _IDENT_RE.match(k) else "<non-identifier>"


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- input handling
def load_manifest(input_dir):
    path = Path(input_dir) / "manifest.json"
    if not path.is_file():
        raise InputRefused(f"{path.name} not found in {input_dir}; run probe_p1_extract.py first")
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise InputRefused(f"{path.name} is not valid UTF-8 JSON")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise InputRefused(f"{path.name} has no 'cases' list")
    return manifest, _md5_bytes(raw)


def load_case(input_dir, case_id, manifest):
    """Return (case, md5). Every refusal message is free of product id and month."""
    if case_id not in CASE_IDS:
        raise InputRefused(f"case {case_id} is not one of {list(CASE_IDS)}")
    fname = f"case_{case_id}.json"
    path = Path(input_dir) / fname
    if not path.is_file():
        raise InputRefused(f"{fname} not found in {input_dir}")
    raw = path.read_bytes()
    entry = next((e for e in manifest["cases"] if isinstance(e, dict) and e.get("case_id") == case_id), None)
    if entry is None:
        raise InputRefused(f"case {case_id} is not listed in manifest.json")
    md5 = _md5_bytes(raw)
    if entry.get("md5") != md5:
        raise InputRefused(
            f"{fname}: md5 {md5} does not match manifest.json ({entry.get('md5')}); "
            f"the case file changed after it was built. Rebuild with probe_p1_extract.py"
        )
    try:
        case = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise InputRefused(f"{fname} is not valid UTF-8 JSON")

    if not isinstance(case, dict) or set(case) != set(CASE_KEYS):
        raise InputRefused(f"{fname}: top-level keys are not the probe-input-v0 layout")
    if case["schema"] != INPUT_SCHEMA_VERSION:
        raise InputRefused(f"{fname}: schema is not {INPUT_SCHEMA_VERSION}")
    if case["case_id"] != case_id:
        raise InputRefused(f"{fname}: case_id inside the file does not match its name")
    segs = case["segments"]
    if not isinstance(segs, list) or not segs:
        raise InputRefused(f"{fname}: no segments")
    seen, phase = set(), 0  # phase 0 = still in current, 1 = in prior
    n_cur = n_cur_unc = n_pri = n_pri_unc = 0
    for i, s in enumerate(segs):
        if not isinstance(s, dict) or set(s) != set(SEGMENT_KEYS):
            raise InputRefused(f"{fname}: segments[{i}] does not have the probe-input-v0 keys")
        if not _is_int(s["segment_id"]):
            raise InputRefused(f"{fname}: segments[{i}] segment_id is not an integer")
        if s["segment_id"] in seen:
            raise InputRefused(f"{fname}: segments[{i}] repeats an earlier segment_id")
        seen.add(s["segment_id"])
        if s["period"] not in PERIODS:
            raise InputRefused(f"{fname}: segments[{i}] period is not 'current' or 'prior'")
        if s["period"] == "prior":
            phase = 1
        elif phase == 1:
            raise InputRefused(f"{fname}: segments[{i}] is current but follows a prior segment")
        if not isinstance(s["unclustered"], bool):
            raise InputRefused(f"{fname}: segments[{i}] unclustered is not true/false")
        if not isinstance(s["theme_groups"], list) or s["unclustered"] != (len(s["theme_groups"]) == 0):
            raise InputRefused(f"{fname}: segments[{i}] unclustered disagrees with theme_groups")
        if not _is_str(s["text"]):
            raise InputRefused(f"{fname}: segments[{i}] text is empty or not a string")
        if s["period"] == "current":
            n_cur += 1
            n_cur_unc += s["unclustered"]
        else:
            n_pri += 1
            n_pri_unc += s["unclustered"]
    counts = case["counts"]
    if not isinstance(counts, dict):
        raise InputRefused(f"{fname}: the counts block is not an object")
    want = (counts.get("current"), counts.get("current_unclustered"),
            counts.get("prior_kept"), counts.get("prior_unclustered_kept"))
    if want != (n_cur, n_cur_unc, n_pri, n_pri_unc):
        raise InputRefused(f"{fname}: segment counts do not match the counts block")
    if n_cur == 0:
        raise InputRefused(f"{fname}: no current-period segments")
    return case, md5


def model_view(case):
    """What the model sees. An explicit allow-list; nothing is computed here."""
    return {"segments": [{k: s[k] for k in MODEL_FACING} for s in case["segments"]]}


def render_user_message(view):
    """One compact JSON object per physical line, so a reader can eyeball or diff it.
    json.loads(render_user_message(v)) == v."""
    rows = [json.dumps(seg, ensure_ascii=False) for seg in view["segments"]]
    return '{"segments": [\n' + ",\n".join("  " + r for r in rows) + "\n]}"


def build_messages(system_prompt, view):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": render_user_message(view)},
    ]


def prompt_label(arm, prompt_file):
    stem = Path(prompt_file).stem
    prefix = PROMPT_FILE_PREFIXES[arm]
    core = stem[len(prefix):] if stem.startswith(prefix) else stem
    return "prompt-" + re.sub(r"[^A-Za-z0-9.\-]+", "-", core)


def load_prompt(prompt_file):
    path = Path(prompt_file)
    if not path.is_file():
        raise InputRefused(f"prompt file {path} not found")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputRefused(f"prompt file {path} is not UTF-8: {exc}")
    if not text.strip():
        raise InputRefused(f"prompt file {path} is empty")
    return text, _md5_bytes(raw)


# ------------------------------------------------------------------ response formats
def _obj(props):
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


def build_response_format(arm):
    """Strict JSON schema for the arm's answer. Counts and lengths are checked in code."""
    if arm == "A":
        point = _obj({
            "claim": {"type": "string"},
            "basis": {"type": "string", "enum": list(BASIS_VALUES)},
            "evidence": {
                "type": "array",
                "items": _obj({
                    "segment_id": {"type": "integer"},
                    "quote": {"type": "string"},
                    "gloss": {"type": "string"},
                }),
            },
        })
        schema = _obj({
            "schema": {"type": "string", "enum": [READOUT_SCHEMA_VERSION]},
            "overall": _obj({
                "stands_out": {"type": "string", "enum": list(STANDS_OUT_VALUES)},
                "summary": {"type": "string"},
            }),
            "what_reviewers_say": {"type": "array", "items": point},
            "what_looks_new": {"type": "array", "items": point},
            "in_unclustered_segments": {"type": "array", "items": point},
        })
        assert tuple(schema["properties"]) == READOUT_TOP_KEYS
        name = "probe_readout_v0"
    elif arm == "B":
        schema = _obj({
            "schema": {"type": "string", "enum": [LABELING_SCHEMA_VERSION]},
            "labels": {
                "type": "array",
                "items": _obj({
                    "label_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "note": {"type": "string"},
                    "example_segment_id": {"type": "integer"},
                    "example_quote": {"type": "string"},
                }),
            },
            "assignments": {
                "type": "array",
                "items": _obj({"segment_id": {"type": "integer"}, "label_id": {"type": "integer"}}),
            },
        })
        assert tuple(schema["properties"]) == LABELING_TOP_KEYS
        name = "probe_labeling_v0"
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


# ------------------------------------------------------------------- quote checking
def _normalise(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def quote_status(quote, text):
    """How a quoted string relates to the text it cites.

    verbatim                     copied exactly and inside one line
    spans_line_break             copied exactly but crosses a line break (rule 5 forbids it)
    whitespace_or_unicode_differs  matches only after Unicode/whitespace normalisation
    not_found                    does not match even then
    """
    if quote and quote in text:
        return "spans_line_break" if ("\n" in quote or "\r" in quote) else "verbatim"
    nq = _normalise(quote)
    if nq and nq in _normalise(text):
        return "whitespace_or_unicode_differs"
    return "not_found"


def _quote_check(where, seg, quote, flags):
    """Return one quote_checks entry and append flags. `seg` may be None (id not in the case)."""
    entry = {"where": where, "segment_id": None, "status": None, "chars": len(quote)}
    if seg is None:
        entry["status"] = "segment_not_in_case"
        return entry
    entry["segment_id"] = seg["segment_id"]
    entry["status"] = quote_status(quote, seg["text"])
    if entry["status"] != "verbatim":
        flags.append(f"{where}: quote_{entry['status']}")
    if not (CAPS["quote_min_chars"] <= len(quote) <= CAPS["quote_max_chars"]):
        flags.append(f"{where}: quote_length_{len(quote)}_outside_"
                     f"{CAPS['quote_min_chars']}_{CAPS['quote_max_chars']}")
    return entry


# ------------------------------------------------------------------ structure checks
def _index(case):
    return {s["segment_id"]: s for s in case["segments"]}


def check_readout(parsed, case):
    """Arm A. Returns dict(problems, flags, quote_checks, derived).

    problems (fatal, status structure_fail): wrong shape or types, or a cited segment id that
    is not in the case. flags (not fatal, counted by probe_p4_checks.py): rule breaches the prompt states
    (caps, evidence from a prior-period text, an unclustered section citing a themed text,
    stands_out disagreeing with what_looks_new, quote problems).
    No string in problems/flags carries model-written text.
    """
    problems, flags, qchecks = [], [], []
    out = {"problems": problems, "flags": flags, "quote_checks": qchecks, "derived": {}}
    if not isinstance(parsed, dict):
        problems.append("output_is_not_an_object")
        return out
    for k in READOUT_TOP_KEYS:
        if k not in parsed:
            problems.append(f"missing_key:{k}")
    for k in parsed:
        if k not in READOUT_TOP_KEYS:
            problems.append(f"unexpected_key:{_safe_key(k)}")
    if "schema" in parsed and parsed["schema"] != READOUT_SCHEMA_VERSION:
        problems.append("wrong_schema_value")

    idx = _index(case)
    stands_out = None
    overall = parsed.get("overall")
    if "overall" in parsed:
        if not isinstance(overall, dict) or set(overall) != {"stands_out", "summary"}:
            problems.append("overall_bad_shape")
        else:
            if overall["stands_out"] not in STANDS_OUT_VALUES:
                problems.append("overall.stands_out_not_yes_or_no")
            else:
                stands_out = overall["stands_out"]
            if not _is_str(overall["summary"]):
                problems.append("overall.summary_not_a_nonempty_string")
            elif _words(overall["summary"]) > CAPS["summary_words"]:
                flags.append(f"overall.summary: {_words(overall['summary'])} words > {CAPS['summary_words']}")

    n_points = {}
    for sec in READOUT_SECTIONS:
        if sec not in parsed:
            continue
        pts = parsed[sec]
        if not isinstance(pts, list):
            problems.append(f"{sec}_not_a_list")
            continue
        n_points[sec] = len(pts)
        if len(pts) > CAPS["points_per_section"]:
            flags.append(f"{sec}: {len(pts)} points > {CAPS['points_per_section']}")
        for i, p in enumerate(pts):
            where = f"{sec}[{i}]"
            if not isinstance(p, dict) or set(p) != set(POINT_KEYS):
                problems.append(f"{where}_bad_shape")
                continue
            if not _is_str(p["claim"]):
                problems.append(f"{where}.claim_not_a_nonempty_string")
            elif _words(p["claim"]) > CAPS["claim_words"]:
                flags.append(f"{where}.claim: {_words(p['claim'])} words > {CAPS['claim_words']}")
            if p["basis"] not in BASIS_VALUES:
                problems.append(f"{where}.basis_not_in_enum")
            ev = p["evidence"]
            if not isinstance(ev, list):
                problems.append(f"{where}.evidence_not_a_list")
                continue
            if not (CAPS["evidence_min"] <= len(ev) <= CAPS["evidence_max"]):
                flags.append(f"{where}: {len(ev)} evidence entries outside "
                             f"{CAPS['evidence_min']}..{CAPS['evidence_max']}")
            for j, e in enumerate(ev):
                ewhere = f"{where}.evidence[{j}]"
                if not isinstance(e, dict) or set(e) != set(EVIDENCE_KEYS):
                    problems.append(f"{ewhere}_bad_shape")
                    continue
                if not _is_int(e["segment_id"]):
                    problems.append(f"{ewhere}.segment_id_not_an_integer")
                    continue
                if not _is_str(e["quote"]):
                    problems.append(f"{ewhere}.quote_not_a_nonempty_string")
                    continue
                if not isinstance(e["gloss"], str):
                    problems.append(f"{ewhere}.gloss_not_a_string")
                elif _words(e["gloss"]) > CAPS["gloss_words"]:
                    flags.append(f"{ewhere}.gloss: {_words(e['gloss'])} words > {CAPS['gloss_words']}")
                seg = idx.get(e["segment_id"])
                if seg is None:
                    problems.append(f"{ewhere}_segment_not_in_case")
                    qchecks.append(_quote_check(ewhere, None, e["quote"], flags))
                    continue
                if seg["period"] != "current":
                    flags.append(f"{ewhere}: cites a prior-period text")
                if sec == "in_unclustered_segments" and not seg["unclustered"]:
                    flags.append(f"{ewhere}: unclustered section cites a themed text")
                qchecks.append(_quote_check(ewhere, seg, e["quote"], flags))

    if stands_out is not None and "what_looks_new" in n_points:
        if (stands_out == "yes") != (n_points["what_looks_new"] > 0):
            flags.append("overall.stands_out disagrees with what_looks_new")
    out["derived"] = {"points_per_section": n_points}
    return out


def check_labeling(parsed, case):
    """Arm B. Returns dict(problems, flags, quote_checks, derived). Same conventions as
    check_readout. derived.label_counts is computed here by code, never by the model."""
    problems, flags, qchecks = [], [], []
    out = {"problems": problems, "flags": flags, "quote_checks": qchecks, "derived": {}}
    if not isinstance(parsed, dict):
        problems.append("output_is_not_an_object")
        return out
    for k in LABELING_TOP_KEYS:
        if k not in parsed:
            problems.append(f"missing_key:{k}")
    for k in parsed:
        if k not in LABELING_TOP_KEYS:
            problems.append(f"unexpected_key:{_safe_key(k)}")
    if "schema" in parsed and parsed["schema"] != LABELING_SCHEMA_VERSION:
        problems.append("wrong_schema_value")

    idx = _index(case)
    label_ids = []
    labels = parsed.get("labels")
    if "labels" in parsed:
        if not isinstance(labels, list) or not labels:
            problems.append("labels_not_a_nonempty_list")
        else:
            if len(labels) > CAPS["labels_max"]:
                flags.append(f"labels: {len(labels)} labels > {CAPS['labels_max']}")
            for i, lab in enumerate(labels):
                where = f"labels[{i}]"
                if not isinstance(lab, dict) or set(lab) != set(LABEL_KEYS):
                    problems.append(f"{where}_bad_shape")
                    continue
                if not _is_int(lab["label_id"]):
                    problems.append(f"{where}.label_id_not_an_integer")
                elif lab["label_id"] in label_ids:
                    problems.append(f"{where}.label_id_repeats")
                else:
                    label_ids.append(lab["label_id"])
                if not _is_str(lab["name"]):
                    problems.append(f"{where}.name_not_a_nonempty_string")
                elif _words(lab["name"]) > CAPS["label_name_words"]:
                    flags.append(f"{where}.name: {_words(lab['name'])} words > {CAPS['label_name_words']}")
                if not _is_str(lab["note"]):
                    problems.append(f"{where}.note_not_a_nonempty_string")
                elif _words(lab["note"]) > CAPS["label_note_words"]:
                    flags.append(f"{where}.note: {_words(lab['note'])} words > {CAPS['label_note_words']}")
                if not _is_int(lab["example_segment_id"]):
                    problems.append(f"{where}.example_segment_id_not_an_integer")
                    continue
                if not _is_str(lab["example_quote"]):
                    problems.append(f"{where}.example_quote_not_a_nonempty_string")
                    continue
                seg = idx.get(lab["example_segment_id"])
                if seg is None:
                    problems.append(f"{where}_example_segment_not_in_case")
                    qchecks.append(_quote_check(where, None, lab["example_quote"], flags))
                else:
                    qchecks.append(_quote_check(where, seg, lab["example_quote"], flags))
            if sorted(label_ids) != list(range(1, len(labels) + 1)) and len(label_ids) == len(labels):
                flags.append("label_ids are not 1..n")

    counts = {}
    assigned_ok = 0
    assignments = parsed.get("assignments")
    if "assignments" in parsed:
        if not isinstance(assignments, list):
            problems.append("assignments_not_a_list")
        else:
            got_ids, shape_bad = [], 0
            for i, a in enumerate(assignments):
                where = f"assignments[{i}]"
                if not isinstance(a, dict) or set(a) != set(ASSIGNMENT_KEYS):
                    problems.append(f"{where}_bad_shape")
                    shape_bad += 1
                    continue
                if not _is_int(a["segment_id"]) or not _is_int(a["label_id"]):
                    problems.append(f"{where}_ids_not_integers")
                    shape_bad += 1
                    continue
                got_ids.append(a["segment_id"])
                if a["label_id"] not in label_ids:
                    problems.append(f"{where}_label_id_not_defined")
                    continue
                seg = idx.get(a["segment_id"])
                if seg is not None:
                    c = counts.setdefault(a["label_id"], {"current": 0, "prior": 0})
                    c[seg["period"]] += 1
            want_ids = [s["segment_id"] for s in case["segments"]]
            missing = len(set(want_ids) - set(got_ids))
            extra = len(set(got_ids) - set(want_ids))
            dupes = len(got_ids) - len(set(got_ids))
            if missing or extra or dupes:
                problems.append(f"assignments_do_not_cover_input_once_each:"
                                f"missing={missing},extra={extra},repeated={dupes}")
            elif got_ids != want_ids and not shape_bad:
                flags.append("assignments are not in the input's order")
            assigned_ok = len(set(got_ids) & set(want_ids))
    out["derived"] = {
        "n_segments": len(case["segments"]),
        "n_labels": len(labels) if isinstance(labels, list) else 0,
        "n_assigned_once_or_more": assigned_ok,
        "label_counts": {str(k): counts[k] for k in sorted(counts)},
    }
    return out


def check_output(arm, parsed, case):
    return (check_readout if arm == "A" else check_labeling)(parsed, case)


# ------------------------------------------------------------------ files and safety
_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?\s*```\s*$", re.DOTALL)


def strip_code_fences(text):
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1), True
    return text, False


_SK_RE = re.compile(r"sk-[A-Za-z0-9_\-\*\.]{6,}")


def redact(text, secrets=()):
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "[REDACTED]")
    return _SK_RE.sub("sk-[REDACTED]", text)


def record_path(out_dir, case_id, arm, label, n):
    return Path(out_dir) / f"probe_output_case{case_id}_{arm}_{label}_r{n}.json"


def next_run_index(out_dir, case_id, arm, label):
    d = Path(out_dir)
    if not d.is_dir():
        return 1
    pat = re.compile(rf"^probe_output_case{case_id}_{arm}_{re.escape(label)}_r(\d+)\.json$")
    found = [int(m.group(1)) for f in d.iterdir() if (m := pat.match(f.name))]
    return (max(found) + 1) if found else 1


def check_prompt_unchanged(out_dir, arm, label, md5):
    """A prompt label, once used for an arm, must keep the same bytes."""
    d = Path(out_dir)
    if not d.is_dir():
        return
    for f in sorted(d.glob(f"probe_output_case*_{arm}_{label}_r*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        old = rec.get("prompt_md5")
        if old and old != md5:
            raise InputRefused(
                f"prompt '{label}' for arm {arm} was already used ({f.name}) with md5 {old}, "
                f"but the file now has md5 {md5}. Save your edit under a new name, e.g. "
                f"{PROMPT_FILE_PREFIXES[arm]}v1.txt, so every output stays tied to its exact prompt."
            )


def write_record(path, record):
    """Exclusive create: raises FileExistsError rather than overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_usage(usage):
    if usage is None:
        return None
    return {
        "prompt_tokens": _get(usage, "prompt_tokens"),
        "completion_tokens": _get(usage, "completion_tokens"),
        "total_tokens": _get(usage, "total_tokens"),
        "cached_tokens": _get(_get(usage, "prompt_tokens_details"), "cached_tokens"),
        "reasoning_tokens": _get(_get(usage, "completion_tokens_details"), "reasoning_tokens"),
    }


def estimate_cost(u):
    if not u or u.get("prompt_tokens") is None or u.get("completion_tokens") is None:
        return None
    cached = u.get("cached_tokens") or 0
    fresh = u["prompt_tokens"] - cached
    cost = (
        fresh * LIST_PRICE_USD_PER_M["input"]
        + cached * LIST_PRICE_USD_PER_M["cached_input"]
        + u["completion_tokens"] * LIST_PRICE_USD_PER_M["output"]
    ) / 1_000_000
    return round(cost, 5)


# --------------------------------------------------------------------- the call
def call_once(
    client,
    *,
    arm,
    case,
    case_md5,
    manifest_md5,
    messages,
    response_format,
    model,
    reasoning_effort,
    max_completion_tokens,
    prompt_meta,
    secrets=(),
):
    """One API call -> one record. Never raises for API or parse problems.

    The record holds no product id and no month; the case is identified by its number and
    the md5 of its file only.
    """
    segs = case["segments"]
    request_params = {
        "model": model,
        "temperature": "not passed",
        "reasoning_effort": reasoning_effort if reasoning_effort else "not passed (API default)",
        "max_completion_tokens": max_completion_tokens if max_completion_tokens else "not passed",
        "response_format": f"json_schema strict: {response_format['json_schema']['name']}",
    }
    record = {
        "record_schema": RECORD_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "case_id": case["case_id"],
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "output_schema_version": OUTPUT_SCHEMA_BY_ARM[arm],
        "prompt_label": prompt_meta["label"],
        "prompt_file": prompt_meta["file"],
        "prompt_md5": prompt_meta["md5"],
        "case_file_md5": case_md5,
        "manifest_md5": manifest_md5,
        "model_view_md5": _md5_text(messages[1]["content"]),
        "n_segments": len(segs),
        "n_current": sum(1 for s in segs if s["period"] == "current"),
        "requested_model": model,
        "request_params": request_params,
        "openai_sdk_version": getattr(sys.modules.get("openai"), "__version__", None),
        "started_at_utc": utc_now_iso(),
        "finished_at_utc": None,
        "duration_s": None,
        "served_model": None,
        "response_id": None,
        "finish_reason": None,
        "usage": None,
        "est_cost_usd": None,
        "est_cost_basis": PRICE_BASIS,
        "raw_response": None,
        "fences_stripped": False,
        "parsed": None,
        "status": None,
        "structure_problems": [],
        "flags": [],
        "quote_checks": [],
        "derived": {},
        "error": None,
    }
    kwargs = {"model": model, "messages": messages, "response_format": response_format}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_completion_tokens:
        kwargs["max_completion_tokens"] = max_completion_tokens

    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - every API failure becomes a record
        record["status"] = "api_error"
        record["error"] = redact(f"{type(exc).__name__}: {exc}", secrets)
        record["duration_s"] = round(time.monotonic() - t0, 2)
        record["finished_at_utc"] = utc_now_iso()
        return record
    record["duration_s"] = round(time.monotonic() - t0, 2)
    record["finished_at_utc"] = utc_now_iso()

    record["served_model"] = _get(resp, "model")
    record["response_id"] = _get(resp, "id")
    record["usage"] = extract_usage(_get(resp, "usage"))
    record["est_cost_usd"] = estimate_cost(record["usage"])
    choices = _get(resp, "choices") or []
    if not choices:
        record["status"] = "parse_error"
        record["error"] = "response had no choices"
        return record
    choice = choices[0]
    msg = _get(choice, "message")
    content, refusal = _get(msg, "content"), _get(msg, "refusal")
    record["finish_reason"] = _get(choice, "finish_reason")
    record["raw_response"] = {"content": content, "refusal": refusal}

    if refusal:
        record["status"] = "refusal"
        record["error"] = redact(str(refusal), secrets)
        return record
    if not isinstance(content, str) or not content.strip():
        record["status"] = "parse_error"
        record["error"] = f"empty content (finish_reason={record['finish_reason']})"
        return record
    text, fenced = strip_code_fences(content.strip())
    record["fences_stripped"] = fenced
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        record["status"] = "parse_error"
        record["error"] = f"JSONDecodeError: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        return record
    record["parsed"] = parsed
    res = check_output(arm, parsed, case)
    record["structure_problems"] = res["problems"]
    record["flags"] = res["flags"]
    record["quote_checks"] = res["quote_checks"]
    record["derived"] = res["derived"]
    record["status"] = "structure_fail" if res["problems"] else "structure_ok"
    return record


# ----------------------------------------------------------------------- logging
def status_lines(case_id, arm, n, record):
    """The only per-call log output: status, sizes, counts. Never model text, never a
    segment's text. `record["error"]` is shown only for api_error (a client message, redacted)
    and parse_error (a runner-written string); a refusal's text stays in the record."""
    u = record["usage"] or {}
    lines = [
        f"case {case_id} arm {arm} r{n}: {record['status']}"
        f" | finish={record['finish_reason']} served={record['served_model']}"
        f" | in={u.get('prompt_tokens')} out={u.get('completion_tokens')}"
        f" reasoning={u.get('reasoning_tokens')}"
        f" | {record['duration_s']}s | ~${record['est_cost_usd']}"
    ]
    qc = record["quote_checks"]
    if record["parsed"] is not None:
        verb = sum(1 for q in qc if q["status"] == "verbatim")
        parts = [f"quotes verbatim {verb}/{len(qc)}"]
        if arm == "B":
            d = record["derived"]
            parts.insert(0, f"assigned {d.get('n_assigned_once_or_more', '?')}/{d.get('n_segments', '?')}"
                            f" | labels {d.get('n_labels', '?')}")
        else:
            pps = record["derived"].get("points_per_section", {})
            parts.insert(0, "points " + "/".join(str(pps.get(s, "?")) for s in READOUT_SECTIONS))
        parts.append(f"flags {len(record['flags'])} | problems {len(record['structure_problems'])}")
        lines.append("  " + " | ".join(parts))
    if record["structure_problems"]:
        lines.append(f"  STRUCTURE PROBLEMS: {record['structure_problems']}")
    if record["flags"]:
        lines.append(f"  rule flags (not failures): {record['flags']}")
    if record["status"] == "refusal":
        lines.append("  refusal recorded; its text is in the record file, not in this log")
    elif record["error"]:
        lines.append(f"  ERROR: {record['error']}")
    return lines


def run_calls(
    plan,
    *,
    cases,
    case_md5s,
    manifest_md5,
    out_dir,
    model,
    reasoning_effort,
    max_completion_tokens,
    repeat,
    prompts,
    prompt_metas,
    client,
    secrets=(),
    log=print,
):
    """plan = [(case_id, arm), ...] in call order. Each record is written at once."""
    results = []
    for case_id, arm in plan:
        case = cases[case_id]
        messages = build_messages(prompts[arm], model_view(case))
        response_format = build_response_format(arm)
        meta = prompt_metas[arm]
        for _ in range(repeat):
            n = next_run_index(out_dir, case_id, arm, meta["label"])
            path = record_path(out_dir, case_id, arm, meta["label"], n)
            record = call_once(
                client,
                arm=arm,
                case=case,
                case_md5=case_md5s[case_id],
                manifest_md5=manifest_md5,
                messages=messages,
                response_format=response_format,
                model=model,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_completion_tokens,
                prompt_meta=meta,
                secrets=secrets,
            )
            record["run_index"] = n
            write_record(path, record)
            results.append((path, record))
            for line in status_lines(case_id, arm, n, record):
                log(line)
            log(f"  saved: {path}")
    return results


# ---------------------------------------------------------------------------- CLI
def make_openai_client():
    """Lazy: only reached when a real call is about to happen."""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(f"Missing package ({exc}). Run: pip install openai python-dotenv")
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (expected in .env next to this script)")
    return OpenAI(api_key=key)


def _utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def build_parser():
    p = argparse.ArgumentParser(
        description="Send probe cases to the LLM and record the raw answers."
    )
    p.add_argument("--case", nargs="+", action="extend", type=int, required=True, choices=CASE_IDS,
                   metavar="N", help="one or more of 1..5 (case 5 is the sealed control)")
    p.add_argument("--arm", nargs="+", action="extend", required=True, choices=ARMS,
                   help="A = prose readout, B = segment labeling")
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--readout-prompt", default=DEFAULT_PROMPT_FILES["A"], help="prompt file for arm A")
    p.add_argument("--labeling-prompt", default=DEFAULT_PROMPT_FILES["B"], help="prompt file for arm B")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"default {DEFAULT_MODEL!r} is a placeholder from OpenAI's model page; "
                        "the first call confirms it")
    p.add_argument("--reasoning-effort", choices=REASONING_EFFORTS, default=DEFAULT_REASONING_EFFORT,
                   help=f"default {DEFAULT_REASONING_EFFORT!r} (as in run_report_model.py)")
    p.add_argument("--max-completion-tokens", type=int, default=None,
                   help="omit for no cap (reasoning tokens count against a cap)")
    p.add_argument("--repeat", type=int, default=1,
                   help=f"calls per case and arm (1..{MAX_REPEAT}); use 2 for the S7 repeats")
    p.add_argument("--allow-many", action="store_true",
                   help=f"needed when cases x arms x repeat > {MAX_CALLS_WITHOUT_ALLOW_MANY}")
    p.add_argument("--print-request", action="store_true",
                   help="print what would be sent and exit: no API call, no key read, no files "
                        "written. Case 5's user message is always withheld.")
    p.add_argument("--show-user-message", action="store_true",
                   help="with --print-request: print every segment of every case except case 5 "
                        "(default: the first 3 and the last)")
    return p


def _dedupe(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def print_request(args, arms, cases_order, cases, prompts, prompt_metas, n_calls, printer=print):
    printer(f"MODEL: {args.model} | reasoning_effort={args.reasoning_effort or 'not passed'} "
            f"| max_completion_tokens={args.max_completion_tokens or 'not passed'} "
            f"| temperature not passed | calls={n_calls} (repeat={args.repeat})")
    for arm in arms:
        meta, text = prompt_metas[arm], prompts[arm]
        printer(f"\nARM {arm} ({ARM_NAMES[arm]}) PROMPT: {meta['file']} md5={meta['md5']} "
                f"{len(text.split())} words, {len(text)} chars (system message)")
        printer("RESPONSE_FORMAT:")
        printer(json.dumps(build_response_format(arm), ensure_ascii=False, indent=2))
    for cid in cases_order:
        case = cases[cid]
        view = model_view(case)
        msg = render_user_message(view)
        if cid == CONTROL_CASE:
            printer(f"\n=== CASE {cid} USER MESSAGE WITHHELD (sealed control) | "
                    f"model view md5={_md5_text(msg)} ===")
            continue
        segs = view["segments"]
        n_cur = sum(1 for s in segs if s["period"] == "current")
        printer(f"\n=== CASE {cid} USER MESSAGE: {len(segs)} segments ({n_cur} current, "
                f"{len(segs) - n_cur} prior), {len(msg)} chars, model view md5={_md5_text(msg)} ===")
        rows = [json.dumps(s, ensure_ascii=False) for s in segs]
        if args.show_user_message or len(rows) <= 4:
            for r in rows:
                printer("  " + r)
        else:
            for r in rows[:3]:
                printer("  " + r)
            printer(f"  ... {len(rows) - 4} more segments; pass --show-user-message to print them all ...")
            printer("  " + rows[-1])
    printer("\n(no API call made, no key read, nothing written)")


def main(argv=None, client_factory=None):
    _utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _main(parser, args, client_factory)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - never print a traceback that could carry case text
        print(f"\nUNEXPECTED ERROR ({type(exc).__name__}): details withheld so nothing from a case "
              f"leaks into this log. Records written so far are kept. To debug, set PROBE_DEBUG=1 "
              f"and run again.")
        if os.environ.get("PROBE_DEBUG") == "1":
            traceback.print_exc()
        return 2


def _main(parser, args, client_factory):
    try:
        if args.show_user_message and not args.print_request:
            raise InputRefused("--show-user-message only works with --print-request")
        cases_order = _dedupe(args.case)
        arms = _dedupe(args.arm)
        if not (1 <= args.repeat <= MAX_REPEAT):
            raise InputRefused(f"--repeat must be between 1 and {MAX_REPEAT}")
        n_calls = len(cases_order) * len(arms) * args.repeat
        if n_calls > MAX_CALLS_WITHOUT_ALLOW_MANY and not args.allow_many:
            raise InputRefused(
                f"This would make {n_calls} calls. Test on a subset first (dry-run before full "
                f"run); pass --allow-many to confirm."
            )
        prompt_files = {"A": args.readout_prompt, "B": args.labeling_prompt}
        prompts, prompt_metas = {}, {}
        for arm in arms:
            text, md5 = load_prompt(prompt_files[arm])
            prompts[arm] = text
            prompt_metas[arm] = {"label": prompt_label(arm, prompt_files[arm]),
                                 "file": Path(prompt_files[arm]).name, "md5": md5}
        manifest, manifest_md5 = load_manifest(args.input_dir)
        cases, case_md5s = {}, {}
        for cid in cases_order:
            cases[cid], case_md5s[cid] = load_case(args.input_dir, cid, manifest)
        if not args.print_request:
            for arm in arms:
                check_prompt_unchanged(args.out_dir, arm, prompt_metas[arm]["label"], prompt_metas[arm]["md5"])
    except InputRefused as exc:
        parser.error(str(exc))

    if args.print_request:
        print_request(args, arms, cases_order, cases, prompts, prompt_metas, n_calls)
        return 0

    plan = [(cid, arm) for cid in cases_order for arm in arms]
    client = (client_factory or make_openai_client)()
    secrets = (os.environ.get("OPENAI_API_KEY", ""),)
    results = run_calls(
        plan,
        cases=cases,
        case_md5s=case_md5s,
        manifest_md5=manifest_md5,
        out_dir=args.out_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        repeat=args.repeat,
        prompts=prompts,
        prompt_metas=prompt_metas,
        client=client,
        secrets=secrets,
    )
    counts = {}
    for _, rec in results:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    total = sum((rec["est_cost_usd"] or 0) for _, rec in results)
    print(f"\nDONE: {len(results)} call(s) {counts} | est. total ~${round(total, 4)} ({PRICE_BASIS})")
    print("structure_ok checks SHAPE only. Quote checks and rule flags are counted by probe_p4_checks.py; "
          "usefulness is judged in the blind review. No model text is printed here: the records are in "
          f"{args.out_dir}/ (do not open case 5's before the blind review).")
    return 0 if all(rec["status"] == "structure_ok" for _, rec in results) else 1


if __name__ == "__main__":
    sys.exit(main())
