#!/usr/bin/env python3
"""
run_report_model.py  --  ask the LLM to narrate one month's report input.

What this does
    Reads report_inputs/report_input_YYYY-MM.json (built by build_report_input.py),
    sends a trimmed "model view" of it plus the system prompt file to OpenAI, and
    saves the raw answer and its provenance to report_outputs/. It changes nothing
    else. The first real call is the human's (this script reads OPENAI_API_KEY from
    .env only when it is about to call the API).

What this does NOT do (the renderer and gates.py do that)
    No number checking, no vocabulary checking, no retries, no rendering, no Slack.
    "structure_ok" means only that the JSON has the agreed shape.
    Read every output yourself.

Design rules kept here
    * The model gets only what it may quote: `_display` strings, names, ids, flags.
      Evidence phrases, caveats, the data-history note and coverage figures are not
      sent; the renderer owns them.
    * No temperature is passed (matches llm_topic_triage.py). reasoning_effort defaults
      to "high" (override with --reasoning-effort). No output
      cap is set unless you pass --max-completion-tokens.
    * No retry on our side. A first-attempt parse failure is data. (The OpenAI SDK's
      own transport retries on 429/5xx/connection errors are left at its default.)
    * Never overwrites: each call writes a new file ..._r1.json, ..._r2.json, ...
    * A prompt file, once used, is immutable: edit it and save it under a new name
      (phase4_system_prompt_v2.txt) or the run is refused.
    * More than 6 calls in one command needs --allow-many (dry-run before full run).

Usage (from the project folder)
    python run_report_model.py --as-of 2026-06 --print-request      # no API call
    python run_report_model.py --as-of 2026-06                       # one real call
    python run_report_model.py --as-of 2026-06 2025-12 2026-02
    python run_report_model.py --as-of 2026-06 --repeat 2            # same month twice
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUNNER_VERSION = "phase4-4.2-run-v1"
RECORD_SCHEMA_VERSION = "phase4-4.2-record-v1"
OUTPUT_SCHEMA_VERSION = "phase4-out-v1"
PAYLOAD_SCRIPT_VERSION = "phase4-4.1-v3"

# Placeholder-grade: taken from OpenAI's model page on 2026-09-24. The first real
# call confirms it (the record stores requested_model and served_model).
DEFAULT_MODEL = "gpt-5.6-sol"
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
# Run at "high". Cost note: reasoning tokens are billed as
# output tokens, so "high" costs more per call than the API default.
DEFAULT_REASONING_EFFORT = "high"

DEFAULT_PROMPT_FILE = "phase4_system_prompt_v1.txt"
DEFAULT_INPUT_DIR = "report_inputs"
DEFAULT_OUT_DIR = "report_outputs"

WINDOW_FIRST_MONTH = "2025-07"
WINDOW_LAST_MONTH = "2026-06"
MAX_REPEAT = 5
MAX_CALLS_WITHOUT_ALLOW_MANY = 6

# Starting word caps. Over-cap is flagged, not failed.
WORD_CAPS = {
    "headline": 40,
    "watchlist_item": 30,
    "category_shift": 50,
    "brand_vs_category": 50,
    "complaint_lead": 30,
    "complaint_theme": 30,
}

# List prices per million tokens, from OpenAI's model page on 2026-09-24. An
# estimate for planning only; the invoice is the truth.
LIST_PRICE_USD_PER_M = {"input": 4.00, "cached_input": 0.40, "output": 20.00}
PRICE_BASIS = "OpenAI model page 2026-09-24 list prices; estimate only"

# ------------------------------------------------------------------ model view
# Explicit allow-lists: the model sees exactly these fields and nothing else.
WATCH_SUMMARY_KEYS = (
    "headline_is_quiet",
    "no_flags_at_any_tier",
    "tier12_flagged_count_display",
    "tier3_dip_only_count_display",
)
WATCH_PRODUCT_KEYS = (
    "product_id",
    "brand_name_ko",
    "product_name_ko",
    "highest_tier",
    "newly_flagged_this_month",
    "consecutive_months_flagged_tier_1_2_display",
    "net_sentiment_display",
    "review_count_display",
    "decline_since_display",
    "notable_month_dip_drop_display",
)
MOM_KEYS = ("available", "prior_net_sentiment_display", "delta_display")
# Fields that only make sense for one kind of row. A tier-3 row is "no risk flag, just a
# one-month dip", so it must not carry flag-history fields the model could turn into
# "flagged for the first time" / "flagged for N months". Tier 1-2 rows do not need the
# dip figure (they get the month-over-month change instead).
TIER12_ONLY_KEYS = (
    "newly_flagged_this_month",
    "consecutive_months_flagged_tier_1_2_display",
    "decline_since_display",
)
TIER3_ONLY_KEYS = ("notable_month_dip_drop_display",)
CATEGORY_KEYS = (
    "category_name",
    "shift_available",
    "prior_mean_display",
    "recent_mean_display",
    "shift_display",
    "rank_worst_first",
)
BRAND_KEYS = (
    "brand_name_ko",
    "category_name",
    "brand_mean_display",
    "category_mean_display",
    "gap_display",
    "n_products_display",
)
COMPLAINT_TOP_KEYS = (
    "available",
    "recent_window_display",
    "prior_window_display",
    "materiality_floor_display",
    "reported_rising_count",
)
THEME_KEYS = (
    "theme_group",
    "rank_rise",
    "share_prior_display",
    "share_recent_display",
    "shift_display",
    "reported_as_rising",
)


def _pick(d, keys):
    return {k: d[k] for k in keys if k in d}


def model_view(payload):
    """The trimmed input the model sees. Deterministic; no numbers are computed here."""
    env = payload["envelope"]
    wl = payload["watchlist"]
    products = []
    for p in wl["products"]:
        row = _pick(p, WATCH_PRODUCT_KEYS)
        for k in (TIER12_ONLY_KEYS if p.get("highest_tier") == 3 else TIER3_ONLY_KEYS):
            row.pop(k, None)
        row["month_over_month"] = _pick(p.get("month_over_month") or {}, MOM_KEYS)
        products.append(row)

    bvc = payload["brand_vs_category"]
    cm = payload["complaint_movement"]
    if cm.get("available"):
        reported = [g for g in cm.get("theme_groups") or [] if g.get("reported_as_rising")]
        reported.sort(key=lambda g: g["rank_rise"])
        cm_view = _pick(cm, COMPLAINT_TOP_KEYS)
        cm_view["theme_groups"] = [_pick(g, THEME_KEYS) for g in reported]
    else:
        cm_view = {"available": False}

    return {
        "as_of_month": env["as_of_month"],
        "as_of_month_display": env["as_of_month_display"],
        "watchlist": {
            "summary": _pick(wl["summary"], WATCH_SUMMARY_KEYS),
            "products": products,
        },
        "category_shift": [_pick(c, CATEGORY_KEYS) for c in payload["category_shift"]],
        "brand_vs_category": {
            "largest_below_category": [_pick(r, BRAND_KEYS) for r in bvc["largest_below_category"]],
            "largest_above_category": [_pick(r, BRAND_KEYS) for r in bvc["largest_above_category"]],
        },
        "complaint_movement": cm_view,
    }


# ------------------------------------------------------------- request building
def _obj(props):
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


OUTPUT_TOP_KEYS = (
    "schema",
    "as_of_month",
    "headline",
    "watchlist",
    "category_shift",
    "brand_vs_category",
    "complaint_movement",
)


def build_response_format():
    """Strict JSON schema for phase4-out-v1 (same style as llm_topic_triage.py)."""
    schema = _obj(
        {
            "schema": {"type": "string", "enum": [OUTPUT_SCHEMA_VERSION]},
            "as_of_month": {"type": "string"},
            "headline": {"type": "string"},
            "watchlist": {
                "type": "array",
                "items": _obj({"product_id": {"type": "integer"}, "text": {"type": "string"}}),
            },
            "category_shift": {"type": "string"},
            "brand_vs_category": {"type": "string"},
            "complaint_movement": _obj(
                {
                    "lead": {"type": ["string", "null"]},
                    "themes": {
                        "type": "array",
                        "items": _obj(
                            {"theme_group": {"type": "string"}, "text": {"type": "string"}}
                        ),
                    },
                }
            ),
        }
    )
    assert tuple(schema["properties"]) == OUTPUT_TOP_KEYS
    return {
        "type": "json_schema",
        "json_schema": {"name": "phase4_monthly_report", "strict": True, "schema": schema},
    }


def build_messages(system_prompt, view):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(view, ensure_ascii=False, indent=2)},
    ]


# -------------------------------------------------------------- input handling
class InputRefused(Exception):
    """A payload or setting the runner will not send."""


def validate_month(s):
    if not re.fullmatch(r"\d{4}-\d{2}", s or ""):
        raise InputRefused(f"'{s}' is not a month in YYYY-MM form")
    if not (WINDOW_FIRST_MONTH <= s <= WINDOW_LAST_MONTH):
        raise InputRefused(
            f"{s} is outside the report window {WINDOW_FIRST_MONTH} .. {WINDOW_LAST_MONTH}"
        )
    return s


def payload_path(input_dir, month):
    return Path(input_dir) / f"report_input_{month}.json"


def load_payload(input_dir, month):
    path = payload_path(input_dir, month)
    if not path.is_file():
        raise InputRefused(f"{path} not found; run build_report_input.py first")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputRefused(f"{path} is not valid UTF-8 JSON: {exc}")
    env = payload.get("envelope") or {}
    if env.get("script_version") != PAYLOAD_SCRIPT_VERSION:
        raise InputRefused(
            f"{path}: script_version {env.get('script_version')!r}, expected "
            f"{PAYLOAD_SCRIPT_VERSION!r}; rebuild with build_report_input.py"
        )
    if env.get("as_of_month") != f"{month}-01":
        raise InputRefused(f"{path}: as_of_month is {env.get('as_of_month')!r}, expected {month}-01")
    val = payload.get("validation") or {}
    if val.get("warnings"):
        raise InputRefused(f"{path}: the builder recorded validation warnings: {val['warnings']}")
    if not val.get("gates_passed"):
        raise InputRefused(f"{path}: no validation gates recorded as passed")
    return payload, hashlib.md5(raw).hexdigest()


def prompt_label(prompt_file):
    stem = Path(prompt_file).stem
    prefix = "phase4_system_prompt_"
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
    return text, hashlib.md5(raw).hexdigest()


# ------------------------------------------------------------------ output files
def record_path(out_dir, month, label, n):
    return Path(out_dir) / f"report_output_{month}_{label}_r{n}.json"


def next_run_index(out_dir, month, label):
    d = Path(out_dir)
    if not d.is_dir():
        return 1
    pat = re.compile(rf"^report_output_{re.escape(month)}_{re.escape(label)}_r(\d+)\.json$")
    found = [int(m.group(1)) for f in d.iterdir() if (m := pat.match(f.name))]
    return (max(found) + 1) if found else 1


def check_prompt_unchanged(out_dir, label, md5):
    """A prompt label, once used, must keep the same bytes."""
    d = Path(out_dir)
    if not d.is_dir():
        return
    for f in sorted(d.glob(f"report_output_*_{label}_r*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        old = rec.get("prompt_md5")
        if old and old != md5:
            raise InputRefused(
                f"prompt '{label}' was already used ({f.name}) with md5 {old}, but the file "
                f"now has md5 {md5}. Save your edit under a new name, e.g. "
                f"phase4_system_prompt_v2.txt, so every output stays tied to its exact prompt."
            )


def write_record(path, record):
    """Exclusive create: raises FileExistsError rather than overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


# -------------------------------------------------------------------- checking
def _is_str(x):
    return isinstance(x, str) and x.strip() != ""


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _words(s):
    return len(s.split())


def _cap(flags, label, text, cap_key):
    n = _words(text)
    cap = WORD_CAPS[cap_key]
    if n > cap:
        flags.append(f"{label}: {n} words > {cap}")


def expected_watchlist_ids(payload):
    return [p["product_id"] for p in payload["watchlist"]["products"]]


def expected_theme_groups(payload):
    cm = payload["complaint_movement"]
    if not cm.get("available"):
        return []
    rows = [g for g in cm.get("theme_groups") or [] if g.get("reported_as_rising")]
    rows.sort(key=lambda g: g["rank_rise"])
    return [g["theme_group"] for g in rows]


def structure_check(parsed, payload):
    """Shape and cross-reference checks against the payload. Returns (problems, flags).

    problems are fatal (status structure_fail); flags are word-cap overruns.
    Numbers, wording and vocabulary are NOT checked here; that is the renderer's gates.
    """
    problems, flags = [], []
    if not isinstance(parsed, dict):
        return ["output_is_not_an_object"], flags

    for k in OUTPUT_TOP_KEYS:
        if k not in parsed:
            problems.append(f"missing_key:{k}")
    for k in parsed:
        if k not in OUTPUT_TOP_KEYS:
            problems.append(f"unexpected_key:{k}")

    if "schema" in parsed and parsed["schema"] != OUTPUT_SCHEMA_VERSION:
        problems.append("wrong_schema_value")
    if "as_of_month" in parsed and parsed["as_of_month"] != payload["envelope"]["as_of_month"]:
        problems.append("as_of_month_mismatch")

    for key in ("headline", "category_shift", "brand_vs_category"):
        if key in parsed:
            if not _is_str(parsed[key]):
                problems.append(f"{key}_not_a_nonempty_string")
            else:
                _cap(flags, key, parsed[key], key)

    if "watchlist" in parsed:
        wl = parsed["watchlist"]
        if not isinstance(wl, list):
            problems.append("watchlist_not_a_list")
        else:
            got_ids = []
            for i, e in enumerate(wl):
                if not isinstance(e, dict) or set(e) != {"product_id", "text"}:
                    problems.append(f"watchlist[{i}]_bad_shape")
                    got_ids.append(None)
                    continue
                got_ids.append(e["product_id"] if _is_int(e["product_id"]) else None)
                if not _is_str(e["text"]):
                    problems.append(f"watchlist[{i}]_text_empty")
                else:
                    _cap(flags, f"watchlist[{i}] product_id={e['product_id']}", e["text"], "watchlist_item")
            if got_ids != expected_watchlist_ids(payload):
                problems.append("watchlist_ids_differ_from_input")

    if "complaint_movement" in parsed:
        cm = parsed["complaint_movement"]
        pcm = payload["complaint_movement"]
        if not isinstance(cm, dict) or set(cm) != {"lead", "themes"}:
            problems.append("complaint_movement_bad_shape")
        else:
            lead, themes = cm["lead"], cm["themes"]
            if not isinstance(themes, list):
                problems.append("themes_not_a_list")
                themes = []
            if not pcm.get("available"):
                if lead is not None:
                    problems.append("lead_not_null_but_comparison_unavailable")
                if themes:
                    problems.append("themes_not_empty_but_comparison_unavailable")
            else:
                if not _is_str(lead):
                    problems.append("lead_missing_but_comparison_available")
                else:
                    for disp in (pcm["recent_window_display"], pcm["prior_window_display"]):
                        if disp not in lead:
                            problems.append(f"lead_missing_window_display:{disp}")
                    _cap(flags, "complaint lead", lead, "complaint_lead")
                got_groups = []
                for i, t in enumerate(themes):
                    if not isinstance(t, dict) or set(t) != {"theme_group", "text"}:
                        problems.append(f"themes[{i}]_bad_shape")
                        got_groups.append(None)
                        continue
                    got_groups.append(t["theme_group"])
                    if not _is_str(t["text"]):
                        problems.append(f"themes[{i}]_text_empty")
                    else:
                        _cap(flags, f"themes[{i}] {t['theme_group']}", t["text"], "complaint_theme")
                if got_groups != expected_theme_groups(payload):
                    problems.append("themes_differ_from_reported_groups")
    return problems, flags


_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?\s*```\s*$", re.DOTALL)


def strip_code_fences(text):
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1), True
    return text, False


# ------------------------------------------------------------ safety and usage
_SK_RE = re.compile(r"sk-[A-Za-z0-9_\-\*\.]{6,}")


def redact(text, secrets=()):
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "[REDACTED]")
    return _SK_RE.sub("sk-[REDACTED]", text)


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


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- the call
def call_once(
    client,
    *,
    payload,
    payload_md5,
    view,
    messages,
    response_format,
    model,
    reasoning_effort,
    max_completion_tokens,
    prompt_meta,
    secrets=(),
):
    """One API call -> one record. Never raises for API or parse problems."""
    request_params = {
        "model": model,
        "temperature": "not passed",
        "reasoning_effort": reasoning_effort if reasoning_effort else "not passed (API default)",
        "max_completion_tokens": max_completion_tokens if max_completion_tokens else "not passed",
        "response_format": "json_schema strict: phase4_monthly_report",
    }
    record = {
        "record_schema": RECORD_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "prompt_label": prompt_meta["label"],
        "prompt_file": prompt_meta["file"],
        "prompt_md5": prompt_meta["md5"],
        "as_of_month": payload["envelope"]["as_of_month"],
        "input_file_md5": payload_md5,
        "model_view_md5": hashlib.md5(messages[1]["content"].encode("utf-8")).hexdigest(),
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
        "word_cap_flags": [],
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
        record["error"] = f"JSONDecodeError: {exc}"
        return record
    record["parsed"] = parsed
    problems, flags = structure_check(parsed, payload)
    record["structure_problems"], record["word_cap_flags"] = problems, flags
    record["status"] = "structure_fail" if problems else "structure_ok"
    return record


def format_readable(parsed):
    if not isinstance(parsed, dict):
        return "  (no parsed object)"
    lines = []
    lines.append(f"  HEADLINE: {parsed.get('headline')}")
    lines.append("  WATCHLIST:")
    for e in parsed.get("watchlist") or []:
        if isinstance(e, dict):
            lines.append(f"    [{e.get('product_id')}] {e.get('text')}")
    lines.append(f"  CATEGORY SHIFT: {parsed.get('category_shift')}")
    lines.append(f"  BRAND VS CATEGORY: {parsed.get('brand_vs_category')}")
    cm = parsed.get("complaint_movement")
    if isinstance(cm, dict):
        lines.append(f"  COMPLAINT LEAD: {cm.get('lead')}")
        for t in cm.get("themes") or []:
            if isinstance(t, dict):
                lines.append(f"    - {t.get('theme_group')}: {t.get('text')}")
    return "\n".join(lines)


def run_months(
    months,
    *,
    input_dir,
    out_dir,
    model,
    reasoning_effort,
    max_completion_tokens,
    repeat,
    system_prompt,
    prompt_meta,
    client,
    secrets=(),
    log=print,
    loaded=None,
):
    """Call the model for each month (month-major), writing each record at once."""
    results = []
    for month in months:
        payload, md5 = (loaded or {}).get(month) or load_payload(input_dir, month)
        view = model_view(payload)
        messages = build_messages(system_prompt, view)
        response_format = build_response_format()
        for _ in range(repeat):
            n = next_run_index(out_dir, month, prompt_meta["label"])
            path = record_path(out_dir, month, prompt_meta["label"], n)
            record = call_once(
                client,
                payload=payload,
                payload_md5=md5,
                view=view,
                messages=messages,
                response_format=response_format,
                model=model,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_completion_tokens,
                prompt_meta=prompt_meta,
                secrets=secrets,
            )
            record["run_index"] = n
            write_record(path, record)
            results.append((path, record))
            u = record["usage"] or {}
            log(
                f"{month} r{n}: {record['status']}"
                f" | finish={record['finish_reason']} served={record['served_model']}"
                f" | in={u.get('prompt_tokens')} out={u.get('completion_tokens')}"
                f" reasoning={u.get('reasoning_tokens')}"
                f" | {record['duration_s']}s | ~${record['est_cost_usd']}"
            )
            if record["status"] in ("structure_ok", "structure_fail"):
                log(format_readable(record["parsed"]))
            if record["structure_problems"]:
                log(f"  STRUCTURE PROBLEMS: {record['structure_problems']}")
            if record["word_cap_flags"]:
                log(f"  word-cap flags (not failures): {record['word_cap_flags']}")
            if record["error"]:
                log(f"  ERROR: {record['error']}")
            log(f"  saved: {path}")
    return results


# ------------------------------------------------------------------------ CLI
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
        description="Send report inputs to the LLM and record the raw answers."
    )
    p.add_argument("--as-of", nargs="+", action="extend", required=True, metavar="YYYY-MM",
                   help=f"one or more months in {WINDOW_FIRST_MONTH} .. {WINDOW_LAST_MONTH}")
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"default {DEFAULT_MODEL!r} is a placeholder from OpenAI's model page; "
                        "the first call confirms it")
    p.add_argument("--reasoning-effort", choices=REASONING_EFFORTS, default=DEFAULT_REASONING_EFFORT,
                   help=f"default {DEFAULT_REASONING_EFFORT!r}")
    p.add_argument("--max-completion-tokens", type=int, default=None,
                   help="omit for no cap (reasoning tokens count against a cap)")
    p.add_argument("--repeat", type=int, default=1, help=f"calls per month (1..{MAX_REPEAT})")
    p.add_argument("--allow-many", action="store_true",
                   help=f"needed when months x repeat > {MAX_CALLS_WITHOUT_ALLOW_MANY}")
    p.add_argument("--print-request", action="store_true",
                   help="print what would be sent and exit: no API call, no key read, no files written")
    return p


def main(argv=None, client_factory=None):
    _utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        months = []
        for m in args.as_of:
            validate_month(m)
            if m not in months:
                months.append(m)
        if not (1 <= args.repeat <= MAX_REPEAT):
            raise InputRefused(f"--repeat must be between 1 and {MAX_REPEAT}")
        n_calls = len(months) * args.repeat
        if n_calls > MAX_CALLS_WITHOUT_ALLOW_MANY and not args.allow_many:
            raise InputRefused(
                f"This would make {n_calls} calls. Test on a subset first (dry-run before full "
                f"run); pass --allow-many to confirm."
            )
        system_prompt, prompt_md5 = load_prompt(args.prompt_file)
        label = prompt_label(args.prompt_file)
        prompt_meta = {"label": label, "file": Path(args.prompt_file).name, "md5": prompt_md5}
        loaded = {m: load_payload(args.input_dir, m) for m in months}
        if not args.print_request:
            check_prompt_unchanged(args.out_dir, label, prompt_md5)
    except InputRefused as exc:
        parser.error(str(exc))

    if args.print_request:
        rf = build_response_format()
        print(f"PROMPT: {prompt_meta['file']} md5={prompt_md5} "
              f"{len(system_prompt.split())} words, {len(system_prompt)} chars (system message)")
        print(f"MODEL: {args.model} | reasoning_effort={args.reasoning_effort or 'not passed'} "
              f"| max_completion_tokens={args.max_completion_tokens or 'not passed'} "
              f"| temperature not passed | calls={n_calls}")
        print("RESPONSE_FORMAT:")
        print(json.dumps(rf, ensure_ascii=False, indent=2))
        for m in months:
            view = model_view(loaded[m][0])
            msg = build_messages(system_prompt, view)[1]["content"]
            print(f"\n=== USER MESSAGE for {m} ({len(msg)} chars) ===")
            print(msg)
        print("\n(no API call made, no key read, nothing written)")
        return 0

    client = (client_factory or make_openai_client)()
    secrets = (os.environ.get("OPENAI_API_KEY", ""),)
    results = run_months(
        months,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_completion_tokens=args.max_completion_tokens,
        repeat=args.repeat,
        system_prompt=system_prompt,
        prompt_meta=prompt_meta,
        client=client,
        secrets=secrets,
        loaded=loaded,
    )
    counts = {}
    for _, rec in results:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    total = sum((rec["est_cost_usd"] or 0) for _, rec in results)
    print(f"\nDONE: {len(results)} call(s) {counts} | est. total ~${round(total, 4)} ({PRICE_BASIS})")
    print("structure_ok checks SHAPE only. Numbers, wording and banned words are checked by the renderer's gates: "
          "read every output.")
    return 0 if all(rec["status"] == "structure_ok" for _, rec in results) else 1


if __name__ == "__main__":
    sys.exit(main())
