#!/usr/bin/env python3
"""Posts the rendered monthly reports to a Slack incoming webhook.

    python post_report.py --all                        # DRY RUN (the default): checks all twelve months, says what would be posted, posts nothing
    python post_report.py 2026-06                      # dry run for one month
    python post_report.py 2026-06 --post               # POSTS 2026-06 (the first post to a webhook must be a single month: the trial)
    python post_report.py --all --post                 # POSTS every month not yet posted to this webhook, oldest first, 1.1 s apart
    python post_report.py 2026-06 --post --again       # posts a month again although it was posted before (or its earlier attempt has no recorded result)

THE WEBHOOK URL is a secret. It is read from the environment variable SLACK_WEBHOOK_URL (put there from .env when the command runs, as probe_run_model.py does
with the API key), must begin https://hooks.slack.com/services/, and is never printed, never written to a file and never taken from the command line. What is
recorded is a fingerprint of it (8 hex digits of its SHA-256), so a month posted to a test channel is not taken for a month posted to the real one.

WHAT MAY BE POSTED. Nothing posts until ALL twelve months (2025-07 to 2026-06) are ready, whichever months are asked for. A month is ready when
its run record (render_report.py) says `passed` with no failed gate; its report files are there and still have the md5 the run record holds; the run record's
code stamps and payload md5s equal the code and payloads that exist now (so a report rendered before a change is not posted after it); and its Slack payload
is valid JSON with blocks. Nothing later than 2026-06 is accepted.

THE LOOP. Months go oldest first, at least 1.1 seconds apart. Before each send a record is written to reports/report_YYYY-MM.posted.json
("sending", the md5 of the bytes about to be sent); after the reply it is completed (the HTTP status, Slack's reply, the time). A month whose record still says
"sending" has an outcome nobody knows (the process died): it is not posted again until you have looked at the channel and pass --again. Slack's reply must be
HTTP 200 and the text `ok`; anything else stops the loop at once and the months after it stay unposted (run it again: months already posted to this webhook are
skipped, so it resumes where it stopped). Nothing is ever retried here. What a stop means depends on what came back: an HTTP rejection (400, 404, 410, 429, 5xx) is
recorded `failed` (Slack did not post it, and a rerun tries it again); no answer at all (a timeout, a reset) or an HTTP 200 whose text is not `ok` is recorded
`unclear` (Slack may have posted it): that month is held like a "sending" one until you have looked at the channel and pass --again. (Incoming webhooks cannot delete
a message they posted; Slack's own documentation says so. A wrong post is deleted by hand in Slack.)

THE TRIAL. The first post to a webhook (one with no successful post on record) must be a single month. Read it in Slack: Korean text, the bold values,
the colour bars, the wrapping of the three top lines. If the colour bars do not show, re-render with `render_report.py --no-bars --write --force` and trial again.

Nothing here calls a model, opens a database or contacts anything but the webhook (and only with --post). The only files it writes are the posted records.

Exit code: 0 done (or a dry run that found every month ready and a sound plan); 1 a month is not ready, the plan was refused, or a post failed; 2 it could not run (a bad
argument, a missing folder, no valid webhook URL when --post is given).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import replay_extract as rx
import render_report as rr

MONTHS = tuple(rx.EXPECTED_MONTHS)
LAST_MONTH = rx.LAST_MONTH
WEBHOOK_ENV = "SLACK_WEBHOOK_URL"
WEBHOOK_PREFIX = "https://hooks.slack.com/services/"
MIN_GAP_S = 1.1
TIMEOUT_S = 20
REPLY_KEPT = 200                                   # characters of Slack's reply kept in the record
HELD = ("sending", "unclear")                      # a month whose record says this may already be in the channel: it is not posted again without --again


class BadUrl(Exception):
    """The webhook URL is missing or is not a Slack webhook URL. The message never contains the URL."""


class SendError(Exception):
    """The request did not complete (no HTTP answer). `kind` is a class name, never the URL."""
    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


class MarkerProblem(Exception):
    """A posted record exists and cannot be read."""


# ------------------------------------------------------------------ the secret
def validate_url(url):
    if not url:
        raise BadUrl(f"{WEBHOOK_ENV} is not set")
    if not url.startswith(WEBHOOK_PREFIX) or len(url) <= len(WEBHOOK_PREFIX) or any(c.isspace() for c in url):
        raise BadUrl(f"{WEBHOOK_ENV} is set but does not look like a Slack webhook URL (it must begin {WEBHOOK_PREFIX})")
    return url


def fingerprint(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]


def scrub(text, url):
    """Text with every trace of the URL taken out (the whole URL and its secret path), for anything that is printed or recorded."""
    if not url:
        return text
    text = text.replace(url, "<webhook>")
    secret = url[len(WEBHOOK_PREFIX):] if url.startswith(WEBHOOK_PREFIX) else url
    parts = {secret, *[p for p in secret.split("/") if len(p) >= 6]}
    for part in sorted(parts, key=len, reverse=True):              # the whole secret first, then its parts: the order must not depend on the run
        if part:
            text = text.replace(part, "<secret>")
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None                               # a redirect is an HTTPError here, never followed with the body lost


def http_post(url, body, timeout=TIMEOUT_S):
    """POST the bytes as JSON. Returns (HTTP status, reply text). Any HTTP status is an answer; only a failure to get one raises SendError (class name only)."""
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json; charset=utf-8"})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", "replace")
        except Exception:
            text = ""
        return e.code, text
    except urllib.error.URLError as e:
        raise SendError(type(e.reason).__name__ if not isinstance(e.reason, str) else "URLError")
    except Exception as e:                                           # a timeout, a reset, anything else: the class name only
        raise SendError(type(e).__name__)


def default_load_env(env):
    """Puts the variables of .env (next to where the command is run) into the process environment, without overriding one that is already set. Only for the real
    environment; a missing python-dotenv is not an error (the variable may be set already)."""
    if env is not os.environ:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


# ------------------------------------------------------------------ what is ready
def md5_bytes(data):
    return hashlib.md5(data).hexdigest()


def file_md5(path):
    with open(path, "rb") as f:
        return md5_bytes(f.read())


def report_names(month):
    return {"md": f"report_{month}.md", "slack": f"report_{month}.slack.json", "run": f"report_{month}.run.json", "posted": f"report_{month}.posted.json"}


def preflight_month(month, reports, inputs, code=None):
    """Is the month's report ready to post? {"month", "ok", "problems", "blocks", "attachments", "bytes", "flags", "slack_md5", "color_bars"}."""
    code = code if code is not None else rr.code_stamps()
    names = report_names(month)
    out = {"month": month, "ok": False, "problems": [], "blocks": None, "attachments": None, "bytes": None, "flags": None, "slack_md5": None, "color_bars": None}
    problems = out["problems"]
    run_path = os.path.join(reports, names["run"])
    if not os.path.exists(run_path):
        problems.append(f"no run record {names['run']} (render_report.py {month} --write)")
        return out
    try:
        with open(run_path, encoding="utf-8") as f:
            rec = json.load(f)
    except (ValueError, OSError):
        problems.append(f"{names['run']} cannot be read")
        return out
    if rec.get("as_of") != month:
        problems.append(f"the run record is for {rec.get('as_of')}, not {month}")
    if rec.get("status") != "passed":
        problems.append(f"the run record says {rec.get('status')}, not passed")
    failed = sorted(g for g, v in (rec.get("gates") or {}).items() if isinstance(v, dict) and v.get("status") == "fail")
    if failed:
        problems.append(f"gate(s) failed: {', '.join(failed)}")
    if not rec.get("gates"):
        problems.append("the run record lists no gates")
    outputs = rec.get("outputs") or {}
    for kind in ("md", "slack"):
        name = names[kind]
        path = os.path.join(reports, name)
        held = outputs.get(name)
        if not held:
            problems.append(f"the run record holds no md5 for {name}")
        elif not os.path.exists(path):
            problems.append(f"{name} is missing")
        else:
            with open(path, "rb") as f:
                data = f.read()
            if md5_bytes(data) != held.get("md5") or len(data) != held.get("bytes"):
                problems.append(f"{name} changed since it was rendered (md5 differs from the run record)")
    slack_path = os.path.join(reports, names["slack"])
    if os.path.exists(slack_path):
        out["slack_md5"] = file_md5(slack_path)
        try:
            with open(slack_path, encoding="utf-8") as f:
                kit = json.load(f)
            if not isinstance(kit, dict) or not isinstance(kit.get("blocks"), list) or not kit["blocks"]:
                problems.append(f"{names['slack']} has no blocks")
            else:
                attachments = kit.get("attachments") or []
                out["blocks"] = len(kit["blocks"]) + sum(len(a.get("blocks") or []) for a in attachments if isinstance(a, dict))
                out["attachments"] = len(attachments)
                out["bytes"] = os.path.getsize(slack_path)
        except (ValueError, OSError):
            problems.append(f"{names['slack']} is not valid JSON")
    stamped = rec.get("code") or {}
    if set(stamped) != set(code):
        problems.append("the run record's list of code files is not the current one")
    changed = sorted(m for m in stamped if m in code and stamped[m] != code[m])
    if changed:
        problems.append(f"rendered by different code than now ({', '.join(changed)}): render it again")
    inp = rec.get("inputs") or {}
    payload = inp.get("payload") or {}
    payload_files = [(payload.get("file"), payload.get("md5"))] + [(f"report_input_{p.get('month')}.json", p.get("md5")) for p in inp.get("previous_payloads") or []]
    for fname, md5 in payload_files:
        path = os.path.join(inputs, fname or "")
        if not fname or not os.path.exists(path):
            problems.append(f"payload {fname} is not in {inputs}/")
        elif file_md5(path) != md5:
            problems.append(f"payload {fname} changed since the month was rendered")
    out["flags"] = len(rec.get("flags") or [])
    out["color_bars"] = (rec.get("block_kit") or {}).get("color_bars")
    out["ok"] = not problems
    return out


# ------------------------------------------------------------------ the posted records
def marker_path(reports, month):
    return os.path.join(reports, report_names(month)["posted"])


def read_marker(reports, month):
    """{"month", "posts": [...]} (empty when the month was never attempted). Raises MarkerProblem when the file cannot be read."""
    path = marker_path(reports, month)
    if not os.path.exists(path):
        return {"month": month, "posts": []}
    try:
        with open(path, encoding="utf-8") as f:
            marker = json.load(f)
        if not isinstance(marker, dict) or not isinstance(marker.get("posts"), list) or marker.get("month") != month:
            raise ValueError("shape")
        return marker
    except (ValueError, OSError):
        raise MarkerProblem(f"{os.path.basename(path)} cannot be read; look at it before posting {month}")


def write_marker(reports, month, marker):
    """Writes the record whole, then moves it into place, so a crash leaves the old record or the new one and never half of one."""
    path = marker_path(reports, month)
    data = (json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def state_for(marker, fp):
    """('done' | 'unknown' | 'new', number of attempts) for this webhook fingerprint."""
    mine = [p for p in marker["posts"] if p.get("webhook") == fp]
    if any(p.get("status") == "ok" for p in mine):
        return "done", len(mine)
    if any(p.get("status") in HELD for p in mine):
        return "unknown", len(mine)
    return "new", len(mine)


def webhook_has_a_post(reports, fp):
    """True when any month has a successful post on record for this fingerprint (the trial has happened)."""
    for month in MONTHS:
        try:
            marker = read_marker(reports, month)
        except MarkerProblem:
            continue
        if any(p.get("webhook") == fp and p.get("status") == "ok" for p in marker["posts"]):
            return True
    return False


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ the command
def build_parser():
    p = argparse.ArgumentParser(description="Post the rendered monthly reports to a Slack webhook (dry run unless --post).")
    p.add_argument("months", nargs="*", metavar="YYYY-MM", help="the months to post")
    p.add_argument("--all", action="store_true", help="every replay month, 2025-07 to 2026-06")
    p.add_argument("--post", action="store_true", help="actually post (the default is a dry run that contacts nothing)")
    p.add_argument("--again", action="store_true", help="post a month again although it was posted to this webhook before, or an earlier attempt has no recorded result")
    p.add_argument("--reports", default="reports")
    p.add_argument("--inputs", default="report_inputs")
    return p


def main(argv=None, out=None, env=None, load_env=None, send=None, sleep=None, clock=None, now=None, code=None, required=None):
    """`env`, `load_env`, `send`, `sleep`, `clock`, `now`, `code` and `required` (the months that must all be ready) exist so that tests run without the real
    environment, network, clock or code; the command line never sets them."""
    out = out or sys.stdout
    env = os.environ if env is None else env
    load_env = load_env or default_load_env
    send = send or http_post
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    now = now or now_utc
    required = tuple(required) if required else MONTHS
    url_seen = []                                          # the URL, once read, only so that anything printed can be scrubbed of it

    def say(text=""):
        print(scrub(text, url_seen[0] if url_seen else None), file=out)

    args = build_parser().parse_args(argv)
    if args.all and args.months:
        say("name the months or give --all, not both")
        return 2
    months = sorted(set(required if args.all else args.months))
    if not months:
        say("name the months to post (YYYY-MM) or give --all")
        return 2
    bad = [m for m in months if m not in required]
    if bad:
        later = [m for m in bad if len(m) == 7 and m > LAST_MONTH]
        say((f"nothing later than {LAST_MONTH} is accepted: {', '.join(later)}" if later else f"not a replay month ({required[0]} to {required[-1]}): {', '.join(bad)}"))
        return 2
    for folder in (args.reports, args.inputs):
        if not os.path.isdir(folder):
            say(f"no folder {folder}/ (run it from the project folder)")
            return 2

    say(("POST" if args.post else "DRY RUN (nothing is posted, no connection is made)") + f" · {len(months)} month(s) asked for · {len(required)} must be ready\n")

    # ---- 1. every month must be ready, whichever are asked for
    code = code if code is not None else rr.code_stamps()
    ready = {m: preflight_month(m, args.reports, args.inputs, code) for m in required}
    say(f"  {'month':<9}{'ready':<7}{'blocks':>7}{'bars':>6}{'bytes':>8}{'flags':>7}  note")
    for m in required:
        r = ready[m]
        say(f"  {m:<9}{'yes' if r['ok'] else 'NO':<7}{r['blocks'] if r['blocks'] is not None else '-':>7}{r['attachments'] if r['attachments'] is not None else '-':>6}"
            f"{r['bytes'] if r['bytes'] is not None else '-':>8}{r['flags'] if r['flags'] is not None else '-':>7}  {'; '.join(r['problems'])}")
    not_ready = [m for m in required if not ready[m]["ok"]]
    if not_ready:
        say(f"\nNOT READY: {len(not_ready)} of {len(required)} month(s) cannot be posted ({', '.join(not_ready)}). Nothing posts before every month is ready.")
        return 1
    flagged = [f"{m} ({ready[m]['flags']})" for m in required if ready[m]["flags"]]
    if flagged:
        say(f"\n  read the flagged wording before posting: {', '.join(flagged)} (each run record's `flags`)")

    # ---- 2. the webhook
    load_env(env)
    url = env.get(WEBHOOK_ENV)
    if url:
        url_seen.append(url)
    try:
        validate_url(url)
        fp = fingerprint(url)
        say(f"\n  webhook: {WEBHOOK_ENV} is set and looks like a Slack webhook (fingerprint {fp}); it is never printed or saved")
    except BadUrl as why:
        if args.post:
            say(f"\n{why}")
            return 2
        fp = None
        say(f"\n  webhook: not ready ({why}); a dry run does not need it, --post does")

    # ---- 3. the plan
    plan = []
    for m in months:
        try:
            marker = read_marker(args.reports, m)
        except MarkerProblem as why:
            say(f"\nREFUSED: {why}")
            return 1
        state = state_for(marker, fp)[0] if fp else ("new" if not marker["posts"] else "cannot-tell")
        if state == "cannot-tell":
            plan.append((m, "post", "earlier attempts are on record; which webhook they went to cannot be told without the URL"))
        elif state == "done" and not args.again:
            plan.append((m, "skip", "already posted to this webhook (--again to post it once more)"))
        elif state == "unknown" and not args.again:
            plan.append((m, "blocked", "an earlier attempt has no clear result (it may be in the channel); look at the channel, then --again"))
        else:
            plan.append((m, "post", "again" if state in ("done", "unknown") else ""))
    to_post = [p for p in plan if p[1] == "post"]
    blocked = [p for p in plan if p[1] == "blocked"]
    say("\n  plan:")
    for m, action, why in plan:
        say(f"    {m}  {action.upper() if action != 'skip' else 'skip':<8}{why}")
    if blocked:
        say(f"\nREFUSED: {len(blocked)} month(s) have no clear outcome ({', '.join(p[0] for p in blocked)}).")
        return 1
    if len(to_post) > 1 and fp and not webhook_has_a_post(args.reports, fp):
        say("\nREFUSED: the first post to a webhook is a single trial. Post one month, read it in Slack, then run this again for the rest.")
        return 1
    if not args.post:
        trial = len(to_post) == 1 and fp and not webhook_has_a_post(args.reports, fp)
        say(f"\nDry run: {len(to_post)} month(s) would be posted" + (", as the single trial." if trial else ".") + " Add --post to post.")
        return 0

    # ---- 4. the loop
    last_start = None
    posted = []
    for m, _, _ in to_post:
        if last_start is not None:
            wait = MIN_GAP_S - (clock() - last_start)
            if wait > 0:
                sleep(wait)
        body_path = os.path.join(args.reports, report_names(m)["slack"])
        with open(body_path, "rb") as f:
            body = f.read()
        sent_md5 = md5_bytes(body)
        if sent_md5 != ready[m]["slack_md5"]:
            say(f"\nSTOPPED before {m}: the Slack payload changed after it was checked. Nothing was sent for {m}.")
            return 1
        marker = read_marker(args.reports, m)
        entry = {"attempt": len([p for p in marker["posts"] if p.get("webhook") == fp]) + 1, "webhook": fp, "started_utc": now(), "md5_sent": sent_md5, "bytes_sent": len(body),
                 "status": "sending", "http_status": None, "reply": None, "error": None, "finished_utc": None}
        marker["posts"].append(entry)
        write_marker(args.reports, m, marker)                       # before the send: a crash leaves "sending", never nothing
        last_start = clock()
        try:
            status, reply = send(url, body)
            error = None
        except SendError as e:
            status, reply, error = None, None, scrub(e.kind, url)
        entry["finished_utc"] = now()
        entry["http_status"] = status
        entry["reply"] = scrub(reply, url)[:REPLY_KEPT] if isinstance(reply, str) else None
        entry["error"] = error
        if status == 200 and isinstance(reply, str) and reply.strip() == "ok":
            entry["status"] = "ok"
        elif status is None or status == 200:
            entry["status"] = "unclear"                             # no answer, or a 200 that is not `ok`: Slack may have posted it
        else:
            entry["status"] = "failed"                              # Slack answered with a rejection: it did not post it
        write_marker(args.reports, m, marker)                       # before the next month is attempted
        if entry["status"] == "ok":
            posted.append(m)
            say(f"  posted {m}: HTTP 200 ok · {ready[m]['blocks']} blocks · {len(body)} bytes · md5 {sent_md5[:8]}")
            continue
        detail = f"HTTP {status}, reply {entry['reply']!r}" if status is not None else f"no answer ({error})"
        rest = [p[0] for p in to_post if p[0] > m]
        say(f"\nSTOPPED at {m}: {detail}. Recorded in {report_names(m)['posted']}. Not posted: {', '.join([m] + rest)}. Nothing was retried.")
        if entry["status"] == "unclear":
            say(f"{m} may or may not be in the channel: look there. It is held until you pass --again.")
        return 1
    say(f"\nPosted {len(posted)} month(s): {', '.join(posted) if posted else 'none'}.")
    if posted and not any(p[1] == "skip" for p in plan) and len(posted) == 1:
        say("Read it in Slack now (Korean text, bold values, colour bars, the wrapping of the three top lines) before posting the rest.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
