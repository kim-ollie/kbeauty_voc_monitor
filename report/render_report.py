#!/usr/bin/env python3
"""Renders the monthly VoC report from the saved payloads and the saved model records.

    python render_report.py 2025-08 2026-03 2026-06            # DRY RUN (the default): builds each message, runs every gate, writes nothing
    python render_report.py 2026-06 --write                    # writes reports/report_2026-06.md, .slack.json and .run.json
    python render_report.py 2026-06 --write --force            # ... replacing files that already exist and differ
    python render_report.py 2026-06 --no-bars | --section-style bold   # the two Slack fallbacks (no colour bars; bold section names instead of header blocks)
    python render_report.py 2026-06 --explain                  # also prints the number gate's working, line by line
    python render_report.py 2026-06 --records probe_inputs:probe_outputs   # the model records to read, as IN_FOLDER:OUT_FOLDER (repeat for several)

What it does, per month: reads the month's payload and every earlier payload from report_inputs/, loads the model records of each flagged
product from probe_inputs/ and probe_outputs/, dry_inputs/ and dry_outputs/, and replay_inputs/ and replay_outputs/ (the probe's, the dry run's
and the replay's own records, whichever folders exist; the run of record is the first run that passed its shape check, see records.py), builds the message
(message.py), builds its Slack payload (block_kit.py) and runs every gate (gates.py). A month whose window holds a single complaint month (2025-07) is
built with no complaint block and needs no records (the block rule in message.py). A month that passes every blocking gate is written as three files:

    reports/report_YYYY-MM.md          the Markdown form (the README-facing file)
    reports/report_YYYY-MM.slack.json  the Slack Block Kit payload, exactly as it would be posted
    reports/report_YYYY-MM.run.json    the run record: the md5 of every input and of every rendering module, the model records' stamps, each
                                       gate's verdict, the number gate's tally, the word counts and the md5 of the two files above

A month that a blocking gate stops gets the run record only, with status "blocked" and the gates' reasons, so the stop is on record and
no report file exists to be posted by mistake. A month whose model records are missing cannot be built at all: it is reported and skipped.

Nothing here posts to Slack, calls a model, opens a database or reads .env. The only files it writes are the three above, and only with
--write. It never replaces a file that differs unless --force is given. Two runs on the same inputs write identical bytes: there is no
timestamp in any output (the files' own dates say when).

Exit code: 0 every requested month passed; 1 at least one month was blocked by a gate; 2 at least one month could not be built or written.
"""
import argparse
import hashlib
import json
import os
import sys

import block_kit as bk
import gates
import message as msg
import number_gate as ng
import records as rc

RENDER_VERSION = "phase4-4.4-v3"      # v3: the Slack layout after the trial post (header blocks, small captions, split top lines); the run record names the section style
HERE = os.path.dirname(os.path.abspath(__file__))
# The code whose output the report is: a change to any of these changes what a report can say, so each one's md5 goes in the run record.
CODE_MODULES = ("render_report.py", "message.py", "system_rows.py", "top_line.py", "flag_line.py", "brand_block.py", "mix_shift.py", "complaint_block.py",
                "records.py", "block_kit.py", "gates.py", "number_gate.py", "report_config.py", "probe_p4_checks.py", "probe_run_model.py")


def md5_bytes(data):
    return hashlib.md5(data).hexdigest()


def md5_path(path):
    with open(path, "rb") as f:
        return md5_bytes(f.read())


def code_stamps(folder=HERE):
    return {name: (md5_path(os.path.join(folder, name)) if os.path.exists(os.path.join(folder, name)) else None) for name in CODE_MODULES}


def load_payloads(folder):
    """[(month 'YYYY-MM', file name, md5, payload)] for every saved payload, oldest first."""
    out = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("report_input_") and name.endswith(".json"):
            path = os.path.join(folder, name)
            with open(path, "rb") as f:
                raw = f.read()
            payload = json.loads(raw.decode("utf-8-sig"))
            out.append((payload["envelope"]["as_of_month"][:7], name, md5_bytes(raw), payload))
    return sorted(out, key=lambda t: t[0])


def dumps(obj):
    """A run record as text: sorted keys, Korean as Korean, one trailing newline."""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_month(month, loaded, store, color_bars=True, allow_no_blocks=False, section_style="header"):
    """Everything about one month, without touching a file:
    {"month", "status": "passed" | "blocked", "message", "gates", "markdown", "slack_json", "record", "files": {name: bytes}}.
    Raises rc.MissingRecords when a flagged product has no saved model records."""
    months = [t[0] for t in loaded]
    i = months.index(month)
    _, payload_name, payload_md5, payload = loaded[i]
    previous = [t[3] for t in loaded[:i]]
    message = msg.build_message(payload, previous, store=store, strict_records=False, require_blocks=not allow_no_blocks)
    result = gates.run_gates(message, payload, previous, store, color_bars=color_bars, section_style=section_style)
    kit = result["block_kit"]
    passed = result["passed"]
    markdown, slack_json = msg.to_markdown(message), bk.to_json(kit)
    files = {}
    if passed:
        files[f"report_{month}.md"] = markdown.encode("utf-8")
        files[f"report_{month}.slack.json"] = slack_json.encode("utf-8")
    record = {
        "render_version": RENDER_VERSION,
        "as_of": month,
        "status": "passed" if passed else "blocked",
        "inputs": {
            "payload": {"file": payload_name, "md5": payload_md5, "script_version": payload["envelope"]["script_version"],
                        "mart_snapshot_date": payload["envelope"]["mart_snapshot_date"]},
            "previous_payloads": [{"month": t[0], "md5": t[2]} for t in loaded[:i]],
            "records": {str(pid): st for pid, st in sorted(message["records"].items())},
        },
        "code": code_stamps(),
        "gates": result["gates"],
        "flags": result["flags"],
        "numbers": {"passed": result["numbers"]["passed"], "tokens": result["numbers"]["tokens"], "failures": result["numbers"]["failures"]} if result["numbers"] else None,
        "sizes": {"message_words": result["sizes"]["message_words"], "block_words": {str(k): v for k, v in sorted(result["sizes"]["block_words"].items())}},
        "block_kit": {"blocks": len(bk.all_blocks(kit)), "attachments": len(kit.get("attachments", [])), "color_bars": color_bars, "section_style": section_style,
                      "dashboard_link": any(p["type"] == "dashboard" for p in message["parts"]), "bytes": len(slack_json.encode("utf-8"))},
        "build_notes": message["checks"]["notes"],
        "outputs": {name: {"md5": md5_bytes(data), "bytes": len(data)} for name, data in sorted(files.items())},
    }
    files[f"report_{month}.run.json"] = dumps(record).encode("utf-8")
    return {"month": month, "status": record["status"], "message": message, "gates": result, "markdown": markdown, "slack_json": slack_json, "record": record, "files": files}


def write_month(rendered, out_dir, force):
    """Writes a month's files, all or none. Returns ("written" | "unchanged" | "refused", [messages])."""
    plan, notes, refused = [], [], False
    for name, data in sorted(rendered["files"].items()):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                current = f.read()
            if current == data:
                notes.append(f"{name}: unchanged")
                continue
            if not force:
                notes.append(f"{name}: exists and differs; not replaced (use --force)")
                refused = True
                continue
            notes.append(f"{name}: replaced")
        else:
            notes.append(f"{name}: written")
        plan.append((path, data))
    if rendered["status"] == "blocked":
        for suffix in (".md", ".slack.json"):
            name = f"report_{rendered['month']}{suffix}"
            if os.path.exists(os.path.join(out_dir, name)):
                notes.append(f"NOTE: {name} is left from an earlier run; this run blocked the month, so it must not be posted")
    if refused:
        notes = [f"{n.split(':')[0]}: not written (another file of this month differs; use --force)" if n.endswith((": written", ": replaced")) else n for n in notes]
        return "refused", notes
    if plan:
        os.makedirs(out_dir, exist_ok=True)
    for path, data in plan:
        with open(path, "wb") as f:
            f.write(data)
    return ("written" if plan else "unchanged"), notes


def build_parser():
    p = argparse.ArgumentParser(description="Render the monthly VoC report from the saved payloads and model records (dry run unless --write).")
    p.add_argument("months", nargs="+", metavar="YYYY-MM", help="the months to render")
    p.add_argument("--write", action="store_true", help="write the report files (the default is a dry run that writes nothing)")
    p.add_argument("--force", action="store_true", help="with --write: replace a file that already exists and differs")
    p.add_argument("--explain", action="store_true", help="print the number gate's working for each month")
    p.add_argument("--no-bars", action="store_true", help="Slack payload without colour bars (no attachments)")
    p.add_argument("--section-style", choices=bk.SECTION_STYLES, default="header", help="section names as header blocks (default) or as bold lines at body size "
                   "(the fallback if Slack does not show a header block inside a colour bar)")
    p.add_argument("--no-blocks", action="store_true", help="build without complaint blocks (previews only; not allowed with --write)")
    p.add_argument("--out", default="reports")
    p.add_argument("--inputs", default="report_inputs")
    p.add_argument("--records", action="append", metavar="IN:OUT", help="a folder of model-call inputs and the folder of their records; repeat for several "
                   "(default: probe_inputs:probe_outputs, dry_inputs:dry_outputs and replay_inputs:replay_outputs, the ones that exist)")
    return p


def main(argv=None, out=None):
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    if args.no_blocks and args.write:
        print("--no-blocks builds a preview without the complaint blocks; it cannot be written as a report", file=out)
        return 2
    loaded = load_payloads(args.inputs)
    have = [t[0] for t in loaded]
    try:
        store = None if args.no_blocks else rc.open_store(rc.parse_pairs(args.records))
    except ValueError as why:
        print(f"--records: {why}", file=out)
        return 2
    print(("WRITE" if args.write else "DRY RUN (nothing is written)") + f" · render {RENDER_VERSION} · {len(args.months)} month(s)\n", file=out)
    worst, rows = 0, []
    for month in args.months:
        if month not in have:
            print(f"{month}: no saved payload in {args.inputs}/ (have {have[0] if have else 'none'} to {have[-1] if have else 'none'})\n", file=out)
            worst, rows = max(worst, 2), rows + [(month, "no payload", "", "")]
            continue
        try:
            r = render_month(month, loaded, store, color_bars=not args.no_bars, allow_no_blocks=args.no_blocks, section_style=args.section_style)
        except (rc.MissingRecords, rc.RecordProblem) as why:
            print(f"{month}: cannot be built: {why}\n", file=out)
            worst, rows = max(worst, 2), rows + [(month, "cannot be built", "", "")]
            continue
        print(f"== {month}: {r['status'].upper()}", file=out)
        print("\n".join(gates.summary_lines(r["gates"])), file=out)
        if args.explain and r["gates"]["numbers"]:
            print("\nnumber gate:\n" + ng.explanation_table(r["gates"]["numbers"]), file=out)
        code = 0 if r["status"] == "passed" else 1
        if args.write:
            state, notes = write_month(r, args.out, args.force)
            print("\n".join(f"  {os.path.join(args.out, n)}" for n in notes), file=out)
            if state == "refused":
                code = 2
        else:
            print("  would write: " + ", ".join(os.path.join(args.out, n) for n in sorted(r["files"])), file=out)
        print("", file=out)
        worst = max(worst, code)
        rows.append((month, r["status"], f"{r['gates']['sizes']['message_words']} words", f"{r['record']['block_kit']['blocks']} blocks"))
    print("month     result           words        blocks", file=out)
    for month, status, words, blocks in rows:
        print(f"{month:<9} {status:<16} {words:<12} {blocks}", file=out)
    return worst


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
