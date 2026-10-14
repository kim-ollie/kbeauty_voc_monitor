#!/usr/bin/env python3
"""Builds the model-call inputs for the replay months that have none yet.

    python replay_extract.py                     # DRY RUN (the default): plans, builds and checks everything, writes nothing
    python replay_extract.py --write             # writes replay_inputs/case_N.json and replay_inputs/manifest.json
    python replay_extract.py --write --force     # ... replacing a case file that would change (refused once the case has saved records)

The plan (no data needed): the twelve saved payloads (report_inputs/) say, per month, which flagged products the message shows a complaint block for
(message.py: the first READOUT_MAX_PRODUCTS flagged rows, and none for a month under the block rule). Each of those product-months needs a saved case
and two saved model answers. The ones that already have a case in the probe's or the dry run's folders are left alone; the others are numbered
1, 2, 3 ... oldest first and built into replay_inputs/ (at most MAX_CASES: the runner's case 5 is the sealed control and is never used here).

The build reuses dry_extract.py's own builder (and through it the probe's loader), from the same local files, so the loader's three rules
exist in one place. Before it builds anything it proves it still builds the way the probe and the dry run did: probe cases 1-4 must rebuild to
probe_inputs/manifest.json's md5s, and the dry run's cases to dry_inputs/manifest.json's.

LOCAL FILES ONLY. No database, no network, no model call; .env is never read. Read-only over its sources; writes only under replay_inputs/, and only
with --write. Two runs on the same inputs write identical bytes (no timestamps).

Exit code: 0 every check passed; 1 a check failed or a write was refused; 2 it could not run (a payload or a source file is missing).
"""
import argparse
import hashlib
import json
import os
import sys

import message as msg
import records as rc
import render_report as rr
import system_rows as sr

EXPECTED_MONTHS = [f"2025-{m:02d}" for m in range(7, 13)] + [f"2026-{m:02d}" for m in range(1, 7)]
LAST_MONTH = EXPECTED_MONTHS[-1]
OUT_DIR, OUT_RECORDS = "replay_inputs", "replay_outputs"
MAX_CASES = 4                    # runner case ids are 1-5; 5 is the sealed control and is never a replay case
ARMS_PER_CASE = 2
MAX_CALLS_WITHOUT_ALLOW_MANY = 6  # probe_run_model.MAX_CALLS_WITHOUT_ALLOW_MANY
COMPARE_PAIRS = (("probe_inputs", "probe_outputs"), ("dry_inputs", "dry_outputs"))   # the folders whose cases already exist
TIER_NAME = {1: "strict", 2: "broad_only"}


class Refused(Exception):
    """The plan cannot be carried out (too many cases, an uncovered product-month)."""


# ------------------------------------------------------------------ the plan (payloads and folders only; no data files, no pandas)
def flagged_rows(payload):
    """[(product_id, highest_tier, position)] for the tier 1-2 rows the message shows, in row order (position counts from 0)."""
    tiers = {e["product_id"]: e["highest_tier"] for e in payload["watchlist"]["products"]}
    built = sr.build_sections(payload)
    pids = [r["product_id"] for s in built["sections"] for r in s["rows"] if r["tier"] in (1, 2)]
    return [(pid, tiers[pid], i) for i, pid in enumerate(pids)]


def blocks_needed(payload):
    """The product ids whose complaint block the month's message shows, in row order: the first READOUT_MAX_PRODUCTS flagged rows, and none at all
    for a month under the block rule. (test_replay_extract checks this against message.build_message for all twelve payloads.)"""
    if payload["coverage"]["complaint_months_in_window"] < msg.MIN_COMPLAINT_MONTHS_FOR_BLOCK:
        return []
    return [pid for pid, _, _ in flagged_rows(payload)][:msg.READOUT_MAX_PRODUCTS]


def existing_cases(pairs):
    """{(month, product_id): path of its case file} over the folders that exist (two cases for one product-month are refused by the store)."""
    stores = [rc.RecordStore(i, o) for i, o in pairs if os.path.isdir(i)]
    if not stores:
        return {}
    return {key: value[1] for key, value in rc.MultiStore(stores).cases.items()}


def plan(payloads, have):
    """payloads: [(month, payload)] oldest first. have: {(month, pid): case path} that exist already.
    Returns {"rows": [{month, product_id, tier, position, status, where}], "cases": [{id, month, asof, product_id, tier}]}.
    status: "block rule" | "beyond the cap" | "has a case" | "NEW". Raises Refused when more than MAX_CASES are needed."""
    rows, new = [], []
    for month, payload in payloads:
        shown = set(blocks_needed(payload))
        window = payload["coverage"]["complaint_months_in_window"]
        for pid, tier, pos in flagged_rows(payload):
            if window < msg.MIN_COMPLAINT_MONTHS_FOR_BLOCK:
                status, where = "block rule", f"{window} complaint month(s) in the window"
            elif pid not in shown:
                status, where = "beyond the cap", f"only the first {msg.READOUT_MAX_PRODUCTS} flagged products get a block"
            elif (month, pid) in have:
                status, where = "has a case", have[(month, pid)]
            else:
                status, where = "NEW", ""
                new.append({"month": month, "product_id": pid, "tier": tier})
            rows.append({"month": month, "product_id": pid, "tier": tier, "position": pos, "status": status, "where": where})
    if len(new) > MAX_CASES:
        raise Refused(f"{len(new)} product-months need a case; the most a replay folder may hold is {MAX_CASES} (the runner's case 5 is the sealed control)")
    cases = [{"id": i + 1, "month": c["month"], "asof": f"{c['month']}-01", "product_id": c["product_id"], "tier": c["tier"]} for i, c in enumerate(new)]
    for r in rows:
        if r["status"] == "NEW":
            c = next(c for c in cases if (c["month"], c["product_id"]) == (r["month"], r["product_id"]))
            r["where"] = f"case {c['id']}"
    return {"rows": rows, "cases": cases}


def runner_commands(cases, in_dir=OUT_DIR, out_dir=OUT_RECORDS):
    """The exact commands for the model calls: a look at the request first (no call), then the calls. None when there are no cases."""
    if not cases:
        return None
    ids = " ".join(str(c["id"]) for c in cases)
    calls = len(cases) * ARMS_PER_CASE
    base = f"python probe_run_model.py --input-dir {in_dir} --out-dir {out_dir}"
    return {"calls": calls, "look": f"{base} --case {cases[0]['id']} --arm A --print-request",
            "run": f"{base} --case {ids} --arm A B" + (" --allow-many" if calls > MAX_CALLS_WITHOUT_ALLOW_MANY else "")}


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_path(path):
    with open(path, "rb") as f:
        return md5_bytes(f.read())


def has_records(records_dir, case_id):
    """True when any saved record of this case id exists in the output folder (a case file with records must never change)."""
    if not os.path.isdir(records_dir):
        return False
    return any(n.startswith(f"probe_output_case{case_id}_") for n in os.listdir(records_dir))


def stale_files(out_dir, cases):
    """Case files in the output folder that the plan does not contain (a plan that changed since they were written). The manifest is rewritten with them, never stale."""
    if not os.path.isdir(out_dir):
        return []
    want = {f"case_{c['id']}.json" for c in cases}
    return sorted(n for n in os.listdir(out_dir) if n.startswith("case_") and n not in want)


def write_cases(files, out_dir, records_dir, force):
    """files: {name: bytes} (the case files and manifest.json). All or none. Returns ("written" | "unchanged" | "refused", [notes])."""
    plan_, notes, refused = [], [], False
    for name, data in sorted(files.items()):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                current = f.read()
            if current == data:
                notes.append(f"{name}: unchanged")
                continue
            m = name.startswith("case_") and name[5:-5].isdigit()
            if m and has_records(records_dir, int(name[5:-5])):
                notes.append(f"{name}: exists, differs, and case {name[5:-5]} has saved records; a case file with records is never replaced")
                refused = True
                continue
            if not force:
                notes.append(f"{name}: exists and differs; not replaced (use --force)")
                refused = True
                continue
            notes.append(f"{name}: replaced")
        else:
            notes.append(f"{name}: written")
        plan_.append((path, data))
    if refused:
        notes = [f"{n.split(':')[0]}: not written (another file differs; nothing was written)" if n.endswith((": written", ": replaced")) else n for n in notes]
        return "refused", notes
    if plan_:
        os.makedirs(out_dir, exist_ok=True)
    for path, data in plan_:
        with open(path, "wb") as f:
            f.write(data)
    return ("written" if plan_ else "unchanged"), notes


def prior_months_expected(months, asof, prior_months):
    """How many prior months a case has: the three before the as-of month, or fewer when the corpus does not reach back that far."""
    return min(prior_months, months.index(asof))


def manifest_of(cases, built, src_md5, script_md5, seed, prior_cap):
    """The manifest the runner reads (`cases` with case_id and md5), plus what the reader needs to trust the files. No timestamp: a rerun writes the same bytes."""
    out = {"script": "replay_extract.py", "script_md5": script_md5, "purpose": "replay cases", "seed": seed,
           "prior_cap": prior_cap, "sources_md5": src_md5, "cases": []}
    for c in cases:
        body, text = built[c["id"]][0], built[c["id"]][1]
        out["cases"].append({"case_id": c["id"], "file": f"case_{c['id']}.json", "md5": md5_bytes(text.encode("utf-8")), "bytes": len(text.encode("utf-8")),
                             "product_id": c["product_id"], "as_of_month": c["month"], "tier": TIER_NAME[c["tier"]], "counts": body["counts"],
                             "prior_sampling": body["prior_sampling"]})
    return out


# ------------------------------------------------------------------ the build (needs the local data files and pandas)
def case_checks(check, c, body, ptot, pm, ctx):
    """The per-case checks of dry_extract.py, with 'three prior months' relaxed to 'as many as the corpus has, up to three'.
    ctx: dict(o=overview frame, seg=segments, months=[...], mentions, uncl_mentions, ext_m, ext_u, prior_months=3, prior_cap=200, tf=..., tier_of=...)."""
    import pandas as pd
    cid, pid, asof = c["id"], c["product_id"], c["asof"]
    k = body["counts"]
    o, months = ctx["o"], ctx["months"]
    ov = o[(o.product_id == pid) & (o.review_month == asof)]
    check(f"case {cid}: the overview extract has exactly one row for the product-month", len(ov) == 1)
    if len(ov) != 1:
        return
    ov = ov.iloc[0]
    check(f"case {cid}: tier at the as-of month is {TIER_NAME[c['tier']]} (overview extract) as the payload says", ctx["tier_of"](ov) == TIER_NAME[c["tier"]])
    check(f"case {cid}: clears the review-count gate", bool(ctx["tf"](pd.Series([ov.meets_review_count_gate]))[0]))
    want = prior_months_expected(months, asof, ctx["prior_months"])
    check(f"case {cid}: {want} prior month(s), consecutive, all before the as-of month ({[x[:7] for x in pm]})",
          len(pm) == want and want >= 1 and all(months.index(a) + 1 == months.index(b) for a, b in zip(pm + [asof], (pm + [asof])[1:])))
    check(f"case {cid}: prior kept respects the cap", k["prior_kept"] == min(ptot, ctx["prior_cap"]))
    used = [asof] + pm
    check(f"case {cid}: theme mentions equal the mart extract, every month used",
          all(int(ctx["mentions"].get((pid, mo), 0)) == int(ctx["ext_m"].get((pid, mo), 0)) for mo in used))
    check(f"case {cid}: unclustered mentions equal the mart extract, every month used",
          all(int(ctx["uncl_mentions"].get((pid, mo), 0)) == int(ctx["ext_u"].get((pid, mo), 0)) for mo in used))
    check(f"case {cid}: at most the month's review_count texts", k["current"] <= int(ov.review_count))
    check(f"case {cid}: there are texts this month and in the prior period", k["current"] > 0 and k["prior_kept"] > 0, f"{k['current']} now, {k['prior_kept']} prior")


def main(argv=None, out=None):
    out = out or sys.stdout

    def say(text=""):
        print(text, file=out)

    ap = argparse.ArgumentParser(description="Build the replay's model-call inputs for the flagged product-months that have none (dry run unless --write).")
    ap.add_argument("--write", action="store_true", help="write replay_inputs/ (the default is a dry run that writes nothing)")
    ap.add_argument("--force", action="store_true", help="with --write: replace a case file that would change (never one that has saved records)")
    ap.add_argument("--inputs", default="report_inputs", help="folder of the twelve payloads")
    ap.add_argument("--out", default=OUT_DIR, help="folder the case files go to")
    ap.add_argument("--records-out", default=OUT_RECORDS, help="folder the runner saves the answers to (checked before a case file is replaced)")
    ap.add_argument("--records", action="append", metavar="IN:OUT", help="folders whose cases already exist and are left alone; repeat for several "
                    "(default: probe_inputs:probe_outputs and dry_inputs:dry_outputs, the ones that exist)")
    args = ap.parse_args(argv)
    check = _Checker(say)

    say(("WRITE " + args.out + "/" if args.write else "DRY RUN (nothing is written)") + "; local files only, no database, no model call, no Slack, .env not read")
    if not os.path.isdir(args.inputs):
        say(f"\nno folder {args.inputs}/ (run it from the project folder)")
        return 2
    loaded = rr.load_payloads(args.inputs)
    months = [t[0] for t in loaded]
    try:
        pairs = rc.parse_pairs(args.records) or list(COMPARE_PAIRS)
        have = existing_cases(pairs)
    except (ValueError, rc.RecordProblem) as why:
        say(f"\n--records: {why}")
        return 2

    say(f"\n1. The plan (from the {len(loaded)} payloads in {args.inputs}/ and the case folders that exist)")
    check("the payloads are the twelve replay months, in order", months == EXPECTED_MONTHS, f"{months[:1]}..{months[-1:]}")
    check(f"no month later than {LAST_MONTH}", all(m <= LAST_MONTH for m in months))
    try:
        p = plan([(t[0], t[3]) for t in loaded], have)
    except Refused as why:
        say(f"REFUSED: {why}")
        return 1
    say(f"\n  {'month':<8}{'product':>9}  tier  {'result':<15} note")
    for r in p["rows"]:
        say(f"  {r['month']:<8}{r['product_id']:>9}  {r['tier']:<4}  {r['status']:<15} {r['where']}")
    say(f"\n  {len(p['cases'])} case(s) to build, {len(p['cases']) * ARMS_PER_CASE} model call(s) needed")
    check("every product-month the messages show a block for has a case (existing or planned)",
          all(r["status"] in ("block rule", "beyond the cap", "has a case") or (r["status"] == "NEW" and r["where"].startswith("case ")) for r in p["rows"]))
    stale = stale_files(args.out, p["cases"])
    check(f"{args.out}/ holds nothing the plan does not contain", not stale, ", ".join(stale))
    cmds = runner_commands(p["cases"], args.out, args.records_out)
    if not p["cases"]:
        say(f"\nNothing to build: every product-month the messages show a block for already has a case.")
        return 0 if not check.fails else 1
    if check.fails:
        say(f"\n=== RESULT ===\nFAILED checks ({len(check.fails)}): {check.fails}\nNothing written.")
        return 1

    # ---- the data
    import dry_extract as de
    import probe_p1_extract as p1
    import pandas as pd
    sources = [p1.GATE3, p1.SOFT, p1.OVERVIEW, p1.EXTRACT, p1.META]
    missing = [s for s in sources + [de.PROBE_MANIFEST] if not os.path.exists(s)]
    if missing:
        say("\nthe local source files are not here (run it from the project folder): " + ", ".join(missing))
        return 2
    say("\n2. Sources (md5)")
    src_md5 = {}
    for s in sources:
        src_md5[s] = md5_path(s)
        say(f"  {src_md5[s]}  {s}")
    g = pd.read_csv(p1.GATE3, encoding="utf-8-sig")
    s_ = pd.read_csv(p1.SOFT, encoding="utf-8-sig")
    o = pd.read_csv(p1.OVERVIEW, encoding="utf-8-sig")
    e = pd.read_csv(p1.EXTRACT, encoding="utf-8-sig")
    theme_of = de.load_theme_of(p1.META)

    say("\n3. Segment status (whole corpus, the loader's three rules, via the probe's functions)")
    m = de.derive(g, s_, theme_of)
    check("rows joined", len(m) == 179263, f"{len(m):,}")
    rcount = m.reason.value_counts().to_dict()
    for k_, v in p1.EXPECT_REASONS.items():
        check(f"reason {k_}", rcount.get(k_, 0) == v, f"{rcount.get(k_, 0):,}")
    scount = m.status.value_counts().to_dict()
    for k_, v in p1.EXPECT_STATUS.items():
        check(f"status {k_}", scount.get(k_, 0) == v, f"{scount.get(k_, 0):,}")
    kept = de.kept_topic_pairs(m)
    recon = kept.groupby(["topic_id", "product_id", "review_month"]).review_id.nunique().rename("r").reset_index()
    j = recon.merge(e[["topic_id", "product_id", "review_month", "mention_count"]], on=["topic_id", "product_id", "review_month"], how="outer", indicator=True)
    check("mart cells reproduced (26,624), zero differences", len(e) == 26624 and int((j._merge != "both").sum()) == 0 and int((j.r != j.mention_count).sum()) == 0)
    mentions = kept.groupby(["product_id", "review_month"]).size().rename("mentions")
    uncl_mentions = kept[kept.topic_id == -1].groupby(["product_id", "review_month"]).size().rename("uncl_mentions")
    ext_m = e.groupby(["product_id", "review_month"]).mention_count.sum()
    ext_u = e[e.topic_id == -1].groupby(["product_id", "review_month"]).mention_count.sum()
    seg = m[m.status != "excluded_only"].copy()
    corpus_months = sorted(m.review_month.unique())

    say("\n4. Same builder as the probe and the dry run: their cases rebuilt and compared with their manifests")
    with open(de.PROBE_MANIFEST, encoding="utf-8") as f:
        pman = {c["case_id"]: c["md5"] for c in json.load(f)["cases"]}
    for pc in de.PROBE_CASES:
        _, text, _, _ = de.build_case(m, seg, corpus_months, pc["pid"], pc["asof"], pc["id"])
        check(f"probe case {pc['id']} rebuilds to the same md5", md5_bytes(text.encode("utf-8")) == pman.get(pc["id"]))
    dry_manifest = os.path.join("dry_inputs", "manifest.json")
    if os.path.exists(dry_manifest):
        with open(dry_manifest, encoding="utf-8") as f:
            dman = {c["case_id"]: c["md5"] for c in json.load(f)["cases"]}
        for dc in de.DRY_CASES:
            _, text, _, _ = de.build_case(m, seg, corpus_months, dc["pid"], dc["asof"], dc["id"])
            check(f"dry-run case {dc['id']} rebuilds to the same md5", md5_bytes(text.encode("utf-8")) == dman.get(dc["id"]))
    else:
        say("  (dry_inputs/manifest.json is not here: the dry run's cases are not compared)")

    say("\n5. The replay cases")
    built = {}
    for c in p["cases"]:
        body, text, pm, ptot = de.build_case(m, seg, corpus_months, c["product_id"], c["asof"], c["id"])
        built[c["id"]] = (body, text, pm, ptot)
    first = p["cases"][0]
    again = de.build_case(m, seg, corpus_months, first["product_id"], first["asof"], first["id"])
    check(f"sampling is reproducible (case {first['id']} built twice, identical bytes)", again[1] == built[first["id"]][1])
    say(f"\n  {'case':<7}{'now':>6}{'uncl':>6}{'prior in file':>15}{'prior kept':>12}{'chars now':>11}{'chars prior':>13}  {'bytes':>9}  md5")
    for c in p["cases"]:
        body, text, pm, ptot = built[c["id"]]
        k = body["counts"]
        say(f"  case {c['id']:<2}{k['current']:>6}{k['current_unclustered']:>6}{ptot:>15}{k['prior_kept']:>12}{k['chars_current']:>11,}{k['chars_prior']:>13,}"
            f"  {len(text.encode()):>9,}  {md5_bytes(text.encode('utf-8'))}")
    for c in p["cases"]:
        body, _, pm, _ = built[c["id"]]
        per_month = {mo: int(((seg.product_id == c["product_id"]) & (seg.review_month == mo)).sum()) for mo in pm}
        say(f"  case {c['id']}: {c['product_id']} as of {c['month']} (tier {c['tier']}) | prior months {[x[:7] for x in pm]} "
            f"model-input segments per prior month {list(per_month.values())}")

    say("\n6. Checks")
    ctx = dict(o=o, months=corpus_months, mentions=mentions, uncl_mentions=uncl_mentions, ext_m=ext_m, ext_u=ext_u, prior_months=p1.PRIOR_MONTHS,
               prior_cap=p1.PRIOR_CAP, tf=p1.tf, tier_of=de.tier_of)
    for c in p["cases"]:
        body, _, pm, ptot = built[c["id"]]
        case_checks(check, c, body, ptot, pm, ctx)
    check("every case file holds only the allowed keys (no nicknames, user ids or profile fields)",
          all(list(built[c["id"]][0].keys()) == p1.CASE_KEYS and all(list(x.keys()) == p1.SEGMENT_KEYS for x in built[c["id"]][0]["segments"]) for c in p["cases"]))
    check("segment ids are unique within each case",
          all(len({x["segment_id"] for x in built[c["id"]][0]["segments"]}) == len(built[c["id"]][0]["segments"]) for c in p["cases"]))
    say("\n  PII scan of the text (counts only; nothing is edited or removed):")
    any_hit = False
    for c in p["cases"]:
        segs = built[c["id"]][0]["segments"]
        hits = {k_: [x["segment_id"] for x in segs if pat.search(x["text"])] for k_, pat in p1.PII.items()}
        any_hit |= sum(len(v) for v in hits.values()) > 0
        say(f"    case {c['id']}: " + ", ".join(f"{k_} {len(v)}" for k_, v in hits.items()))
        for k_, v in hits.items():
            if v:
                say(f"      {k_} segment ids: {v}")
    say("    (no hits)" if not any_hit else "    hits listed above")

    # ---- the files
    files = {f"case_{c['id']}.json": built[c["id"]][1].encode("utf-8") for c in p["cases"]}
    manifest = manifest_of(p["cases"], built, src_md5, md5_path(os.path.abspath(__file__)), p1.SEED, p1.PRIOR_CAP)
    files["manifest.json"] = p1.dumps(manifest).encode("utf-8")
    if check.fails:
        say(f"\n=== RESULT ===\nFAILED checks ({len(check.fails)}): {check.fails}\nNothing written.")
        return 1
    code = 0
    if args.write:
        state, notes = write_cases(files, args.out, args.records_out, args.force)
        say("\n" + "\n".join(f"  {os.path.join(args.out, n)}" for n in notes))
        if state == "refused":
            say("\n=== RESULT ===\nREFUSED: nothing was written.")
            return 1
    else:
        say("\nDry run: would write " + ", ".join(os.path.join(args.out, n) for n in sorted(files)) + ".")
    say(f"\n=== RESULT ===\nAll {check.n} checks passed.")
    say(f"\nNext: {cmds['calls']} model call(s). First look at what would be sent (no call):\n  {cmds['look']}\nthen:\n  {cmds['run']}")
    return code


class _Checker:
    def __init__(self, say):
        self.fails, self.n, self.say = [], 0, say

    def __call__(self, name, ok, detail=""):
        self.n += 1
        if not ok:
            self.fails.append(name)
        self.say(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        return ok


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
