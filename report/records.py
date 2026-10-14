"""Finding and reading the saved model records of a product-month.

A "case" is the input the model was given for one flagged product-month (`case_N.json`: schema, case_id, product_id, as_of_month, the
month's complaint segments); each case has two saved records, Arm A (the readout) and Arm B (the labelling), named
`probe_output_case{N}_{A|B}_{label}_r{run}.json`. A case and its records live in a pair of folders (probe_inputs and probe_outputs for the probe's
cases, dry_inputs and dry_outputs for the dry run, replay_inputs and replay_outputs for the replay's own); `MultiStore` reads several pairs as one. This module only finds, checks and hands over what is
saved; it calls nothing (no model, no database, no network) and writes nothing.

THE RUN OF RECORD: the first run, in run order, whose status is `structure_ok`. A failed run is never overwritten (the runner writes each call
to a new r1, r2, ...), so a retry is simply the next run, and it replaces the failed one for reading; the failed run stays on disk and is listed under
`earlier_attempts` in the stamps. If no run passed, the latest one is handed over, so the problem named is the latest.
`run=1` asks for run 1 only. When the first run passes, nothing changes: it is still the run of record.

A record is refused unless it is `structure_ok`: a response that failed the structure or id checks is retried once by the replay loop, and a
second failure stops the run, so the renderer never builds a block from one. `strict=False` exists for one reason: a case whose
every run failed its checks can still be shown, with the failure named in the problems; the caller must ask for that on purpose and the message
records it.
"""
import glob
import hashlib
import json
import os
import re

ARMS = {"A": "readout", "B": "labelling"}
FIRST_OK = "first_ok"
RECORD_NAME = "probe_output_case{case}_{arm}_{label}_r{run}.json"


class MissingRecords(Exception):
    """The month has a flagged product but no saved case or record for it."""


class RecordProblem(Exception):
    """A record exists but cannot be used (wrong status, wrong case, input changed since)."""


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class RecordStore:
    def __init__(self, input_dir, output_dir, label="prompt-v0", run=FIRST_OK):
        self.input_dir, self.output_dir, self.label, self.run = input_dir, output_dir, label, run
        self.cases = {}
        for path in sorted(glob.glob(os.path.join(input_dir, "case_*.json"))):
            if not re.search(r"case_\d+\.json$", path):
                continue
            c = read_json(path)
            key = (c["as_of_month"][:7], int(c["product_id"]))
            if key in self.cases:
                raise RecordProblem(f"two cases for {key}: {self.cases[key][1]} and {path}")
            self.cases[key] = (int(c["case_id"]), path)

    def saved_runs(self, case_id, arm):
        """{run number: path} of the saved runs of one case and arm: every run under the default rule, or just the one asked for."""
        if self.run != FIRST_OK:
            path = os.path.join(self.output_dir, RECORD_NAME.format(case=case_id, arm=arm, label=self.label, run=self.run))
            return {self.run: path} if os.path.exists(path) else {}
        out = {}
        for path in glob.glob(os.path.join(self.output_dir, RECORD_NAME.format(case=case_id, arm=arm, label=self.label, run="*"))):
            m = re.search(r"_r(\d+)\.json$", path)
            if m:
                out[int(m.group(1))] = path
        return dict(sorted(out.items()))

    def choose(self, case_id, arm):
        """(run, path, earlier attempts) for the run of record, or None when nothing is saved. `earlier` lists the runs before it (run, file, status)."""
        runs = self.saved_runs(case_id, arm)
        if not runs:
            return None
        status = {n: read_json(path).get("status") for n, path in runs.items()}
        if self.run != FIRST_OK:
            chosen = self.run
        else:
            chosen = next((n for n in runs if status[n] == "structure_ok"), max(runs))
        return chosen, runs[chosen], [{"run": n, "file": os.path.basename(runs[n]), "status": status[n]} for n in runs if n < chosen]

    def record_path(self, case_id, arm):
        c = self.choose(case_id, arm)
        if c:
            return c[1]
        return os.path.join(self.output_dir, RECORD_NAME.format(case=case_id, arm=arm, label=self.label, run=1 if self.run == FIRST_OK else self.run))

    def has(self, month, product_id):
        key = (month, int(product_id))
        if key not in self.cases:
            return False
        cid = self.cases[key][0]
        return all(self.choose(cid, a) is not None for a in ARMS)

    def load(self, month, product_id, strict=True):
        """{'case', 'case_path', 'readout' (Arm A parsed), 'labeling' (Arm B parsed), 'records': {'A', 'B'}, 'record_files': {'A', 'B'},
        'earlier': {'A': [...], 'B': [...]}, 'problems': [...]}."""
        key = (month, int(product_id))
        if key not in self.cases:
            raise MissingRecords(f"no case file for product {product_id} in {month} under {self.input_dir}")
        cid, case_path = self.cases[key]
        recs, files, earlier, problems = {}, {}, {}, []
        for arm in ARMS:
            chosen = self.choose(cid, arm)
            if chosen is None:
                raise MissingRecords(f"no record {RECORD_NAME.format(case=cid, arm=arm, label=self.label, run='*' if self.run == FIRST_OK else self.run)} under {self.output_dir}")
            _, path, earlier[arm] = chosen
            rec = read_json(path)
            if rec.get("case_id") != cid or rec.get("arm") != arm:
                raise RecordProblem(f"{os.path.basename(path)}: it is case {rec.get('case_id')} arm {rec.get('arm')}, expected case {cid} arm {arm}")
            if rec.get("case_file_md5") != md5_file(case_path):
                raise RecordProblem(f"{os.path.basename(path)}: the case file changed after the model was called (md5 differs)")
            if rec.get("status") != "structure_ok":
                msg = f"{os.path.basename(path)}: status {rec.get('status')}: {rec.get('structure_problems')}"
                if strict:
                    raise RecordProblem(msg)
                problems.append(msg)
            recs[arm], files[arm] = rec, os.path.basename(path)
        return {"case": read_json(case_path), "case_path": case_path, "readout": recs["A"]["parsed"], "labeling": recs["B"]["parsed"],
                "records": recs, "record_files": files, "earlier": earlier, "problems": problems}


class MultiStore:
    """Several RecordStores read as one. A product-month is looked up in the store that holds its case; two stores holding the same one are refused."""
    def __init__(self, stores):
        self.stores = list(stores)
        self.cases, self.owner = {}, {}
        for store in self.stores:
            for key, value in store.cases.items():
                if key in self.cases:
                    raise RecordProblem(f"two cases for {key}: {self.cases[key][1]} and {value[1]}")
                self.cases[key], self.owner[key] = value, store

    def has(self, month, product_id):
        store = self.owner.get((month, int(product_id)))
        return store is not None and store.has(month, product_id)

    def load(self, month, product_id, strict=True):
        store = self.owner.get((month, int(product_id)))
        if store is None:
            raise MissingRecords(f"no case file for product {product_id} in {month} under {', '.join(s.input_dir for s in self.stores)}")
        return store.load(month, product_id, strict=strict)


DEFAULT_PAIRS = (("probe_inputs", "probe_outputs"), ("dry_inputs", "dry_outputs"), ("replay_inputs", "replay_outputs"))


def open_store(pairs=None, **kw):
    """A store over the given (input folder, output folder) pairs: the probe's, the dry run's and the replay's when none are given (only those that exist)."""
    pairs = list(pairs) if pairs else [p for p in DEFAULT_PAIRS if os.path.isdir(p[0])]
    stores = [RecordStore(i, o, **kw) for i, o in pairs]
    return stores[0] if len(stores) == 1 else MultiStore(stores)


def parse_pairs(specs):
    """['in:out', ...] -> [(in, out), ...]; None or [] when there are none."""
    out = []
    for spec in specs or []:
        if spec.count(":") != 1 or not all(spec.split(":")):
            raise ValueError(f"expected IN_FOLDER:OUT_FOLDER, got {spec!r}")
        out.append(tuple(spec.split(":")))
    return out


def stamps(loaded):
    """What the run record keeps of the two model calls."""
    out = {"case_id": loaded["case"]["case_id"], "case_dir": os.path.basename(os.path.dirname(loaded["case_path"])),
           "case_file_md5": md5_file(loaded["case_path"]), "problems": list(loaded["problems"])}
    for arm, rec in loaded["records"].items():
        out[arm] = {"file": loaded["record_files"][arm], "earlier_attempts": loaded["earlier"][arm], "prompt_file": rec["prompt_file"], "prompt_md5": rec["prompt_md5"], "output_schema_version": rec["output_schema_version"],
                    "requested_model": rec["requested_model"], "served_model": rec["served_model"], "run_index": rec["run_index"],
                    "response_id": rec["response_id"], "finished_at_utc": rec["finished_at_utc"], "status": rec["status"]}
    return out
