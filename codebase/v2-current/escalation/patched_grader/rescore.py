"""Re-grade the LCB-100 runs with the special judges in checkers.py.

READ-ONLY with respect to everything that already exists: it reads the recorded results
JSONs under runs/ and writes fresh copies elsewhere. No run is re-executed against the
model -- only the saved code is re-run against the tests.

Two defects are corrected:

  1. WRONG JUDGE. Four problems accept more than one right answer (or a float within
     tolerance) and LiveCodeBench string-compares. checkers.py supplies the real judge.

  2. NO sys.stdout.buffer. The *stdin* half of this was already fixed upstream of here
     (MockStdinWithBuffer in codebase/livecodebench/lcb_runner/evaluation/testing_util.py,
     replayed into the *.regraded.json files). The OUTPUT half is still live: `Capturing`
     swaps sys.stdout for a StringIO, which has no .buffer, so `sys.stdout.buffer.write()`
     dies with AttributeError before emitting a byte. Here each submission runs as a real
     subprocess with real pipes, like the original judge, so neither stream can misbehave.

Only recorded FAILURES are re-graded, and only where one of the defects can apply: the four
mis-judged ids, plus any failure whose source mentions stdin.buffer (which in practice is
the same code that reaches for stdout.buffer). A pass under exact match is still a pass
under a strictly more lenient judge, so passes are carried over untouched rather than
re-run. The baseline reported as `old_pass@1` is the arm's *.regraded.json score wherever
one exists -- that is the number the paper plotted before the patch.

Outputs, both regenerated from scratch on every run:

  runs/patched_grader/<run>__<arm>.json   per-arm detail: every record's final verdict plus
                                          the tag of whatever the re-run did (ok, timeout,
                                          runtime_error, wrong_answer, empty)
  runs/patched_verdicts.json              what the charts consume, via
                                          paper_plot_script/make_patched_files.py

This repo carries no environment of its own; run it against the one that produced the runs:

    uv run --project /home/persis/model-test python \\
        codebase/v2-current/escalation/patched_grader/rescore.py
"""
import os
import sys
import json
import glob
import time
import resource
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
# patched_grader -> escalation -> v2-current -> codebase -> repo root
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(REPO, "codebase", "livecodebench"))
sys.path.insert(0, HERE)

from checkers import check, CHECKERS  # noqa: E402

# The four problems whose judge is wrong (see the memo in checkers.py). abc385_f is graded
# by default_check's float tolerance rather than by a bespoke checker.
MISJUDGED = set(CHECKERS) | {"abc385_f"}
PER_TEST_TIMEOUT = float(os.environ.get("PATCH_TIMEOUT", "10"))
MEM_LIMIT_BYTES = 4 * 1024 ** 3
WORKERS = int(os.environ.get("PATCH_WORKERS", "12"))
OUT_DIR = os.path.join(REPO, "runs", "patched_grader")
VERDICTS_PATH = os.path.join(REPO, "runs", "patched_verdicts.json")

# The five runs the paper reports. Regrading exactly these reproduces patched_verdicts.json.
PAPER_RUNS = [
    "128k-reasoning-off-1pass",
    "128k-reasoning-on-1pass",
    "16k-reasoning-off-5pass",
    "fable5-128k-reasoning-on-5pass",
    "firstparty-128k-reasoning-on-5pass",
]
# PATCH_RUNS: comma-separated run directory names under runs/ to re-grade instead.
# PATCH_MODELS: comma-separated substrings; if set, only arms whose model matches are kept.
# PATCH_GLOBS: raw comma-separated globs, for runs like results_think_high whose JSONs sit
# at the top of the run directory instead of under results/.
if os.environ.get("PATCH_GLOBS"):
    RESULT_GLOBS = [g.strip() for g in os.environ["PATCH_GLOBS"].split(",")]
else:
    runs = [d.strip() for d in os.environ.get("PATCH_RUNS", ",".join(PAPER_RUNS)).split(",")]
    RESULT_GLOBS = [os.path.join(REPO, "runs", d, "results", "*.json") for d in runs]
MODEL_FILTER = [m.strip() for m in os.environ.get("PATCH_MODELS", "").split(",") if m.strip()]


def _limit():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))


def run_one(code_path, stdin_text):
    """Run a submission on one test. Returns (ok, stdout, tag)."""
    try:
        p = subprocess.run([sys.executable, code_path], input=stdin_text,
                           capture_output=True, text=True,
                           timeout=PER_TEST_TIMEOUT, preexec_fn=_limit)
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    if p.returncode != 0:
        return False, p.stdout, "runtime_error"
    return True, p.stdout, "ok"


def work(job):
    """Worker entry point. Runs in a separate PROCESS: the special judges are pure Python
    and would otherwise serialise on the GIL behind the subprocess calls."""
    arm, qid, code, tests = job
    passed, tag = regrade(qid, code, tests)
    return arm, qid, passed, tag


def regrade(qid, code, tests):
    """Run `code` against every test of `qid` under the patched judge."""
    if not (code or "").strip():
        return False, "empty"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sol.py")
        with open(path, "w") as f:
            f.write(code)
        for inp, exp in tests:
            ok, out, tag = run_one(path, inp)
            if not ok:
                return False, tag
            if not check(qid, inp, exp, out):
                return False, "wrong_answer"
    return True, "ok"


def main():
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    os.makedirs(OUT_DIR, exist_ok=True)

    dataset = load_code_generation_dataset(
        release_version=os.environ.get("LCB_RELEASE", "release_v6"))
    by_id = {p.question_id: p for p in dataset}

    tests_cache = {}

    def tests_for(qid):
        if qid not in tests_cache:
            io = by_id[qid].get_evaluation_sample()["input_output"]
            io = json.loads(io) if isinstance(io, str) else io
            tests_cache[qid] = list(zip(io["inputs"], io["outputs"]))
        return tests_cache[qid]

    files = []
    for g in RESULT_GLOBS:
        # `.regraded.json` twins are not separate arms -- they are the BASELINE for the
        # matching base file, and hold identical `code` (regrade.py only re-scores).
        # `.patched.json` twins are this script's own output replayed by
        # paper_plot_script/make_patched_files.py.
        files += [f for f in sorted(glob.glob(g))
                  if ".regraded" not in f and ".patched" not in f]

    jobs, arms = [], {}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        # Placeholders carry no records: {"skipped": ...} (opus 128k reasoning-off) and
        # {"note": ...} (the nemotron stubs).
        if "lcb" not in d or not d["lcb"].get("records"):
            print(f"  skip (no records): {os.path.basename(f)}", flush=True)
            continue
        model = d.get("model") or ""
        if MODEL_FILTER and not any(m in model for m in MODEL_FILTER):
            continue
        parent = os.path.dirname(f)   # some runs keep JSONs in results/, others at the top
        run = os.path.basename(os.path.dirname(parent) if os.path.basename(parent) == "results"
                               else parent)
        name = os.path.basename(f)[:-5]
        arm = f"{run}__{name}"
        # Older runs predate the n_graded/n_infra split and their records carry only
        # question_id/code/passed -- no `status`, so nothing is infra-excluded there.
        lcb = d["lcb"]
        # The baseline to beat is the regraded score where one exists -- that is what the
        # paper plots. Some base files (q38's cap128k replay) carry pass@1 = null and only
        # the regraded twin has a number at all.
        rg = f.replace(".json", ".regraded.json")
        old, old_src = lcb["pass@1"], "raw"
        if os.path.exists(rg):
            try:
                old, old_src = json.load(open(rg))["lcb"]["pass@1"], "regraded"
            except Exception:
                pass
        arms[arm] = {"file": f, "run": run, "name": name, "model": d.get("model"),
                     "engine": d.get("engine"), "records": lcb["records"],
                     "old": old, "old_src": old_src,
                     "n_graded": lcb.get("n_graded", len(lcb["records"])),
                     "n_infra": lcb.get("n_infra", 0)}
        for r in arms[arm]["records"]:
            if r.get("passed"):
                continue
            qid, code = r["question_id"], r.get("code") or ""
            if qid not in by_id:             # id not in this release; leave the verdict alone
                continue
            if by_id[qid].starter_code:      # call-based problem; this grader is stdin-only
                continue
            if qid in MISJUDGED or "stdin.buffer" in code or "stdout.buffer" in code:
                jobs.append((arm, qid, code, tests_for(qid)))

    print(f"{len(arms)} arms, {len(jobs)} failed records eligible for re-grading", flush=True)

    flips = {}
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for arm, qid, passed, tag in ex.map(work, jobs, chunksize=1):
            done += 1
            flips[(arm, qid)] = (passed, tag)
            if passed:
                print(f"  FLIP {arm:32s} {qid:12s} -> PASS", flush=True)
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(jobs) - done) / rate if rate else 0
                print(f"  ... {done}/{len(jobs)}  {el/60:.1f}m elapsed, ~{eta/60:.1f}m left",
                      flush=True)

    summary, verdicts = [], {}
    for arm, a in arms.items():
        recs = []
        for r in a["records"]:
            qid = r["question_id"]
            passed = bool(r.get("passed"))
            tag = None
            if (arm, qid) in flips:
                passed, tag = flips[(arm, qid)]
            recs.append({"question_id": qid, "passed": passed,
                         "status": r.get("status"), "patched_tag": tag})
        graded = [r for r in recs if r["status"] != "infra"]
        newp = 100.0 * sum(r["passed"] for r in graded) / max(1, len(graded))
        delta = None if a["old"] is None else newp - a["old"]
        summary.append({"arm": arm, "run": a["run"], "model": a["model"], "engine": a["engine"],
                        "old_pass@1": a["old"], "old_src": a["old_src"], "new_pass@1": newp,
                        "delta": delta, "n_graded": len(graded)})
        with open(os.path.join(OUT_DIR, f"{arm}.json"), "w") as f:
            json.dump({"arm": arm, "run": a["run"], "model": a["model"], "engine": a["engine"],
                       "old_pass@1": a["old"], "old_src": a["old_src"], "new_pass@1": newp,
                       "records": recs}, f, indent=2)
        # The charts want one verdict per question, keyed by run and by the arm's own file
        # name -- the run prefix is what the enclosing dict already says.
        verdicts.setdefault(a["run"], {})[a["name"]] = {
            "old_pass@1": a["old"], "old_src": a["old_src"], "new_pass@1": newp,
            "verdicts": {r["question_id"]: r["passed"] for r in recs}}

    with open(VERDICTS_PATH, "w") as f:
        json.dump(verdicts, f, indent=1)

    summary.sort(key=lambda s: (s["model"] or "", s["engine"] or "", s["arm"]))
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'arm':50s} {'model':34s} {'old':>6s} {'new':>6s} {'delta':>7s}")
    for s in summary:
        o = "  n/a" if s['old_pass@1'] is None else f"{s['old_pass@1']:6.1f}"
        dl = "    n/a" if s['delta'] is None else f"{s['delta']:+7.1f}"
        print(f"{s['arm']:50s} {(s['model'] or '')[:34]:34s} "
              f"{o:>6s} {s['new_pass@1']:6.1f} {dl:>7s}")
    print(f"\nWrote {VERDICTS_PATH}, {OUT_DIR}/summary.json and one JSON per arm.")


if __name__ == "__main__":
    main()
