#!/usr/bin/env python3
"""Re-grade stored generations against the CURRENT evaluator, without re-running any model."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "livecodebench"))

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

import numpy as np


def _passed(res0):
    if isinstance(res0, (list, tuple)):
        return bool(np.all(np.array(res0) > 0)) and len(res0) > 0
    return bool(res0)


def regrade_all(paths, problems):
    kept = []
    for path in paths:
        blob = json.load(open(path))
        if "lcb" not in blob or not blob["lcb"].get("records"):
            print(f"  skip (no records): {os.path.basename(path)}")
            continue
        kept.append((path, blob))
    paths = [p for p, _ in kept]

    blobs, index = [], []
    samples, generations = [], []
    for bi, (path, blob) in enumerate(kept):
        blobs.append(blob)
        for ri, r in enumerate(blob["lcb"]["records"]):
            if not (r.get("code") or "").strip():
                r["passed_before_regrade"] = bool(r.get("passed"))
                r["passed"] = False
                continue
            qid = r["question_id"]
            if qid not in problems:
                raise SystemExit(f"{path}: id {qid!r} is not in the loaded dataset")
            index.append((bi, ri))
            samples.append(problems[qid].get_evaluation_sample())
            generations.append([r["code"]])

    nproc = int(os.environ.get("REGRADE_PROCS", min(96, max(8, (os.cpu_count() or 8) // 2))))
    print(f"grading {len(generations)} generations from {len(paths)} files "
          f"with {nproc} workers", flush=True)
    _, results, _ = codegen_metrics(samples, generations, k_list=[1], num_process_evaluate=nproc)

    for n, (bi, ri) in enumerate(index):
        rec = blobs[bi]["lcb"]["records"][ri]
        rec["passed_before_regrade"] = bool(rec.get("passed"))
        rec["passed"] = _passed(results[n][0])

    out = []
    for path, blob in zip(paths, blobs):
        recs = blob["lcb"]["records"]
        graded = [r for r in recs if r.get("status") != "infra"]
        npass = sum(bool(r["passed"]) for r in graded)
        old = blob["lcb"]["pass@1"]
        blob["lcb"]["pass@1_before_regrade"] = old
        blob["lcb"]["pass@1"] = 100.0 * npass / max(1, len(graded))
        dst = path.replace(".json", ".regraded.json")
        json.dump(blob, open(dst, "w"))
        flips = [(r["question_id"], r["passed_before_regrade"], r["passed"])
                 for r in recs if "passed_before_regrade" in r
                 and r["passed_before_regrade"] != r["passed"]]
        out.append((path, old, blob["lcb"]["pass@1"], flips))
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    os.environ.setdefault("LCB_RELEASE", "release_v6")
    problems = {p.question_id: p
                for p in load_code_generation_dataset(release_version=os.environ["LCB_RELEASE"])}
    print(f"loaded {len(problems)} problems from {os.environ['LCB_RELEASE']}\n")
    for path, old, new, flips in regrade_all(paths, problems):
        gained = sum(1 for _, w, n in flips if n and not w)
        lost = sum(1 for _, w, n in flips if w and not n)
        was = f"{old:5.1f}" if old is not None else "    -"
        print(f"{os.path.basename(path):34} {was} -> {new:5.1f}  "
              f"({gained:+3d} newly passing, {lost:3d} newly failing)")
        for qid, was, now in flips:
            if was and not now:
                print(f"    !! now failing: {qid}")


if __name__ == "__main__":
    main()
