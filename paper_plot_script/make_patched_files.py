#!/usr/bin/env python3
"""Write the *.patched.json twins the plot scripts read."""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VERDICTS = os.path.join(ROOT, "runs", "patched_verdicts.json")

RUN_DIRS = {
    "128k-reasoning-on-1pass": "runs/128k-reasoning-on-1pass/results",
    "128k-reasoning-off-1pass": "runs/128k-reasoning-off-1pass/results",
    "16k-reasoning-off-5pass": "runs/16k-reasoning-off-5pass/results",
    "firstparty-128k-reasoning-on-5pass": "runs/firstparty-128k-reasoning-on-5pass/results",
    "fable5-128k-reasoning-on-5pass": "runs/fable5-128k-reasoning-on-5pass/results",
}


def main():
    data = json.load(open(VERDICTS))
    written = skipped = 0
    for run, arms in data.items():
        base = os.path.join(ROOT, RUN_DIRS[run])
        for arm, rec in arms.items():
            src = os.path.join(base, f"{arm}.regraded.json")
            if not os.path.exists(src):
                print(f"  no regraded twin, skipping: {run}/{arm}")
                skipped += 1
                continue
            blob = json.load(open(src))
            verdicts = rec["verdicts"]
            recs = blob["lcb"]["records"]
            for r in recs:
                if r["question_id"] in verdicts:
                    r["passed"] = verdicts[r["question_id"]]
            graded = [r for r in recs if r.get("status") != "infra"]
            blob["lcb"]["pass@1"] = 100.0 * sum(bool(r["passed"]) for r in graded) / max(1, len(graded))
            dst = os.path.join(base, f"{arm}.patched.json")
            with open(dst, "w") as f:
                json.dump(blob, f, indent=2)
            written += 1
    print(f"wrote {written} *.patched.json files, skipped {skipped}")


if __name__ == "__main__":
    main()
