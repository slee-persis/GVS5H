#!/usr/bin/env python3
"""Per-problem, per-pass token counts for the three LCB-100 arms -> runs/per_problem_tokens.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "runs/per_problem_tokens.json")
PASSES = [1, 2, 3, 4, 5]

ARMS = {
    "q38_single": f"{ROOT}/runs/4models-1pass-reason-on/results/q38_single_p%d.json",
    "q38_multi": f"{ROOT}/runs/4models-1pass-reason-on/results/q38_multiagent_p%d.json",
    "fable_single": f"{ROOT}/runs/fable5-5pass-single/results/fable5_single_p%d.json",
    "luna_single": f"{ROOT}/runs/4models-1pass-reason-on/results/luna_single_p%d.json",
    "luna_multi": f"{ROOT}/runs/4models-1pass-reason-on/results/luna_multiagent_p%d.json",
    "terra_single": f"{ROOT}/runs/4models-1pass-reason-on/results/terra_single_p%d.json",
    "terra_multi": f"{ROOT}/runs/4models-1pass-reason-on/results/terra_multiagent_p%d.json",
}


def tokens(ws):
    got = [0, 0, 0, 0]
    with open(os.path.join(ws, "transcript.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            if "completion_tokens" not in d and "prompt_tokens" not in d:
                continue
            i = 2 if d.get("discarded") else 0
            got[i] += d.get("prompt_tokens") or 0
            got[i + 1] += d.get("completion_tokens") or 0
    return got


def main():
    out = {}
    for arm, pattern in ARMS.items():
        qids, rows = None, []
        for p in PASSES:
            recs = json.load(open(pattern % p))["lcb"]["records"]
            ids = [r["question_id"] for r in recs]
            assert qids is None or ids == qids, f"{arm} p{p}: question_id order differs"
            qids = ids
            rows.append([tokens(r["ws"]) for r in recs])
            print(f"  {arm} p{p}: {len(recs)} problems")
        out[arm] = dict(qids=qids, tokens=rows)
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT, f"({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
