#!/usr/bin/env python3
"""Per-problem, per-pass token counts for the three LCB-100 arms -> runs/per_problem_tokens.json.

The results JSONs carry `completion_tokens` only for the single arms (the manager records
None: its problem-level record has no single call to attribute tokens to), and neither arm
records prompt tokens. The workspace transcripts have both, per call, including the calls
that were retried and discarded -- so they are the only complete source, and the only one
that can attribute a manager problem's whole worker fan-out to that problem.

Each record's `ws` field names its own workspace directory, so question_id -> tokens is
exact rather than positional. `discarded` rows are counted separately: a discarded attempt
was still generated and still billed, but it is not the work that produced the answer.

Reading every transcript is ~650MB of JSON, so this writes a small extract that the cost
plot reads instead of re-walking the tree on every run.

    uv run --project /home/persis/model-test python \\
        paper_plot_script/extract_tokens.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root, one level up from paper_plot_script/
OUT = os.path.join(ROOT, "runs/per_problem_tokens.json")
PASSES = [1, 2, 3, 4, 5]

ARMS = {
    "q38_single": f"{ROOT}/runs/4models-1pass-reason-on/results/q38_single_p%d.json",
    "q38_multi": f"{ROOT}/runs/4models-1pass-reason-on/results/q38_multiagent_p%d.json",
    "fable_single": f"{ROOT}/runs/fable5-5pass-single/results/fable5_single_p%d.json",
    # The GPT-5.6 arms. The cost chart prices only the two managers, but the cost-vs-score
    # scatter covers ALL seven pinned-backend arms and reads its tokens from this same file,
    # so dropping the single twins here would break that figure rather than shrink it.
    "luna_single": f"{ROOT}/runs/4models-1pass-reason-on/results/luna_single_p%d.json",
    "luna_multi": f"{ROOT}/runs/4models-1pass-reason-on/results/luna_multiagent_p%d.json",
    "terra_single": f"{ROOT}/runs/4models-1pass-reason-on/results/terra_single_p%d.json",
    "terra_multi": f"{ROOT}/runs/4models-1pass-reason-on/results/terra_multiagent_p%d.json",
}


def tokens(ws):
    """-> (in, out, discarded_in, discarded_out) over every call in one problem's workspace."""
    got = [0, 0, 0, 0]
    with open(os.path.join(ws, "transcript.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            if "completion_tokens" not in d and "prompt_tokens" not in d:
                continue                       # _meta header row
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
        # [pass][problem][in, out, disc_in, disc_out]
        out[arm] = dict(qids=qids, tokens=rows)
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT, f"({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
