#!/usr/bin/env python3
"""Rebuild Qwen3.8's 128k cap-matched single arm from the 250k generations.

The single arm was generated at a 250k cap. This replays each generation truncated to
128,000 output tokens and re-extracts the solution from that prefix, so the arm can be
compared like-for-like against the manager arm (which ran natively at 128k). It writes new
results files that the ordinary regrade path then scores, so the cap-matched numbers come
off the same fixed evaluator as everything else.

Truncation is token-exact, using the serving stack's own tokenizer (vLLM /tokenize for
`Qwen/Qwen3.8-27B-FP8`) rather than a character-count approximation -- a proportional
guess is wrong here precisely because the interesting cases are ones where a complete code
block sits just before or just after the boundary.

What gets truncated is the model's whole output stream, reasoning THEN answer, because
that is what the 250k cap bounded. Where the cut lands decides what the extractor sees:
past the reasoning, it sees a truncated answer; inside the reasoning, the answer does not
exist at all and the harness's empty-content fallback hands it the truncated reasoning
instead -- which is why so many cut-off generations still yield code (S2.2).

    uv run --no-project --python 3.12 --with 'datasets<4' --with numpy python escalation/capmatch_q38.py

Writes <name>.cap128k.json next to each source file; grade them with regrade.py.
"""
import glob
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "livecodebench"))
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LMStyle

CAP = 128_000
MODEL = os.environ.get("CAPMATCH_MODEL", "Qwen/Qwen3.8-27B-FP8")
BASE = os.environ.get("CAPMATCH_BASE", "http://localhost:8215")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))   # repo root, three up from escalation/
RESULTS = os.path.join(ROOT, "runs/4models-1pass-reason-on/results")
WS = os.path.join(ROOT, "runs/4models-1pass-reason-on/ws")


def _post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def tokens(text):
    return _post("/tokenize", {"model": MODEL, "prompt": text})["tokens"]


def detokenize(ids):
    return _post("/detokenize", {"model": MODEL, "tokens": ids})["prompt"]


def call_of(ws):
    """The single arm makes one model call per problem; return its transcript record."""
    tr = os.path.join(ws, "transcript.jsonl")
    if not os.path.exists(tr):
        return None
    for line in open(tr):
        rec = json.loads(line)
        if not rec.get("_meta"):
            return rec
    return None


def truncated_answer(call):
    """What the harness would have graded had the cap been 128k instead of 250k.

    Returns None when the generation fits inside the cap and nothing changes.
    """
    reasoning = call.get("reasoning") or ""
    response = call.get("response") or ""
    if reasoning and reasoning == response:
        # openai_chat's empty-content fallback stores ONE stream in both fields (142 of the
        # 500 records here), so tokenizing both and adding would double its length.
        ids = tokens(response)
        return None if len(ids) <= CAP else detokenize(ids[:CAP])
    r_ids = tokens(reasoning) if reasoning else []
    c_ids = tokens(response) if response else []
    if len(r_ids) + len(c_ids) <= CAP:
        return None
    if len(r_ids) >= CAP:
        # The cut lands inside the reasoning: no answer was ever emitted, so the harness
        # falls back to the reasoning text (orchestrator.openai_chat does this when
        # content is empty).
        return detokenize(r_ids[:CAP])
    return detokenize(c_ids[:CAP - len(r_ids)])


def main():
    for p in range(1, 6):
        src = f"{RESULTS}/q38_single_p{p}.json"
        blob = json.load(open(src))
        recs = blob["lcb"]["records"]
        cut = changed = 0
        for r in recs:
            ws = r.get("ws")
            if ws:
                ws = os.path.join(WS, *ws.split("/")[-2:])   # <config>/<hash>, re-rooted under runs/
            if not ws or not os.path.isdir(ws):
                continue
            call = call_of(ws)
            if call is None:
                continue
            raw = truncated_answer(call)
            if raw is None:
                continue
            cut += 1
            new_code = extract_code(raw, LMStyle.ClaudeCode)
            if new_code != r.get("code"):
                changed += 1
            r["code"] = new_code
            r["cap_matched_to"] = CAP
        # pass@1 is recomputed by regrade.py; clear it so a stale value cannot be read.
        blob["lcb"]["pass@1"] = None
        dst = src.replace(".json", ".cap128k.json")
        json.dump(blob, open(dst, "w"))
        n_empty = sum(1 for r in recs if not (r.get("code") or "").strip())
        print(f"p{p}: {cut:3} generations over the cap, {changed:3} with different code, "
              f"{n_empty:3} now empty -> {os.path.basename(dst)}")


if __name__ == "__main__":
    main()
