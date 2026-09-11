"""Benchmark driver"""

import os
import re
import sys
import json
import argparse
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # repo root, three up from escalation/
LCB = os.path.join(ROOT, "codebase", "livecodebench")
RUNS = os.path.join(ROOT, "runs")
sys.path.insert(0, LCB)
sys.path.insert(0, HERE)

from orchestrator import (escalate, CODE_SPEC, MATH_SPEC, MATH500_SPEC, GPQA_SPEC, HLE_SPEC, LADDER)

SOLVE = escalate
PARALLEL = int(os.environ.get("BENCH_PARALLEL", "1"))
HARDEST = os.environ.get("BENCH_HARDEST", "0") == "1"
HLE_JUDGE_MODEL = os.environ.get("HLE_JUDGE_MODEL", "")
HLE_TEXT_ONLY = os.environ.get("HLE_TEXT_ONLY", "1") == "1"
HLE_PARQUET = os.environ.get("HLE_PARQUET", os.path.join(
    HERE, "data", "hle", "data", "test-00000-of-00001.parquet"))
HLE_MCQ_ONLY = os.environ.get("HLE_MCQ_ONLY", "1") == "1"


def log(msg):
    print(msg, flush=True)


def _parallel_map(fn, items):
    if PARALLEL <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        return list(ex.map(fn, items))


def _status_counts(records):
    c = {}
    for r in records:
        c[r.get("status")] = c.get(r.get("status"), 0) + 1
    return c


def _classify_status(parseable, status):
    if parseable:
        return "ok"
    if status.get("infra_exhausted") or status.get("infra_fail"):
        return "infra"
    if status.get("error"):
        return "error"
    if status.get("finish_reason") == "length" or status.get("truncated_calls", 0) > 0:
        return "truncated"
    return "empty_stop"


# --------------------------------------------------------------------------
# LiveCodeBench (code generation)
# --------------------------------------------------------------------------

_FMT_STARTER = ("You will use the following starter code to write the solution to the "
                "problem and enclose your code within delimiters.")
_FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout "
              "(do not directly test on the sample inputs). Enclose your code within "
              "delimiters as follows. Ensure that when the python program runs, it reads "
              "the inputs, runs the algorithm and writes output to STDOUT.")


def build_code_prompt(problem):
    p = f"### Question\n{problem.question_content}\n\n"
    if problem.starter_code:
        p += f"### Format: {_FMT_STARTER}\n"
        p += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        p += f"### Format: {_FMT_STDIN}\n\n"
    return p


def run_lcb(n, ids_file):
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.lm_styles import LMStyle

    dataset = load_code_generation_dataset(release_version=os.environ.get("LCB_RELEASE", "release_v1"))
    by_id = {p.question_id: p for p in dataset}

    if HARDEST:
        hard = [p for p in dataset if p.difficulty.value == "hard"]
        after = os.environ.get("LCB_AFTER")
        if after:
            from datetime import datetime
            cut = datetime.strptime(after, "%Y-%m-%d")
            hard = [p for p in hard if p.contest_date >= cut]
        picked = sorted(hard, key=lambda p: p.contest_date, reverse=True)[:n]
        span = f"{picked[-1].contest_date.date()}..{picked[0].contest_date.date()}" if picked else "-"
        log(f"\n=== LiveCodeBench (latest {len(picked)} hard{', after ' + after if after else ''}, dates {span}) ===")
    else:
        with open(ids_file) as f:
            wanted = json.load(f)
        picked = [by_id[i] for i in wanted if i in by_id][:n]
        log(f"\n=== LiveCodeBench: {len(picked)} problems ===")

    def _solve_lcb(prob):
        plog = lambda m: log(f"[{prob.question_id}] {m}")
        plog(f"start ({prob.difficulty.value})")
        status = {}
        try:
            raw = SOLVE(build_code_prompt(prob), CODE_SPEC, log=plog, status_out=status,
                        tests=prob.public_test_cases)
        except Exception as e:
            plog(f"ERROR: {e}")
            status["error"] = str(e)
            raw = ""
        code = extract_code(raw, LMStyle.ClaudeCode)
        status["class"] = _classify_status(bool(code.strip()), status)
        plog(f"done ({status['class']})")
        return code, status

    solved = _parallel_map(_solve_lcb, picked)
    codes = [c for c, _ in solved]
    samples = [p.get_evaluation_sample() for p in picked]
    generations = [[c] for c in codes]
    records = [{"question_id": p.question_id, "code": c, "status": s.get("class"),
                "finish_reason": s.get("finish_reason"), "completion_tokens": s.get("completion_tokens"),
                "truncated_calls": s.get("truncated_calls"), "n_calls": s.get("n_calls"),
                "ws": s.get("ws")}
               for p, (c, s) in zip(picked, solved)]

    metrics, results, _ = codegen_metrics(samples, generations, k_list=[1], num_process_evaluate=8)
    import numpy as np

    def _passed(res0):
        if isinstance(res0, (list, tuple)):
            return bool(np.all(np.array(res0) > 0)) and len(res0) > 0
        return bool(res0)

    passed = [_passed(results[idx][0]) for idx in range(len(picked))]
    for rec, p in zip(records, passed):
        rec["passed"] = bool(p)
        log(f"  {rec['question_id']}: {'PASS' if p else 'FAIL'}")
    graded = [r for r in records if r.get("status") != "infra"]
    n_infra = len(records) - len(graded)
    npass = sum(r["passed"] for r in graded)
    passk = 100.0 * npass / max(1, len(graded))
    log(f"LiveCodeBench pass@1 = {passk:.1f}%  ({npass}/{len(graded)}"
        + (f"; {n_infra} infra-excluded of {len(records)}" if n_infra else "") + ")")
    log(f"  status breakdown: {_status_counts(records)}")
    return {"benchmark": "livecodebench", "pass@1": passk, "records": records,
            "n_graded": len(graded), "n_infra": n_infra}


# --------------------------------------------------------------------------
# AIME (integer-answer math)
# --------------------------------------------------------------------------
def extract_answer_int(text):
    if not text:
        return None
    m = re.findall(r"ANSWER:\s*(-?\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m[-1])
    # fallback: last integer in \boxed{} or last standalone integer
    m = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", text)
    if m:
        return int(m[-1])
    m = re.findall(r"(-?\d+)", text)
    return int(m[-1]) if m else None


def run_aime(n):
    from datasets import load_dataset

    ds = list(load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")) \
        + list(load_dataset("opencompass/AIME2025", "AIME2025-II", split="test"))
    if HARDEST:
        idx = [10, 11, 12, 13, 14, 25, 26, 27, 28, 29]
        picked = [ds[i] for i in idx if i < len(ds)][:n]
    else:
        picked = ds[:n]
    log(f"\n=== AIME 2025-I: {len(picked)} problems ===")

    def _solve_aime(item):
        i, ex = item
        gm = re.search(r"-?\d+", str(ex["answer"]))
        gold = int(gm.group()) if gm else -1
        plog = lambda m: log(f"[AIME {i + 1}] {m}")
        plog(f"start gold={gold}")
        prompt = f"Solve the following AIME problem.\n\n{ex['question']}"
        status = {}
        try:
            raw = SOLVE(prompt, MATH_SPEC, log=plog, status_out=status)
        except Exception as e:
            plog(f"ERROR: {e}")
            status["error"] = str(e)
            raw = ""
        pred = extract_answer_int(raw)
        ok = pred is not None and pred == gold
        status["class"] = _classify_status(pred is not None, status)
        plog(f"pred={pred}  {'PASS' if ok else 'FAIL'} ({status['class']})")
        return {"idx": i, "gold": gold, "pred": pred, "passed": ok,
                "status": status.get("class"), "finish_reason": status.get("finish_reason")}

    records = _parallel_map(_solve_aime, list(enumerate(picked)))
    acc = 100.0 * sum(r["passed"] for r in records) / max(1, len(records))
    log(f"AIME pass@1 = {acc:.1f}%  ({sum(r['passed'] for r in records)}/{len(records)})")
    log(f"  status breakdown: {_status_counts(records)}")
    return {"benchmark": "aime2025", "pass@1": acc, "records": records}


# --------------------------------------------------------------------------
# MATH-500 and GPQA (free-form string answers)
# --------------------------------------------------------------------------
def _strip_boxed(s):
    s = str(s)
    key = "\\boxed{"
    i = s.rfind(key)
    if i < 0:
        return s.strip()
    j = i + len(key)
    depth, start = 1, j
    while j < len(s) and depth:
        depth += (s[j] == "{") - (s[j] == "}")
        j += 1
    return s[start:j - 1].strip()


def _norm(s):
    s = _strip_boxed(s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("\\dfrac", "\\frac"), ("$", ""), ("\\text", ""), ("{", ""), ("}", ""),
                 ("\\ ", ""), (" ", "")]:
        s = s.replace(a, b)
    return s.strip().rstrip(".").lower()


def extract_answer_text(text):
    if not text:
        return ""
    m = re.findall(r"ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()
    if "\\boxed{" in text:
        return _strip_boxed(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _answer_match(pred, gold):
    p, g = _norm(pred), _norm(gold)
    if not p:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        pass
    return g in p or p in g


def _run_qa(name, spec, picked, gold_of, prompt_of):
    def _solve(item):
        i, ex = item
        gold = gold_of(ex)
        plog = lambda m: log(f"[{name} {i + 1}] {m}")
        plog(f"start gold={gold!r}")
        status = {}
        try:
            raw = SOLVE(prompt_of(ex), spec, log=plog, status_out=status)
        except Exception as e:
            plog(f"ERROR: {e}")
            status["error"] = str(e)
            raw = ""
        pred = extract_answer_text(raw)
        ok = _answer_match(pred, gold)
        status["class"] = _classify_status(bool(pred.strip()), status)
        plog(f"pred={pred!r}  {'PASS' if ok else 'FAIL'} ({status['class']})")
        return {"idx": i, "gold": str(gold), "pred": pred, "passed": ok,
                "status": status.get("class"), "finish_reason": status.get("finish_reason")}

    records = _parallel_map(_solve, list(enumerate(picked)))
    acc = 100.0 * sum(r["passed"] for r in records) / max(1, len(records))
    log(f"{name} pass@1 = {acc:.1f}%  ({sum(r['passed'] for r in records)}/{len(records)})")
    log(f"  status breakdown: {_status_counts(records)}")
    return {"benchmark": name.lower(), "pass@1": acc, "records": records}


def run_math500(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if HARDEST:  # MATH level 5 = hardest
        picked = [x for x in ds if str(x["level"]) == "5"][:n]
    else:
        picked = [ds[i] for i in range(min(n, len(ds)))]
    log(f"\n=== MATH-500{' (level 5)' if HARDEST else ''}: {len(picked)} problems ===")
    return _run_qa("MATH500", MATH500_SPEC, picked,
                   gold_of=lambda ex: ex["answer"],
                   prompt_of=lambda ex: f"Solve the following problem.\n\n{ex['problem']}")


def run_gpqa(n):
    import csv
    import random
    path = os.environ.get("GPQA_CSV", os.path.join(HERE, "data", "gpqa_diamond.csv"))
    rows = list(csv.DictReader(open(path)))
    if HARDEST: 
        def _hard(r):
            t = (r.get("Writer's Difficulty Estimate", "") or "").lower()
            return next((s for s, k in [(4, "post-grad"), (4, "graduate"), (3, "hard undergrad"),
                                        (2, "easy undergrad"), (1, "high school")] if k in t), 0)
        rows = sorted(rows, key=_hard, reverse=True)
    rows = rows[:n]
    log(f"\n=== GPQA-diamond (4-way MCQ{', hardest' if HARDEST else ''}): {len(rows)} problems ===")

    rng = random.Random(42)
    items = []
    for row in rows:
        opts = [row["Correct Answer"].strip(), row["Incorrect Answer 1"].strip(),
                row["Incorrect Answer 2"].strip(), row["Incorrect Answer 3"].strip()]
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        letters = "ABCD"
        gold_letter = letters[order.index(0)]  # where the correct option landed
        body = "\n".join(f"{letters[i]}) {opts[order[i]]}" for i in range(4))
        prompt = f"{row['Question'].strip()}\n\n{body}"
        items.append({"prompt": prompt, "gold": gold_letter})

    def _solve(it):
        i, ex = it
        plog = lambda m: log(f"[GPQA {i + 1}] {m}")
        plog(f"start gold={ex['gold']}")
        status = {}
        try:
            raw = SOLVE(ex["prompt"], GPQA_SPEC, log=plog, status_out=status)
        except Exception as e:  # noqa: BLE001
            plog(f"ERROR: {e}")
            status["error"] = str(e)
            raw = ""
        m = re.findall(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?", raw)
        pred = m[-1].upper() if m else ""
        ok = pred == ex["gold"]
        status["class"] = _classify_status(bool(pred), status)
        plog(f"pred={pred!r}  {'PASS' if ok else 'FAIL'} ({status['class']})")
        return {"idx": i, "gold": ex["gold"], "pred": pred, "passed": ok,
                "status": status.get("class"), "finish_reason": status.get("finish_reason")}

    records = _parallel_map(_solve, list(enumerate(items)))
    acc = 100.0 * sum(r["passed"] for r in records) / max(1, len(records))
    log(f"GPQA pass@1 = {acc:.1f}%  ({sum(r['passed'] for r in records)}/{len(records)})")
    return {"benchmark": "gpqa", "pass@1": acc, "records": records}


# --------------------------------------------------------------------------
# Humanity's Last Exam (expert-level free-form + MCQ, LLM-judged)
# --------------------------------------------------------------------------
_HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not \
based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Focus ONLY on whether the answer extracted from [response] matches [correct_answer]. Do not \
attempt to solve the problem, do not argue for a different answer, and do not comment on \
background. Answer 'yes' only if they match, or are within a small margin of error for \
numerical problems; answer 'no' if there is any inconsistency, ambiguity, or non-equivalency.

Reply with exactly these four lines and nothing else:
extracted_final_answer: <the exact final answer from [response], or None>
reasoning: <one or two sentences>
correct: <yes|no>
confidence: <the confidence score 0-100 stated in [response], or 100 if absent>"""


def _judge_model():
    return HLE_JUDGE_MODEL or os.environ.get("MULTIAGENT_MODEL") or LADDER[-1]


def _hle_judge(question, gold, response, plog):
    from orchestrator import chat
    if not response.strip():
        return False, "", 0
    try:
        raw = chat(_judge_model(), [{"role": "user", "content": _HLE_JUDGE_PROMPT.format(
            question=question, response=response[:20000], correct_answer=gold)}], temperature=0.0)
    except Exception as e:
        plog(f"JUDGE ERROR: {e}")
        return False, "", 0
    ok = re.findall(r"^\s*correct\s*:\s*(yes|no)", raw, re.I | re.M)
    if not ok:
        plog(f"JUDGE UNPARSEABLE: {raw[:200]!r}")
    ext = re.findall(r"^\s*extracted_final_answer\s*:\s*(.+)", raw, re.I | re.M)
    conf = re.findall(r"^\s*confidence\s*:\s*(\d+)", raw, re.I | re.M)
    return (bool(ok) and ok[-1].lower() == "yes",
            ext[-1].strip() if ext else "",
            int(conf[-1]) if conf else 0)


_MCQ_RE = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-Z])\$?\b")
_CONF_RE = re.compile(r"(?im)^\s*CONFIDENCE[ \t]*:[ \t]*(\d+)")


def _hle_grade_mcq(response, gold):
    m = _MCQ_RE.findall(response or "")
    pred = m[-1].upper() if m else ""
    conf = _CONF_RE.findall(response or "")
    return bool(pred) and pred == gold.strip().upper(), pred, int(conf[-1]) if conf else 0


def run_hle(n):
    import random
    from datasets import load_dataset

    if os.path.exists(HLE_PARQUET):
        ds = load_dataset("parquet", data_files=HLE_PARQUET, split="train")
    else:
        ds = load_dataset("cais/hle", split="test")
    ds = ds.remove_columns([c for c in ("image_preview", "rationale_image")
                            if c in ds.column_names])
    n_total = len(ds)
    if HLE_TEXT_ONLY:
        ds = ds.filter(lambda im: not im, input_columns="image")
    ds = ds.remove_columns(["image"])
    n_text = len(ds)
    if HLE_MCQ_ONLY:
        ds = ds.filter(lambda t: t == "multipleChoice", input_columns="answer_type")

    idx = list(range(len(ds)))
    if 0 < n < len(idx):
        idx = sorted(random.Random(42).sample(idx, n))
    picked = [ds[i] for i in idx]
    tag = "HLE" + (" text-only" if HLE_TEXT_ONLY else "") + (" MCQ" if HLE_MCQ_ONLY else "")
    log(f"\n=== {tag}: {len(picked)} of {len(ds)} eligible "
        f"({n_total - n_text} multimodal, {n_text - len(ds)} non-MCQ excluded of {n_total}) ===")
    n_judged = sum(1 for p in picked if p.get("answer_type") != "multipleChoice")
    log(f"    grading: {len(picked) - n_judged} deterministic (MCQ letter match)"
        + (f" + {n_judged} LLM-judged by {_judge_model()}" if n_judged else ", no LLM judge"))

    def _solve(item):
        i, ex = item
        gold = str(ex["answer"])
        plog = lambda m: log(f"[HLE {i + 1}] {m}")
        plog(f"start {ex.get('category', '?')} / {ex.get('answer_type', '?')}")
        try:
            raw = SOLVE(ex["question"], HLE_SPEC, log=plog)
        except Exception as e:
            plog(f"ERROR: {e}")
            raw = ""
        if ex.get("answer_type") == "multipleChoice":
            ok, extracted, conf = _hle_grade_mcq(raw, gold)
        else:
            ok, extracted, conf = _hle_judge(ex["question"], gold, raw, plog)
        plog(f"pred={extracted!r} conf={conf}  {'PASS' if ok else 'FAIL'}"
             + ("  (no answer letter found)" if not extracted else ""))
        return {"idx": i, "id": ex.get("id", ""), "category": ex.get("category", ""),
                "answer_type": ex.get("answer_type", ""), "gold": gold,
                "pred": extracted, "confidence": conf, "passed": ok}

    records = _parallel_map(_solve, list(enumerate(picked)))
    acc = 100.0 * sum(r["passed"] for r in records) / max(1, len(records))
    unparsed = sum(1 for r in records if not r["pred"])
    log(f"HLE pass@1 = {acc:.1f}%  ({sum(r['passed'] for r in records)}/{len(records)})")
    if unparsed: 
        log(f"    WARNING: {unparsed}/{len(records)} responses had no extractable answer")
    name = "hle" + ("_text_only" if HLE_TEXT_ONLY else "") + ("_mcq" if HLE_MCQ_ONLY else "")
    return {"benchmark": name, "pass@1": acc, "unparsed": unparsed,
            "judge": _judge_model() if n_judged else "deterministic", "records": records}


def main():
    global SOLVE, PARALLEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcb", type=int, default=10)
    ap.add_argument("--aime", type=int, default=10)
    ap.add_argument("--math500", type=int, default=100)
    ap.add_argument("--gpqa", type=int, default=198)
    ap.add_argument("--hle", type=int, default=200, help="0 = all eligible questions")
    ap.add_argument("--ids-file", default=os.path.join(HERE, "lcb100_hardest_v6.json"))
    ap.add_argument("--out", default=os.path.join(RUNS, "results.json"))
    ap.add_argument("--only", choices=["lcb", "aime", "math500", "gpqa", "hle"], default=None)
    ap.add_argument("--engine", choices=["escalate", "multiagent", "single"], default="escalate")
    ap.add_argument("--parallel", type=int, default=PARALLEL, help="# problems solved concurrently")
    args = ap.parse_args()
    PARALLEL = args.parallel
    if args.engine == "multiagent":
        from multiagent import multiagent_solve, MODEL, MAX_ITERS
        SOLVE = multiagent_solve
        log(f"Engine: multiagent  model={MODEL}  max_iters={MAX_ITERS}")
        out = {"engine": "multiagent", "model": MODEL}
    elif args.engine == "single":
        from multiagent import single_solve, MODEL
        SOLVE = single_solve
        log(f"Engine: single-shot  model={MODEL}")
        out = {"engine": "single", "model": MODEL}
    else:
        log(f"Engine: escalate  ladder={LADDER}")
        out = {"engine": "escalate", "ladder": LADDER}
    log(f"Parallel: {PARALLEL} problems at once")
    run_map = {
        "lcb": lambda: run_lcb(args.lcb, args.ids_file),
        "aime": lambda: run_aime(args.aime),
        "math500": lambda: run_math500(args.math500),
        "gpqa": lambda: run_gpqa(args.gpqa),
        "hle": lambda: run_hle(args.hle),
    }
    try:
        if args.only:
            out[args.only] = run_map[args.only]()
        else: 
            if args.lcb > 0:
                out["lcb"] = run_map["lcb"]()
            if args.aime > 0:
                out["aime"] = run_map["aime"]()
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        log(f"\nWrote {args.out}")
    log("\n==== SUMMARY ====")
    for key in ("lcb", "aime", "math500", "gpqa", "hle"):
        if key in out:
            log(f"{out[key]['benchmark']:>16}: pass@1 = {out[key]['pass@1']:.1f}%")


if __name__ == "__main__":
    main()
