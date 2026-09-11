"""
Multi-agent collaborative solver: a primary manager plus worker subagents, coordinating over a shared workspace. 

Every role is the same model invoked in a fresh context.
"""

import os
import re
import sys
import time
import json
import hashlib
import subprocess

from orchestrator import chat

MODEL = os.environ.get("MULTIAGENT_MODEL", "groq:qwen/qwen3.6-27b")
MAX_ITERS = int(os.environ.get("MULTIAGENT_MAX_ITERS", "10"))
STRICT_FORMAT = os.environ.get("MULTIAGENT_STRICT_FORMAT", "auto")


def _strict():
    return STRICT_FORMAT == "1" or (
        STRICT_FORMAT == "auto" and any(s in MODEL.lower() for s in ("muse", "glimmer")))


def _format_rule(first, *rest):
    others = "".join(f", then the literal line '### {r}'" for r in rest)
    return ("\n\nTHE FORMAT IS MANDATORY AND YOUR REPLY IS PARSED BY A PROGRAM. "
            f"Begin your reply with the literal line '### {first}'{others}. "
            "Write nothing before the first header and nothing outside these sections. "
            "Do NOT solve the problem and do NOT write code -- a later worker does that. "
            "A reply without these exact header lines is discarded.")
MAX_TASKS = int(os.environ.get("MULTIAGENT_MAX_TASKS", "12"))  # cap on live task list
WS_ROOT = os.environ.get("MULTIAGENT_WS", os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "runs", "ws")))


# --- workspace helpers -----------------------------------------------------

def _slug(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _read(ws, name):
    p = os.path.join(ws, name)
    return open(p).read() if os.path.exists(p) else ""


def _write(ws, name, content):
    with open(os.path.join(ws, name), "w") as f:
        f.write(content)


def _append(ws, name, content):
    with open(os.path.join(ws, name), "a") as f:
        f.write(content)


# --- transcript logging ----------------------------------------------------
# every agent call for a problem is appended to <ws>/transcript.jsonl
def _record(ws, rec):
    with open(os.path.join(ws, "transcript.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _chat(ws, role, messages, temperature, meta=None):
    m = meta if meta is not None else {}
    resp = chat(MODEL, messages, temperature=temperature, meta=m)
    _record(ws, {"t": time.time(), "role": role, "request": messages, "response": resp,
                 "reasoning": m.get("reasoning"), "thinking_blocks": m.get("thinking_blocks"),
                 # Anthropic returns a SUMMARY of the thinking; vLLM/DashScope return the real
                 # chain of thought in the same `reasoning` field. Without this flag the two are
                 # indistinguishable on disk, and any later analysis of thinking length or
                 # content would silently compare a summary against a transcript.
                 "reasoning_is_summary": m.get("reasoning_is_summary"),
                 "finish_reason": m.get("finish_reason"), "completion_tokens": m.get("completion_tokens"),
                 "prompt_tokens": m.get("prompt_tokens"), "provider": m.get("provider"),
                 "attempts": m.get("attempts"), "discarded": m.get("discarded"),
                 "infra_exhausted": m.get("infra_exhausted")})
    return resp


# --- parsing helpers -------------------------------------------------------

def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code(text):
    return re.sub(r"```.*?```", "[code omitted -- approach only]", text, flags=re.DOTALL).strip()


def _sections(text):
    out, cur, buf = {}, None, []
    for line in _strip_think(text).splitlines():
        s = line.strip()
        m = (re.match(r"^#{1,3}\s*([A-Za-z_]+)\s*$", s)
             or re.match(r"^\*\*([A-Za-z_]+)\*\*\s*:?\s*$", s)
             or re.match(r"^([A-Z_]{3,})\s*:\s*$", s))
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).upper(), []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def _bullets(text):
    items = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)", line)
        if m:
            items.append(m.group(1).strip())
    return items


def _extract_py(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", _strip_think(text), re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_tasks(text):
    out = []
    for b in _bullets(text):
        m = re.match(r"\[\s*([a-z_ ]+?)\s*\]\s*(.*)", b, re.I)
        if m:
            raw = m.group(1).lower().replace(" ", "").replace("_", "")
            desc = m.group(2).strip()
            status = "done" if raw == "done" else ("in_progress" if raw in ("wip", "inprogress") else "pending")
        else:
            desc, status = b, "pending"
        if desc:
            out.append({"id": len(out) + 1, "desc": desc, "status": status, "result": ""})
    return out[:MAX_TASKS]


ANS_RE = re.compile(r"ANSWER:\s*\S", re.I)
MAX_ANSWER_CHARS = 20000
MAX_PLAN_CHARS = 4000


def _has_answer(ws, spec):
    """True if the workspace already holds a usable final artifact."""
    if spec["kind"] == "code":
        return bool(_read(ws, "solution.py").strip())
    return bool(ANS_RE.search(_read(ws, "answer.md")))


# --- roles -----------------------------------------------------------------

def _save_tasks(ws, tasks):
    _write(ws, "tasks.json", json.dumps(tasks, indent=2))


def _add_tasks(tasks, descs):
    have = {t["desc"].lower() for t in tasks}
    nid = max([t["id"] for t in tasks], default=0)
    for d in descs:
        d = d.strip()
        if d and d.lower() not in have and len(tasks) < MAX_TASKS:
            nid += 1
            tasks.append({"id": nid, "desc": d, "status": "pending", "result": ""})
            have.add(d.lower())


def _primary_plan(problem, spec, ws, log):
    kind = spec["kind"]
    sys = (
        "You are the PRIMARY orchestrator (manager) of a small team of workers, all "
        f"expert at {'competitive programming' if kind == 'code' else 'olympiad mathematics'}. "
        "Given a problem, produce a short overarching plan to solve it, then a task list "
        "the workers can pick up. Respond with EXACTLY these sections:\n"
        "### PLAN\n<3-6 sentence strategy>\n"
        "### TASKS\n<3-6 bullet tasks, each a concrete unit of work>"
    )
    if _strict():
        sys += _format_rule("PLAN", "TASKS")
    reply = _chat(ws, "primary_plan", [
        {"role": "system", "content": sys},
        {"role": "user", "content": problem},
    ], temperature=0.3)
    sec = _sections(reply)
    _write(ws, "plan.md", (sec.get("PLAN") or reply.strip())[:MAX_PLAN_CHARS])
    tasks = []
    _add_tasks(tasks, _bullets(sec.get("TASKS", "")))
    log(f"    [primary] plan + {len(tasks)} initial tasks")
    return tasks


def _ideation_worker(problem, spec, ws, log):
    sys = (
        "You are the FIRST WORKER. Do NOT solve the problem and do NOT write any code. "
        "Just think about it: identify the core difficulty, then list SEVERAL DISTINCT "
        "candidate approaches (genuinely different algorithms / data structures / problem "
        "reductions, not variations of one idea), and note pitfalls for each. Describe each "
        "approach in prose only -- absolutely no code blocks; a later worker will implement. "
        "Respond with EXACTLY:\n### NOTES\n<your analysis>\n"
        "### NEXT\n<bullet list of distinct approaches to try next>"
    )
    if _strict():
        sys += _format_rule("NOTES", "NEXT")
    reply = _chat(ws, "ideation", [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"PROBLEM:\n{problem}\n\nPLAN:\n{_read(ws, 'plan.md')}"},
    ], temperature=0.4)
    sec = _sections(reply)
    _append(ws, "notes.md",
            f"\n## ideation\n{_strip_code(sec.get('NOTES') or reply.strip())[:MAX_PLAN_CHARS * 2]}\n")
    proposals = [_strip_code(p) for p in _bullets(sec.get("NEXT", ""))]
    log(f"    [worker:ideate] proposed {len(proposals)} approaches")
    return proposals


def _primary_manage(problem, spec, ws, tasks, proposals, last_summary, log):
    kind = spec["kind"]
    cur = _read(ws, "solution.py" if kind == "code" else "answer.md").strip()
    task_lines = "\n".join(
        f"- [{'done' if t['status'] == 'done' else 'todo'}] {t['desc']}" for t in tasks
    ) or "(none)"
    prop_lines = "\n".join(f"- {p}" for p in proposals) or "(none)"
    sys = (
        "You are the PRIMARY orchestrator and manager. You OWN the task list and decide "
        "when the problem is solved. Review the current progress and the latest worker's "
        "result, then:\n"
        "- The LATEST WORKER RESULT may include a SAMPLE TESTS verdict from actually running "
        "the code. Treat it as ground truth: only set STATUS 'done' if the solution PASSED the "
        "sample tests; if it FAILED, you MUST set STATUS 'continue' and choose a task that "
        "fixes the failing case or switches to a different approach.\n"
        "- If the current solution/answer is complete and correct, set STATUS to 'done'.\n"
        "- Otherwise CURATE the task list: merge duplicates, drop finished or irrelevant "
        "items, mark completed ones [done], and fold in ONLY genuinely new sub-tasks from "
        "the proposals. Then choose the single most valuable next task.\n"
        "- IMPORTANT: if the current solution keeps failing, or the last worker made no real "
        "progress, do NOT keep refining the same idea. Switch to a DIFFERENT approach "
        "(a different algorithm / data structure / reduction) from the notes, or ask for a "
        "new one. You have many rounds -- use them to try distinct approaches, not to polish "
        "a stuck one.\n"
        "Respond with EXACTLY these sections:\n"
        "### STATUS\n<done|continue>\n"
        "### NEXT\n<exact text of the ONE task to do next; omit if done>\n"
        "### TASKS\n<curated list, one per line, each '- [done] ...' or '- [todo] ...'>"
    )
    if _strict():
        sys += _format_rule("STATUS", "NEXT", "TASKS")
    reply = _chat(ws, "primary_manage", [
        {"role": "system", "content": sys},
        {"role": "user", "content": (
            f"PROBLEM:\n{problem}\n\nCURRENT {'SOLUTION' if kind == 'code' else 'ANSWER'}:\n"
            f"{cur or '(none yet)'}\n\nNOTES:\n{_read(ws, 'notes.md')}\n\n"
            f"CURRENT TASK LIST:\n{task_lines}\n\nLATEST WORKER RESULT: {last_summary}\n\n"
            f"PROPOSED NEW STEPS:\n{prop_lines}"
        )},
    ], temperature=0.2)
    sec = _sections(reply)
    curated = _parse_tasks(sec.get("TASKS", "")) or tasks
    status = "done" if sec.get("STATUS", "").strip().lower().startswith("done") else "continue"
    if status == "done" and not cur:  # invariant: can't be done with no answer produced yet
        status = "continue"
    nxt = (_bullets(sec.get("NEXT", ""))[:1] or [sec.get("NEXT", "").strip()])[0]
    nxt = re.sub(r"^[-*]\s*", "", nxt).strip()
    if status == "continue" and not nxt:  # manager wants to continue but named no task
        todo = [t for t in curated if t["status"] != "done"]
        nxt = todo[0]["desc"] if todo else ""
    if not cur and not nxt:
        status, nxt = "continue", "Implement the full working solution for the most promising approach in the notes."
    log(f"    [primary-manage] {status}" + (f", next: {nxt[:60]}" if status == "continue" else ""))
    return status, nxt, curated


def _summarize_cutoff(ws, reply, task_desc):
    txt = _strip_think(reply) or reply
    snippet = txt if len(txt) <= 9000 else txt[:3500] + "\n...[middle omitted]...\n" + txt[-5500:]
    sysmsg = (
        "A worker's solution attempt was CUT OFF when it hit the token limit. Summarize its "
        "partial attempt in 3-5 sentences: which approach it was pursuing, what it established "
        "or ruled out, how far it got, and what remained unfinished. Be concrete so another "
        "worker can resume or judge it. Do NOT try to finish the solution yourself."
    )
    s = _chat(ws, "cutoff_summary", [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": f"TASK: {task_desc}\n\nCUT-OFF ATTEMPT:\n{snippet}"},
    ], temperature=0.2)
    return _strip_think(s).strip() or "(the cut-off attempt could not be summarized)"


def _worker(problem, spec, ws, task, log, finalize=False):
    kind = spec["kind"]
    cur = _read(ws, "solution.py" if kind == "code" else "answer.md")
    notes_fmt = (
        "### NOTES\n<the COMPLETE notes file, rewritten. You are shown the current NOTES "
        "above: fold your findings into them, keep what still matters, and DELETE anything "
        "superseded, disproven, or now obvious. This REPLACES the file, so whatever you omit "
        "is gone. Organise it as '- **Topic:** ...' bullets, under ~800 words. Do NOT use "
        "markdown headings (#, ##), bold-only lines, or ALLCAPS: lines anywhere inside this "
        "section -- the reply is split on those, so they would truncate your notes.>\n"
    )
    if kind == "code":
        out_fmt = (
            "### CODE\n```python\n<the FULL updated self-contained program>\n```\n"
            + notes_fmt +
            "### NEXT\n<bullet list of remaining steps, or 'none'>\n### STATUS\n<solved|continue>"
        )
    else:
        out_fmt = (
            "### ANSWER\n<full worked solution, ending with the exact final "
            "ANSWER: line your instructions require>\n"
            + notes_fmt +
            "### NEXT\n<bullet list of remaining checks, or 'none'>\n### STATUS\n<solved|continue>"
        )
    goal = (
        "Produce the DEFINITIVE final solution now, using all notes and current work."
        if finalize else f"Complete this task: {task['desc']}"
    )
    sys = (
        f"You are a WORKER subagent, {spec['solver_system']} "
        "You share a workspace with the team. Build on the current work and notes where "
        "useful -- but if your task is to try a different approach, write a FRESH solution "
        "for that approach instead of patching the stuck one. "
        f"Respond with EXACTLY these sections:\n{out_fmt}"
    )
    wmeta = {}
    reply = _chat(ws, "finalize" if finalize else f"worker:{task['id']}", [
        {"role": "system", "content": sys},
        {"role": "user", "content": (
            f"PROBLEM:\n{problem}\n\nPLAN:\n{_read(ws, 'plan.md')}\n\n"
            f"NOTES:\n{_read(ws, 'notes.md')}\n\nCURRENT WORK:\n{cur or '(none yet)'}\n\n"
            f"YOUR TASK: {goal}"
        )},
    ], temperature=0.2, meta=wmeta)
    sec = _sections(reply)
    wrote = False
    if kind == "code":
        code = _extract_py(sec.get("CODE", "")) or _extract_py(reply)
        if code:
            _write(ws, "solution.py", code)
            wrote = True
    else:
        ans = sec.get("ANSWER", "").strip()
        if not ans and ANS_RE.search(reply) and len(reply) < MAX_ANSWER_CHARS:
            ans = _strip_think(reply).strip()
        if ans and (ANS_RE.search(ans) or not _read(ws, "answer.md").strip()):
            _write(ws, "answer.md", ans)
            wrote = True
    if sec.get("NOTES"):
        if wmeta.get("finish_reason") != "length":
            _write(ws, "notes.md", sec["NOTES"].strip()[:MAX_PLAN_CHARS * 2] + "\n")
    nexts = [b for b in _bullets(sec.get("NEXT", "")) if b.lower() != "none"]
    status = sec.get("STATUS", "continue").strip().lower()
    if wmeta.get("finish_reason") == "length" and not finalize:
        digest = _summarize_cutoff(ws, reply, task["desc"])
        summary = (f"worker EXCEEDED THE TOKEN LIMIT and was cut off before finishing task: "
                   f"{task['desc'][:80]}. No complete solution was produced -- the approach was "
                   f"too long to finish in one call, so prefer a simpler or different approach "
                   f"next. Summary of its partial thinking: {digest[:500]}")
        log(f"    [worker] task {task['id']} -> CUT OFF at token limit; summarized thinking, told manager")
        return "continue", nexts, summary, wrote
    summary = f"worker reported '{status}' on: {task['desc'][:80]}. " + (sec.get("NOTES", "")[:200])
    log(f"    [worker] task {task['id']} -> {status}, +{len(nexts)} next steps")
    return status, nexts, summary, wrote


# --- sample-test execution (feedback so the manager knows if the code actually works) -----

def _run_samples(ws, tests):
    code = _read(ws, "solution.py")
    stdin_tests = [t for t in (tests or [])
                   if "stdin" in str(getattr(t, "testtype", "")).lower()]
    if not code.strip() or not stdin_tests:
        return {"ran": False}
    sol = os.path.join(ws, "solution.py")
    passed, fail = 0, None
    for t in stdin_tests:
        inp = getattr(t, "input", "") or ""
        exp = (getattr(t, "output", "") or "").strip()
        try:

            r = subprocess.run([sys.executable, sol], input=inp, capture_output=True,
                               text=True, timeout=10)
            got = (r.stdout or "").strip()
            if r.returncode != 0 and not got:
                got = f"<runtime error: {(r.stderr or '')[:200]}>"
        except subprocess.TimeoutExpired:
            got = "<timed out (>10s)>"
        except Exception as e:
            got = f"<error: {e}>"
        if got == exp:
            passed += 1
        elif fail is None:
            fail = {"input": inp[:600], "expected": exp[:400], "got": got[:400]}
    return {"ran": True, "passed": passed, "total": len(stdin_tests), "fail": fail}


def _sample_feedback(res):
    if not res or not res.get("ran"):
        return ""
    if res["passed"] == res["total"]:
        return f"[SAMPLE TESTS: PASSED all {res['total']} public samples -- the solution looks correct.] "
    f = res.get("fail") or {}
    return ("[SAMPLE TESTS: FAILED -- passed {p}/{t}. The current solution is WRONG. First failing "
            "case: input={i!r} expected={e!r} got={g!r}. Fix the bug or, if this approach keeps "
            "failing, switch to a DIFFERENT approach.] ").format(
                p=res["passed"], t=res["total"], i=f.get("input", ""),
                e=f.get("expected", ""), g=f.get("got", ""))


# --- public entry point ----------------------------------------------------

def multiagent_solve(problem_text, spec, log=None, status_out=None, tests=None):
    log = log or (lambda *a, **k: None)
    ws = os.path.join(WS_ROOT, _slug(problem_text))
    os.makedirs(ws, exist_ok=True)
    for f in ("task.md", "notes.md", "transcript.jsonl", "plan.md",
              "solution.py", "answer.md", "tasks.json"):
        _write(ws, f, "")
    _write(ws, "task.md", problem_text)
    _record(ws, {"_meta": True, "t": time.time(), "model": MODEL,
                 "kind": spec["kind"], "max_iters": MAX_ITERS, "problem": problem_text})

    tasks = _primary_plan(problem_text, spec, ws, log)
    proposals = _ideation_worker(problem_text, spec, ws, log)
    status, next_desc, tasks = _primary_manage(
        problem_text, spec, ws, tasks, proposals, "ideation complete", log)
    _save_tasks(ws, tasks)

    iters, prev_desc = 0, None
    while status == "continue" and next_desc and iters < MAX_ITERS:
        if prev_desc is not None and next_desc.strip().lower() == prev_desc.strip().lower():
            log(f"    [primary] reissued the same task; no progress, stopping after {iters} iters")
            break
        prev_desc = next_desc
        iters += 1
        task = {"id": iters, "desc": next_desc, "status": "in_progress", "result": ""}
        _, nexts, summary, wrote = _worker(problem_text, spec, ws, task, log)
        res = None
        if spec["kind"] == "code" and tests and wrote:
            res = _run_samples(ws, tests)
            if res.get("ran"):
                summary = _sample_feedback(res) + summary
                log(f"    [samples] {res['passed']}/{res['total']} public tests passed")
        status, next_desc, tasks = _primary_manage(
            problem_text, spec, ws, tasks, nexts, summary, log)
        if res and res.get("ran") and res["passed"] < res["total"] and status == "done":
            status = "continue"
            if not next_desc:
                next_desc = "The solution fails the public sample tests; fix it or try a different approach."
            log("    [primary-manage] overriding 'done' -- sample tests still failing")
        _save_tasks(ws, tasks)

    if status == "done" and _has_answer(ws, spec):
        log("    [finalize] skipped (primary marked done)")
    else:
        _worker(problem_text, spec, ws, {"id": 0, "desc": "finalize"}, log, finalize=True)

    if status_out is not None:
        status_out["ws"] = ws
        recs = []
        for ln in open(os.path.join(ws, "transcript.jsonl")):
            r = json.loads(ln)
            if not r.get("_meta"):
                recs.append(r)
        status_out["finish_reason"] = recs[-1].get("finish_reason") if recs else None
        status_out["truncated_calls"] = sum(1 for r in recs if r.get("finish_reason") == "length")
        status_out["n_calls"] = len(recs)
        final_empty = (not _read(ws, "solution.py").strip() if spec["kind"] == "code"
                       else not ANS_RE.search(_read(ws, "answer.md")))
        if final_empty and any(r.get("infra_exhausted") for r in recs):
            status_out["infra_fail"] = True

    if spec["kind"] == "code":
        code = _read(ws, "solution.py")
        return f"```python\n{code}\n```" if code else ""
    return _read(ws, "answer.md")


def single_solve(problem_text, spec, log=None, status_out=None, tests=None):
    log = log or (lambda *a, **k: None)
    ws = os.path.join(WS_ROOT, _slug(problem_text))
    os.makedirs(ws, exist_ok=True)
    _write(ws, "task.md", problem_text)
    _write(ws, "transcript.jsonl", "")
    _record(ws, {"_meta": True, "t": time.time(), "model": MODEL,
                 "kind": spec["kind"], "engine": "single", "problem": problem_text})
    meta = {}
    answer = _chat(ws, "single", [
        {"role": "system", "content": spec["solver_system"]},
        {"role": "user", "content": problem_text},
    ], temperature=0.2, meta=meta)
    if status_out is not None:
        status_out.update(meta)
        status_out["ws"] = ws
    log(f"    [single] {len(answer)} chars  finish={meta.get('finish_reason')}")
    return answer
