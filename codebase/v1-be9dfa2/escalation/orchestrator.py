"""Hierarchical multi-model escalation"""

import os
import sys
import json
import time
import threading
import urllib.request
import urllib.error

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_REASONING = os.environ.get("ESCALATION_GROQ_REASONING", "none")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REASONING = os.environ.get("ESCALATION_OR_REASONING", "none")

LADDER = os.environ.get(
    "ESCALATION_LADDER",
    "qwen3.5:2b,qwen3.5:9b,qwen3.5:35b,qwen3.5:122b",
).split(",")

MAX_ROUNDS = int(os.environ.get("ESCALATION_MAX_ROUNDS", "2"))
THINK = os.environ.get("ESCALATION_THINK", "0") == "1"
REQUEST_TIMEOUT = int(os.environ.get("ESCALATION_TIMEOUT", "1200"))
CLOUD_TIMEOUT = int(os.environ.get("ESCALATION_CLOUD_TIMEOUT", "120"))
CLOUD_MAX_TOKENS = int(os.environ.get("ESCALATION_CLOUD_MAX_TOKENS", "8000"))


def ollama_chat(model, messages, temperature=0.2, num_ctx=16384, meta=None):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": THINK,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=data, headers={"Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                out = json.loads(r.read())
            if meta is not None:
                meta["finish_reason"] = out.get("done_reason")
                meta["completion_tokens"] = out.get("eval_count")
            return out["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    return ""


def openai_chat(url, api_key, model, messages, temperature=0.2, extra=None, meta=None):
    body = {"model": model, "messages": messages, "stream": False, "temperature": temperature}
    if CLOUD_MAX_TOKENS > 0:
        body["max_tokens"] = CLOUD_MAX_TOKENS
    if extra:
        body.update(extra)
    data = json.dumps(body).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "model-test/1.0",
    }

    def _do(box):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=CLOUD_TIMEOUT) as r:
                box["out"] = json.loads(r.read())
        except Exception as e:
            box["err"] = e

    for attempt in range(6):
        box = {}
        th = threading.Thread(target=_do, args=(box,), daemon=True)
        th.start()
        th.join(CLOUD_TIMEOUT + 5)
        if th.is_alive():
            if attempt == 5:
                raise TimeoutError(f"cloud call exceeded {CLOUD_TIMEOUT}s on every attempt")
            print(f"    [cloud] hard timeout >{CLOUD_TIMEOUT}s, abandoning provider (attempt {attempt + 1})", file=sys.stderr, flush=True)
            continue
        if "out" in box:
            choice = box["out"]["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason")
            content = msg.get("content") or ""
            if not content and finish == "stop":
                content = msg.get("reasoning") or ""
            if finish and finish != "stop":
                ntok = (box["out"].get("usage") or {}).get("completion_tokens")
                print(f"    [cloud] finish_reason={finish} ({len(content)} chars, "
                      f"{ntok} completion tokens)", file=sys.stderr, flush=True)
                if not content:
                    print("    [cloud] truncated before any content; discarding reasoning-only reply",
                          file=sys.stderr, flush=True)
            if meta is not None:
                meta["finish_reason"] = finish
                meta["completion_tokens"] = (box["out"].get("usage") or {}).get("completion_tokens")
            return content
        e = box.get("err")
        if isinstance(e, urllib.error.HTTPError) and e.code == 429 and attempt < 5:
            ra = e.headers.get("retry-after")
            try:
                wait = float(ra) if ra else min(30.0, 5.0 * (attempt + 1))
            except ValueError:
                wait = min(30.0, 5.0 * (attempt + 1))
            print(f"    [cloud] 429 rate-limited, sleep {wait:.0f}s (attempt {attempt + 1})", file=sys.stderr, flush=True)
            time.sleep(wait + 1)
            continue
        if attempt == 5:
            raise e
        print(f"    [cloud] {type(e).__name__} on attempt {attempt + 1}, retrying", file=sys.stderr, flush=True)
        time.sleep(3 * (attempt + 1))
    return ""


def chat(model, messages, temperature=0.2, num_ctx=16384, meta=None):
    if model.startswith("groq:"):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set but ladder uses a groq: model")
        extra = {"reasoning_effort": GROQ_REASONING} if GROQ_REASONING else None
        return openai_chat(GROQ_URL, GROQ_API_KEY, model[len("groq:"):], messages, temperature, extra, meta)
    if model.startswith("openrouter:"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set but model uses an openrouter: prefix")
        extra = {"provider": {"sort": "throughput"}}
        if OPENROUTER_REASONING == "none":
            extra["reasoning"] = {"enabled": False}
        elif OPENROUTER_REASONING in ("low", "medium", "high"):
            extra["reasoning"] = {"effort": OPENROUTER_REASONING}
        return openai_chat(OPENROUTER_URL, OPENROUTER_API_KEY, model[len("openrouter:"):], messages, temperature, extra, meta)
    return ollama_chat(model, messages, temperature=temperature, num_ctx=num_ctx, meta=meta)


# --- Task specifications ---------------------------------------------------
# A task spec adapts the loop to a domain (code vs math) via prompt fragments.

CODE_SPEC = {
    "kind": "code",
    "solver_system": (
        "You are an elite competitive programmer. Solve the given problem in Python. "
        "Think carefully about algorithmic complexity and edge cases. "
        "Output EXACTLY ONE complete, self-contained Python program inside a single "
        "```python ...``` fenced block, and nothing else after it."
    ),
    "critic_system": (
        "You are a ruthless code reviewer for competitive programming. You are given a "
        "problem statement and a candidate Python solution. Check for: wrong algorithm, "
        "incorrect edge cases, off-by-one errors, wrong time/space complexity for the "
        "constraints, and INPUT/OUTPUT FORMAT mismatches (stdin/stdout vs function). "
        "If and only if the solution is fully correct and complete, reply with exactly "
        "'APPROVED' on the first line. Otherwise reply 'REJECTED' on the first line "
        "followed by a numbered list of concrete, fixable problems."
    ),
}

MATH_SPEC = {
    "kind": "math",
    "solver_system": (
        "You are an elite mathematician solving an AIME problem. The answer is an integer "
        "from 0 to 999. Reason step by step, then on the FINAL line output exactly "
        "'ANSWER: <integer>'."
    ),
    "critic_system": (
        "You are a meticulous math grader. You are given an AIME problem and a candidate "
        "solution. Check the reasoning and the final integer answer for errors. If the "
        "solution is fully correct, reply with exactly 'APPROVED' on the first line. "
        "Otherwise reply 'REJECTED' on the first line followed by the specific errors."
    ),
}


MATH500_SPEC = {
    "kind": "math",
    "solver_system": (
        "You are an elite mathematician. Solve the problem. Reason step by step, then on the FINAL "
        "line output exactly 'ANSWER: <final answer>' with the answer in simplest form (a number or "
        "simplified expression, written as it would appear in a textbook)."
    ),
    "critic_system": (
        "You are a meticulous math grader. Given a problem and a candidate solution, check the "
        "reasoning and final answer. If fully correct, reply exactly 'APPROVED' on the first line; "
        "otherwise reply 'REJECTED' then the specific errors."
    ),
}

GPQA_SPEC = {
    "kind": "math",
    "solver_system": (
        "You are a PhD-level expert in physics, chemistry, and biology answering a multiple-choice "
        "question with options A, B, C, D. Reason carefully, then on the FINAL line output exactly "
        "'ANSWER: <letter>' where <letter> is the single correct option (A, B, C, or D)."
    ),
    "critic_system": (
        "You are a rigorous science reviewer. Given a multiple-choice question and a candidate "
        "answer, verify the reasoning and the chosen letter. If fully correct reply 'APPROVED' on "
        "the first line; otherwise reply 'REJECTED' then the errors."
    ),
}

HLE_SPEC = {
    "kind": "math",
    "solver_system": (
        "You are answering a question from Humanity's Last Exam: an extremely difficult, "
        "expert-level question that may come from any academic discipline. It may be "
        "multiple-choice or require an exact free-form answer (a number, expression, name, or "
        "short phrase). If it is multiple-choice, answer with the option LETTER only -- note "
        "there may be more than four options. Reason carefully, then finish with exactly "
        "these two lines:\n"
        "ANSWER: <your succinct, exact final answer -- no explanation, no units unless asked>\n"
        "CONFIDENCE: <integer 0-100>"
    ),
    "critic_system": (
        "You are a rigorous expert reviewer across all academic fields. Given a question and a "
        "candidate answer, check the reasoning and the exact final answer for errors. If fully "
        "correct reply 'APPROVED' on the first line; otherwise reply 'REJECTED' then the errors."
    ),
}


def _is_approved(critique):
    return critique.strip().upper().startswith("APPROVED")


def solve_layer(model, problem_text, spec, prior_answer=None, temperature=0.2, log=None):
    """Run the solver<->critic loop for one model layer. Returns final answer text."""
    log = log or (lambda *a, **k: None)

    solver_msgs = [{"role": "system", "content": spec["solver_system"]}]
    if prior_answer:
        user = (
            f"{problem_text}\n\n"
            "A smaller model produced the following candidate solution. Treat it as a "
            "hint only: verify it, fix any mistakes, and produce YOUR OWN best solution.\n\n"
            f"--- candidate ---\n{prior_answer}\n--- end candidate ---"
        )
    else:
        user = problem_text
    solver_msgs.append({"role": "user", "content": user})

    answer = chat(model, solver_msgs, temperature=temperature)
    solver_msgs.append({"role": "assistant", "content": answer})

    for rnd in range(MAX_ROUNDS):
        critique = chat(
            model,
            [
                {"role": "system", "content": spec["critic_system"]},
                {
                    "role": "user",
                    "content": f"PROBLEM:\n{problem_text}\n\nCANDIDATE SOLUTION:\n{answer}",
                },
            ],
            temperature=0.1,
        )
        log(f"    [{model}] round {rnd + 1} critic: {'APPROVED' if _is_approved(critique) else 'rejected'}")
        if _is_approved(critique):
            break
        # solver revises given the critique
        solver_msgs.append(
            {
                "role": "user",
                "content": (
                    "A reviewer raised the following issues. Fix them and output the full "
                    f"corrected solution again in the required format.\n\n{critique}"
                ),
            }
        )
        answer = chat(model, solver_msgs, temperature=temperature)
        solver_msgs.append({"role": "assistant", "content": answer})

    return answer


def escalate(problem_text, spec, ladder=None, log=None, status_out=None):
    """Run the full ladder small->large, threading each layer's answer upward."""
    log = log or (lambda *a, **k: None)
    ladder = ladder or LADDER
    prior = None
    for model in ladder:
        log(f"  layer {model}")
        prior = solve_layer(model, problem_text, spec, prior_answer=prior, log=log)
    if status_out is not None:
        status_out.setdefault("finish_reason", None)
    return prior
