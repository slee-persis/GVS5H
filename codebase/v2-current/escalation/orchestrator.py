"""Hierarchical multi-model escalation"""

import os
import sys
import json
import time
import subprocess
import threading
import urllib.request
import urllib.error

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get(
    "ESCALATION_OPENAI_BASE", "https://api.groq.com/openai/v1/chat/completions")
GROQ_REASONING = os.environ.get("ESCALATION_GROQ_REASONING", "none")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REASONING = os.environ.get("ESCALATION_OR_REASONING", "none")
OPENROUTER_PROVIDERS = [p.strip() for p in os.environ.get("ESCALATION_OR_PROVIDERS", "").split(",") if p.strip()]


LADDER = os.environ.get(
    "ESCALATION_LADDER",
    "qwen3.5:2b,qwen3.5:9b,qwen3.5:35b,qwen3.5:122b",
).split(",")

MAX_ROUNDS = int(os.environ.get("ESCALATION_MAX_ROUNDS", "2"))
THINK = os.environ.get("ESCALATION_THINK", "0") == "1"
REQUEST_TIMEOUT = int(os.environ.get("ESCALATION_TIMEOUT", "1200"))
CLOUD_TIMEOUT = int(os.environ.get("ESCALATION_CLOUD_TIMEOUT", "120"))
CLOUD_MAX_TOKENS = int(os.environ.get("ESCALATION_CLOUD_MAX_TOKENS", "8000"))

CLAUDE_TIMEOUT = int(os.environ.get("ESCALATION_CLAUDE_TIMEOUT", "1800"))
CLAUDE_ATTEMPTS = int(os.environ.get("ESCALATION_CLAUDE_ATTEMPTS", "3"))

# --- first-party providers -------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_REASONING = os.environ.get("ESCALATION_OPENAI_REASONING", "")

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_URL = os.environ.get(
    "ESCALATION_DASHSCOPE_BASE",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")

DASHSCOPE_THINKING_BUDGET = int(os.environ.get("ESCALATION_DASHSCOPE_THINKING_BUDGET", "0"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MAX_OUTPUT = 128000
ANTHROPIC_EFFORT = "high"


def ollama_chat(model, messages, temperature=0.2, num_ctx=16384, meta=None):
    num_ctx = int(os.environ.get("ESCALATION_NUM_CTX", num_ctx))
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
                meta["prompt_tokens"] = out.get("prompt_eval_count")
                meta["reasoning"] = (out.get("message") or {}).get("thinking") or ""
            return out["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    return ""


def openai_chat(url, api_key, model, messages, temperature=0.2, extra=None, meta=None,
                token_param="max_tokens", send_temperature=True):
    base = {"model": model, "messages": messages, "stream": False}
    if send_temperature:
        base["temperature"] = temperature
    if CLOUD_MAX_TOKENS > 0:
        base[token_param] = CLOUD_MAX_TOKENS
    if extra:
        base.update(extra)
    prov0 = dict(base.get("provider") or {})
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "model-test/1.0",
    }

    def _do(box, data):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=CLOUD_TIMEOUT) as r:
                box["out"] = json.loads(r.read())
        except urllib.error.HTTPError as e:
            box["err"] = e
            box["code"] = e.code
            try:
                box["body"] = e.read().decode()[:300]
            except Exception:
                box["body"] = ""
        except Exception as e:
            box["err"] = e

    ATTEMPTS = 16
    only = list(prov0.get("only") or [])
    cur_max = base.get(token_param)
    ignore, last, got_response = [], "", False
    for attempt in range(ATTEMPTS):
        body = dict(base)
        if cur_max:
            body[token_param] = cur_max
        prov = dict(prov0)
        if only:
            k = attempt % len(only)
            prov["order"] = only[k:] + only[:k]
        if ignore:
            prov["ignore"] = list(ignore)
        if prov:
            body["provider"] = prov
        box = {}
        th = threading.Thread(target=_do, args=(box, json.dumps(body).encode()), daemon=True)
        th.start()
        th.join(CLOUD_TIMEOUT + 5)
        if th.is_alive():
            print(f"    [cloud] hard timeout >{CLOUD_TIMEOUT}s, abandoning provider (attempt {attempt + 1})", file=sys.stderr, flush=True)
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "why": f"hard timeout >{CLOUD_TIMEOUT}s (no response)"})
            continue
        out = box.get("out")
        if out is None:
            e = box.get("err")
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "why": f"{type(e).__name__}: {str(e)[:200]}",
                     "http_code": box.get("code"), "body": box.get("body")})
            body_txt = (box.get("body") or "").lower()
            ctx_err = any(s in body_txt for s in (
                "context", "maximum context length", "too long", "exceeds", "max_model_len"))
            if box.get("code") == 400 and ctx_err and cur_max and cur_max > 40000:
                cur_max = max(40000, cur_max - 24000)
                print(f"    [cloud] HTTP 400 (prompt+max_tokens > context); reducing {token_param} to {cur_max}", file=sys.stderr, flush=True)
                continue
            if box.get("code") == 400 and ctx_err:
                print(f"    [cloud] prompt alone exceeds the context window at {token_param}"
                      f"={cur_max}; not retryable, aborting", file=sys.stderr, flush=True)
                break
            if box.get("code") in (400, 401, 402, 403, 404) and not ctx_err:
                print(f"    [cloud] HTTP {box.get('code')} is not retryable, aborting: "
                      f"{(box.get('body') or '')[:200]}", file=sys.stderr, flush=True)
                break
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                ra = e.headers.get("retry-after")
                try:
                    wait = float(ra) if ra else min(30.0, 5.0 * (attempt + 1))
                except ValueError:
                    wait = min(30.0, 5.0 * (attempt + 1))
                print(f"    [cloud] 429 rate-limited, sleep {wait:.0f}s (attempt {attempt + 1})", file=sys.stderr, flush=True)
                time.sleep(wait + 1)
                continue
            print(f"    [cloud] {type(e).__name__} on attempt {attempt + 1}, retrying", file=sys.stderr, flush=True)
            time.sleep(3 * (attempt + 1))
            continue
        served = out.get("provider")
        choices = out.get("choices")
        if not choices:
            print(f"    [cloud] no choices from {served} ({str(out.get('error'))[:80]}); rerouting", file=sys.stderr, flush=True)
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "provider": served, "why": "no choices in response",
                     "body": str(out.get("error"))[:500]})
            if served:
                ignore.append(served)
            time.sleep(2)
            continue
        got_response = True
        choice = choices[0]
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason")
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if not content:
            content = reasoning
        usage = out.get("usage") or {}
        ntok = usage.get("completion_tokens")
        if meta is not None:
            meta["finish_reason"] = finish
            meta["completion_tokens"] = ntok
            meta["prompt_tokens"] = usage.get("prompt_tokens")
            meta["reasoning"] = reasoning
            meta["provider"] = served
            meta["attempts"] = attempt + 1
        cap_used = cur_max or CLOUD_MAX_TOKENS
        clamped = finish == "length" and cap_used and (ntok or 0) < 0.9 * cap_used
        if content.strip() and not clamped:
            return content
        why = f"clamped at {ntok} tok (< cap)" if clamped else f"empty reply (finish={finish})"
        print(f"    [cloud] {why} from {served}; rerouting to another provider (attempt {attempt + 1})", file=sys.stderr, flush=True)
        if meta is not None:
            meta.setdefault("discarded", []).append(
                {"attempt": attempt + 1, "provider": served, "why": why, "finish_reason": finish,
                 "completion_tokens": ntok, "content": content, "reasoning": reasoning})
        last = content or last
        if served:
            ignore.append(served)
        time.sleep(1)
    n_tried = len((meta or {}).get("discarded") or []) or ATTEMPTS
    print(f"    [cloud] gave up after {n_tried} attempt(s) without a clean completion "
          f"(got_response={got_response})", file=sys.stderr, flush=True)
    if meta is not None and not last.strip():
        meta["infra_exhausted"] = True
    return last


def claude_cli_chat(model, messages, temperature=0.2, meta=None):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    body = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose",
           "--no-session-persistence", "--tools", "", "--model", model]
    if system:
        cmd += ["--append-system-prompt", system]
    for attempt in range(CLAUDE_ATTEMPTS):
        try:
            proc = subprocess.run(cmd, input=body, capture_output=True, text=True,
                                  timeout=CLAUDE_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"    [claude] timeout >{CLAUDE_TIMEOUT}s (attempt {attempt + 1})",
                  file=sys.stderr, flush=True)
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "why": f"timeout >{CLAUDE_TIMEOUT}s (no response)"})
            continue
        text, usage, stop, err, turns = "", {}, None, False, None
        thinking, blocks = [], 0
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "assistant":
                for b in (ev.get("message") or {}).get("content") or []:
                    if b.get("type") == "thinking":
                        blocks += 1
                        if b.get("thinking"):
                            thinking.append(b["thinking"])
            if ev.get("type") == "result":
                text = ev.get("result") or ""
                usage = ev.get("usage") or {}
                stop = ev.get("stop_reason")
                err = bool(ev.get("is_error"))
                turns = ev.get("num_turns")
        if meta is not None:
            meta["finish_reason"] = ("length" if stop == "max_tokens"
                                     else "error" if err else stop or "stop")
            meta["completion_tokens"] = usage.get("output_tokens")
            meta["prompt_tokens"] = usage.get("input_tokens")
            meta["reasoning"] = "\n\n".join(thinking)
            meta["thinking_blocks"] = blocks
            meta["num_turns"] = turns
            meta["attempts"] = attempt + 1
        if text.strip() and not err:
            return text
        print(f"    [claude] empty/error reply (stop={stop} rc={proc.returncode}); "
              f"retrying (attempt {attempt + 1})", file=sys.stderr, flush=True)
        if meta is not None:
            meta.setdefault("discarded", []).append(
                {"attempt": attempt + 1, "why": f"empty/error (stop={stop} rc={proc.returncode})",
                 "finish_reason": stop, "completion_tokens": usage.get("output_tokens"),
                 "content": text, "reasoning": "\n\n".join(thinking), "stderr": proc.stderr})
        time.sleep(3 * (attempt + 1))
    if meta is not None:
        meta["infra_exhausted"] = True
    return ""


def anthropic_chat(model, messages, temperature=0.2, meta=None):
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=CLOUD_TIMEOUT)
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    msgs = [{"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] != "system"]
    want = CLOUD_MAX_TOKENS or ANTHROPIC_MAX_OUTPUT
    max_out = min(want, ANTHROPIC_MAX_OUTPUT)
    for attempt in range(CLAUDE_ATTEMPTS):
        try:
            kw = {"model": model, "max_tokens": max_out, "messages": msgs,
                  "thinking": {"type": "adaptive", "display": "summarized"},
                  "output_config": {"effort": ANTHROPIC_EFFORT}}
            if system:
                kw["system"] = system
            with client.messages.stream(**kw) as stream:
                msg = stream.get_final_message()
        except Exception as e:  # noqa: BLE001 - recorded, then retried
            print(f"    [anthropic] {type(e).__name__} on attempt {attempt + 1}: {str(e)[:200]}",
                  file=sys.stderr, flush=True)
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "why": f"{type(e).__name__}: {str(e)[:200]}"})
            time.sleep(3 * (attempt + 1))
            continue
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        thinking = "\n\n".join(getattr(b, "thinking", "") or ""
                               for b in msg.content if getattr(b, "type", "") == "thinking")
        if meta is not None:
            meta["finish_reason"] = "length" if msg.stop_reason == "max_tokens" else msg.stop_reason
            meta["completion_tokens"] = msg.usage.output_tokens
            meta["prompt_tokens"] = msg.usage.input_tokens
            meta["reasoning"] = thinking
            meta["reasoning_is_summary"] = True  # NOT the raw chain of thought
            meta["attempts"] = attempt + 1
            if want > ANTHROPIC_MAX_OUTPUT:
                meta["cap_clamped_to"] = ANTHROPIC_MAX_OUTPUT
        if msg.stop_reason == "refusal":
            print("    [anthropic] stop_reason=refusal", file=sys.stderr, flush=True)
        return text
    if meta is not None:
        meta["infra_exhausted"] = True
    return ""


def chat(model, messages, temperature=0.2, num_ctx=16384, meta=None):
    if model.startswith("groq:"):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set but ladder uses a groq: model")
        extra = {"reasoning_effort": GROQ_REASONING} if GROQ_REASONING else None
        return openai_chat(GROQ_URL, GROQ_API_KEY, model[len("groq:"):], messages, temperature, extra, meta)
    if model.startswith("claude:"):  # subscription auth via the Claude Code CLI
        return claude_cli_chat(model[len("claude:"):], messages, temperature, meta)
    if model.startswith("openai:"):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set but model uses an openai: prefix")
        name = model[len("openai:"):]
        gpt5 = name.startswith("gpt-5")
        extra = {"reasoning_effort": OPENAI_REASONING} if OPENAI_REASONING else None
        return openai_chat(OPENAI_URL, OPENAI_API_KEY, name, messages, temperature, extra, meta,
                           token_param="max_completion_tokens" if gpt5 else "max_tokens",
                           send_temperature=not gpt5)
    if model.startswith("dashscope:"):
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY not set but model uses a dashscope: prefix")
        extra = {"thinking_budget": DASHSCOPE_THINKING_BUDGET} if DASHSCOPE_THINKING_BUDGET else None
        if not extra:
            print("    [dashscope] WARNING: ESCALATION_DASHSCOPE_THINKING_BUDGET is unset -- "
                  "max_tokens does not bound thinking on this provider, so this call is "
                  "effectively uncapped", file=sys.stderr, flush=True)
        return openai_chat(DASHSCOPE_URL, DASHSCOPE_API_KEY, model[len("dashscope:"):],
                           messages, temperature, extra, meta)
    if model.startswith("anthropic:"):
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set but model uses an anthropic: prefix")
        return anthropic_chat(model[len("anthropic:"):], messages, temperature, meta)
    if model.startswith("openrouter:"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set but model uses an openrouter: prefix")
        prov = {"sort": "throughput"}
        if OPENROUTER_PROVIDERS:
            prov["only"] = OPENROUTER_PROVIDERS
        extra = {"provider": prov}
        if OPENROUTER_REASONING == "none":
            extra["reasoning"] = {"enabled": False}
        elif OPENROUTER_REASONING in ("low", "medium", "high"):
            extra["reasoning"] = {"effort": OPENROUTER_REASONING}
        elif OPENROUTER_REASONING.startswith("budget:"):
            extra["reasoning"] = {"max_tokens": int(OPENROUTER_REASONING.split(":", 1)[1])}
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


def escalate(problem_text, spec, ladder=None, log=None, status_out=None, tests=None):
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
