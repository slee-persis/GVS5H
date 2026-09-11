#!/usr/bin/env bash

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root
PYTHON=(uv run --no-project --python 3.12 --with datasets --with numpy --with anthropic python)

export LCB_RELEASE=release_v6
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MULTIAGENT_MAX_ITERS=10
export MULTIAGENT_MAX_TASKS=12

CAP=${CAP:-128000}
IDS=escalation/lcb100_hardest_v6.json
OUT="$ROOT/runs/4models-1pass-reason-on/results"
WS="$ROOT/runs/4models-1pass-reason-on/ws"
mkdir -p "$OUT" "$WS"

set -a; [ -f escalation/.env ] && . escalation/.env; set +a

export OPENAI_API_KEY="${OPENAI_API_KEY:-${openai_api_key:-}}"

{ echo "sha=$(git rev-parse HEAD)"
  git status --porcelain -- escalation/ | sed 's/^/dirty: /'
  echo "cap=$CAP max_iters=$MULTIAGENT_MAX_ITERS release=$LCB_RELEASE ids=$IDS reasoning=on"
} > "$OUT/run_config.txt"

check() {
  IDS="$IDS" python3 -c '
import json, os, sys
want = json.load(open(os.environ["IDS"]))
got = [r["question_id"] for r in json.load(open(sys.argv[1]))["lcb"]["records"]]
if got != want:
    print(f"  !! MISMATCH in {sys.argv[1]}: {len(got)}/{len(want)} ids, "
          f"missing={sorted(set(want)-set(got))[:5]}", file=sys.stderr)
    sys.exit(1)
print(f"  ids OK ({len(got)}/100, order matches)")' "$1"
}

run() {
  local tag=$1 eng=$2 par=$3; shift 3
  local out="$OUT/${tag}_${eng}.json"
  [ -f "$out" ] && { echo "[skip done] ${tag}_${eng}"; return; }
  echo "[$(date +%H:%M:%S)] start ${tag}_${eng}"
  env "$@" MULTIAGENT_WS="$WS/${tag}_${eng}" \
    "${PYTHON[@]}" escalation/run_bench.py \
    --engine "$eng" --only lcb --lcb 100 --ids-file "$IDS" --parallel "$par" --out "$out" \
    > "/tmp/4m_${tag}_${eng}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  ${tag}_${eng}"
  check "$out" || echo "  (see /tmp/4m_${tag}_${eng}.log)"
}


have() {
  [ -n "${!1:-}" ] && return 0
  echo "[skip] $2: \$$1 is not set"; return 1
}

# --- arm 1: qwen3.6-27b, local vLLM -------------------------------------------------------
LITELLM_CONFIG="${LITELLM_CONFIG:-/home/persis/litellm/config.yaml}"
LITELLM_KEY="${LITELLM_KEY:-$(grep -m1 'master_key:' "$LITELLM_CONFIG" 2>/dev/null | awk '{print $2}')}"
if curl -sf -m 20 http://localhost:8216/v1/chat/completions \
     -H "Authorization: Bearer $LITELLM_KEY" -H 'Content-Type: application/json' \
     -d '{"model":"qwen3.6-27b-vllm","messages":[{"role":"user","content":"ok"}],"max_tokens":4}' \
     > /dev/null; then

  for eng in single multiagent; do
    run muse $eng 48 \
      ESCALATION_OPENAI_BASE="http://localhost:8216/v1/chat/completions" \
      GROQ_API_KEY="$LITELLM_KEY" \
      ESCALATION_GROQ_REASONING="" \
      ESCALATION_CLOUD_MAX_TOKENS="$CAP" \
      ESCALATION_CLOUD_TIMEOUT=7200 \
      MULTIAGENT_STRICT_FORMAT=1 \
      MULTIAGENT_MODEL="groq:small-model"
  done
else
  echo "[skip] q36l: litellm probe failed on :8216"
fi

# --- arm 2: GPT-5.6 Luna, OpenAI ----------------------------------------------------------
if have OPENAI_API_KEY "luna (gpt-5.6-luna)"; then
  for eng in single multiagent; do
    run luna $eng 32 \
      OPENAI_API_KEY="$OPENAI_API_KEY" \
      ESCALATION_CLOUD_MAX_TOKENS="$CAP" \
      ESCALATION_CLOUD_TIMEOUT=7200 \
      MULTIAGENT_MODEL="openai:gpt-5.6-luna"
  done
fi

# --- arm 3: Qwen3.8-Max, Alibaba Cloud Model Studio (first-party) --------------------------
if have DASHSCOPE_API_KEY "q38 (qwen3.8-max)"; then
  for eng in single multiagent; do
    run q38 $eng 4 \
      DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
      ESCALATION_CLOUD_MAX_TOKENS="$CAP" \
      ESCALATION_DASHSCOPE_THINKING_BUDGET="${THINK_BUDGET:-100000}" \
      ESCALATION_CLOUD_TIMEOUT=7200 \
      MULTIAGENT_MODEL="dashscope:qwen3.8-max"
  done
fi

# --- arm 4: Opus 5, Anthropic -------------------------------------------------------------
if have ANTHROPIC_API_KEY "opus5 (claude-opus-5)"; then
  for eng in single multiagent; do
    run opus5 $eng 4 \
      ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
      ESCALATION_CLOUD_MAX_TOKENS="$CAP" \
      ESCALATION_CLOUD_TIMEOUT=7200 \
      MULTIAGENT_MODEL="anthropic:claude-opus-5"
  done
fi

echo "[$(date +%H:%M:%S)] ALL 4-MODEL RUNS DONE (skipped arms listed above)"
