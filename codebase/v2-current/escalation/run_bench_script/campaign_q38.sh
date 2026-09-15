#!/usr/bin/env bash
# Qwen3.8-27B-FP8 (local vLLM via litellm), 5 passes x {single, manager}, LCB-100 pinned ids.
# PIPELINED: the next arm-run starts once the current one has only REMAIN problems left,
# rather than waiting for its tail. The last handful of problems in a pass run nearly alone
# (concurrency collapses to single digits while a few hard problems consume all 10 manager
# rounds), which is 30-60 min of a mostly idle GPU per pass.
#
# CAPS ARE PER-ARM: single 250,000 (against this model's 262,144 window, ~12k left for the
# prompt), manager 128,000. Set in launch() -- see the rationale there. The single arm is
# NOT comparable to Luna/Terra, which are hard-capped at 128,000 by the provider; report it
# as its own condition.
#
# Parallelism follows the cap (12 single / 24 manager). vLLM reports 610,388 KV tokens per
# replica and "Maximum concurrency for 262,144 tokens per request: 2.33x" -- only ~7
# full-length sequences fit across all three replicas. At a 128k cap and --parallel 36 the
# server already queued 22 and preempted; preemption at these lengths is expensive recompute
# (100k+ tokens), so err low and raise at a checkpoint if preemptions stay at zero.
#
# NO format gate: Qwen3.8-27B emits the required '### PLAN' / '### TASKS' headers correctly
# (verified before this run), unlike Muse-Glimmer which needed MULTIAGENT_STRICT_FORMAT=1.
# Reasoning is ON by default in this model's chat template; we send no reasoning_effort,
# which litellm's config notes only prepends a sentence to the system prompt anyway.
#
# The previous commit holds this script exactly as it ran, from a working tree outside this
# repo. This commit adapts the six machine-specific things -- working directory, PATH,
# HF_HOME, OUT/WS, the litellm key lookup and the uv invocation -- to the repo layout used by
# the sibling run_*.sh scripts. Every experimental setting is unchanged: the 250k/128k
# per-arm caps, --parallel 12/24, the model alias, the timeout and the iteration limits.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1     # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                               # repo root
# uv lives in ~/.local/bin, which is on the INTERACTIVE PATH but not the one systemd gives a
# user service. Scheduling this under a timer without it failed instantly with
# "env: 'uv': No such file or directory" -- and, because wait_almost only polled for progress
# and never checked whether the child was alive, the driver then slept for 40 hours.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv not on PATH ($PATH)"; exit 1; }
PYTHON=(uv run --no-project --python 3.12 --with 'datasets<4' --with numpy --with anthropic python)
export LCB_RELEASE=release_v6
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MULTIAGENT_MAX_ITERS=10 MULTIAGENT_MAX_TASKS=12
OUT="$ROOT/runs/4models-1pass-reason-on/results"
WS="$ROOT/runs/4models-1pass-reason-on/ws"
IDS=escalation/lcb100_hardest_v6.json
mkdir -p "$OUT" "$WS"

# The proxy key. Put LITELLM_KEY in escalation/.env (gitignored) or the environment; the
# litellm config is only a fallback for the machine this originally ran on.
set -a; [ -f escalation/.env ] && . escalation/.env; set +a
LITELLM_CONFIG="${LITELLM_CONFIG:-/home/persis/litellm/config.yaml}"
K="${LITELLM_KEY:-$(grep -m1 'master_key:' "$LITELLM_CONFIG" 2>/dev/null | awk '{print $2}')}"
TAG=q38
REMAIN=${REMAIN:-8}     # release the next run when this many problems are left (92/100)
TOTAL=100
PAR=${PAR:-12}

launch() {  # eng pass
  local eng=$1 p=$2
  local out="$OUT/${TAG}_${eng}_p${p}.json"
  if [ -f "$out" ]; then echo "[skip done] ${TAG}_${eng}_p${p}"; return 1; fi
  # PER-ARM CAP. Single gets 250k, manager 128k. The point is that NEITHER cap should bind:
  # a single shot at 128k hit the ceiling on 27.8% of problems (median 72,051 tokens), while
  # manager calls are short by construction -- the previous local model averaged ~25k per
  # call across ~6 calls -- so 128k is far above where the manager actually stops. Equalising
  # the number would not equalise the constraint; it would just re-impose the ceiling on the
  # arm that needs headroom, and the measured delta would be truncation relief again.
  # PER-ARM PARALLELISM follows from the cap: shorter manager sequences use less KV, so more
  # of them fit in the 610,388 tokens/replica pool.
  local cap=250000 par=12
  if [ "$eng" = "multiagent" ]; then cap=128000; par=24; fi
  echo "[$(date +%H:%M:%S)] start ${TAG}_${eng}_p${p} (cap=${cap} parallel=${par})"
  env ESCALATION_OPENAI_BASE="http://localhost:8216/v1/chat/completions" \
      GROQ_API_KEY="$K" \
      ESCALATION_GROQ_REASONING="" \
      ESCALATION_CLOUD_MAX_TOKENS="$cap" \
      ESCALATION_CLOUD_TIMEOUT=14400 \
      MULTIAGENT_MODEL="groq:small-model" \
      MULTIAGENT_WS="$WS/${TAG}_${eng}_p${p}" \
    "${PYTHON[@]}" escalation/run_bench.py \
      --engine "$eng" --only lcb --lcb "$TOTAL" --ids-file "$IDS" --parallel "$par" --out "$out" \
      > "/tmp/camp_${TAG}_${eng}_p${p}.log" 2>&1 &
  LAST_PID=$!
  return 0
}

wait_almost() {  # eng pass -- return at <=REMAIN left, or when the run has finished
  local eng=$1 p=$2
  local out="$OUT/${TAG}_${eng}_p${p}.json"
  local log="/tmp/camp_${TAG}_${eng}_p${p}.log"
  while true; do
    [ -f "$out" ] && { echo "[$(date +%H:%M:%S)] ${TAG}_${eng}_p${p} finished"; return; }
    # LIVENESS: a run that died without writing a result must not be waited on forever.
    # Without this the driver polls a dead process indefinitely -- which is exactly what
    # happened when uv was missing from the systemd PATH (40 hours of sleep 60).
    if ! kill -0 "$LAST_PID" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] !! ${TAG}_${eng}_p${p} EXITED without writing $out"
      echo "   last log lines:"; tail -5 "$log" 2>/dev/null | sed 's/^/   /'
      return 1
    fi
    # grep -c prints 0 AND exits 1 on no match, so `|| echo 0` would append a second 0 and
    # make the comparison a syntax error -- which fails closed and stalls the pipeline.
    local d=0
    if [ -f "$log" ]; then d=$(grep -c 'done (' "$log" 2>/dev/null) || d=0; fi
    if [ "$d" -ge $((TOTAL - REMAIN)) ]; then
      echo "[$(date +%H:%M:%S)] ${TAG}_${eng}_p${p} at ${d}/${TOTAL} -- releasing next"
      return
    fi
    sleep 60
  done
}

for p in 1 2 3 4 5; do
  for eng in single multiagent; do
    # A run that dies without a result is treated as fatal: the failure modes seen so far
    # (uv missing from PATH, backend unreachable) are systematic, so continuing would burn
    # hours and yield a partial grid that looks like real data.
    if launch "$eng" "$p"; then
      wait_almost "$eng" "$p" || { echo "[$(date +%H:%M:%S)] ABORTING campaign"; exit 1; }
    fi
  done
done
while pgrep -f "run_bench.py --engine .* --lcb ${TOTAL}" >/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] ${TAG} 5 PASSES DONE"
