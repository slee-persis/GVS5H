#!/usr/bin/env bash

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root
PYTHON=(uv run --no-project --python 3.12 --with datasets --with numpy --with anthropic python)

export LCB_RELEASE=release_v6
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CAP=${CAP:-128000}
PAR=${PAR:-50}
REMAIN=${REMAIN:-8}
export ESCALATION_CLAUDE_ATTEMPTS=${ESCALATION_CLAUDE_ATTEMPTS:-10}
IDS=escalation/lcb100_hardest_v6.json
OUT="$ROOT/runs/fable5-5pass-single/results"
WS="$ROOT/runs/fable5-5pass-single/ws"
mkdir -p "$OUT" "$WS"

set -a; [ -f escalation/.env ] && . escalation/.env; set +a

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "FATAL: ANTHROPIC_API_KEY is not set (not in the environment, not in escalation/.env)."
  echo "       This arm is first-party Anthropic; there is no other credential path."
  exit 1
fi

"${PYTHON[@]}" - <<'PY' || exit 1
import os, sys, anthropic
c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300)
try:
    with c.messages.stream(model="claude-fable-5", max_tokens=1024,
                           messages=[{"role": "user", "content": "Reply with just: ok"}]) as s:
        m = s.get_final_message()
except Exception as e:
    print(f"FATAL: probe failed: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
    sys.exit(1)
print(f"  probe OK: model={m.model} stop={m.stop_reason} out_tokens={m.usage.output_tokens}")
PY

{ echo "sha=$(git rev-parse HEAD)"
  git status --porcelain -- escalation/ | sed 's/^/dirty: /'
  echo "model=anthropic:claude-fable-5 cap=$CAP parallel=$PAR release=$LCB_RELEASE ids=$IDS"
  echo "engine=single passes=5 thinking=adaptive(always on) effort=high(pinned) fallbacks=none"
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


launch() {
  local p=$1 out="$OUT/fable5_single_p${p}.json"
  [ -f "$out" ] && { echo "[skip done] fable5_single_p${p}"; return 1; }
  echo "[$(date +%H:%M:%S)] start fable5_single_p${p}"
  env ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
      ESCALATION_CLOUD_MAX_TOKENS="$CAP" \
      ESCALATION_CLOUD_TIMEOUT=7200 \
      MULTIAGENT_MODEL="anthropic:claude-fable-5" \
      MULTIAGENT_WS="$WS/fable5_single_p${p}" \
    "${PYTHON[@]}" escalation/run_bench.py \
      --engine single --only lcb --lcb 100 --ids-file "$IDS" --parallel "$PAR" --out "$out" \
      > "/tmp/fable5_single_p${p}.log" 2>&1 &
  return 0
}

wait_almost() {
  local p=$1 out="$OUT/fable5_single_p${p}.json" log="/tmp/fable5_single_p${p}.log" d
  while true; do
    [ -f "$out" ] && { echo "[$(date +%H:%M:%S)] fable5_single_p${p} finished"; return; }
    d=0
    if [ -f "$log" ]; then d=$(grep -c 'done (' "$log" 2>/dev/null) || d=0; fi
    if [ "$d" -ge $((100 - REMAIN)) ]; then
      echo "[$(date +%H:%M:%S)] fable5_single_p${p} at ${d}/100 -- releasing next"
      return
    fi
    sleep 30
  done
}

for p in 1 2 3 4 5; do
  launch "$p" && wait_almost "$p"
done
wait

for p in 1 2 3 4 5; do
  out="$OUT/fable5_single_p${p}.json"
  [ -f "$out" ] && { printf 'p%s: ' "$p"; check "$out" || echo "  (see /tmp/fable5_single_p${p}.log)"; }
done

echo "[$(date +%H:%M:%S)] FABLE5 5 SINGLE PASSES DONE"
