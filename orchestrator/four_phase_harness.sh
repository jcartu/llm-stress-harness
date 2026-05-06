#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# four_phase_harness.sh — Production-grade four-phase validation harness for
#                        OpenAI-compatible LLM endpoints.
#
# This is the orchestrator that runs against a single, already-launched
# inference container and produces a complete validation artifact bundle:
#
#   Phase 1: Functional gates  — 4 binary correctness probes
#                                (Fibonacci × 5 reps, tool calling, simple
#                                arithmetic, multi-turn coherence)
#   Phase 2: Throughput matrix — 3×3 grid (concurrency × prefix-context)
#                                with N=2 reseeded runs per cell, 60 s
#                                steady-state + 20 s warmup, mean ± stdev
#   Phase 3: HumanEval         — 164 problems, c=8, full failure-mode
#                                classification via stress_harness.py
#   Phase 4: MBPP              — 257 problems, c=8, same harness
#
# Total wall time per config: ~60 minutes (mostly Phase 3+4).
#
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   four_phase_harness.sh CONFIG_LABEL KV_BUDGET OUT_DIR
#
# Required environment (override defaults as needed):
#   PORT           — vLLM port (default: 11435)
#   MODEL          — model id sent in /v1/chat/completions (default: qwen3.6-27b)
#   BENCH          — path to llm-inference-bench Python (default: ./venv/bin/python)
#   SCRIPT         — path to llm_decode_bench.py
#   HARNESS        — path to stress_harness.py (default: ../harness/stress_harness.py)
#   PROBLEMS_DIR   — directory containing humaneval.jsonl & mbpp.jsonl
#
# Output:
#   $OUT_DIR/gates.log              — Phase 1 stdout
#   $OUT_DIR/gates.json             — Phase 1 machine-readable summary
#   $OUT_DIR/throughput-matrix/...  — Phase 2 per-cell results.json + bench.log
#   $OUT_DIR/humaneval.jsonl        — Phase 3 per-problem records
#   $OUT_DIR/humaneval_summary.json — Phase 3 aggregate
#   $OUT_DIR/mbpp.jsonl             — Phase 4 per-problem records
#   $OUT_DIR/mbpp_summary.json      — Phase 4 aggregate
#
# Why these four phases and not just HumanEval?
#   HumanEval alone gives you a single number. You cannot tell from it
#   whether a regression came from the model, the engine, the chat
#   template, or KV-cache pressure. The four phases are designed to triage
#   *before* the expensive Phase 3+4 — Phase 1 catches broken templates in
#   <60 s, Phase 2 catches throughput regressions in ~5 min, only then do
#   you commit ~50 min to coding correctness.
#
# Why N=2 reseeded runs in Phase 2?
#   N=1 measures noise. N=3+ wastes 50% of the budget for marginal stdev
#   precision at 9 cells × multiple configs. N=2 with reported (mean, stdev)
#   bands lets us reject sub-2% throughput regressions while keeping the
#   sweep tractable.
#
# Why duration=60 / warmup=20 / decode-only?
#   Per the upstream `llm_decode_bench` author's recommendation: 60 s of
#   measured decode is enough to drown out CUDA-graph jitter and prefix-
#   cache transients; 20 s of warmup ensures the engine has reached
#   steady-state KV occupancy before measurement begins; --skip-prefill
#   isolates decode throughput from prefill batching effects, which are
#   measured separately.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIG_LABEL="${1:?usage: four_phase_harness.sh CONFIG_LABEL KV_BUDGET OUT_DIR}"
KV_BUDGET="${2:?need KV budget (run-1 of llm_decode_bench reports it; rerun with this value)}"
OUT_DIR="${3:?need OUT_DIR}"

# Tunables
PORT="${PORT:-11435}"
MODEL="${MODEL:-qwen3.6-27b}"
BENCH="${BENCH:-python3}"
SCRIPT="${SCRIPT:-./llm_decode_bench.py}"
HARNESS="${HARNESS:-$(dirname "$0")/../harness/stress_harness.py}"
PROBLEMS_DIR="${PROBLEMS_DIR:-./problems}"

mkdir -p "$OUT_DIR"
START=$(date +%s)

echo ""
echo "═══════════════════════════════════════════════"
echo "  CONFIG: $CONFIG_LABEL"
echo "  Endpoint: http://localhost:$PORT  Model: $MODEL"
echo "  KV budget: $KV_BUDGET tokens"
echo "  Started: $(date +%H:%M:%S)"
echo "  Phase params: 60s decode + 20s warmup, N=2 reseeded runs per cell"
echo "═══════════════════════════════════════════════"

# ────────────────────────────────────────────────────────────────────────────
# Phase 1 — Functional gates (4 binary probes, ~60 s)
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] PHASE 1: Functional gates"
GATE_LOG="$OUT_DIR/gates.log"
python3 - <<PY 2>&1 | tee "$GATE_LOG"
import requests, json
URL = 'http://localhost:${PORT}/v1/chat/completions'
H = {'Content-Type': 'application/json'}
def ask(msgs, max_tokens=4096, tools=None):
    p = {'model':'${MODEL}','messages':msgs,'temperature':0.0,'max_tokens':max_tokens}
    if tools: p['tools']=tools; p['tool_choice']='auto'
    return requests.post(URL,headers=H,json=p,timeout=300).json()['choices'][0]['message']

results = {}

# Gate 1: Reproducibility — same prompt 5x must produce same canonical Fibonacci
ok = 0
for i in range(5):
    m=ask([{'role':'user','content':'Output the first 10 Fibonacci numbers as a comma-separated list (start: 1, 1).'}])
    c=(m.get('content') or '').strip()
    if '1, 1, 2, 3, 5, 8, 13, 21, 34, 55' in c: ok += 1
results['fib_5x'] = (ok, 5)
print(f"Gate 1 (Fibonacci 5x reproducibility): {ok}/5 -> {'PASS' if ok==5 else 'FAIL'}")

# Gate 2: Tool calling — must emit a tool_call with structured args
m=ask([{'role':'user','content':'What is the current weather in Tokyo? Use the tool.'}],
    tools=[{'type':'function','function':{'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}])
tcs=m.get('tool_calls') or []
g2 = any(tc['function']['name']=='get_weather' and 'tokyo' in tc['function']['arguments'].lower() for tc in tcs)
results['tool_call'] = g2
print(f"Gate 2 (Tool call structured emission): {'PASS' if g2 else 'FAIL'}")

# Gate 3: Reasoning — verify a non-trivial arithmetic answer survives <think>
m=ask([{'role':'user','content':'What is 47 times 83? Show the result as a number only on the last line.'}], 8192)
c=(m.get('content') or '').strip()
g3 = '3901' in c
results['reasoning_47x83'] = g3
print(f"Gate 3 (47*83=3901 in content): {'PASS' if g3 else 'FAIL'}")

# Gate 4: Multi-turn — context preservation across 3 turns
msgs=[{'role':'user','content':'Imagine the temperature in Tokyo is 28C. Just acknowledge.'}]
t1=ask(msgs,2048); t1c=(t1.get('content') or '').strip(); msgs.append({'role':'assistant','content':t1c})
msgs.append({'role':'user','content':'Now imagine Berlin is at 18C. Just acknowledge.'})
t2=ask(msgs,2048); t2c=(t2.get('content') or '').strip(); msgs.append({'role':'assistant','content':t2c})
msgs.append({'role':'user','content':'Which of the two cities I mentioned is warmer? Answer in one short sentence.'})
t3=ask(msgs,4096); t3c=(t3.get('content') or '').strip()
g4 = 'tokyo' in t3c.lower() and 'warm' in t3c.lower()
results['multi_turn'] = g4
print(f"Gate 4 (Multi-turn context preservation): {'PASS' if g4 else 'FAIL'} -- T3: {t3c[:80]!r}")

total = sum([results['fib_5x'][0]==5, results['tool_call'], results['reasoning_47x83'], results['multi_turn']])
print(f"\nGates passed: {total}/4")
with open('$OUT_DIR/gates.json','w') as f:
    json.dump({'results':results,'gates_passed':total,'gates_total':4}, f, indent=2)
PY

# ────────────────────────────────────────────────────────────────────────────
# Phase 2 — Throughput matrix (3 concurrency × 3 prefix-ctx, N=2 each)
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] PHASE 2: Throughput matrix (9 cells, N=2, 60s+20s warmup)"
mkdir -p "$OUT_DIR/throughput-matrix/runs"

# Cell grid: concurrency ∈ {1, 2, 4} × prefix_ctx ∈ {0, 32k, 131k}
declare -a CELLS=(
  "1 0" "1 32000" "1 131072"
  "2 0" "2 32000" "2 131072"
  "4 0" "4 32000" "4 131072"
)

for CELL in "${CELLS[@]}"; do
  CONC=$(echo $CELL | cut -d' ' -f1)
  CTX=$(echo $CELL | cut -d' ' -f2)
  LBL="c${CONC}_ctx${CTX}"
  for run in 1 2; do
    OUT_RUN="$OUT_DIR/throughput-matrix/runs/${LBL}_run${run}"
    mkdir -p "$OUT_RUN"
    if [ -f "$OUT_RUN/results.json" ]; then continue; fi
    if ! curl -s -m 3 http://localhost:$PORT/v1/models 2>/dev/null | grep -q "${MODEL%/*}"; then
      echo "  CONTAINER DOWN at $LBL run$run — aborting Phase 2"; break 2
    fi
    "$BENCH" "$SCRIPT" \
      --host localhost --port $PORT --model "$MODEL" \
      --concurrency $CONC --contexts $CTX \
      --duration 60 --decode-warmup-seconds 20 \
      --kv-budget $KV_BUDGET --skip-prefill \
      --display-mode plain --no-hw-monitor \
      --output "$OUT_RUN/results.json" > "$OUT_RUN/bench.log" 2>&1 || echo "    run$run FAILED"
    sleep 1
  done
  python3 - <<PY
import json, glob, statistics
runs=[]; accepts=[]
for r in sorted(glob.glob('$OUT_DIR/throughput-matrix/runs/${LBL}_run*/results.json')):
    try:
        d=json.load(open(r))['results'][0]
        if d['aggregate_tps']>5: runs.append(d['aggregate_tps']); accepts.append(d.get('server_spec_accept_rate',0))
    except: pass
if runs:
    m=statistics.mean(runs); s=statistics.stdev(runs) if len(runs)>1 else 0
    print(f"  $LBL: {m:.1f}±{s:.2f}  accept={statistics.mean(accepts):.1%}")
else: print(f"  $LBL: NO DATA")
PY
done

# ────────────────────────────────────────────────────────────────────────────
# Phase 3 — HumanEval (164 problems, c=8, full failure-mode classification)
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] PHASE 3: HumanEval (164 problems, c=8)"
python3 "$HARNESS" \
  --url http://localhost:$PORT/v1/chat/completions \
  --model "$MODEL" \
  --config-label "$CONFIG_LABEL" \
  --benchmark humaneval \
  --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
  --output "$OUT_DIR/humaneval.jsonl" \
  --concurrency 8 --max-tokens 8192 --request-timeout 600 2>&1 | tail -10

# ────────────────────────────────────────────────────────────────────────────
# Phase 4 — MBPP (257 problems, c=8, same harness)
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] PHASE 4: MBPP (257 problems, c=8)"
python3 "$HARNESS" \
  --url http://localhost:$PORT/v1/chat/completions \
  --model "$MODEL" \
  --config-label "$CONFIG_LABEL" \
  --benchmark mbpp \
  --problems-file "$PROBLEMS_DIR/mbpp.jsonl" \
  --output "$OUT_DIR/mbpp.jsonl" \
  --concurrency 8 --max-tokens 8192 --request-timeout 600 2>&1 | tail -10

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "═══════════════════════════════════════════════"
echo "  CONFIG $CONFIG_LABEL DONE in ${ELAPSED}s ($((ELAPSED/60)) min)"
echo "  Artifacts in: $OUT_DIR"
echo "═══════════════════════════════════════════════"
