#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# wait_vllm_ready.sh — Poll an OpenAI-compatible endpoint until a named model
#                     is served, then issue a warm-up completion.
#
# Usage:
#   wait_vllm_ready.sh [TIMEOUT_SECONDS] [MODEL_NAME] [PORT]
#
# Defaults:
#   TIMEOUT_SECONDS = 600      (vLLM cold init typically 90-210s; FP8+spec ≈ 120s)
#   MODEL_NAME      = Qwen3.6-27B
#   PORT            = 11435
#
# Exit codes:
#   0 — model is being served and warm-up succeeded
#   1 — timeout waiting for /v1/models to expose MODEL_NAME
#
# Why warm-up after readiness?
#   /v1/models returning the model means the API server is up and the engine
#   has loaded weights, but it does NOT mean the first decode CUDA-graph is
#   compiled or that flashinfer kernels are JIT-cached. The warm-up
#   completion forces both. Without it, the first benchmark request takes
#   15–25 s of "engine compile" time that pollutes p95 latency stats.
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

TIMEOUT="${1:-600}"
EXPECTED_MODEL="${2:-Qwen3.6-27B}"
PORT="${3:-11435}"
URL="http://localhost:${PORT}"

T0=$(date +%s)
echo "[wait_vllm_ready] target=$URL model=$EXPECTED_MODEL timeout=${TIMEOUT}s"

while true; do
    if curl -s -m 3 "${URL}/v1/models" 2>/dev/null \
       | grep -q "\"${EXPECTED_MODEL}\""; then
        ELAPSED=$(($(date +%s) - T0))
        echo "[wait_vllm_ready] vLLM ready: ${EXPECTED_MODEL} served (${ELAPSED}s)"
        echo "[wait_vllm_ready] issuing warm-up completion..."
        curl -s -m 30 "${URL}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${EXPECTED_MODEL}\",\"prompt\":\"hello\",\"max_tokens\":5,\"temperature\":0.0}" \
            > /dev/null 2>&1
        WARMUP_ELAPSED=$(($(date +%s) - T0))
        echo "[wait_vllm_ready] warm-up done. Total: ${WARMUP_ELAPSED}s"
        exit 0
    fi
    ELAPSED=$(($(date +%s) - T0))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[wait_vllm_ready] TIMEOUT after ${TIMEOUT}s waiting for ${EXPECTED_MODEL}"
        exit 1
    fi
    sleep 5
done
