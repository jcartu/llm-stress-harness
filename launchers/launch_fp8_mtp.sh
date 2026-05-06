#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_fp8_mtp.sh — Launch vLLM with FP8 base + native Multi-Token Prediction
#                    (MTP) speculative decoding, parametric draft length.
#
# Pattern: FP8-quantized base model with built-in MTP heads (Qwen's native
# self-speculative architecture — no separate drafter model). MTP is the
# production-recommended configuration for Qwen3.6-27B-FP8 ("ROCK SOLID")
# due to its stability and freedom from external drafter weight loads.
#
# Usage:
#   NUM_SPEC=3 ./launch_fp8_mtp.sh
#
# Required env:
#   NUM_SPEC          — draft length (production: 3; experimental: 5)
#
# Optional env:
#   MAX_NUM_SEQS      — concurrent sequence ceiling (default: 128)
#   MAX_BATCHED       — max batched tokens (default: 32758)
#   NAME              — container name (auto-generated from params)
#   PORT              — host port (default: 11435)
#   BASE_MODEL        — HF model id (default: Qwen/Qwen3.6-27B-FP8)
#   TP_SIZE           — tensor parallel size (default: 2)
#   IMAGE             — vLLM container image (default: repne/vllm:latest)
#
# ─────────────────────────────────────────────────────────────────────────────
# Why MTP and not DFlash for production?
#   MTP heads ship inside the FP8 model checkpoint — no second-model
#   weight load, no segfault risk from precision-mismatched drafters,
#   no version-skew between base and drafter. In measured runs, MTP=3
#   delivers ~241 tok/s vs DFlash N=7 at ~244 tok/s (~1% gap), but with
#   substantially lower operational risk. We use MTP=3 in production.
#
# Why `--load-format instanttensor`?
#   Halves cold start time (210s → 105s for FP8 27B-TP=2). Safe for
#   FP8+MTP (segfault only manifests with FP8+BF16 mixed-precision drafters).
#
# About `--speculative-config.draft_sample_method gumbel`:
#   This flag is deprecated upstream as of May 2026. Older vLLM versions
#   may still accept it. Modern versions ignore or warn. Future versions
#   will error — remove if upgrading vLLM.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NUM_SPEC="${NUM_SPEC:?usage: NUM_SPEC=3 ./launch_fp8_mtp.sh}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_BATCHED="${MAX_BATCHED:-32758}"
NAME="${NAME:-qwen-vllm-fp8-mtp-N${NUM_SPEC}-seqs${MAX_NUM_SEQS}-bt${MAX_BATCHED}}"
PORT="${PORT:-11435}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B-FP8}"
TP_SIZE="${TP_SIZE:-2}"
IMAGE="${IMAGE:-repne/vllm:latest}"
HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')"

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  --volume "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --volume "$HOME/.cache/vllm:/root/.cache/vllm" \
  --volume "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  --volume "$HOME/.triton/cache:/root/.triton/cache" \
  --env OMP_NUM_THREADS=16 \
  --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --env NCCL_P2P_LEVEL=SYS --env NCCL_NET_GDR_LEVEL=SYS --env NCCL_MIN_NCHANNELS=8 \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  "$IMAGE" \
    -O3 --model "$BASE_MODEL" \
    --served-model-name Qwen3.6-27B qwen3.6-27b \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" --gpu-memory-utilization 0.85 \
    --max-model-len 262144 --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
    --max-cudagraph-capture-size 256 --language-model-only --enable-auto-tool-choice \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-prefix-caching \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens "$NUM_SPEC" \
    --attention-backend flashinfer --load-format instanttensor \
    --default-chat-template-kwargs.preserve_thinking true >/dev/null

START=$(date +%s)
DEADLINE=$(( START + 360 ))
while [ $(date +%s) -lt $DEADLINE ]; do
  if curl -s -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q 'Qwen3.6-27B'; then
    echo "[FP8+MTP N=$NUM_SPEC seqs=$MAX_NUM_SEQS bt=$MAX_BATCHED READY] in $(( $(date +%s) - START ))s"
    docker logs "$NAME" 2>&1 | grep -iE 'GPU KV cache size|Maximum concurrency' | tail -2
    exit 0
  fi
  if docker logs --tail 5 "$NAME" 2>&1 | grep -qiE 'TypeError|ValueError|RuntimeError|Engine core init.*failed|Exited'; then
    echo "[FAIL]"; docker logs --tail 30 "$NAME" 2>&1 | tail -25; exit 1
  fi
  sleep 5
done
echo "[TIMEOUT]"; exit 1
