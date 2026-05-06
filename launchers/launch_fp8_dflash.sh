#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_fp8_dflash.sh — Launch vLLM with FP8 base + BF16 DFlash drafter
#                       speculative decoding, parametric draft length.
#
# Pattern: FP8-quantized base model accelerated by a BF16 DFlash drafter
# (Discrete Flash, an EAGLE-class single-step parallel sampler) producing
# N candidate tokens per cycle, verified by the FP8 base.
#
# Usage:
#   NUM_SPEC=8 ./launch_fp8_dflash.sh
#
# Required env:
#   NUM_SPEC          — draft length (typically 7, 8, or 15)
#
# Optional env (defaults shown):
#   NAME              — container name (default: qwen-vllm-fp8-dflash-N${NUM_SPEC})
#   PORT              — host port (default: 11435)
#   BASE_MODEL        — HF model id (default: Qwen/Qwen3.6-27B-FP8)
#   DRAFT_MODEL       — HF drafter id (default: z-lab/Qwen3.6-27B-DFlash)
#   TP_SIZE           — tensor parallel size (default: 2)
#   IMAGE             — vLLM container image (default: repne/vllm:latest)
#
# ─────────────────────────────────────────────────────────────────────────────
# Why FP8 base + BF16 drafter?
#   FP8 base gives 2x throughput vs BF16 base on Blackwell. BF16 drafter
#   keeps acceptance rate high (FP8 drafters drop 3-5pp accept-rate due to
#   quantization noise on logits used for rejection sampling). Mixed
#   precision is a Pareto improvement.
#
# Why no `--load-format instanttensor`?
#   FP8 base + BF16 DFlash drafter triggers a segfault during drafter weight
#   load when instanttensor is enabled (cuMemcpyDtoDAsync_v2 in the drafter
#   weight-load path, traced May 2026 against vLLM 0.1.dev16434). Using
#   default tensor loader is ~30 s slower at startup but reliable.
#
# Why no `--speculative-config.draft_sample_method gumbel`?
#   Deprecated upstream as of May 2026. Default sampler is now equivalent
#   or better in measured acceptance rates. Including the flag will produce
#   warnings; future versions may error.
#
# Post-ready 60s settle:
#   /v1/models returning 200 means the API server is up but the engine
#   has not necessarily finished compiling all CUDA graphs. The first
#   inference can take 15-25s of compile time. The 60s settle ensures
#   subsequent benchmarks see steady-state behavior.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NUM_SPEC="${NUM_SPEC:?usage: NUM_SPEC=8 ./launch_fp8_dflash.sh}"
NAME="${NAME:-qwen-vllm-fp8-dflash-N${NUM_SPEC}}"
PORT="${PORT:-11435}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-27B-FP8}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.6-27B-DFlash}"
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
  --env OMP_NUM_THREADS=8 \
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
    --max-model-len 262144 \
    --max-num-seqs 128 --max-num-batched-tokens 32758 \
    --max-cudagraph-capture-size 256 --language-model-only \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    --speculative-config.method dflash \
    --speculative-config.model "$DRAFT_MODEL" \
    --speculative-config.num_speculative_tokens "$NUM_SPEC" \
    --speculative-config.attention_backend flash_attn \
    --speculative-config.use_local_argmax_reduction true \
    --attention-backend flashinfer \
    --default-chat-template-kwargs.preserve_thinking true >/dev/null

START=$(date +%s)
DEADLINE=$(( START + 600 ))
while [ $(date +%s) -lt $DEADLINE ]; do
  if curl -s -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q 'Qwen3.6-27B'; then
    echo "[FP8+DFlash N=$NUM_SPEC READY] in $(( $(date +%s) - START ))s"
    docker logs "$NAME" 2>&1 | grep -iE 'GPU KV cache size|Maximum concurrency' | tail -2
    echo "[$(date +%H:%M:%S)] Post-ready settle: 60s for CUDA graph compile..."
    sleep 60
    echo "[$(date +%H:%M:%S)] Settle complete, ready to bench."
    exit 0
  fi
  if docker logs --tail 5 "$NAME" 2>&1 | grep -qiE 'TypeError|ValueError|RuntimeError|Engine core init.*failed|Exited'; then
    echo "[FAIL]"; docker logs --tail 50 "$NAME" 2>&1 | tail -40; exit 1
  fi
  sleep 5
done
echo "[TIMEOUT]"; docker logs --tail 30 "$NAME" 2>&1; exit 1
