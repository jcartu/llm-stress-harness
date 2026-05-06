# Launchers — Parametric vLLM Container Templates

Reference launchers for vLLM in three speculative-decoding configurations.
These are **templates**, not generic deployment scripts. They encode
specific decisions made during the May 2026 Qwen3.6-27B production
validation (see [stress-validation report][study]) and should be adapted
to your model and hardware.

[study]: https://github.com/jcartu/qwen36-27b-blackwell-stress-validation

## What's here

| Script | Pattern | Use case |
|--------|---------|----------|
| `launch_fp8_mtp.sh` | FP8 base + native MTP heads | **Production-recommended** for Qwen3.6-27B. Stable, no external drafter. |
| `launch_fp8_dflash.sh` | FP8 base + BF16 DFlash drafter | Experimental. Higher peak throughput at high concurrency, more failure modes. |
| `launch_bf16_dflash.sh` | BF16 base + BF16 DFlash drafter | Quality reference. Use to isolate FP8 quantization effects from spec-decode config. |

## Why these three patterns

These represent the **non-overlapping points on the speculative-decoding
design space** that matter for production:

```
                        Throughput
                            ▲
                            │
                  FP8+DFlash N=8 ●─── (peak, but risky)
                            │
                  FP8+MTP=3 ●────── (RECOMMENDED — best stability/speed)
                            │
                            │
                 BF16+DFlash ●───── (quality reference; -22% throughput vs FP8)
                            │
                            │
                            └────────────► Stability/Quality
```

Other configurations (FP8+MTP=5, BF16+MTP, FP8+DFlash N=15) were measured
during the validation study and found to be Pareto-dominated by these three.

## Common parameters

All three launchers share the same core invocation pattern:

```bash
docker run --gpus all --network host \
  --volume "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ... \
  repne/vllm:latest \
    -O3 --model <BASE_MODEL> \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 32758 \
    --enable-auto-tool-choice \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    --speculative-config.method <mtp|dflash> \
    --speculative-config.num_speculative_tokens "$NUM_SPEC" \
    --attention-backend flashinfer \
    ...
```

The differences between the three launchers are surgical and consequential:

### `launch_fp8_mtp.sh`
- `--speculative-config.method mtp` (uses native heads inside the FP8 checkpoint)
- `--load-format instanttensor` (halves cold start)
- No drafter model loaded (no second weight-load risk)

### `launch_fp8_dflash.sh`
- `--speculative-config.method dflash`
- `--speculative-config.model z-lab/Qwen3.6-27B-DFlash` (separate drafter)
- `--speculative-config.attention_backend flash_attn` (drafter uses different backend)
- **Omits** `--load-format instanttensor` (segfault with mixed-precision drafter)

### `launch_bf16_dflash.sh`
- Same as FP8 DFlash, but `--model Qwen/Qwen3.6-27B` (BF16 base)
- `OMP_NUM_THREADS=8` (higher threading helps less without FP8 tensor cores active)

## Readiness pattern

Every launcher embeds the same readiness loop:

```bash
START=$(date +%s)
DEADLINE=$(( START + 600 ))
while [ $(date +%s) -lt $DEADLINE ]; do
  if curl -s -m 3 http://localhost:11435/v1/models 2>/dev/null | grep -q 'Qwen3.6-27B'; then
    # Optionally: post-ready settle for CUDA graph compile
    sleep 60
    exit 0
  fi
  if docker logs --tail 5 "$NAME" 2>&1 | grep -qiE 'TypeError|...|Engine core init.*failed|Exited'; then
    exit 1   # fast-fail on init crash
  fi
  sleep 5
done
exit 1   # timeout
```

This is the same pattern as `utils/wait_vllm_ready.sh`, inlined for
self-containment. The 60s post-ready settle in `launch_fp8_dflash.sh`
is **mandatory** if you immediately benchmark — it gives the engine time
to compile remaining CUDA graphs before measurement begins.

## Override points

Every launcher exposes the same environment variable contract:

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `NUM_SPEC` | Yes | Speculative draft length |
| `NAME` | No | Container name (auto-generated from params) |
| `PORT` | No | Host port (default 11435) |
| `BASE_MODEL` | No | HF model id for base |
| `DRAFT_MODEL` | No | HF drafter id (DFlash launchers only) |
| `TP_SIZE` | No | Tensor parallel size |
| `IMAGE` | No | vLLM container image |
| `MAX_NUM_SEQS` | No | (MTP only) concurrent sequence ceiling |
| `MAX_BATCHED` | No | (MTP only) max batched tokens |

This makes them adaptable to other Qwen-family models or other GPU configs
without forking the script.

## ⚠️ Caveats

1. **These launchers assume** an `~/.cache/huggingface/token` file exists with
   your HF token. If your model is gated, this is required.

2. **`repne/vllm:latest`** is a non-public image used during the validation
   study. Substitute `vllm/vllm-openai:latest` (or your preferred image) by
   passing `IMAGE=...`.

3. **`Qwen/Qwen3.6-27B-FP8`** is a placeholder for the model under test.
   Substitute via `BASE_MODEL=...`.

4. **`z-lab/Qwen3.6-27B-DFlash`** is the official DFlash drafter checkpoint
   for this model family. Other Qwen models may have different drafters.

5. **`--max-num-batched-tokens 32758`** and **`--gpu-memory-utilization 0.85`**
   are tuned for `2× RTX Pro 6000 Blackwell` (96 GB each). Lower these on
   smaller GPUs.
