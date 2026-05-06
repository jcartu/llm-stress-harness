# Orchestrator — `four_phase_harness.sh`

A single-script orchestrator that runs the complete validation suite against
an already-launched OpenAI-compatible LLM endpoint and produces a full
artifact bundle suitable for cross-config comparison.

## What it produces

```
$OUT_DIR/
├── gates.log                  # Phase 1 stdout (human-readable)
├── gates.json                 # Phase 1 machine-readable summary
├── throughput-matrix/
│   └── runs/
│       ├── c1_ctx0_run1/
│       │   ├── results.json   # llm_decode_bench output
│       │   └── bench.log
│       ├── c1_ctx0_run2/
│       ├── c1_ctx32000_run1/
│       └── ... (9 cells × 2 runs = 18 dirs)
├── humaneval.jsonl            # 164 records, one per problem
├── humaneval_summary.json     # aggregate (pass_rate, p95, failure_modes)
├── mbpp.jsonl                 # 257 records
└── mbpp_summary.json
```

## Wall time

| Phase | Typical duration | Cumulative |
|-------|------------------|------------|
| 1 — Functional gates | ~60 s | 1 min |
| 2 — Throughput matrix (9 cells × 2 runs × 80 s) | ~25 min | 26 min |
| 3 — HumanEval (164 problems @ c=8) | ~5 min | 31 min |
| 4 — MBPP (257 problems @ c=8) | ~25 min | 56 min |

Total: **~60 min per config** on `2× RTX Pro 6000 Blackwell`. Scales with
GPU.

## Why four phases and not just HumanEval?

HumanEval alone produces **a single number** (`pass_rate`). When that number
regresses, you cannot tell whether the cause is:

- The model degraded (real correctness regression)
- The chat template changed (no_code spike)
- KV cache pressure caused truncation (empty_response spike)
- The engine got slower (timeout spike, identical pass distribution)
- Infrastructure broke (http_error spike)

The four phases form a **diagnostic ladder**, ordered by execution cost and
failure-mode specificity:

1. **Phase 1 (60 s)** catches broken chat templates, broken tool parsers,
   broken reasoning parsers, broken multi-turn handling — **before** you
   spend an hour on Phases 3–4 with a misconfigured engine.

2. **Phase 2 (25 min)** measures pure decode throughput across 9
   `(concurrency × prefix-context)` cells. Catches throughput regressions
   that don't surface as correctness errors. The 3×3 grid covers the
   regimes that matter:
   - `c=1, ctx=0` — single-user latency baseline
   - `c=4, ctx=131072` — agent workload at near-max context
   - `c=2, ctx=32000` — typical chat workload mid-range

3. **Phases 3 & 4** measure end-to-end correctness on canonical benchmarks
   under realistic concurrent load (`c=8`).

If Phase 1 fails, Phases 2–4 are likely meaningless. If Phase 2 shows
throughput regression but Phases 3–4 pass, you've isolated the issue to
the engine layer. This separation is what makes regressions diagnosable.

## Methodology constants

These are **deliberate** choices, not accidents:

| Constant | Value | Why |
|----------|-------|-----|
| Phase 2 duration | 60 s | Drowns out CUDA-graph jitter; <60 s shows bimodal distributions |
| Phase 2 warmup | 20 s | Engine reaches steady-state KV occupancy by ~15 s |
| Phase 2 N reseeded runs | 2 | Sufficient for 2σ regression detection at 9 cells × ~5 configs |
| Phase 3/4 concurrency | 8 | Saturates prefill on dual Blackwell; exposes prefix-cache pressure |
| Phase 3/4 max_tokens | 8192 | Budget for `<think>` reasoning + final code (Qwen3 reasoning is verbose) |
| Phase 3/4 timeout | 600 s | p99 of correct generations; rare slow cases not artificially failed |

If you tune any of these, **document the change** in your output bundle's
README. Cross-config comparisons require methodology constants to match.

## Usage

```bash
# Launch your container first, e.g.:
NUM_SPEC=3 ../launchers/launch_fp8_mtp.sh

# Then run the harness:
./four_phase_harness.sh \
  fp8_mtp3_baseline \
  1178863 \
  ./out/fp8_mtp3_baseline
```

The `KV_BUDGET` arg is the GPU KV cache size in tokens, which `llm_decode_bench`
needs to size its prefix-context tests. It's printed in vLLM's startup log
as `GPU KV cache size: 1,178,863 tokens` — copy that value.

## Tunables (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `11435` | vLLM port |
| `MODEL` | `qwen3.6-27b` | Model id sent in `/v1/chat/completions` |
| `BENCH` | `python3` | Python executable for `llm_decode_bench` |
| `SCRIPT` | `./llm_decode_bench.py` | Path to `llm_decode_bench.py` |
| `HARNESS` | `../harness/stress_harness.py` | Path to stress harness |
| `PROBLEMS_DIR` | `./problems` | Directory containing `humaneval.jsonl` and `mbpp.jsonl` |
