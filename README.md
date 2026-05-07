[![← qwen-bench hub](https://img.shields.io/badge/%E2%86%90-qwen--bench_hub-blueviolet?style=for-the-badge)](https://github.com/jcartu/qwen-bench)

> Part of the [`qwen-bench`](https://github.com/jcartu/qwen-bench) ongoing benchmark series.
> See the hub for the current SOTA leaderboard and a chronological index of all studies.

---

<div align="center">

<img src="docs/images/hero.png" alt="LLM Stress Harness — diagnostic instrumentation for self-hosted inference" width="100%" />

# `llm-stress-harness`

### A diagnostic toolkit for self-hosted LLM inference stacks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Made with asyncio](https://img.shields.io/badge/asyncio-native-blueviolet?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/asyncio.html)
[![vLLM compatible](https://img.shields.io/badge/vLLM-compatible-orange?style=for-the-badge)](https://github.com/vllm-project/vllm)
[![SGLang compatible](https://img.shields.io/badge/SGLang-compatible-red?style=for-the-badge)](https://github.com/sgl-project/sglang)

> **Tools for separating models that *look* good in chat from models that *are* good under load.**

[What is this?](#what-is-this) ·
[Who is this for?](#who-is-this-for) ·
[What's in the box](#whats-in-the-box) ·
[Quickstart](#quickstart) ·
[Failure taxonomy](#the-failure-taxonomy) ·
[Methodology](#methodology--statistical-rigor)

</div>

---

## What is this?

`llm-stress-harness` is a small, opinionated toolkit for **measuring whether
a self-hosted LLM inference deployment actually works** — not on a marketing
slide, not on a single curated prompt, but under the kind of concurrent,
long-context, real-shape load that breaks production systems.

It is the distilled instrument behind the
[**Qwen3.6-27B Blackwell stress-validation study**][study] — a multi-week
production validation of `Qwen3.6-27B-FP8` on `2× RTX Pro 6000` (Blackwell)
across 8 speculative-decoding configurations. Every chart, every failure mode,
every regression isolation in that report came out of the four scripts in
this repo.

The toolkit answers four questions, in order of cost-to-execute:

1. **Does the endpoint *work*?** (chat templates intact, tool calls parseable, reasoning preserved)
2. **Is it *fast*?** (decode throughput across a concurrency × prefix-context grid)
3. **Is the model *correct*?** (HumanEval, 164 problems under load)
4. **Is the model *robust*?** (MBPP, 257 problems, larger surface area)

[study]: https://github.com/jcartu/qwen36-27b-blackwell-stress-validation

<div align="center">
<img src="docs/images/phases.png" alt="Four-phase diagnostic ladder" width="85%" />
<br/>
<em>The four-phase diagnostic ladder. Each phase is more expensive than the last; each catches a class of failures the previous phase missed.</em>
</div>

---

## Who is this for?

| You are… | This toolkit is for you if… |
|---|---|
| **An ML/infra engineer** running self-hosted vLLM, SGLang, TensorRT-LLM, or any OpenAI-compatible endpoint | You need evidence (not vibes) that a config change improved or regressed the deployment |
| **A model evaluator** comparing quantizations (FP8 vs BF16), draft strategies (MTP vs DFlash), or batch sizes | You want apples-to-apples numbers across configs, not single-run leaderboard scores |
| **A reliability engineer** investigating intermittent failures in an agent platform | You need to classify *why* requests fail, not just count that they did |
| **A researcher** publishing inference-systems results | You need a reproducible methodology with documented constants, deterministic seeds, and forensic per-request telemetry |
| **An open-source maintainer** of an inference engine | You want a third-party reference implementation of "what good looks like" for stress testing |

This toolkit is **not** a leaderboard, **not** a chat-quality benchmark, and
**not** a sampling-based pass@k tool. It is a single-shot, deterministic
(`temperature=0`, `seed=42`), failure-taxonomic load tester.

---

## Why this exists

Most LLM benchmarks are *correctness probes*: they ask a model to solve a
problem, score the answer, average the scores, publish a leaderboard. They
treat the inference server as a black-box oracle that always returns a
syntactically-valid JSON body containing a syntactically-valid completion
containing a syntactically-valid code block.

That assumption holds for hosted, managed APIs (OpenAI, Anthropic, Google).
It does *not* hold for self-hosted inference stacks under realistic
production-shaped load. A 27B-parameter model running on a tensor-parallel
vLLM cluster with speculative decoding, FP8 quantization, prefix caching, and
flash-attention will fail in a *zoo* of subtle, partial, intermittent ways:

- The HTTP layer 502s because the engine ran out of KV cache mid-request
- The completion arrives but `content` is `null` because the model spent its
  entire token budget thinking inside `<think>` tags
- The response is well-formed but contains no fenced code block
- The code is fenced and well-formed but raises a `SyntaxError` on parse
- The code parses but throws `RecursionError` after 30 seconds of CPU
- Five concurrent requests pass, the sixth times out at 600 s
- The first 80 problems pass, then KV-cache pressure causes the 81st to
  silently truncate at `finish_reason="length"`

A "pass rate" number that collapses all of those into a single percentage
*hides the bugs you actually need to find.* This toolkit exists to refuse that
collapse — to **classify every request into one of seven mutually-exclusive
failure modes** and emit per-request telemetry so you can answer the question
*"why did config X regress from 73 % to 62 %?"* with evidence, not vibes.

<div align="center">
<img src="docs/images/failure_taxonomy.png" alt="Failure-mode classification funnel" width="80%" />
<br/>
<em>Every request is sorted into one of seven mutually-exclusive failure-mode bins. The aggregate pass rate is for dashboards; the breakdown is for engineers.</em>
</div>

---

## What's in the box

```
llm-stress-harness/
├── harness/
│   └── stress_harness.py         # 322-line async failure-taxonomic correctness probe
├── orchestrator/
│   ├── four_phase_harness.sh     # End-to-end 4-phase validation suite (~60 min/config)
│   └── README.md                 # Phase-by-phase methodology documentation
├── launchers/
│   ├── launch_fp8_mtp.sh         # FP8 + native MTP heads (production-recommended)
│   ├── launch_fp8_dflash.sh      # FP8 + BF16 DFlash drafter (peak throughput)
│   ├── launch_bf16_dflash.sh     # BF16 + BF16 DFlash drafter (quality reference)
│   └── README.md                 # Speculative-decoding configuration guide
├── utils/
│   └── wait_vllm_ready.sh        # Parameterized vLLM readiness probe with warm-up
├── docs/
│   └── images/                   # Hero & section illustrations
├── README.md                     # ← you are here
├── LICENSE                       # MIT
└── requirements.txt
```

### The four components, in plain English

#### 🔬 `harness/stress_harness.py` — *the microscope*
A 322-line single-file async correctness probe. Hits an OpenAI-compatible
chat endpoint with N concurrent requests, runs the returned code in
sandboxed subprocesses, classifies every request into one of seven failure
modes, and emits both per-request JSONL telemetry and an aggregate summary.
**This is the core instrument.** Everything else exists to feed it or
contextualize its output.

#### 🎼 `orchestrator/four_phase_harness.sh` — *the conductor*
A bash orchestrator that runs the full validation ladder against a
running endpoint: functional gates (60 s) → throughput matrix
(25 min, 9 cells × 2 reseeded runs) → HumanEval-164 (5 min) → MBPP-257
(25 min). Total wall time ≈ 1 hour per config on `2× RTX Pro 6000 Blackwell`.
Produces a single artifact bundle suitable for cross-config comparison.
**This is what you run when you want to characterize a config end-to-end.**

#### 🚀 `launchers/launch_*.sh` — *the test harness rigs*
Three parametric vLLM container templates, one per speculative-decoding
strategy that survived the production validation as Pareto-optimal:

- `launch_fp8_mtp.sh` — **the boring choice that ships.** FP8 base with
  native Multi-Token Prediction heads. No external drafter, no
  precision-mismatch segfaults, ~241 tok/s. This is what runs in production.
- `launch_fp8_dflash.sh` — **the experimental choice.** FP8 base with a
  separate BF16 DFlash drafter. ~244 tok/s (1 % faster than MTP=3) but more
  failure modes; documented `instanttensor` segfault.
- `launch_bf16_dflash.sh` — **the quality reference.** BF16 base with BF16
  DFlash drafter. ~22 % slower than FP8 but useful for isolating quantization
  effects from speculative-decoding configuration.

Every launcher is env-parameterized (`NUM_SPEC`, `BASE_MODEL`, `TP_SIZE`,
`IMAGE`, `PORT`, `NAME`, …) and embeds an inline readiness loop with
fast-fail on engine init crashes.

#### 🩺 `utils/wait_vllm_ready.sh` — *the readiness probe*
A 55-line bash script that polls `/v1/models`, watches container logs
for crash signatures, and adds a configurable post-ready settle window
(60 s default) for CUDA graph compilation to finish. **Use this between
launching a container and running benchmarks.** If you don't, you'll
measure cold-start jitter and blame the model.

<div align="center">
<img src="docs/images/speculative.png" alt="Speculative decoding — drafter proposes, verifier accepts/rejects" width="80%" />
<br/>
<em>Speculative decoding: a small drafter network proposes <code>k</code> candidate tokens, a larger verifier network accepts or rejects each. The launchers in this repo cover the three Pareto-optimal points on the throughput/stability frontier for this technique.</em>
</div>

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/jcartu/llm-stress-harness
cd llm-stress-harness
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch an endpoint (or skip if you already have one)

```bash
# Production-recommended: FP8 + MTP=3
NUM_SPEC=3 ./launchers/launch_fp8_mtp.sh

# Or, experimental: FP8 + DFlash N=8
NUM_SPEC=8 ./launchers/launch_fp8_dflash.sh

# Or, quality reference: BF16 + DFlash N=8
NUM_SPEC=8 ./launchers/launch_bf16_dflash.sh
```

### 3. Wait for warm-up

```bash
./utils/wait_vllm_ready.sh
# polls /v1/models, watches for crash signatures, adds 60s settle window
```

### 4a. Run a single benchmark

```bash
python harness/stress_harness.py \
  --url http://localhost:11435/v1/chat/completions \
  --model qwen3.6-27b \
  --config-label baseline \
  --benchmark humaneval \
  --problems-file problems/humaneval.jsonl \
  --output runs/baseline-humaneval.jsonl \
  --concurrency 8 \
  --max-tokens 8192 \
  --request-timeout 600
```

### 4b. Or run the full 4-phase suite

```bash
./orchestrator/four_phase_harness.sh \
  fp8_mtp3_baseline \
  1178863 \
  ./out/fp8_mtp3_baseline
#       ^                ^               ^
#       config label     KV budget       output dir
#                        (from vLLM
#                         startup log)
```

> **Problem files are not bundled.** Use the canonical
> [HumanEval][humaneval] (164 problems) and [MBPP][mbpp] (257 sanitized
> problems) JSONL files. Each line must contain `task_id`, `prompt`, and
> either `test` + `entry_point` (HumanEval-style) or `test_list` (MBPP-style).

[humaneval]: https://github.com/openai/human-eval
[mbpp]: https://github.com/google-research/google-research/tree/master/mbpp

---

## Design philosophy

Three commitments shape every line in this repo:

### 1. **Asynchronous all the way down, but bounded.**
The harness uses `aiohttp` + `asyncio.Semaphore` to maintain *exactly* `N`
concurrent in-flight HTTP requests against the endpoint. There is no thread
pool for I/O, no per-request task spawning beyond the semaphore-gated
coroutine. This faithfully simulates a load-balanced production deployment
where downstream code holds a connection open until the model responds —
which is the regime where speculative-decoding accept-rate degradations,
prefix-cache eviction storms, and CUDA-graph re-capture stalls actually
manifest. A `ThreadPoolExecutor`-based harness *cannot* see these because it
serializes requests behind the GIL during JSON parsing.

### 2. **Sandboxed test execution must not block the event loop.**
Generated code is executed in `subprocess.run` calls — not `exec()`, not
`ast.literal_eval`, not a Python sandbox library. We chose subprocess
isolation for three reasons: (a) malicious or runaway code (`while True:`,
fork bombs, `os.system('rm -rf /')`) cannot escape; (b) `SIGALRM`-based
timeouts work cleanly on a child process; (c) memory leaks from
re-importing user code do not accumulate in the harness process. The
`subprocess.run` call is wrapped in `loop.run_in_executor(None, ...)` so the
event loop continues dispatching new HTTP requests during the ~10 s test
window. Without this, a single slow test would freeze all `N`
concurrent slots.

### 3. **Incremental persistence. Crashes are first-class events.**
After every completed problem, the *entire* result set is rewritten to
`args.output` as JSONL. This is `O(n²)` in I/O but `n` is bounded by ~257
(MBPP) or ~164 (HumanEval), so the cost is negligible (~50 ms per write at
257 records). The benefit is total: if the GPU OOMs at problem 213, you
keep the data for problems 1–212 and restart with `--limit` adjusted. We
have lost zero benchmark runs to crashes since adopting this pattern.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ARGPARSE / I/O                                  │
│   --url --model --benchmark --concurrency --max-tokens --request-timeout    │
│                              --problems-file                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     ASYNCIO.SEMAPHORE(concurrency)                           │
│              ┌─────────────────────────────────────────┐                     │
│              │    asyncio.as_completed(N tasks)        │                     │
│              └─────────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
                  ▼                                     ▼
       ┌──────────────────┐                  ┌──────────────────┐
       │   HTTP STAGE     │                  │   HTTP STAGE     │   ... × N
       │  aiohttp.post    │                  │  aiohttp.post    │
       │  POST /v1/chat   │                  │  POST /v1/chat   │
       └──────────────────┘                  └──────────────────┘
                  │                                     │
                  ▼                                     ▼
       ┌──────────────────┐                  ┌──────────────────┐
       │  PARSE STAGE     │                  │  PARSE STAGE     │
       │  json.loads,     │                  │  classify        │
       │  extract code    │                  │  failure_mode    │
       └──────────────────┘                  └──────────────────┘
                  │                                     │
                  ▼                                     ▼
       ┌──────────────────┐                  ┌──────────────────┐
       │  EXEC STAGE      │                  │  EXEC STAGE      │
       │  subprocess.run  │                  │  (in default     │
       │  in executor     │                  │   thread pool)   │
       └──────────────────┘                  └──────────────────┘
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
              ┌──────────────────────────────────────────┐
              │   ProblemResult (dataclass) → JSONL      │
              │   incremental write to args.output       │
              └──────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────┐
              │    Aggregate → summary.json              │
              │    pass_rate · p95 latency · TPS         │
              │    failure_mode_breakdown                │
              └──────────────────────────────────────────┘
```

### Three-stage pipeline per problem

Every problem flows through exactly three stages, each of which can fail
independently and must be classified separately for diagnostic value:

| Stage | What it does | Time budget | Can fail with |
|-------|--------------|-------------|---------------|
| **HTTP** | `POST /v1/chat/completions` with `aiohttp.ClientTimeout(total=…)` | `--request-timeout` (default 300 s) | `http_error`, `timeout`, `exception` |
| **Parse** | `json.loads` body → walk `choices[0].message` → regex-extract fenced code | <10 ms | `empty_response`, `no_code`, `exception` |
| **Exec** | Splice code + tests into a string → `subprocess.run` with stdin closed, `PYTHONDONTWRITEBYTECODE=1` | 10 s (hardcoded) | `test_fail` |

Why hardcode the test timeout to 10 s? Because the **median MBPP/HumanEval
test runs in <100 ms** and the **p99 runs in <2 s**; anything past 10 s is
diagnostic of generated code with `O(2ⁿ)` recursion or an infinite loop, not
a slow-but-correct solution. You want this to be a hard ceiling, not a
configurable knob you'll tune away to "improve" results.

---

## The failure taxonomy

This is the conceptual heart of the harness. Every request is classified
into exactly one of seven `failure_mode` values, ordered roughly from
"infrastructure broken" to "model is wrong":

```
ok ────────────────────────────────────────────────► test passed, code correct
test_fail ─────────────────────────────────────────► code ran, assertions failed
no_code ───────────────────────────────────────────► response had no fenced ```python``` block
empty_response ────────────────────────────────────► HTTP 200 but content="" (often: thinking ate budget)
http_error ────────────────────────────────────────► HTTP 4xx/5xx response
timeout ───────────────────────────────────────────► aiohttp.ClientTimeout fired before response
exception ─────────────────────────────────────────► JSON parse error or other client-side fault
```

These are *not* arbitrary buckets. They form a strict diagnostic ladder:

> #### **A regression in `ok` rate is meaningless without seeing which non-`ok` mode absorbed the loss.**

A drop from 88.7 % → 73.2 % could mean any of:

- **+15 pp `test_fail`** → the model got dumber (real correctness regression)
- **+15 pp `no_code`** → the chat template / reasoning parser changed
- **+15 pp `empty_response`** → `<think>` budget overflowed (raise `max_tokens`)
- **+15 pp `timeout`** → the engine is slower (KV pressure, spec accept rate dropped)
- **+15 pp `http_error`** → infrastructure broke (OOM, NCCL hang, container restarted)

The same headline number, five entirely different root causes, five entirely
different fixes. This is why every released summary includes
`failure_mode_breakdown` as a first-class field. The aggregate is for
dashboards; the breakdown is for engineers.

---

## Output schema

### Per-problem JSONL (`args.output`)

Each line is a fully-typed `ProblemResult` dataclass serialized via `asdict`:

```jsonc
{
  "task_id": "HumanEval/42",
  "benchmark": "humaneval",
  "config_label": "baseline",

  // HTTP layer
  "http_status": 200,
  "http_error": null,

  // Wall-clock timing
  "request_start_ts": 1746579812.341,
  "request_end_ts":   1746579814.918,
  "elapsed_s": 2.577,

  // Response shape
  "finish_reason": "stop",         // "stop" | "length" | "tool_calls" | null
  "completion_tokens": 287,
  "prompt_tokens": 162,
  "reasoning_chars": 933,           // <think>...</think> contents
  "content_chars": 412,             // post-think assistant content

  // Code extraction
  "code_extracted": true,
  "code_chars": 312,

  // Sandboxed execution
  "test_run": true,
  "test_passed": true,
  "test_returncode": 0,
  "test_stderr_head": "",

  // Diagnostic classification
  "failure_mode": "ok",

  // Truncated raw content (for forensic review)
  "content_head":   "def add_one(x):\n    return x + 1\n",
  "reasoning_head": "Let me think about this..."
}
```

### Aggregate summary (`<output>_summary.json`)

```jsonc
{
  "config_label": "baseline",
  "benchmark": "humaneval",
  "total_problems": 164,
  "completed": 164,
  "sprint_wall_time_s": 287.4,
  "failure_mode_breakdown": {
    "ok": 120,
    "test_fail": 38,
    "no_code": 4,
    "empty_response": 2
  },
  "pass_rate": 0.7317,
  "mean_elapsed_s": 7.92,
  "median_elapsed_s": 6.31,
  "p95_elapsed_s": 18.4,
  "mean_completion_tokens": 491.2,
  "mean_reasoning_chars": 1820.6,
  "effective_tps": 280.1
}
```

`effective_tps` is `Σ completion_tokens / wall_time` — the *throughput
the user actually saw under concurrent load*, not the per-request decode
speed reported by `vllm bench` or `llm-decode-bench`. These two numbers
diverge significantly under speculative decoding and are *both* useful
metrics; we report the user-observed one.

---

## Methodology & statistical rigor

### Determinism
- `temperature=0.0`, `top_p=1.0`, `seed=42` are passed in every payload.
- For models where `temperature=0.0` is silently rounded up to a small
  epsilon (some MoE backends do this), bit-identical runs are not
  guaranteed; in practice the variance across 3 reseeded runs at `t=0` is
  <1 problem on HumanEval-164 and <2 problems on MBPP-257.

### Concurrency choice
We default to `--concurrency 8` for HumanEval and MBPP. Rationale:
- `c=1` measures latency, not steady-state throughput
- `c=4` does not saturate a 2× Blackwell tensor-parallel deployment
- `c=8` saturates the prefill stage and exposes prefix-cache pressure
- `c=16+` causes head-of-line blocking that is *not* representative of
  agent-style production loads

If you're benchmarking a chat-style deployment with c=1 expected, set
`--concurrency 1`. If you're stress-testing for a swarm-of-agents
deployment, set `--concurrency 32+` and watch `failure_mode_breakdown` light up.

### What this harness does *not* measure

We deliberately do *not*:

- Compute pass@k for k>1. Generation is single-shot at `t=0`. If you need
  sampling-based pass@k, this is the wrong tool.
- Measure prefill-only or decode-only throughput. Use `llm-decode-bench`
  for that. This is an *end-to-end* harness.
- Score reasoning quality. We capture `reasoning_chars` for diagnostic
  purposes, but do not grade `<think>` content.
- Verify model identity via fingerprint. If your endpoint serves the wrong
  model, you'll get bizarrely-low pass rates and need to investigate
  manually. Future work: log `system_fingerprint` per request.

---

## Diagnostic patterns observed in the wild

Real failure modes seen during 8-config validation of Qwen3.6-27B:

| Symptom | Failure-mode signature | Root cause | Fix |
|---------|------------------------|------------|-----|
| Pass rate drops 88 → 65 % | `+25 pp empty_response` | `--reasoning-parser qwen3` + low `max_tokens` → `<think>` ate the budget | Raise `--max-tokens` to 8192+ |
| Pass rate drops 73 → 0 % | `+100 pp http_error` | Container died at problem 47 (KV OOM) | Lower `--max-num-batched-tokens` |
| Pass rate identical, p95 doubles | unchanged distribution, latency shift | Spec-decoding accept rate dropped from 0.85 → 0.55 | Reconfigure draft model; check accept-rate metric |
| `no_code` rate spikes | `+10 pp no_code` | Chat template stripped fenced code blocks | Verify `--chat-template` and `--tool-call-parser` flags |
| Sporadic `timeout` only at high concurrency | `c=1: 0%, c=8: 5%, c=16: 22%` timeouts | Prefix-cache thrashing under concurrent prefill | Bump KV budget or lower concurrency |

These patterns are not theoretical. Every row above corresponds to a real
incident from the [stress-validation report][study].

---

## Mathematical notes

### Throughput vs. accept rate (speculative decoding)

For a speculative-decoding deployment with draft length `k` and acceptance
rate `α(i)` for position `i ∈ [1, k]`, the expected accepted tokens per
draft-verify cycle is:

$$
\mathbb{E}[\text{accepted}] = 1 + \sum_{i=1}^{k} \prod_{j=1}^{i} \alpha(j)
$$

This harness does not compute `α` directly (that's an engine-internal
metric exposed via `/metrics` for vLLM), but `effective_tps` divided by the
non-speculative baseline TPS is a useful proxy for *delivered* speedup —
which is what users feel — vs. the *theoretical* speedup that
`mean_acceptance_length × k` would predict.

### Why `effective_tps = Σ tokens / wall_time` and not per-request mean

Given `N` concurrent requests with completion times `t_i` and token
counts `n_i`, one could report:

- **Per-request throughput:** $\bar{r} = \frac{1}{N} \sum_{i=1}^{N} \frac{n_i}{t_i}$
- **Aggregate throughput:** $r_{\text{agg}} = \frac{\sum_i n_i}{\max_i t_i}$

We report `r_agg`. Reason: $\bar{r}$ over-weights short-completion fast
requests (which are easy because the engine is half-idle) and is a *worse*
predictor of how a production deployment will perform. `r_agg` is what your
infra bill is paying for.

---

## Subdirectory documentation

For deeper material on each component:

- [`harness/`](harness/) — the 322-line probe itself
- [`orchestrator/README.md`](orchestrator/README.md) — phase-by-phase methodology, output bundle layout, wall-time budget
- [`launchers/README.md`](launchers/README.md) — Pareto frontier of speculative-decoding configs, common-parameter table, override points
- [`utils/`](utils/) — readiness-probe internals

---

## Acknowledgments & related work

This harness was built atop and inspired by:

- [**vLLM**](https://github.com/vllm-project/vllm) — the inference engine
  these probes were designed against
- [**HumanEval**](https://github.com/openai/human-eval) — Chen et al. 2021
- [**MBPP**](https://arxiv.org/abs/2108.07732) — Austin et al. 2021
- [**EvalPlus**](https://github.com/evalplus/evalplus) — Liu et al. 2023
  (more rigorous test suites; this harness uses canonical HE/MBPP for
  cross-paper comparability)

The hero and section illustrations were generated with **Gemini 3.1 nano
banana** (`gemini-3.1-flash-image-preview`).

---

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

**Built during the May 2026 Qwen3.6-27B Blackwell production validation.**

If this saves you a debugging session, [⭐ the repo](https://github.com/jcartu/llm-stress-harness)
and [open an issue](https://github.com/jcartu/llm-stress-harness/issues)
about the failure mode you encountered. The taxonomy will grow.

</div>
