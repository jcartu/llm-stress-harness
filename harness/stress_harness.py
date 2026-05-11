#!/usr/bin/env python3
"""Stress-test harness for hard coding benchmarks against vLLM endpoint.

Hammers the model with concurrent requests, tracks every kind of failure
(crashes, timeouts, malformed output, test failures, throughput decay),
and produces machine-readable JSON output.
"""
import argparse, asyncio, json, os, re, signal, subprocess, sys, time, traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import aiohttp

# ---------- Test execution sandbox ----------

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def extract_code(text: str) -> Optional[str]:
    """Extract the largest Python code block from text. Returns None if none found."""
    if not text:
        return None
    matches = CODE_FENCE_RE.findall(text)
    if matches:
        # Pick the longest extracted block
        return max(matches, key=len).strip()
    # Fallback: look for `def ...(...)` patterns
    if 'def ' in text:
        return text.strip()
    return None

def smart_glue_humaneval(code: str, prompt: str) -> str:
    """If model returned body-only (no 'def'), prepend the prompt signature and
    re-indent every body line to 4-space under the def. This fixes a long-standing
    bug where '.strip()' would strip leading indent from line 1, causing the body
    to be parsed as a top-level statement followed by indented continuations,
    yielding IndentationError. See ADDENDUM.md in qwen-bench-2026-05-11-v2-followup.
    """
    if 'def ' in code:
        return code
    indent = '    '
    fixed = []
    for line in code.split('\n'):
        if line.strip() == '':
            fixed.append('')
        elif line.startswith('    ') or line.startswith('\t'):
            fixed.append(line)
        else:
            fixed.append(indent + line)
    return prompt.rstrip() + '\n' + '\n'.join(fixed)

def run_test(code: str, test_code: str, entry_point: str, timeout_s: int = 10) -> dict:
    """Run code + tests in a subprocess. Return {pass, error, stdout, stderr}."""
    full = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True, text=True, timeout=timeout_s,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        return {
            "pass": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:2000],
        }
    except subprocess.TimeoutExpired:
        return {"pass": False, "returncode": -1, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"pass": False, "returncode": -1, "stdout": "", "stderr": f"EXCEPTION: {e}"}

def run_mbpp_test(code: str, tests: list[str], timeout_s: int = 10) -> dict:
    """MBPP-style: just exec code + each assert."""
    full = code + "\n\n" + "\n".join(tests) + "\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True, text=True, timeout=timeout_s,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        return {
            "pass": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:2000],
        }
    except subprocess.TimeoutExpired:
        return {"pass": False, "returncode": -1, "stdout": "", "stderr": "TIMEOUT"}
    except Exception as e:
        return {"pass": False, "returncode": -1, "stdout": "", "stderr": f"EXCEPTION: {e}"}

# ---------- Records ----------

@dataclass
class ProblemResult:
    task_id: str
    benchmark: str
    config_label: str
    
    # HTTP status
    http_status: Optional[int] = None
    http_error: Optional[str] = None
    
    # Timing
    request_start_ts: float = 0.0
    request_end_ts: float = 0.0
    elapsed_s: float = 0.0
    
    # Response shape
    finish_reason: Optional[str] = None
    completion_tokens: int = 0
    prompt_tokens: int = 0
    reasoning_chars: int = 0
    content_chars: int = 0
    
    # Code extraction
    code_extracted: bool = False
    code_chars: int = 0
    
    # Test execution
    test_run: bool = False
    test_passed: bool = False
    test_returncode: Optional[int] = None
    test_stderr_head: str = ""
    
    # Failure mode classification
    failure_mode: str = "ok"   # ok | http_error | timeout | empty_response | no_code | test_fail | exception
    
    # Raw content (truncated for log size)
    content_head: str = ""
    reasoning_head: str = ""

# ---------- Async dispatcher ----------

async def call_endpoint(session, url, model, messages, max_tokens, temperature, timeout_s, extra_body=None):
    payload = {
        "model": model, "messages": messages,
        "temperature": temperature, "top_p": 1.0,
        "max_tokens": max_tokens, "seed": 42,
    }
    if extra_body:
        payload.update(extra_body)
    try:
        t0 = time.time()
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            t1 = time.time()
            text = await r.text()
            return {"status": r.status, "elapsed": t1-t0, "text": text}
    except asyncio.TimeoutError:
        return {"status": -1, "elapsed": timeout_s, "text": "", "error": "TIMEOUT"}
    except Exception as e:
        return {"status": -2, "elapsed": time.time()-t0, "text": "", "error": str(e)}

def build_humaneval_prompt(prob):
    p = prob['prompt']
    return [{
        "role": "user",
        "content": f"Complete the following Python function. Output the COMPLETE function (signature + body) inside a ```python fenced block. No explanation, just the code.\n\n```python\n{p}```",
    }]

def build_mbpp_prompt(prob):
    desc = prob['prompt']
    tests_hint = "\n".join(prob.get('test_list', [])[:3])
    return [{
        "role": "user",
        "content": f"Write a Python function to solve this task. Output ONLY the function inside a ```python fenced block, no explanation.\n\nTask: {desc}\n\nThe function should pass tests like:\n{tests_hint}",
    }]

async def run_one(session, prob, benchmark, config_label, url, model, max_tokens, request_timeout, sem):
    async with sem:
        if benchmark == "humaneval":
            messages = build_humaneval_prompt(prob)
            task_id = prob['task_id']
        else:  # mbpp
            messages = build_mbpp_prompt(prob)
            task_id = f"MBPP/{prob['task_id']}"
        
        result = ProblemResult(task_id=task_id, benchmark=benchmark, config_label=config_label,
                                request_start_ts=time.time())
        
        resp = await call_endpoint(session, url, model, messages, max_tokens, 0.0, request_timeout)
        result.request_end_ts = time.time()
        result.elapsed_s = resp.get('elapsed', 0)
        result.http_status = resp['status']
        
        if resp['status'] != 200:
            result.failure_mode = "http_error" if resp['status'] > 0 else ("timeout" if resp.get('error') == "TIMEOUT" else "exception")
            result.http_error = resp.get('error') or resp.get('text', '')[:500]
            return result
        
        try:
            j = json.loads(resp['text'])
        except Exception as e:
            result.failure_mode = "exception"
            result.http_error = f"JSON parse: {e}"
            return result
        
        if 'choices' not in j or not j['choices']:
            result.failure_mode = "empty_response"
            return result
        
        msg = j['choices'][0]['message']
        result.finish_reason = j['choices'][0].get('finish_reason')
        result.completion_tokens = j.get('usage', {}).get('completion_tokens', 0)
        result.prompt_tokens = j.get('usage', {}).get('prompt_tokens', 0)
        content = (msg.get('content') or '').strip()
        reasoning = (msg.get('reasoning') or '').strip()
        result.content_chars = len(content)
        result.reasoning_chars = len(reasoning)
        result.content_head = content[:500]
        result.reasoning_head = reasoning[:500]
        
        if not content:
            result.failure_mode = "empty_response"
            return result
        
        code = extract_code(content)
        if not code:
            result.failure_mode = "no_code"
            return result
        result.code_extracted = True
        result.code_chars = len(code)
        
        # Run tests in process pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        if benchmark == "humaneval":
            glued = smart_glue_humaneval(code, prob['prompt'])
            result.code_chars = len(glued)
            test_result = await loop.run_in_executor(None, run_test, glued, prob['test'], prob['entry_point'])
        else:  # mbpp
            test_result = await loop.run_in_executor(None, run_mbpp_test, code, prob.get('test_list', []))
        
        result.test_run = True
        result.test_passed = test_result['pass']
        result.test_returncode = test_result['returncode']
        result.test_stderr_head = (test_result.get('stderr','') or '')[:500]
        result.failure_mode = "ok" if test_result['pass'] else "test_fail"
        return result

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:11435/v1/chat/completions')
    ap.add_argument('--model', default='qwen3.6-27b')
    ap.add_argument('--config-label', required=True)
    ap.add_argument('--benchmark', choices=['humaneval', 'mbpp'], required=True)
    ap.add_argument('--problems-file', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--request-timeout', type=int, default=300)
    ap.add_argument('--limit', type=int, default=0, help='limit problem count for smoke testing')
    args = ap.parse_args()
    
    problems = []
    with open(args.problems_file) as f:
        for line in f:
            problems.append(json.loads(line))
    if args.limit > 0:
        problems = problems[:args.limit]
    
    print(f"[{time.strftime('%H:%M:%S')}] Stress harness starting")
    print(f"  config: {args.config_label}")
    print(f"  benchmark: {args.benchmark} ({len(problems)} problems)")
    print(f"  endpoint: {args.url}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  request_timeout: {args.request_timeout}s")
    
    sem = asyncio.Semaphore(args.concurrency)
    sprint_start = time.time()
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            run_one(session, p, args.benchmark, args.config_label,
                    args.url, args.model, args.max_tokens, args.request_timeout, sem)
            for p in problems
        ]
        # Stream results as they complete
        completed = 0
        for fut in asyncio.as_completed(tasks):
            r = await fut
            completed += 1
            results.append(r)
            elapsed_total = time.time() - sprint_start
            mark = '✅' if r.failure_mode == 'ok' else ('❌' if r.failure_mode == 'test_fail' else f'🚨[{r.failure_mode}]')
            print(f"  [{completed:3d}/{len(problems)}] {mark} {r.task_id} fr={r.finish_reason} t={r.elapsed_s:.1f}s")
            # Save incrementally so we don't lose data on crash
            with open(args.output, 'w') as f:
                for r2 in results:
                    f.write(json.dumps(asdict(r2)) + '\n')
    
    elapsed_total = time.time() - sprint_start
    
    # Summary
    summary = {
        'config_label': args.config_label,
        'benchmark': args.benchmark,
        'total_problems': len(problems),
        'completed': len(results),
        'sprint_wall_time_s': elapsed_total,
        'failure_mode_breakdown': {},
        'pass_rate': 0,
        'mean_elapsed_s': 0,
        'median_elapsed_s': 0,
        'p95_elapsed_s': 0,
        'mean_completion_tokens': 0,
        'mean_reasoning_chars': 0,
        'effective_tps': 0,
    }
    
    from collections import Counter
    fm = Counter(r.failure_mode for r in results)
    summary['failure_mode_breakdown'] = dict(fm)
    
    passed = sum(1 for r in results if r.test_passed)
    summary['pass_rate'] = passed / max(len(results), 1)
    
    times = sorted([r.elapsed_s for r in results if r.elapsed_s > 0])
    if times:
        import statistics
        summary['mean_elapsed_s'] = statistics.mean(times)
        summary['median_elapsed_s'] = statistics.median(times)
        summary['p95_elapsed_s'] = times[int(len(times)*0.95)] if len(times) > 20 else times[-1]
    
    total_tokens = sum(r.completion_tokens for r in results)
    summary['mean_completion_tokens'] = total_tokens / max(len(results), 1)
    summary['mean_reasoning_chars'] = sum(r.reasoning_chars for r in results) / max(len(results), 1)
    summary['effective_tps'] = total_tokens / elapsed_total if elapsed_total > 0 else 0
    
    summary_path = args.output.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  Pass rate: {passed}/{len(results)} = {summary['pass_rate']*100:.1f}%")
    print(f"  Failure modes: {dict(fm)}")
    print(f"  Mean elapsed: {summary['mean_elapsed_s']:.1f}s, p95: {summary['p95_elapsed_s']:.1f}s")
    print(f"  Effective tok/s: {summary['effective_tps']:.1f}")
    print(f"  Saved: {args.output}")
    print(f"  Summary: {summary_path}")

if __name__ == '__main__':
    asyncio.run(main())
