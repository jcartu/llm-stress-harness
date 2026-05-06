#!/usr/bin/env python3
"""Generate hero & section images via Gemini 3.1 nano banana."""
import os, sys, pathlib
from google import genai
from google.genai import types

OUT = pathlib.Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-3.1-flash-image-preview"

PROMPTS = {
    "hero": (
        "A wide cinematic 16:9 hero banner for an open-source LLM benchmarking toolkit. "
        "Dual NVIDIA Blackwell-class data-center GPUs glowing teal-cyan inside a dark server chassis, "
        "with translucent overlays of throughput line graphs, latency histograms, and a 3x3 grid of "
        "performance heatmap cells floating in front of them. Subtle neon-green token streams flowing "
        "between the GPUs like data packets. Deep navy and charcoal background, electric cyan and lime "
        "highlights, sharp technical poster aesthetic, no text, no logos, no watermarks. "
        "Style: precise, clean, modern technical illustration meets photorealism."
    ),
    "phases": (
        "A clean horizontal infographic illustration showing four sequential diagnostic phases as glowing "
        "gates. Phase 1 = a small green checkmark gate (functional gates), Phase 2 = a 3x3 grid of "
        "throughput heatmap cells, Phase 3 = a Python function silhouette with a pass/fail badge "
        "(HumanEval), Phase 4 = a stack of unit-test tiles (MBPP). Connected left-to-right by a glowing "
        "cyan token stream. Dark navy background, neon teal and lime accents, isometric technical "
        "diagram style, no text labels, no watermarks."
    ),
    "failure_taxonomy": (
        "A scientific diagnostic illustration: a single LLM response being sorted through a translucent "
        "decision-tree funnel into six labeled bins below. The bins glow different colors: green (pass), "
        "amber (no_code), red (timeout), purple (empty_response), orange (http_error), gray (other_fail). "
        "Above the funnel, a stylized neural network silhouette emits glowing token streams. Dark "
        "background, crisp neon outlines, technical infographic aesthetic, no text or labels rendered, "
        "no watermarks."
    ),
    "speculative": (
        "A technical illustration of speculative decoding: a small fast 'drafter' neural network "
        "(rendered as a compact glowing teal mesh) generating multiple candidate token bubbles in "
        "parallel, with a larger 'verifier' neural network (rendered as a deeper indigo mesh) "
        "accepting some tokens (glowing green) and rejecting others (faded red). Tokens flow "
        "left-to-right across the frame as a luminous stream. Dark background, cinematic depth, "
        "no text or labels, no watermarks."
    ),
}

for name, prompt in PROMPTS.items():
    out_path = OUT / f"{name}.png"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[skip] {out_path} exists ({out_path.stat().st_size} bytes)")
        continue
    print(f"[gen]  {name}: {prompt[:80]}...")
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        wrote = False
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                out_path.write_bytes(part.inline_data.data)
                print(f"[ok]   {out_path} ({out_path.stat().st_size} bytes)")
                wrote = True
                break
        if not wrote:
            print(f"[warn] no image data returned for {name}")
            print(f"       response text: {getattr(resp, 'text', None)}")
    except Exception as e:
        print(f"[err]  {name}: {e}", file=sys.stderr)
        sys.exit(1)

print("done.")
