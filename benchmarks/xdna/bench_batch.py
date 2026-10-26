"""Concurrent requests against the running server (./serve.sh), to see how vLLM's batching scales.

decode: B requests at once, short distinct prompts, N tokens each (ignore_eos).
  per-request rate = (n - 1) / (last token - first token); aggregate = all tokens
  generated while every request was decoding / that window.
prefill: B distinct L-token prompts at once, 1 token each; aggregate = B * L / time
  until the last first token.
Each prompt starts with its own number, so the prefix cache cannot share them.

usage: python bench_batch.py [--decode 1,2,4,8,16,32] [--prefill 1,2,4] [--tokens 128] [--len 1860]
"""
import argparse
import asyncio
import json
import os
import statistics
import time

import httpx
from transformers import AutoTokenizer

MODEL = "google/gemma-4-E2B-it"
URL = os.environ.get("URL", "http://127.0.0.1:8013/v1/completions")
NAME = os.environ.get("SERVED", "gemma-4-e2b-int8-npu")
TEXT = open(os.path.expanduser("~/src/gemma-npu/filler.txt")).read()


def prompts(tok, B, n_tokens, salt):
    body = tok.decode(tok(TEXT)["input_ids"][:n_tokens])
    out = []
    for i in range(B):
        msg = f"Document {salt}-{i}.\n\n{body}\n\nSummarize the text above in detail."
        ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                      return_dict=True)["input_ids"]
        out.append(list(ids))
    return out


async def one(client, ids, n):
    """-> (t_first, t_last, tokens, [(t, completion_tokens)]) relative to the call."""
    body = {"model": NAME, "prompt": ids, "max_tokens": n, "min_tokens": n, "ignore_eos": True,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    t0 = time.perf_counter()
    marks = []
    async with client.stream("POST", URL, json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            u = d.get("usage")
            if u and d.get("choices"):
                marks.append((time.perf_counter() - t0, u["completion_tokens"]))
    return t0, marks


async def run(B, ids_list, n):
    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=B + 4)) as client:
        start = time.perf_counter()
        res = await asyncio.gather(*(one(client, ids, n) for ids in ids_list))
    # Absolute times from the batch start.
    reqs = [[(t0 - start + t, k) for t, k in marks] for t0, marks in res]
    return reqs


def decode_stats(reqs):
    firsts = [m[0][0] for m in reqs]
    lasts = [m[-1][0] for m in reqs]
    per = [(m[-1][1] - m[0][1]) / (m[-1][0] - m[0][0]) for m in reqs]
    lo, hi = max(firsts), min(lasts)  # every request is decoding in [lo, hi]

    def tokens_at(m, t):
        k = 0
        for tt, kk in m:
            if tt <= t:
                k = kk
        return k

    agg = sum(tokens_at(m, hi) - tokens_at(m, lo) for m in reqs) / (hi - lo) if hi > lo else float("nan")
    return statistics.median(per), agg, max(firsts), max(lasts)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", default="1,2,4,8,16,32")
    ap.add_argument("--prefill", default="1,2,4")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--short", type=int, default=190, help="decode prompt length (filler tokens)")
    ap.add_argument("--len", type=int, default=1860, help="prefill prompt length (filler tokens)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    salt = int(time.time())

    await run(1, prompts(tok, 1, 32, f"warm{salt}"), 8)
    for B in [int(b) for b in args.decode.split(",") if b]:
        ids = prompts(tok, B, args.short, f"d{salt}-{B}")
        reqs = await run(B, ids, args.tokens)
        per, agg, first, last = decode_stats(reqs)
        print(f"decode B={B:3d} prompt {len(ids[0])}: per-request {per:5.1f} tok/s, aggregate {agg:6.1f} tok/s, "
              f"last first-token {first * 1e3:6.0f} ms, all done {last:5.1f} s", flush=True)
    for B in [int(b) for b in args.prefill.split(",") if b]:
        ids = prompts(tok, B, args.len, f"p{salt}-{B}")
        reqs = await run(B, ids, 1)
        ttfts = sorted(m[0][0] for m in reqs)
        total = sum(len(i) for i in ids)
        print(f"prefill B={B} x {len(ids[0])} tokens: TTFT first {ttfts[0] * 1e3:5.0f} ms, median "
              f"{statistics.median(ttfts) * 1e3:5.0f} ms, last {ttfts[-1] * 1e3:5.0f} ms -> "
              f"{total / ttfts[-1]:5.0f} prompt tok/s aggregate", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
