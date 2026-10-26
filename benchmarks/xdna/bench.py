"""Gemma 4 E2B on vLLM's CPU backend, the same prompts as ~/src/gemma-npu/gemma_npu.py.

TTFT = a 1-token generate; decode = (n - 1) tokens / (time(n) - time(1)).
Prefix caching is off so every run prefills the whole prompt.

usage: source tools/xdna/env.sh; python bench.py [--quant xdna_int8] [prompt_tokens ...]
(XDNA_VLLM_NPU=0 keeps the int8 GEMMs on the CPU.)
"""
import argparse
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "google/gemma-4-E2B-it"
N_DECODE = 32
TEXT = open(os.path.expanduser("~/src/gemma-npu/filler.txt")).read()


def prompt_text(tok, n_tokens):
    """The user message gemma_npu.prompt_ids builds: the first n tokens of the filler."""
    return tok.decode(tok(TEXT)["input_ids"][:n_tokens]) + "\n\nSummarize the text above in detail."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default=None, help="e.g. xdna_int8")
    ap.add_argument("lengths", type=int, nargs="*", default=[190, 840, 1860, 3640])
    args = ap.parse_args()
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
    tok = AutoTokenizer.from_pretrained(MODEL)
    t = time.perf_counter()
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=max(4608, max(args.lengths) + 600), enable_prefix_caching=False,
              quantization=args.quant, language_model_only=True)
    print(f"engine up in {time.perf_counter() - t:.0f}s", flush=True)

    def gen(ids, n):
        p = SamplingParams(max_tokens=n, min_tokens=n, temperature=0)
        t = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": ids}], p, use_tqdm=False)
        return time.perf_counter() - t, out[0].outputs[0].token_ids

    for n in args.lengths:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt_text(tok, n)}],
                                      add_generation_prompt=True, return_dict=True)["input_ids"]
        ids = list(ids)
        gen(ids, 2)  # warm-up
        t1 = min(gen(ids, 1)[0] for _ in range(2))
        tn, toks = gen(ids, N_DECODE)
        dec = (N_DECODE - 1) / (tn - t1)
        print(f"prompt {len(ids)}: TTFT {t1 * 1e3:.0f} ms ({len(ids) / t1:.0f} tok/s), "
              f"decode {dec:.1f} tok/s, first tokens {list(toks[:8])}", flush=True)


if __name__ == "__main__":
    main()
