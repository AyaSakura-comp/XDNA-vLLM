"""Teacher-forced NLL / perplexity on fixed text windows, bf16 vs int8.

    source tools/xdna/env.sh; python ppl.py [--quant xdna_int8]
Windows: 4 x 2048 tokens of ~/src/gemma-npu/filler.txt (notes) -- the same
for every run, so the numbers compare directly.
"""
import argparse
import math
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "google/gemma-4-E2B-it"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default=None)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--len", type=int, default=2048)
    args = ap.parse_args()
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok(open(os.path.expanduser("~/src/gemma-npu/filler.txt")).read())["input_ids"]
    wins = [[tok.bos_token_id] + ids[i * args.len: (i + 1) * args.len] for i in range(args.windows)]
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=args.len + 64, quantization=args.quant,
              language_model_only=True, enable_prefix_caching=False)
    outs = llm.generate([{"prompt_token_ids": w} for w in wins],
                        SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
    tot, n = 0.0, 0
    for w, o in zip(wins, outs):
        nll = -sum(lp[t].logprob for t, lp in zip(w[1:], o.prompt_logprobs[1:]))
        tot += nll
        n += len(w) - 1
        print(f"window: ppl {math.exp(nll / (len(w) - 1)):.3f}")
    print(f"[{args.quant or 'bf16'}] mean NLL {tot / n:.4f}, ppl {math.exp(tot / n):.3f} over {n} tokens")


if __name__ == "__main__":
    main()
