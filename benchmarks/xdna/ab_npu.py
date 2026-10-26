"""CPU+NPU vs CPU only in one process, interleaved: the NPU split is switched on and off between runs.

Same weights, same caches, same machine load for both, round after round, so
background load hits both alike. Per round and mode:
  prefill B=1: one ~3600-token prompt (one 4096-token chunk), TTFT
  prefill B=4: four ~1900-token prompts at once, time to the last first token
  decode  B=1 / B=4: ~200-token prompts, 128 tokens each, tokens/s per request
Prefix caching is off and every prompt is distinct.

usage: source tools/xdna/env.sh; python benchmarks/xdna/ab_npu.py [rounds]
"""
import os
import statistics
import sys
import time

from transformers import AutoTokenizer

MODEL = "google/gemma-4-E2B-it"
TEXT = open(os.path.expanduser("~/src/gemma-npu/filler.txt")).read()


def prompts(tok, B, n, salt):
    body = tok.decode(tok(TEXT)["input_ids"][:n])
    return [list(tok.apply_chat_template([{"role": "user", "content": f"Document {salt}-{i}.\n\n{body}\n\nSummarize."}],
                                         add_generation_prompt=True, return_dict=True)["input_ids"]) for i in range(B)]


def set_npu(worker, on):
    from vllm.model_executor.layers.quantization import xdna_int8
    xdna_int8.set_npu_enabled(on)


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "2")
    # collective_rpc of a local function (set_npu) is pickled.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(MODEL, dtype="bfloat16", quantization="xdna_int8", language_model_only=True, max_model_len=16384,
              max_num_batched_tokens=4096, enable_prefix_caching=False)

    def gen(ids, n):
        p = SamplingParams(max_tokens=n, min_tokens=n, temperature=0, ignore_eos=True)
        t = time.perf_counter()
        llm.generate([{"prompt_token_ids": i} for i in ids], p, use_tqdm=False)
        return time.perf_counter() - t

    tests = {
        "prefill B=1 (3.6K)": lambda s: gen(prompts(tok, 1, 3600, s), 1),
        "prefill B=4 (4x1.9K)": lambda s: gen(prompts(tok, 4, 1880, s), 1),
        "decode B=1": lambda s: (gen(prompts(tok, 1, 190, s), 128), gen(prompts(tok, 1, 190, s + "x"), 1)),
        "decode B=4": lambda s: (gen(prompts(tok, 4, 190, s), 128), gen(prompts(tok, 4, 190, s + "x"), 1)),
    }
    res = {(t, m): [] for t in tests for m in ("cpu", "npu")}
    for name, fn in tests.items():  # warm-up, both modes
        for on in (True, False):
            llm.collective_rpc(set_npu, args=(on,))
            fn(f"warm{name}{on}")
    for r in range(rounds):
        for name, fn in tests.items():
            for mode in (("npu", "cpu") if r % 2 == 0 else ("cpu", "npu")):
                llm.collective_rpc(set_npu, args=(mode == "npu",))
                v = fn(f"r{r}{name}{mode}")
                if name.startswith("decode"):  # tokens/s per request over the 127 decode steps
                    full, first = v
                    v = 127 / (full - first)
                res[(name, mode)].append(v)
        print(f"round {r + 1}/{rounds} done", flush=True)
    # The two modes of a round ran back to back, so the median of the
    # per-round ratios cancels load that drifts across rounds.
    print(f"\n{'':>22} {'CPU only':>12} {'CPU+NPU':>12} {'paired change':>14} {'rounds NPU better':>18}")
    for name in tests:
        cs, ns = res[(name, "cpu")], res[(name, "npu")]
        c, n = statistics.median(cs), statistics.median(ns)
        ratio = statistics.median(b / a for a, b in zip(cs, ns))
        if name.startswith("decode"):
            better = sum(b > a for a, b in zip(cs, ns))
            print(f"{name:>22} {c:9.1f} t/s {n:9.1f} t/s {100 * (ratio - 1):+13.1f}% {better:>11}/{len(cs)}")
        else:
            better = sum(b < a for a, b in zip(cs, ns))
            print(f"{name:>22} {1e3 * c:9.0f} ms {1e3 * n:9.0f} ms {100 * (ratio - 1):+13.1f}% {better:>11}/{len(cs)}")
    print("samples:", {f"{k[0]}|{k[1]}": [round(x, 3) for x in v] for k, v in res.items()})


if __name__ == "__main__":
    main()
