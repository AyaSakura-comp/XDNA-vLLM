"""Where a prefill or decode step's time goes: vLLM's torch profiler, per op, on the worker.

Scenarios (prefix caching off, distinct prompts):
  prefill1  1 x ~3600-token prompt, 1 token          (one 4096-token chunk)
  prefill4  4 x ~1900-token prompts, 1 token each    (two chunks)
  decode1   1 x ~225-token prompt, 64 tokens
  decode4   4 x ~225-token prompts, 64 tokens each
For each, the worker's op table (self CPU time on its main thread) is saved as
<out>/<scenario>.txt, and the wall time is printed.

usage: source tools/xdna/env.sh; python benchmarks/xdna/profile_ops.py OUT_DIR [scenario ...]
(XDNA_VLLM_NPU=0 for the CPU-only build.)
"""
import glob
import os
import shutil
import sys
import time

from transformers import AutoTokenizer

MODEL = "google/gemma-4-E2B-it"
TEXT = open(os.path.expanduser("~/src/gemma-npu/filler.txt")).read()
SCENARIOS = {"prefill1": (1, 3600, 1), "prefill4": (4, 1880, 1), "decode1": (1, 190, 64), "decode4": (4, 190, 64)}


def prompts(tok, B, n, salt):
    body = tok.decode(tok(TEXT)["input_ids"][:n])
    out = []
    for i in range(B):
        msg = f"Document {salt}-{i}.\n\n{body}\n\nSummarize the text above in detail."
        out.append(list(tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                                return_dict=True)["input_ids"]))
    return out


def main():
    out = os.path.abspath(sys.argv[1])
    names = sys.argv[2:] or list(SCENARIOS)
    os.makedirs(out, exist_ok=True)
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "2")
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(MODEL, dtype="bfloat16", quantization=os.environ.get("QUANT", "xdna_int8") or None,
              language_model_only=True, max_model_len=16384, max_num_batched_tokens=4096,
              enable_prefix_caching=False,
              profiler_config={"profiler": "torch", "torch_profiler_dir": out, "torch_profiler_with_stack": False,
                               "torch_profiler_use_gzip": True, "torch_profiler_record_shapes": False})
    for name in names:
        B, n, gen = SCENARIOS[name]
        p = SamplingParams(max_tokens=gen, min_tokens=gen, temperature=0, ignore_eos=True)
        warm = prompts(tok, B, n, f"warm{name}")
        llm.generate([{"prompt_token_ids": ids} for ids in warm], p, use_tqdm=False)
        ids = prompts(tok, B, n, name)
        llm.start_profile()
        t = time.perf_counter()
        llm.generate([{"prompt_token_ids": i} for i in ids], p, use_tqdm=False)
        dt = time.perf_counter() - t
        llm.stop_profile()
        time.sleep(2)
        for f in glob.glob(os.path.join(out, "*.profiler_out.txt")):
            shutil.move(f, os.path.join(out, f"{name}.txt"))
        print(f"{name}: B={B} x {len(ids[0])} tokens, {gen} generated: {dt * 1e3:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
