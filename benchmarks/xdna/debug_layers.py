"""Per-module output stats for one short prompt, in-process and eager (debugging aid).

    source tools/xdna/env.sh; VLLM_ENABLE_V1_MULTIPROCESSING=0 python debug_layers.py [xdna_int8]
"""
import os
import sys

import torch
from vllm import LLM, SamplingParams


def main():
    quant = sys.argv[1] if len(sys.argv) > 1 else None
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
    llm = LLM("google/gemma-4-E2B-it", dtype="bfloat16", max_model_len=2048, quantization=quant,
              language_model_only=True, enforce_eager=True, enable_prefix_caching=False)
    ex = llm.llm_engine.model_executor
    worker = getattr(ex, "driver_worker", None) or ex.workers[0]
    model = worker.worker.model_runner.model
    rows = []

    def hook(name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel():
                rows.append((name, tuple(t.shape), t.float().abs().mean().item(), bool(torch.isfinite(t).all())))
        return f

    for name, mod in model.named_modules():
        if name.count(".") <= 5 and any(k in name for k in ("embed", "layers.0.", "layers.1.", "layers.34", "norm", "lm_head")):
            mod.register_forward_hook(hook(name))
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
    print("OUTPUT", repr(out[0].outputs[0].text), out[0].outputs[0].token_ids)
    seen = set()
    for name, shape, m, fin in rows:
        if name in seen:
            continue
        seen.add(name)
        print(f"{name:60s} {str(shape):18s} mean|x| {m:10.4f} finite {fin}")


if __name__ == "__main__":
    main()
