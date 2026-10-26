"""One Int8Linear: CPU alone vs split with the NPU, at prefill-sized M.

    source tools/xdna/env.sh; taskset -c 16-30 python bench_split.py
"""
import os
import sys
import time

import torch

from vllm.model_executor.layers.quantization import xdna_int8 as quant

SHAPES = [(24576, 1536, "gate_up 15-34"), (12288, 1536, "gate_up 0-14"), (1536, 12288, "down 15-34"),
          (1536, 6144, "down 0-14"), (5120, 1536, "qkv full"), (2560, 1536, "qkv sliding"), (1536, 2048, "o sliding"),
          (8960, 1536, "PLE model proj")]


def timeit(fn, n=6):
    for _ in range(4):  # compile + let the split settle
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "15")))
    for N, K, what in SHAPES:
        w = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
        wq, ws = quant._quantize(w)
        lin = quant.Int8Linear(wq, ws, what, allow_npu=True)
        for M in (2048, 4096):
            x = torch.randn(M, K).to(torch.bfloat16)
            out = torch.empty(M, N, dtype=torch.bfloat16)
            t_cpu = timeit(lambda: lin.cpu(x, out))
            t_mix = timeit(lambda: lin(x))
            print(f"{what:>15} M={M}: cpu {t_cpu:7.2f} ms | cpu+npu {t_mix:7.2f} ms (npu rows {lin.npu_rows_for(M)}) "
                  f"x{t_cpu / t_mix:.2f}", flush=True)


if __name__ == "__main__":
    main()
