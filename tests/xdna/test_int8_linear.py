"""Int8Linear (CPU oneDNN, and CPU+NPU split) against a float reference.

    source tools/xdna/env.sh; python test_int8_linear.py
"""
import os
import sys

import torch

from vllm.model_executor.layers.quantization import xdna_int8 as quant


def main():
    torch.manual_seed(0)
    ok = True
    for N, K, M in ((2560, 1536, 7), (24576, 1536, 1), (24576, 1536, 3000), (1536, 12288, 2100)):
        w = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
        wq, ws = quant._quantize(w)
        lin = quant.Int8Linear(wq, ws, f"test{N}x{K}", allow_npu=True)
        x = torch.randn(M, K).to(torch.bfloat16)
        ref = x.float() @ w.float().t()
        for _ in range(3):  # rates settle after the first calls
            y = lin(x)
        R = lin.npu_rows_for(M)
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        good = rel < 0.02
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} N={N} K={K} M={M}: NPU rows {R}, rel err {rel:.4f}")
    print("stats", quant.STATS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
