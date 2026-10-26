"""Linear kernels for Gemma 4 E2B shapes, as vLLM's CPU worker would run them.

    source tools/xdna/env.sh; taskset -c 16-30 python bench_gemm.py [--npu]

Per (N, K) and token count M: vLLM's bf16 oneDNN GEMM, oneDNN W8A8 (dynamic
per-token int8), torch._weight_int8pack_mm (W8A16), torch._int_mm (W8A8),
and with --npu the Triton-XDNA int8 kernel on 512-row launches.
"""
import argparse
import os
import sys
import time

import torch

from vllm import _custom_ops as ops

SHAPES = [  # (N, K, what)
    (24576, 1536, "gate_up, layers 15-34"),
    (1536, 12288, "down, layers 15-34"),
    (2560, 1536, "qkv, sliding"),
    (262144, 1536, "LM head"),
]
MS = [1, 4, 16, 64, 256, 1024, 4096]


def timeit(fn, min_time=0.3):
    fn()
    n, t0 = 0, time.perf_counter()
    while True:
        fn()
        n += 1
        t = time.perf_counter() - t0
        if t > min_time and n >= 3:
            return t / n * 1e3


def quant_rows(x):
    s = x.float().abs().amax(-1).clamp(min=1e-8) / 127
    return torch.round(x.float() / s[:, None]).clamp(-127, 127).to(torch.int8), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", action="store_true")
    ap.add_argument("--threads", type=int, default=15)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.npu:
        from vllm.kernels.xdna import gemm_i8 as npu
    print(f"threads {torch.get_num_threads()}", flush=True)
    for N, K, what in SHAPES:
        w = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
        ws = w.float().abs().amax(1).clamp(min=1e-8) / 127
        wq = torch.round(w.float() / ws[:, None]).clamp(-127, 127).to(torch.int8)  # [N, K]
        bf16 = ops.create_onednn_mm(w.t())
        i8 = ops.create_onednn_scaled_mm(wq.t(), ws[:, None].contiguous(), torch.bfloat16, True, False, 32)
        wq_kn = wq.t().contiguous()
        ws_bf = ws.to(torch.bfloat16)
        g = npu.NpuGemm(wq, ws) if args.npu and N < 100000 else None
        gb = N * K / 1e9
        print(f"\n== N={N} K={K} ({what}); int8 weights {gb:.3f} GB", flush=True)
        print(f"{'M':>5} | {'bf16 onednn':>11} | {'w8a8 onednn':>11} | {'int8pack':>9} | {'_int_mm':>8} | {'npu':>7}   (ms)")
        for M in MS:
            x = torch.randn(M, K).to(torch.bfloat16)
            r = {}
            r["bf16"] = timeit(lambda: ops.onednn_mm(bf16, x, None))

            def w8a8():
                xq, xs, _ = ops.onednn_scaled_int8_quant(x)
                out = torch.empty(M, N, dtype=torch.bfloat16)
                return ops.onednn_scaled_mm(i8, xq, out, xs, None, None, None)
            r["w8a8"] = timeit(w8a8)
            r["pack"] = timeit(lambda: torch._weight_int8pack_mm(x, wq, ws_bf)) if M <= 1024 else float("nan")

            def int_mm():
                xq, xs = quant_rows(x)
                return (torch._int_mm(xq, wq_kn).float() * xs[:, None] * ws[None]).to(torch.bfloat16)
            r["int_mm"] = timeit(int_mm) if M > 16 else float("nan")
            r["npu"] = float("nan")
            if g is not None and M >= 512:
                r["npu"] = timeit(lambda: g(x))
                if M == MS[-1]:
                    err = (g(x).float() - x.float() @ w.float().t()).abs().max().item()
                    print(f"      npu vs bf16 reference: max abs err {err:.4f}")
            print(f"{M:>5} | {r['bf16']:>11.3f} | {r['w8a8']:>11.3f} | {r['pack']:>9.3f} | {r['int_mm']:>8.3f} | {r['npu']:>7.3f}",
                  flush=True)
            if M == 1:  # weights smaller than the L3 (64 MB) stay cached across repeats
                print(f"      | {gb * 2e3 / r['bf16']:>7.0f}GB/s | {gb * 1e3 / r['w8a8']:>7.0f}GB/s | {gb * 1e3 / r['pack']:>5.0f}GB/s |"
                      "  (M=1 weight read rate)")


if __name__ == "__main__":
    main()
