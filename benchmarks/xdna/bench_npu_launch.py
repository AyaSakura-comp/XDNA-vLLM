"""One NPU GEMM launch, taken apart: whole launch vs the NPU's own run time, by shape.

For 512-row launches of the int8 K-loop GEMM on resident buffers:
  launch   wall time of gemm_i8.NpuGemm.launch_rows (Python + syncs + run)
  npu      xrt::run start -> wait2 (Triton-XDNA's autotune_time hook)
  C flush  resident.flush of the C view: the cost of one sync of it (launches no longer do any)
Shapes vary N and K so the output size and the work can be told apart.

usage: source tools/xdna/env.sh; python benchmarks/xdna/bench_npu_launch.py
"""
import os
import statistics
import tempfile
import time

import torch

from vllm.kernels.xdna import cpu_ops, gemm_i8, runtime

SHAPES = [  # (N, K, what)
    (24576, 1536, "gate_up 15-34"),
    (24576, 128, "same output, K=128"),
    (12288, 1536, "gate_up 0-14"),
    (1536, 12288, "down 15-34"),
    (1536, 6144, "down 0-14"),
    (512, 1536, "N=512"),
    (512, 128, "smallest"),
]


def median_ms(fn, n=12):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)


def main():
    s = runtime.init()
    from triton.backends.amd_triton_npu import driver

    driver.autotune_time = True  # the launcher writes the run's microseconds to ./data.txt
    os.chdir(tempfile.mkdtemp())
    runtime.use_npu()
    resident = s["resident"]
    print(f"{'shape':>22} {'what':>20} {'launch':>8} {'npu':>8} {'C flush':>8} {'GOP':>6} {'TOPS':>6} "
          f"{'MiB':>6} {'GB/s':>6}")
    for N, K, what in SHAPES:
        w = torch.randint(-127, 128, (N, K), dtype=torch.int8)
        g = gemm_i8.NpuGemm(w, torch.ones(N))
        a = torch.randint(-127, 128, (512, K), dtype=torch.int8)
        g.load_rows(a)
        c = g.results(512)
        launch = median_ms(lambda: g.launch_rows(0))
        npu = []
        for _ in range(6):
            g.launch_rows(0)
            npu.append(float(open("data.txt").read()) / 1e3)
        npu = statistics.median(npu)
        flush = median_ms(lambda: resident.flush(c))
        cpu_ops.invalidate(c)
        ok = torch.equal(c[:4, :N].cpu(), (a[:4].int() @ w.t().int()))
        gop = 2 * 512 * K * g.n_pad / 1e9
        # A is read once per 512-column block, W once, C written once.
        mib = (512 * K * (g.n_pad // 512) + K * g.n_pad + 512 * g.n_pad * 4) / 2**20
        print(f"{f'512x{K}x{N}':>22} {what:>20} {launch:8.2f} {npu:8.2f} {flush:8.2f} {gop:6.1f} "
              f"{gop / npu:6.1f} {mib:6.1f} {mib * 2**20 / npu / 1e6:6.1f}  {'ok' if ok else 'WRONG'}",
              flush=True)


if __name__ == "__main__":
    main()
