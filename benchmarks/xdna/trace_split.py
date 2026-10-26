"""Timeline of one split Int8Linear call: the main thread's steps and the NPU thread's launches.

Runs a linear of a Gemma 4 E2B shape at M rows with R of them on the NPU,
under the OMP binding vLLM's CPU worker uses, and writes the spans of the
median call as JSONL (one span per line: phase, category, lane, rel_start,
duration_sec) for plot_gantt.py, plus a per-step table on stdout.

usage: source tools/xdna/env.sh; python benchmarks/xdna/trace_split.py OUT.jsonl [N K M R]
"""
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("OMP_PLACES", "{" + ",".join(str(c) for c in range(16, 31)) + "}")
os.environ.setdefault("OMP_PROC_BIND", "true")

import torch  # noqa: E402

from vllm.kernels.xdna import gemm_i8 as npu  # noqa: E402
from vllm.model_executor.layers.quantization import xdna_int8 as q  # noqa: E402

SPANS = []
T0 = [0.0]


def span(lane, phase, cat, t0, t1):
    SPANS.append({"lane": lane, "phase": phase, "category": cat, "rel_start": t0 - T0[0], "duration_sec": t1 - t0})


_launch = npu.NpuGemm.launch_rows


def traced_launch(self, r0, n=npu.BLOCK_M):
    t = time.perf_counter()
    _launch(self, r0, n)
    t1 = time.perf_counter()
    span("NPU thread", f"launch rows {r0}-{r0 + n}", "submit", t, t1)
    # The launcher's own timing of xrt::run start -> wait2 (autotune_time).
    run = float(open("data.txt").read()) / 1e6
    SPANS.append({"lane": "NPU", "phase": f"NPU run {r0}-{r0 + n}", "category": "io",
                  "rel_start": t1 - run - T0[0], "duration_sec": run, "approx_start": True})


npu.NpuGemm.launch_rows = traced_launch


def run(lin, x, R):
    """Int8Linear._run with the main thread's steps timed (same calls, same order)."""
    M = x.shape[0]
    T0[0] = t = time.perf_counter()
    out = torch.empty((M, lin.N), dtype=torch.bfloat16)
    g = lin.npu
    sx = g.quantize(x[:R])
    t1 = time.perf_counter(); span("main thread", "quantize NPU rows", "fill", t, t1)
    done, times = q._WORKER.submit(g, R)
    lin.cpu(x[R:], out[R:])
    t2 = time.perf_counter(); span("main thread", f"oneDNN {M - R} rows", "plan", t1, t2)
    for i, r0 in enumerate(range(0, R, npu.BLOCK_M)):
        ta = time.perf_counter()
        done[i].wait()
        tb = time.perf_counter()
        if tb - ta > 1e-4:
            span("main thread", f"wait NPU {i}", "settle", ta, tb)
        r1 = r0 + npu.BLOCK_M
        g.dequant(r0, r1, sx[r0:r1], out[r0:r1])
        span("main thread", f"dequant {r0}-{r1}", "render", tb, time.perf_counter())
    return time.perf_counter() - t


def main():
    path = os.path.abspath(sys.argv[1])
    from triton.backends.amd_triton_npu import driver

    driver.autotune_time = True
    os.chdir(tempfile.mkdtemp())
    N, K, M, R = (int(v) for v in sys.argv[2:6]) if len(sys.argv) > 5 else (24576, 1536, 4096, 1536)
    torch.set_num_threads(15)
    torch.manual_seed(0)
    w = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
    wq, ws = q._quantize(w)
    os.environ["XDNA_VLLM_CALIBRATE_ROWS"] = ""
    q._CALIBRATE_ENV = ""
    lin = q.Int8Linear(wq, ws, "trace", allow_npu=True)
    x = torch.randn(M, K).to(torch.bfloat16)
    out = torch.empty(M, N, dtype=torch.bfloat16)
    for _ in range(3):
        lin.cpu(x, out)
    t = time.perf_counter()
    for _ in range(5):
        lin.cpu(x, out)
    t_cpu = (time.perf_counter() - t) / 5
    calls = []
    for _ in range(9):
        SPANS.clear()
        dt = run(lin, x, R)
        time.sleep(0.05)
        calls.append((dt, list(SPANS)))
    calls.sort(key=lambda c: c[0])
    dt, spans = calls[len(calls) // 2]
    with open(path, "w") as f:
        for s in sorted(spans, key=lambda s: s["rel_start"]):
            f.write(json.dumps(s) + "\n")
    print(f"[{N}, {K}] M={M} R={R}: split call {dt * 1e3:.2f} ms (median of 9), CPU alone {t_cpu * 1e3:.2f} ms, "
          f"x{t_cpu / dt:.2f}")
    for s in sorted(spans, key=lambda s: s["rel_start"]):
        print(f"  {s['lane']:>12}  {s['phase']:<22} {1e3 * s['rel_start']:7.2f} +{1e3 * s['duration_sec']:6.2f} ms")


if __name__ == "__main__":
    main()
