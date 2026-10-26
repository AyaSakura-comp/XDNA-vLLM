# XDNA-vLLM: vLLM on the CPU and the AMD XDNA2 NPU

XDNA-vLLM is vLLM 0.32.0's CPU platform plus kernels for the XDNA2 NPU in
Ryzen AI chips. It was built on Strix Halo (Ryzen AI Max+ 395). The NPU kernels
are ordinary Triton kernels, which [Triton-XDNA](https://github.com/AyaSakura-comp/Triton-XDNA)
(`third_party/triton-xdna`) compiles and launches.

vLLM keeps doing everything else on the CPU:

- scheduling and chunked prefill
- the paged KV cache and prefix caching
- attention
- sampling
- the OpenAI API, and tool-call and reasoning parsing

`main` is v0.32.0 plus the commits listed under [Changes against vLLM v0.32.0](#changes-against-vllm-v0320).
It stays on that release because it runs with that release's prebuilt CPU binaries.

- [Kernels: front end in vLLM, back end in Triton-XDNA](kernels.md) explains
  where a new kernel goes and where every earlier NPU kernel ended up.

## Quick start

```bash
git clone --recursive git@github.com:AyaSakura-comp/XDNA-vLLM.git ~/src/XDNA-vLLM
cd ~/src/XDNA-vLLM
# 1. A Python 3.13 venv at .venv with the vLLM 0.32.0 CPU wheel
#    (GitHub release vllm-0.32.0+cpu-...-manylinux_2_39_x86_64.whl, torch 2.13 cpu)
#    and Triton-XDNA (pip install third_party/triton-xdna, or its prebuilt wheel;
#    see its README). XRT must be at /opt/xilinx/xrt.
# 2. Point that venv at this checkout (binaries copied in, symlinks out):
tools/xdna/link_venv.sh
# 3. Serve:
tools/xdna/serve.sh                     # OpenAI API on 127.0.0.1:8013
# or as a service: cp tools/xdna/vllm-npu.service ~/.config/systemd/user/
```

After `link_venv.sh`, two things in the venv are symlinks into this checkout:

- `site-packages/vllm` points at `vllm/`.
- `site-packages/triton/backends/amd_triton_npu` points at the submodule's backend.

So edits to vLLM, to the kernels, or to Triton-XDNA's Python take effect on the
next start. A change to Triton-XDNA's C++ (the `triton-shared-opt` passes or
Triton itself) needs `pip install third_party/triton-xdna` again.

`tools/xdna/serve.sh` serves `google/gemma-4-E2B-it` with these flags:

- `--quantization xdna_int8 --language-model-only`
- `--max-num-batched-tokens 4096`
- the Gemma 4 tool and reasoning parsers
- `VLLM_CPU_KVCACHE_SPACE=3` (GB)

Memory is about 14.5 GB in all.

## What runs where (`--quantization xdna_int8`)

| part | where | how |
|---|---|---|
| decode linears, LM head, prompts under 1024 tokens | CPU | oneDNN W8A8: per-channel int8 weights, per-token dynamic activations |
| prefill linears with at least 1024 rows: qkv, o, gate/up, down, the PLE model projection | CPU **and** NPU at once | a multiple of 512 rows go to the NPU and the rest to oneDNN; the NPU's int32 results are dequantized on the CPU |
| attention, norms, RoPE, sampling | CPU | vLLM's CPU backend |
| embedding tables | CPU | int8 at load; the LM head reads the same int8 table |

**One split call** (`Int8Linear._run`), for M rows with R of them on the NPU:

1. The main thread quantizes the R rows into the NPU's tiled staging buffer.
2. The NPU launch thread launches them, 1024 rows at a time.
3. Meanwhile the main thread runs oneDNN on the other M − R rows.
4. As each launch returns, a helper thread evicts its result rows from the
   CPU caches and dequantizes them into the output (`cpu_ops.invalidate_dequant`).
5. When the main thread finishes its oneDNN share, it takes any chunks still
   waiting, using all its OpenMP threads.

The launch thread and the helper sit on a core no OpenMP thread uses.

**Split policy** (`_SplitTuner` in `quantization/xdna_int8.py`): the NPU's
share is picked per (N, K, rows/512) by timing whole calls.

- Every candidate (0, 512, … rows) is timed a few times, and the fastest by
  median wins.
- The runner-up is re-timed every 64 calls.
- 0 (CPU only) is always a candidate, so a shape where the NPU does not pay
  stays on the CPU.
- At load, every multiple of 512 rows from 1024 up to the scheduler's chunk
  is calibrated. This also compiles the NPU kernels and takes about 10 s.

The environment variables:

| variable | effect |
|---|---|
| `XDNA_VLLM_NPU=0` | keep every GEMM on the CPU |
| `XDNA_VLLM_NPU_LAYERS` | regex of the linears that get an NPU copy (default `qkv_proj\|o_proj\|gate_up_proj\|down_proj\|per_layer_model_projection`) |
| `XDNA_VLLM_NPU_MIN_ROWS` | default 1024 |
| `XDNA_VLLM_NPU_ROWS` | the NPU's rows per call, at most (default 2048) |
| `XDNA_VLLM_NPU_LAUNCH_ROWS` | rows per NPU launch (default 1024) |
| `XDNA_VLLM_NPU_CPU` | the CPU for the launch thread; default: a core no OpenMP thread uses |
| `XDNA_VLLM_CALIBRATE_ROWS` | the row counts to calibrate at load |
| `XDNA_VLLM_TRACE=1` | log one line per linear call |

## Rooflines and hardware limits (measured on this box)

| | NPU (XDNA2, 32 AIE2P tiles) | CPU (16 Zen 5 cores, 15 used) |
|---|---|---|
| int8 compute peak | 51 TOPS (`xrt-smi validate`) | 9.7 TOPS (`torch._int_mm`) |
| memory bandwidth | about 55 GB/s ceiling; GEMM achieves about 40 GB/s, weight streaming 33–35 GB/s | 118 GB/s; oneDNN reads LM-head weights at 101–114 GB/s |
| fixed cost per launch | 0.41–0.47 ms | |

**Prefill GEMMs.**

- Per 512-row launch of gate_up (24576 × 1536), the NPU runs 3.0 ms, or 12.5–13 TOPS.
- That is a quarter of its compute peak, but close to its memory roof. A
  launch moves 120 MiB: A is read once per 512-column block (36 MiB), W once
  (36 MiB), and the int32 C is written once (48 MiB). At about 40 GB/s that
  is the 3.0 ms.
- About 34 µs of each 512 × 512 block is fixed cost: filling the pipeline and
  writing C back. The schedule does not overlap that with the next block. Of
  gate_up's 3.0 ms, about 1.6 ms is this. Overlapping it needs a different
  mlir-air schedule (back end).
- oneDNN runs these GEMMs at 8.6 TOPS, 89 % of the CPU's int8 peak. Neither
  side has much left on its own, so the gain comes from running both at once.

**What it took to get there.** Each change below was measured on its own:

- XRT's syncs were cache flushes on the launch thread. Under a busy CPU they
  cost more than the NPU's run: 2 × 48 MiB at 1.5–2.5 ms each. The results
  buffer is now device-written and not synced, and the reader evicts the
  lines itself in parallel (Triton-XDNA `set_device_writes(sync_back=False)`).
- The NPU does **not** snoop the CPU caches. Skipping the eviction reads
  0.3–2 % stale results, so the eviction is required.
- Tile-contiguous A and W: 10–26 % less NPU time per launch.
- The launch thread had inherited OpenMP thread 0's CPU. It now runs on a
  core no OpenMP thread uses.

**What does not work on AIE2P.**

- There is no vector int32 → fp32 conversion. The ISA's `VCONV` goes
  fp32 ↔ bf16 or bfp16. So the int32 results cannot be dequantized on the
  NPU cheaply: a bf16 epilogue measured 8× slower.
- An int16 rounding-shift epilogue compiles (with `epilogue=True`) but is not
  faster: 3.07 vs 2.96 ms.

**Decode is a hardware limit, not a software one.**

- A decode step streams about 2 GB of int8 weights, and the CPU reads them
  at about 100 GB/s.
- The NPU reads weights at 33–35 GB/s.
- Every launch costs at least 0.41 ms. A decode-time linear takes 0.03–0.4 ms
  on the CPU, so handing any one of them to the NPU is slower however it is
  split.
- The only candidate is the LM head: 403 MB, 3.5–4 ms a token on the CPU.
  Split about 1:4 with the NPU, its best case saves about 0.8 ms of a ~33 ms
  token (2.4 %), before the two contend for DRAM. It was not done.

## Results (Gemma 4 E2B, 2026-10-26)

TTFT for one request, with prefix caching off (`benchmarks/xdna/bench.py`, 4096-token chunks):

| prompt tokens | vLLM bf16 | xdna_int8, CPU only | xdna_int8 + NPU |
|---|---|---|---|
| 207 | 175 ms | 129 ms | 129 ms |
| 858 | 556 ms | 456 ms | 454 ms |
| 1878 | 1205 ms | 980 ms | 927 ms |
| 3658 | 2627 ms | 2168 ms | 1832 ms |
| 12018 | | 8362 ms | 8159 ms |
| 32018 | | 35.5 s | 34.6 s |

Concurrent requests (`benchmarks/xdna/bench_batch.py`): B distinct
~225-token prompts, 128 tokens each. Throughput is per request / total.

| B | xdna_int8 + NPU | bf16 |
|---|---|---|
| 1 | 30.3 / 30 tok/s | 18.5 / 18.5 tok/s |
| 4 | 29.6 / 121 | 18.9 / 75 |
| 8 | 28.2 / 225 | 18.1 / 144 |
| 16 | 24.7 / 402 | 16.7 / 265 |
| 32 | 17.3 / 648 | 11.4 / 431 |

Decode runs on the CPU only. Perplexity (`benchmarks/xdna/ppl.py`) is 102.06
for bf16 and 103.40 for xdna_int8, 1.3 % higher.

## Changes against vLLM v0.32.0

- `[CPU] Fix the slot-mapping fallback's parameter order`: in the 0.32.0 CPU
  release every request failed with "Boolean value of Tensor … ambiguous".
- `third_party/triton-xdna`: the submodule, following its `main`.
- `vllm/kernels/xdna/`, `vllm/model_executor/layers/quantization/xdna_int8.py`,
  and the registry entry: the kernels and the quantization method.
- `tools/xdna/`, `benchmarks/xdna/`, `tests/xdna/`, `docs/xdna/`.

## Notes

- The HF transformers model backend (`--model-impl transformers`) does not
  load Gemma 4 in vLLM 0.32: first the audio-config token budget fails, then
  the vision-tower weights. So xdna_int8 runs under vLLM's own `gemma4.py`. It
  hooks vLLM's linear layers, which the transformers backend uses too.
- With tcmalloc preloaded, Triton-XDNA loads mlir-air without `RTLD_DEEPBIND`.
  With DEEPBIND, mlir-air's `free()` bypasses tcmalloc and `import triton` segfaults.
- `XDNA_VLLM_CACHE` (default `~/.cache/xdna_vllm`) holds the tiling schedules
  and the intermediate IR. Compiled kernels are in `~/.triton/cache`.
