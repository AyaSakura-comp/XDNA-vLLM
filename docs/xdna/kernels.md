# NPU kernels: front end in vLLM, back end in Triton-XDNA

An NPU kernel in XDNA-vLLM has two halves, and each has one home.

| | lives in | what it is |
|---|---|---|
| **front end** | `vllm/kernels/xdna/` (this repo) | the `@triton.jit` function, its block and tile choices, the Python that stages data and launches it, and the vLLM layer or op that calls it |
| **back end** | `third_party/triton-xdna` ([AyaSakura-comp/Triton-XDNA](https://github.com/AyaSakura-comp/Triton-XDNA), `main`) | what turns any Triton kernel into an NPU launch: the lowering passes (Triton → Linalg → MLIR-AIR → AIE ELF), the transform-schedule generators, the XRT/HSA launchers, and resident tensors |

The rule of thumb is whether a change is specific to one kernel:

- **Specific to one kernel, or to vLLM's use of it** (a new op, a different
  tile, a fused epilogue, a split policy): it goes in this repo.
- **Something the compiler or the runtime lacks** (a pass, a new launch mode,
  a buffer feature, a bug in lowering): it goes in Triton-XDNA. Commit it in
  the submodule on `main`, push it to the fork, then commit the new submodule
  pointer here.

## Adding a kernel

1. Write the kernel in `vllm/kernels/xdna/<op>.py`. Follow `gemm_i8.py`:
   - Define the `@triton.jit` function on first use, after `runtime.init()`,
     so vLLM never imports Triton-XDNA unless the NPU is used.
   - Choose its schedule with `runtime.schedule(name, generator)` and
     `AIR_TRANSFORM_TILING_SCRIPT`.
   - Keep weights and long-lived buffers as resident tensors
     (`resident.to_npu(w, static=True)`, `resident.empty`). Launches on views
     of them copy nothing.
   - Mark outputs `resident.set_device_writes(t, sync_back=False)` and read
     them through `cpu_ops.invalidate_dequant` or `cpu_ops.invalidate`. The
     NPU does not snoop the CPU caches, and XRT's own syncs are
     single-threaded cache flushes on the launching thread.
   - Lay operands out tile-contiguous where you can, so each DMA reads long
     runs. Weights can be reordered once at load.
   - Use one grid per shape: each grid is another compiled kernel and NPU
     hardware context.
   - Launch from one thread that has called `runtime.use_npu()`. The active
     driver is per thread.
2. Call it from vLLM through a layer method or a quantization method, or
   through a `torch.library.custom_op` with a fake implementation, so
   `torch.compile` treats it as one opaque node. `xdna_int8.py` is the example.
3. Add a correctness test in `tests/xdna/` and a benchmark in
   `benchmarks/xdna/`. Report what the kernel reaches against the NPU's roofline:
   - compute: about 51 TOPS int8 (`xrt-smi validate`)
   - memory: about 55 GB/s from DDR
4. If Triton-XDNA needs a change, make it in the submodule, as above.

## Where the earlier NPU kernels went

Before this repo, the NPU kernels lived in per-project directories. All of
them were int8 GEMMs.

| kernel (original place) | now |
|---|---|
| `matmul_i8_k` and `NpuGemm`, `~/src/vllm-npu/xdna_vllm/npu.py` (the vLLM plugin) | `vllm/kernels/xdna/gemm_i8.py`, now `matmul_i8_tiled` (tile-contiguous A/B), with `cpu_ops.py` for the CPU side |
| plugin's quantization method, `~/src/vllm-npu/xdna_vllm/quant.py` | `vllm/model_executor/layers/quantization/xdna_int8.py` |
| `matmul_i8_k`, `~/src/gemma-npu/npu_kernels.py` (HF transformers pipeline) | same kernel as above; that pipeline is replaced by serving through vLLM |
| `matmul_i8_k` copies in `~/src/omnivoice-npu/kbench.py` and `prof_npu.py` | same kernel; the GEMM benchmark is `benchmarks/xdna/bench_gemm.py` |
| `matmul_i8`, whole-K, 256² block, `~/src/egemma-npu/npu_i8.py` | replaced by the K-loop GEMM: 512² block, 64×128 L1 tiles, measured 6.3–7.6 vs ~4 TOPS |
| `matmul_i8_bf16`, an epilogue that casts the int32 result to bf16, same file | not carried over: measured 8× slower, because AIE has no fast i32→f32 |
| `matmul_i8_i16`, an epilogue that shifts and stores int16, same file | not carried over: inexact, and no faster |
| egemma's own pyxrt contexts and buffers (`npu_gemm.py`) | replaced by Triton-XDNA's resident tensors (back end) |

Not merged, because they are not Triton kernels:

- `~/npu-fa`: flash-attention decode and GDN kernels written in C++ against
  mlir-aie IRON.
- The mlir-aie IRON vision models: YOLO, MobileNet.

Triton-XDNA's own `examples/` stay in Triton-XDNA. They are the compiler's
tests and demos: element-wise ops, softmax, norms, matmuls, and the Q4NX LLM
decode demos.
