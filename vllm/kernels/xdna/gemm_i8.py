# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int8 x int8 -> int32 GEMMs on the XDNA2 NPU.

The kernel is an ordinary Triton K-loop GEMM, launched as `kernel[grid](...)`,
each program computing a 512 x 512 block of C in K steps of 128. Both operands
are kept tile-contiguous, so each K step's tile is one 64 KiB run of memory for
the NPU's DMA rather than 512 rows of 128 or 512 bytes far apart (10-26 %
faster per launch, measured):

  A (activations): [rows / 512][K / 128][512][128]   staging, written per call
  B (the weight):  [N / 512][K / 128][128][512]      static, laid out at load
  C (results):     [rows, N] row-major, int32

Weights are static resident tensors (synced to the device once). Activations
and results live in shared resident staging buffers of MAX_ROWS rows: callers
quantize into them with `quantize` and read results with `dequant`. Each
launch covers a multiple of 512 rows of views into them, so every (N, K) is one
compiled kernel (one hardware context) whatever the token count, and a launch
copies and syncs nothing but the A rows it reads.
"""

import os

import torch

from . import cpu_ops, runtime

BLOCK_M = BLOCK_N = 512
BLOCK_K = 128
L1_M, L1_N = 64, 128
MAX_ROWS = int(os.environ.get("XDNA_VLLM_NPU_ROWS", "2048"))  # rows per NPU call, at most

_kernel = None
_staging = {"a": None, "c": None, "max_k": 0, "max_n": 0}


def kernel():
    """The @triton.jit GEMM, defined (and its tiling schedule selected) on first use."""
    global _kernel
    if _kernel is not None:
        return _kernel
    s = runtime.init()
    triton, tl = s["triton"], s["tl"]
    from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform

    # Each core computes an L1_M x L1_N tile of the 512 x 512 block; K streams
    # through L2 in BLOCK_K steps (herd 8 x 4 = the whole array).
    os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = runtime.schedule(
        f"transform_i8_k{BLOCK_K}_{L1_M}x{L1_N}.mlir",
        lambda: generate_matmul_transform(l1_m=L1_M, l1_n=L1_N, l2_k=BLOCK_K, pack_sizes=(8, 8, 8),
                                          accum_type="i32", k_loop=True))

    @triton.jit
    def matmul_i8_tiled(A, B, C, N: tl.constexpr, K: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """C[M, N] = A @ B, A and B tile-contiguous (see the module docstring); M is the grid's."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + pid_m * K * BLOCK_M + offs_m[:, None] * BLOCK_K + offs_k[None, :]
        b_ptrs = B + pid_n * K * BLOCK_N + offs_k[:, None] * BLOCK_N + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for _ in range(0, K // BLOCK_K):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
            a_ptrs += BLOCK_M * BLOCK_K
            b_ptrs += BLOCK_K * BLOCK_N
        tl.store(C + (pid_m * BLOCK_M + offs_m)[:, None] * N + (pid_n * BLOCK_N + offs_n)[None, :], acc)

    _kernel = matmul_i8_tiled
    return _kernel


def _reserve(K, Np):
    """Grow the shared staging buffers to MAX_ROWS rows of (K, Np)."""
    st, resident = _staging, runtime.init()["resident"]
    if K > st["max_k"]:
        st["max_k"] = K
        st["a"] = resident.empty(MAX_ROWS * K, dtype=torch.int8)
    if Np > st["max_n"]:
        st["max_n"] = Np
        st["c"] = resident.empty(MAX_ROWS * Np, dtype=torch.int32)
        # Only the NPU writes the results, and the reader evicts them from the
        # caches itself (cpu_ops), so a launch syncs neither way -- for a
        # gate_up launch that is 2 x 48 MiB of cache flushes off the launching
        # thread.
        resident.set_device_writes(st["c"], sync_back=False)


def tile_rows(a, K):
    """[R, K] row-major (R a multiple of 512) as [R / 512][K / 128][512][128]: a permuted view."""
    R = a.shape[0]
    return a.view(R // BLOCK_M, BLOCK_M, K // BLOCK_K, BLOCK_K).permute(0, 2, 1, 3)


class NpuGemm:
    """One int8 weight [N, K] (per-output-channel scales) resident on the NPU."""

    def __init__(self, wq, sw):
        resident = runtime.init()["resident"]
        kernel()
        cpu_ops.load()
        N, K = wq.shape
        assert K % BLOCK_K == 0, f"K={K} is not a multiple of {BLOCK_K}"
        self.N, self.K = N, K
        self.n_pad = -(-N // BLOCK_N) * BLOCK_N
        wt = torch.zeros((K, self.n_pad), dtype=torch.int8)
        wt[:, :N] = wq.t()
        # [K, Np] -> [Np / 512][K / 128][128][512]
        wt = wt.view(K // BLOCK_K, BLOCK_K, self.n_pad // BLOCK_N, BLOCK_N).permute(2, 0, 1, 3).contiguous()
        self.w = resident.to_npu(wt, static=True)
        self.sw = sw.float().contiguous()
        _reserve(K, self.n_pad)
        self.launches = 0

    def load_rows(self, q):
        """Put int8 rows q [R, K] (R <= MAX_ROWS) into the A staging, tiled and zero-padded to 512."""
        R = q.shape[0]
        assert R <= MAX_ROWS, f"{R} rows > XDNA_VLLM_NPU_ROWS={MAX_ROWS}"
        Rp = -(-R // BLOCK_M) * BLOCK_M
        if R != Rp:
            q = torch.cat([q, q.new_zeros((Rp - R, self.K))])
        a = _staging["a"][: Rp * self.K].view(Rp // BLOCK_M, self.K // BLOCK_K, BLOCK_M, BLOCK_K)
        a.copy_(tile_rows(q, self.K))

    def quantize(self, x):
        """Per-row int8 quantization of bf16 x [R, K] into the A staging; the row scales [R, 1]."""
        R = x.shape[0]
        q = torch.empty((R, self.K), dtype=torch.int8)
        sx = torch.empty((R, 1), dtype=torch.float32)
        torch.ops._C.dynamic_scaled_int8_quant(q, x, sx, None)
        self.load_rows(q)
        return sx

    def results(self, R):
        """The int32 results [R, n_pad] as they lie in memory: stale until cpu_ops.invalidate."""
        return _staging["c"][: R * self.n_pad].view(R, self.n_pad)

    def launch_rows(self, r0, n=BLOCK_M):
        """One launch on staged rows [r0, r0 + n), n a multiple of BLOCK_M. No torch compute."""
        st = _staging
        K, Np = self.K, self.n_pad
        a = st["a"][r0 * K: (r0 + n) * K]
        c = st["c"][r0 * Np: (r0 + n) * Np]
        _kernel[(n // BLOCK_M, Np // BLOCK_N)](a, self.w, c, Np, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                               BLOCK_K=BLOCK_K)
        self.launches += 1

    def launch(self, R):
        """Run the GEMM on the first R staged rows (rounded up to BLOCK_M), 512 rows a launch."""
        for r0 in range(0, -(-R // BLOCK_M) * BLOCK_M, BLOCK_M):
            self.launch_rows(r0)

    def dequant(self, r0, r1, sx, out):
        """out = bf16(results[r0:r1] * sx * sw), after evicting the rows' stale cache lines."""
        c = _staging["c"][r0 * self.n_pad: r1 * self.n_pad].view(r1 - r0, self.n_pad)
        cpu_ops.invalidate_dequant(c, self.N, sx, self.sw, out)

    def __call__(self, x):
        """bf16 x [M, K] -> bf16 [M, N] on the NPU alone (tests, benchmarks)."""
        runtime.use_npu()
        M = x.shape[0]
        sx = self.quantize(x)
        self.launch(M)
        out = torch.empty((M, self.N), dtype=torch.bfloat16)
        self.dequant(0, M, sx, out)
        return out
