# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int8 x int8 -> int32 GEMMs on the XDNA2 NPU.

The kernel is an ordinary Triton K-loop GEMM, launched as `kernel[grid](...)`.
Weights are static resident tensors (synced to the device once). Activations
and results live in shared resident staging buffers of MAX_ROWS rows: callers
quantize straight into `stage(R)` and read results from it, and each launch
covers 512 rows of a view into them, so every (N, K) is one compiled kernel
(one hardware context) whatever the token count, and nothing is copied per
launch beyond the rows' own bytes.
"""

import os

import torch

from . import runtime

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
    def matmul_i8_k(A, B, C, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """C[M, N] = A[M, K] @ B[K, N], row-major; M is the grid's."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for _ in range(0, K // BLOCK_K):
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
        tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)

    _kernel = matmul_i8_k
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


class NpuGemm:
    """One int8 weight [N, K] (per-output-channel scales) resident on the NPU."""

    def __init__(self, wq, sw):
        resident = runtime.init()["resident"]
        kernel()
        N, K = wq.shape
        assert K % BLOCK_K == 0, f"K={K} is not a multiple of {BLOCK_K}"
        self.N, self.K = N, K
        self.n_pad = -(-N // BLOCK_N) * BLOCK_N
        wt = torch.zeros((K, self.n_pad), dtype=torch.int8)
        wt[:, :N] = wq.t()
        self.w = resident.to_npu(wt, static=True)
        self.sw = sw.float().contiguous()
        _reserve(K, self.n_pad)
        self.launches = 0

    def stage(self, R):
        """(int8 rows [R, K] to fill, int32 results [R, n_pad]) in the staging buffers."""
        assert R <= MAX_ROWS, f"{R} rows > XDNA_VLLM_NPU_ROWS={MAX_ROWS}"
        st = _staging
        R = -(-R // BLOCK_M) * BLOCK_M
        return (st["a"][: R * self.K].view(R, self.K), st["c"][: R * self.n_pad].view(R, self.n_pad))

    def launch_rows(self, r0, n=BLOCK_M):
        """One launch on staged rows [r0, r0 + n), n a multiple of BLOCK_M. No torch compute."""
        st = _staging
        K, Np = self.K, self.n_pad
        a = st["a"][r0 * K: (r0 + n) * K].view(n, K)
        c = st["c"][r0 * Np: (r0 + n) * Np].view(n, Np)
        _kernel[(n // BLOCK_M, Np // BLOCK_N)](a, self.w, c, Np, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                               BLOCK_K=BLOCK_K)
        self.launches += 1

    def launch(self, R):
        """Run the GEMM on the first R staged rows (rounded up to BLOCK_M), 512 rows a launch."""
        for r0 in range(0, -(-R // BLOCK_M) * BLOCK_M, BLOCK_M):
            self.launch_rows(r0)

    def __call__(self, x):
        """bf16 x [M, K] -> bf16 [M, N] on the NPU alone (tests, benchmarks)."""
        runtime.use_npu()
        M = x.shape[0]
        a, c = self.stage(M)
        xf = x.float()
        sx = xf.abs().amax(-1).clamp(min=1e-8) / 127
        a[:M] = torch.round(xf / sx[:, None]).clamp(-127, 127)
        self.launch(M)
        return (c[:M, : self.N].float() * sx[:, None] * self.sw[None]).to(x.dtype)
