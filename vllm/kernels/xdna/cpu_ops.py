# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side helpers for NPU kernels (C++, built on first use with torch's cpp_extension).

The NPU does not snoop the CPU's caches: before the host reads what a kernel
wrote, the lines must be evicted, or stale copies are read (measured: 0.3-2 %
of a GEMM's results). XRT's sync does that on the launching thread, one line at
a time, and blocks the next launch -- for a 512 x 24576 int32 output under a
busy CPU, 1.5-2.5 ms a launch. `invalidate_dequant` does it on the caller's
OpenMP threads instead, a block of rows at a time, fused with reading the rows.
"""

import threading

import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <immintrin.h>

// out[r, :n] = bf16(c[r, :n] * sx[r] * sw[:n]) for the rows of c, after
// evicting each block of c's rows from the caches (clflushopt + mfence), so
// the loads see what the NPU wrote. c's rows must start on a cache line
// (row bytes a multiple of 64), as do resident staging buffers.
void invalidate_dequant(torch::Tensor c, int64_t n, torch::Tensor sx, torch::Tensor sw, torch::Tensor out) {
  TORCH_CHECK(c.scalar_type() == torch::kInt32 && c.stride(1) == 1 && (c.stride(0) * 4) % 64 == 0);
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.stride(1) == 1);
  TORCH_CHECK(sx.scalar_type() == torch::kFloat32 && sw.scalar_type() == torch::kFloat32 && sw.is_contiguous());
  const int64_t R = c.size(0), ldc = c.stride(0), ldo = out.stride(0), lds = sx.stride(0);
  const int32_t *C = c.data_ptr<int32_t>();
  const float *SX = sx.data_ptr<float>(), *SW = sw.data_ptr<float>();
  uint16_t *O = reinterpret_cast<uint16_t *>(out.data_ptr<at::BFloat16>());
  at::parallel_for(0, R, 4, [&](int64_t r0, int64_t r1) {
    const char *p = reinterpret_cast<const char *>(C + r0 * ldc);
    const char *end = reinterpret_cast<const char *>(C + r1 * ldc);
    for (; p < end; p += 64) _mm_clflushopt(const_cast<char *>(p));
    _mm_mfence();
    for (int64_t r = r0; r < r1; ++r) {
      const int32_t *cr = C + r * ldc;
      uint16_t *orow = O + r * ldo;
      const __m512 s = _mm512_set1_ps(SX[r * lds]);
      int64_t j = 0;
      for (; j + 16 <= n; j += 16) {
        __m512 v = _mm512_cvtepi32_ps(_mm512_loadu_si512(cr + j));
        v = _mm512_mul_ps(_mm512_mul_ps(v, s), _mm512_loadu_ps(SW + j));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(orow + j), (__m256i)_mm512_cvtneps_pbh(v));
      }
      for (; j < n; ++j) {
        at::BFloat16 b(static_cast<float>(cr[j]) * SX[r * lds] * SW[j]);
        orow[j] = b.x;
      }
    }
  });
}

// Evict t's bytes from the caches (before reading what the NPU wrote there).
void invalidate(torch::Tensor t) {
  TORCH_CHECK(t.is_contiguous());
  const int64_t nbytes = t.numel() * t.element_size();
  const char *base = reinterpret_cast<const char *>(t.data_ptr());
  at::parallel_for(0, (nbytes + 63) / 64, 4096, [&](int64_t l0, int64_t l1) {
    for (int64_t l = l0; l < l1; ++l) _mm_clflushopt(const_cast<char *>(base + l * 64));
    _mm_mfence();
  });
}

TORCH_LIBRARY(xdna_cpu, m) {
  m.def("invalidate_dequant(Tensor c, int n, Tensor sx, Tensor sw, Tensor(a!) out) -> ()");
  m.def("invalidate(Tensor t) -> ()");
}
TORCH_LIBRARY_IMPL(xdna_cpu, CPU, m) {
  m.impl("invalidate_dequant", &invalidate_dequant);
  m.impl("invalidate", &invalidate);
}
"""

_lock = threading.Lock()
_loaded = False


def load():
    """Build (once, cached by torch under ~/.cache/torch_extensions) and register torch.ops.xdna_cpu."""
    global _loaded
    with _lock:
        if not _loaded:
            from torch.utils.cpp_extension import load_inline

            load_inline("xdna_cpu_ops", cpp_sources=[_SRC], functions=None, is_python_module=False,
                        extra_cflags=["-O3", "-mavx512f", "-mavx512bf16", "-mclflushopt", "-fopenmp"],
                        extra_ldflags=["-fopenmp"])
            _loaded = True


def invalidate_dequant(c, n, sx, sw, out):
    """out[:, :n] = bf16(c[:, :n] * sx * sw[:n]), reading c only after evicting its lines."""
    torch.ops.xdna_cpu.invalidate_dequant(c, n, sx, sw, out)


def invalidate(t):
    """Evict a contiguous tensor's lines, so the next reads see what the NPU wrote."""
    torch.ops.xdna_cpu.invalidate(t)
