# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-XDNA (third_party/triton-xdna), once per process.

Kernels in this package are ordinary @triton.jit functions; Triton-XDNA
compiles them for the XDNA2 NPU (Triton -> Linalg -> MLIR-AIR -> AIE ELF) and
launches them through XRT. This module imports it lazily, so vLLM runs without
it until an NPU kernel is first used, and hands out what the kernels share:
the NPU driver (Triton's active driver is per thread in Triton-XDNA) and the
resident-tensor allocator.
"""

import os
import threading

CACHE = os.path.expanduser(os.environ.get("XDNA_VLLM_CACHE", "~/.cache/xdna_vllm"))

_lock = threading.Lock()
_state: dict = {}


def init() -> dict:
    """Import Triton-XDNA and create the NPU driver, once per process."""
    if _state:
        return _state
    with _lock:
        if _state:
            return _state
        import triton
        import triton.language as tl
        from triton.backends.amd_triton_npu import resident
        from triton.backends.amd_triton_npu.driver import NPUDriver

        os.makedirs(CACHE, exist_ok=True)
        # Intermediate IR goes here, not into whatever the server's cwd is.
        os.environ.setdefault("AMD_TRITON_NPU_AIR_PROJECT_PATH", os.path.join(CACHE, "air_project"))
        _state.update(triton=triton, tl=tl, resident=resident, driver=NPUDriver())
    return _state


def use_npu() -> None:
    """Make the NPU Triton's active driver in this thread (call in every launching thread)."""
    s = init()
    s["triton"].runtime.driver.set_active(s["driver"])


def schedule(name: str, text) -> str:
    """Write a transform schedule (an MLIR-AIR tiling script) to the cache once; its path.

    `text` is called only when the file is missing.
    """
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        with open(path + ".tmp", "w") as f:
            f.write(text())
        os.replace(path + ".tmp", path)
    return path


def resident_bytes() -> dict:
    return _state["resident"].usage() if _state else {}
