# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels for the AMD XDNA2 NPU, written in Triton and compiled by Triton-XDNA.

This package is the front end: the @triton.jit functions, their tiling
choices, and the Python that stages data and launches them. The compiler and
the runtime underneath (third_party/triton-xdna) are the back end. A new NPU
kernel goes here; whatever it needs from the compiler or the launcher goes
into Triton-XDNA. See docs/xdna/kernels.md.

Nothing here imports Triton-XDNA at import time.
"""
