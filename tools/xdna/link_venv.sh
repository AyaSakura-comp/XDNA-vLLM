#!/bin/bash
# Point a venv that has the vLLM 0.32.0 CPU wheel and Triton-XDNA installed at
# this checkout, so edits here take effect without rebuilding anything:
#
#   * the wheel's compiled parts (_C*.so, libs/, vllm-rs, _version.py) are
#     copied into vllm/ here (all git-ignored), and site-packages/vllm becomes
#     a symlink to vllm/ here;
#   * site-packages/triton/backends/amd_triton_npu becomes a symlink to the
#     Triton-XDNA submodule's backend (its Python only; a change to
#     Triton-XDNA's C++ needs `pip install third_party/triton-xdna` again).
#
# usage: tools/xdna/link_venv.sh [venv]      (default: .venv)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=$(realpath "${1:-$ROOT/.venv}")
SP=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

if [ ! -L "$SP/vllm" ]; then
    ls -d "$SP"/vllm-*+cpu.dist-info >/dev/null 2>&1 || { echo "$SP/vllm is not a vLLM CPU wheel" >&2; exit 1; }
    ls -d "$SP"/vllm-0.32.0+cpu.dist-info >/dev/null 2>&1 ||
        echo "warning: the wheel is not 0.32.0; its binaries may not match this tree" >&2
    for f in "$SP"/vllm/*.so "$SP/vllm/vllm-rs" "$SP/vllm/_version.py"; do cp -a "$f" "$ROOT/vllm/"; done
    mkdir -p "$ROOT/vllm/libs" && cp -a "$SP"/vllm/libs/* "$ROOT/vllm/libs/"
    mv "$SP/vllm" "$SP/vllm.wheel-backup"
    ln -s "$ROOT/vllm" "$SP/vllm"
    echo "vllm -> $ROOT/vllm (wheel kept at $SP/vllm.wheel-backup)"
fi

BE="$SP/triton/backends/amd_triton_npu"
if [ ! -L "$BE" ]; then
    [ -d "$BE" ] || { echo "Triton-XDNA is not installed in $VENV" >&2; exit 1; }
    mv "$BE" "$SP/triton/backends/.amd_triton_npu.installed"
    ln -s "$ROOT/third_party/triton-xdna/amd_triton_npu/backend" "$BE"
    echo "amd_triton_npu -> third_party/triton-xdna/amd_triton_npu/backend"
fi

# The out-of-tree plugin this replaces would register xdna_int8 twice.
"$VENV/bin/pip" uninstall -y -q xdna-vllm 2>/dev/null || true
