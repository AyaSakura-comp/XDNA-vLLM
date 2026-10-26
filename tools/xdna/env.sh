# Source me (bash): the XDNA-vLLM venv, XRT, the mlir-aie / mlir-air paths
# Triton-XDNA compiles with, and the preloads vLLM's CPU backend wants.
#
#   XDNA_VLLM_VENV   the venv (default: <repo>/.venv)
XDNA_VLLM_ROOT=${XDNA_VLLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
XDNA_VLLM_VENV=${XDNA_VLLM_VENV:-$XDNA_VLLM_ROOT/.venv}
export XDNA_VLLM_ROOT XDNA_VLLM_VENV
source "$XDNA_VLLM_VENV/bin/activate"
source /opt/xilinx/xrt/setup.sh >/dev/null
TRITON_XDNA_ENV_INSTALL=0 source "$XDNA_VLLM_ROOT/third_party/triton-xdna/utils/env_setup.sh" >/dev/null
# tcmalloc and Intel OpenMP, as vLLM's CPU worker expects. (With an allocator
# preloaded, Triton-XDNA loads mlir-air without RTLD_DEEPBIND, which would
# otherwise bind its free() past tcmalloc.)
export LD_PRELOAD="$XDNA_VLLM_ROOT/vllm/libs/libtcmalloc_minimal.so.4:$XDNA_VLLM_VENV/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
