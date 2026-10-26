#!/bin/bash
# Gemma 4 E2B with xdna_int8 (int8 linears; big prefill GEMMs split with the
# XDNA2 NPU). OpenAI API on 127.0.0.1:${PORT:-8013}.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-3}   # GB; ~500k tokens for Gemma 4 E2B
# Under memory pressure, earlyoom should pick this before desktop apps.
# --max-num-batched-tokens: `vllm serve` on CPU defaults to 2048-token prefill
# chunks; 4096 gives the NPU twice the rows per GEMM.
exec choom -n 800 -- vllm serve "${MODEL:-google/gemma-4-E2B-it}" \
    --host 127.0.0.1 --port "${PORT:-8013}" \
    --served-model-name "${SERVED_MODEL_NAME:-gemma-4-e2b-int8-npu}" \
    --quantization xdna_int8 --language-model-only --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    "$@"
