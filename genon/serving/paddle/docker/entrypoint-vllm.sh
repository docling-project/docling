#!/usr/bin/env bash
set -euo pipefail
echo "[vLLM] Serving model=${MODEL_NAME} on :${VLLM_PORT}"
exec paddlex genai vllm serve \
  --model "${MODEL_NAME}" \
  --port "${VLLM_PORT}" \
  --host 0.0.0.0
